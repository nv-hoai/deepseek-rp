"""OpenAI <-> DeepSeek translation (pure functions, no network).

DeepSeek web chat has no native function-calling, so ``tools`` are emulated:
the tool schemas are injected into the prompt and ``<tool_call>`` JSON blocks
in the reply are parsed back into OpenAI ``tool_calls``. The agent framework
executes tools and continues the loop (standard OpenAI contract).
"""

from __future__ import annotations

import json
import re
import time
import uuid

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
# OpenCode harness fallback: models with the harness format in their training
# data sometimes emit DSML instead of <tool_call>. Delimiters use fullwidth
# vertical bars (U+FF5C), e.g. <｜｜DSML｜｜ invoke name="write">.
_DSML_TAG = "｜｜DSML｜｜"
DSML_INVOKE_RE = re.compile(
    rf"<{_DSML_TAG}\s+invoke\s+name=\"([^\"]+)\"\s*>(.*?)"
    rf"</{_DSML_TAG}\s+invoke\s*>",
    re.DOTALL,
)
DSML_PARAM_RE = re.compile(
    rf"<{_DSML_TAG}\s+parameter\s+name=\"([^\"]+)\""
    rf"((?:\s+\w+=\"[^\"]*\")*)\s*>(.*?)"
    rf"</{_DSML_TAG}\s+parameter\s*>",
    re.DOTALL,
)
DSML_ATTR_RE = re.compile(r"(\w+)=\"([^\"]*)\"")
DSML_CALLS_RE = re.compile(
    rf"<{_DSML_TAG}\s+calls\s*>(.*?)</{_DSML_TAG}\s+calls\s*>",
    re.DOTALL,
)


def extract_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if not isinstance(part, dict):
                parts.append(str(part))
            elif part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif "text" in part and isinstance(part["text"], str):
                parts.append(part["text"])
            else:
                parts.append("[unsupported content omitted]")
        return "".join(parts)
    return str(content)


def messages_to_prompt(messages: list[dict]) -> str:
    blocks = []
    for msg in messages:
        role = msg.get("role", "user")
        text = extract_text(msg.get("content")).strip()
        if role == "system":
            blocks.append(f"System: {text}" if text else "System:")
        elif role == "assistant":
            calls = msg.get("tool_calls") or []
            if calls:
                summaries = []
                for call in calls:
                    fn = (call.get("function") or {})
                    summaries.append(
                        f"{fn.get('name', '?')}({fn.get('arguments', '{}')})")
                blocks.append(
                    f"Assistant (tool calls: {'; '.join(summaries)}): {text}")
            else:
                blocks.append(f"Assistant: {text}")
        elif role == "tool":
            label = msg.get("name") or msg.get("tool_call_id") or "tool"
            blocks.append(f"Tool result ({label}): {text}")
        elif role == "function":
            blocks.append(f"Function result ({msg.get('name', '')}): {text}")
        else:
            blocks.append(f"User: {text}")
    return "\n\n".join(blocks)


def _tool_names(tools: list[dict] | None) -> list[str]:
    names = []
    for tool in tools or []:
        fn = (tool.get("function") or {}) if isinstance(tool, dict) else {}
        name = fn.get("name")
        if name:
            names.append(name)
    return names


def build_tool_preamble(tools: list[dict] | None, tool_choice) -> str:
    if not tools or tool_choice == "none":
        return ""
    schemas = []
    for tool in tools:
        fn = tool.get("function", {}) if isinstance(tool, dict) else {}
        schemas.append({
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {"type": "object"}),
        })
    lines = [
        "You have access to these functions:",
        json.dumps(schemas, ensure_ascii=False),
        "To call a function, output EXACTLY one block per call, no other "
        "formatting around it:",
        '<tool_call>{"name": "<function-name>", '
        '"arguments": {<args-object>}}</tool_call>',
        "You may emit multiple <tool_call> blocks in one response. "
        "If no function is needed, answer normally without any <tool_call> block.",
        "Do NOT use any other format: no DSML/XML tags, no <invoke> or "
        "<parameter> blocks, no markdown code fences around calls.",
    ]
    if tool_choice == "required":
        lines.append("You MUST call at least one function in this turn.")
    elif isinstance(tool_choice, dict):
        fn = tool_choice.get("function", {})
        if fn.get("name"):
            lines.append(
                f"You MUST call the function \"{fn['name']}\" in this turn.")
    return "\n".join(lines)


def _new_call(name: str, arguments: dict) -> dict:
    return {
        "id": f"call_{uuid.uuid4().hex[:12]}",
        "name": name,
        "arguments": arguments,
    }


def _coerce_arguments(arguments) -> dict | None:
    if arguments is None:
        return {}
    if isinstance(arguments, str):
        if not arguments:
            return {}
        try:
            arguments = json.loads(arguments)
        except (json.JSONDecodeError, ValueError):
            return None
    return arguments if isinstance(arguments, dict) else None


def parse_tool_calls(text: str, tools: list[dict] | None
                     ) -> tuple[str, list[dict]]:
    """Split reply into ``(clean_text, tool_calls)``.

    Each call is ``{"id", "name", "arguments": dict}``. Unknown names and
    malformed JSON are dropped (blocks still stripped from the text).
    Falls back to DSML ``<invoke>`` blocks when no ``<tool_call>`` matched.
    """
    known = set(_tool_names(tools))
    calls = []

    def _replace(match: re.Match) -> str:
        raw = match.group(1)
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return ""
        if not isinstance(payload, dict):
            return ""
        name = payload.get("name")
        if not name or (known and name not in known):
            return ""
        arguments = _coerce_arguments(payload.get("arguments", {}))
        if arguments is None:
            return ""
        calls.append(_new_call(name, arguments))
        return ""

    clean_text = TOOL_CALL_RE.sub(_replace, text).strip()
    if calls:
        return clean_text, calls
    return parse_dsml_calls(text, tools)


def _dsml_unescape(value: str) -> str:
    return (value.replace("&lt;", "<").replace("&gt;", ">")
            .replace("&amp;", "&").replace("&quot;", "\""))


def parse_dsml_calls(text: str, tools: list[dict] | None
                     ) -> tuple[str, list[dict]]:
    """Fallback parser for OpenCode DSML harness blocks.

    Shape: ``<calls><invoke name="write"><parameter name="path" ...>v
    </parameter>...</invoke></calls>`` with fullwidth-bar delimiters.
    """
    known = set(_tool_names(tools))
    calls = []

    def _replace_invoke(match: re.Match) -> str:
        name, body = match.group(1), match.group(2)
        if known and name not in known:
            return ""
        arguments: dict = {}
        for param in DSML_PARAM_RE.finditer(body):
            pname, attrs, pvalue = param.group(1), param.group(2), param.group(3)
            is_string = dict(DSML_ATTR_RE.findall(attrs)).get("string") == "true"
            value = _dsml_unescape(pvalue.strip())
            if is_string:
                arguments[pname] = value
            else:
                try:
                    arguments[pname] = json.loads(value) if value else ""
                except (json.JSONDecodeError, ValueError):
                    arguments[pname] = value
        if known and name not in known:
            return ""
        calls.append(_new_call(name, arguments))
        return ""

    without_invokes = DSML_INVOKE_RE.sub(_replace_invoke, text)
    clean_text = DSML_CALLS_RE.sub("", without_invokes).strip()
    clean_text = re.sub(r"\n{3,}", "\n\n", clean_text)
    return clean_text, calls


def resolve_flags(model: str, extra: dict) -> tuple[bool, bool]:
    """Map OpenAI model name + extra body to DeepSeek flags."""
    thinking = extra.get("thinking_enabled")
    search = extra.get("search_enabled")
    name = (model or "").lower()
    if thinking is None:
        thinking = not ("chat" in name and "reason" not in name
                        and "r1" not in name and "think" not in name)
        if not name or name == "default":
            thinking = True
    if search is None:
        search = False
    return bool(thinking), bool(search)


def estimate_tokens(text: str) -> int:
    return max(0, len(text or "") // 4)


def openai_tool_calls(calls: list[dict]) -> list[dict]:
    return [{
        "id": call["id"],
        "type": "function",
        "function": {
            "name": call["name"],
            "arguments": json.dumps(call["arguments"], ensure_ascii=False),
        },
    } for call in calls]


def completion_response(request_id: str, created: int, model: str,
                        thinking: str, content: str | None,
                        tool_calls: list[dict],
                        prompt_tokens: int, completion_tokens: int) -> dict:
    message: dict = {"role": "assistant", "content": content}
    if thinking:
        message["reasoning_content"] = thinking
    if tool_calls:
        message["tool_calls"] = openai_tool_calls(tool_calls)
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": "tool_calls" if tool_calls else "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def sse_chunk(request_id: str, created: int, model: str,
              delta: dict, finish_reason: str | None = None) -> str:
    return "data: " + json.dumps({
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta,
                     "finish_reason": finish_reason}],
    }, ensure_ascii=False) + "\n\n"


def now_epoch() -> int:
    return int(time.time())


def new_request_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:12]}"
