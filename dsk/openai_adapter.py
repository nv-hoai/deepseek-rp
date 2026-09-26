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


def _example_arguments(fn: dict) -> dict:
    """Minimal placeholder args from the schema's first required property."""
    params = fn.get("parameters", {}) or {}
    props = params.get("properties", {}) or {}
    required = params.get("required", []) or []
    if required and required[0] in props:
        return {required[0]: "..."}
    if props:
        first = next(iter(props))
        return {first: "..."}
    return {}


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
    first_name = schemas[0]["name"] if schemas else "example_function"
    example = {"name": first_name,
               "arguments": _example_arguments(
                   (tools[0].get("function", {})
                    if isinstance(tools[0], dict) else {}))}
    lines = [
        "You have access to these functions. Call them with <tool_call> blocks:",
        json.dumps(schemas, ensure_ascii=False),
        "Rules:",
        "1. One function call = one block, exactly like this example "
        "(replace values with real ones, keep the structure):",
        f"<tool_call>{json.dumps(example, ensure_ascii=False)}</tool_call>",
        "2. Emit each block separately. Do not nest blocks or wrap them "
        "in markdown code fences.",
        "3. Put any explanation BEFORE the blocks. Put nothing after "
        "the last block.",
        "4. Use only function names from the list above with arguments "
        "matching their schemas.",
        "5. Do NOT use any other format: no DSML/XML tags, no <invoke> or "
        "<parameter> blocks.",
        "6. If no function is needed, answer normally with no blocks.",
    ]
    if tool_choice == "required":
        lines.append("7. Your response MUST consist of at least one <tool_call> "
                     "block and nothing else.")
    elif isinstance(tool_choice, dict):
        fn = tool_choice.get("function", {})
        if fn.get("name"):
            lines.append(
                f"7. Your response MUST consist of a <tool_call> block for "
                f"\"{fn['name']}\" and nothing else.")
    return "\n".join(lines)


def _new_call(name: str, arguments: dict) -> dict:
    return {
        "id": f"call_{uuid.uuid4().hex[:12]}",
        "name": name,
        "arguments": arguments,
    }


def _scan_tool_call_blocks(text: str) -> list[tuple[int, int, Any]]:
    """Find ``<tool_call>`` JSON payloads via balanced decoding.

    Tolerates missing ``</tool_call>`` closers and trailing junk (e.g. model
    mixing DSML closers into the block). Returns ``(start, end, payload)``
    with ``payload=None`` for invalid blocks (still stripped from the text).
    """
    decoder = json.JSONDecoder()
    spans = []
    idx = 0
    while True:
        open_at = text.find("<tool_call>", idx)
        if open_at == -1:
            break
        start = open_at + len("<tool_call>")
        segment = text[start:]
        stripped = segment.lstrip()
        try:
            payload, end = decoder.raw_decode(stripped)
            end_abs = start + (len(segment) - len(stripped)) + end
            spans.append((open_at, end_abs, payload))
            idx = end_abs
            continue
        except ValueError:
            pass
        closer = text.find("</tool_call>", start)
        if closer == -1:
            idx = start
        else:
            spans.append((open_at, closer + len("</tool_call>"), None))
            idx = closer + len("</tool_call>")
    return spans


def _strip_stray_dsml(text: str) -> str:
    text = re.sub(r"</?｜｜DSML｜｜[^>]*>", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


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
    spans = []

    for open_at, end_abs, payload in _scan_tool_call_blocks(text):
        spans.append((open_at, end_abs))
        if not isinstance(payload, dict):
            continue
        name = payload.get("name")
        arguments = _coerce_arguments(payload.get("arguments", {}))
        if not name or (known and name not in known) or arguments is None:
            continue
        calls.append(_new_call(name, arguments))

    if calls:
        kept = []
        cursor = 0
        closer = "</tool_call>"
        for open_at, end_abs in spans:
            kept.append(text[cursor:open_at])
            cursor = end_abs
            tail = text[cursor:]
            stripped = tail.lstrip()
            if stripped.startswith(closer):
                cursor += len(tail) - len(stripped) + len(closer)
        kept.append(text[cursor:])
        return _strip_stray_dsml("".join(kept)), calls
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
    clean_text = DSML_CALLS_RE.sub("", without_invokes)
    return _strip_stray_dsml(clean_text), calls


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


def truncate_to_token_budget(text: str, max_tokens: int | None
                             ) -> tuple[str, bool]:
    """Truncate to ~``max_tokens``; return ``(text, truncated)``."""
    if not max_tokens or max_tokens <= 0:
        return text, False
    budget = max_tokens * 4
    if len(text) <= budget:
        return text, False
    return text[:budget], True


def contains_image(messages: list[dict]) -> bool:
    """Detect image content parts (not supported by DeepSeek web chat)."""
    for msg in messages:
        content = (msg.get("content") if isinstance(msg, dict) else None)
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in (
                        "image_url", "image"):
                    return True
    return False


def truncate_at_stop(text: str, stop) -> str:
    """Cut text at the first stop sequence (str or list of str)."""
    if not stop or not text:
        return text
    sequences = [stop] if isinstance(stop, str) else list(stop)
    cuts = [text.find(seq) for seq in sequences
            if isinstance(seq, str) and seq and seq in text]
    return text[:min(cuts)] if cuts else text


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
