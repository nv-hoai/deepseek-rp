"""Anthropic Messages API translation (pure functions, no network).

Claude Code and other Anthropic-native harnesses POST ``/v1/messages`` with
``tool_use``/``tool_result`` blocks. We convert to the OpenAI-ish shape used
by :mod:`dsk.openai_adapter`, run one DeepSeek turn, and convert back.
"""

from __future__ import annotations

import json
import time
import uuid

from . import openai_adapter as adapter


def to_openai_messages(system, messages: list[dict]) -> list[dict]:
    """Flatten Anthropic blocks to OpenAI-style messages."""
    openai_messages = []
    if system is not None:
        openai_messages.append({"role": "system",
                                "content": _blocks_to_text(system)})
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, str):
            openai_messages.append({"role": role, "content": content})
            continue
        text_parts, tool_uses, tool_results = [], [], []
        for block in content if isinstance(content, list) else []:
            if not isinstance(block, dict):
                text_parts.append(str(block))
                continue
            kind = block.get("type")
            if kind == "text":
                text_parts.append(block.get("text", ""))
            elif kind == "tool_use":
                tool_uses.append({
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(
                            block.get("input", {}), ensure_ascii=False),
                    },
                })
            elif kind == "tool_result":
                result = block.get("content", "")
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id", ""),
                    "content": _blocks_to_text(result),
                })
            elif kind == "image":
                text_parts.append("[unsupported image content omitted]")
            else:
                text_parts.append(str(block))
        if role == "assistant":
            openai_messages.append({
                "role": "assistant",
                "content": "".join(text_parts),
                **({"tool_calls": tool_uses} if tool_uses else {}),
            })
        else:
            if text_parts:
                openai_messages.append({"role": role,
                                        "content": "".join(text_parts)})
            openai_messages.extend(tool_results)
    return openai_messages


def _blocks_to_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, dict) and block.get("type") == "image":
                parts.append("[unsupported image content omitted]")
            else:
                parts.append(adapter.extract_text(block))
        return "".join(parts)
    return str(content)


def to_openai_tools(tools: list[dict] | None) -> list[dict] | None:
    if not tools:
        return None
    converted = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        converted.append({
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema",
                                       {"type": "object"}),
            },
        })
    return converted or None


def to_openai_tool_choice(tool_choice) -> object:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, dict):
        kind = tool_choice.get("type")
        if kind == "none":
            return "none"
        if kind == "any":
            return "required"
        if kind == "tool":
            return {"type": "function",
                    "function": {"name": tool_choice.get("name", "")}}
        return None
    return tool_choice


def has_image(messages: list[dict], system) -> bool:
    def _scan(content) -> bool:
        if isinstance(content, list):
            return any(isinstance(b, dict) and b.get("type") == "image"
                       for b in content)
        return False

    if _scan(system):
        return True
    return any(_scan(m.get("content")) for m in messages
               if isinstance(m, dict))


def apply_max_tokens(text: str, max_tokens: int | None) -> tuple[str, bool]:
    """Truncate to ~``max_tokens``; return ``(text, truncated)``."""
    return adapter.truncate_to_token_budget(text, max_tokens)


def build_response(model: str, thinking: str, content: str | None,
                   calls: list[dict], prompt_tokens: int,
                   completion_tokens: int, truncated: bool = False) -> dict:
    blocks: list[dict] = []
    if thinking:
        blocks.append({"type": "thinking", "thinking": thinking})
    blocks.append({"type": "text", "text": content or ""})
    for call in calls:
        blocks.append({"type": "tool_use", "id": call["id"],
                       "name": call["name"], "input": call["arguments"]})
    stop_reason = "tool_use" if calls else "end_turn"
    if truncated:
        stop_reason = "max_tokens"
    return {
        "id": f"msg_{uuid.uuid4().hex[:12]}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": prompt_tokens,
                  "output_tokens": completion_tokens},
    }


def now_epoch() -> int:
    return int(time.time())
