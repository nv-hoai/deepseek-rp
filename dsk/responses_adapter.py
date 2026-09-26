"""OpenAI Responses API translation (pure functions + response store).

Covers the subset Codex-style harnesses need: ``POST /v1/responses`` with
``input`` items, ``instructions``, function tools, ``tool_choice``,
``previous_response_id`` chaining, and streaming ``response.*`` events.
"""

from __future__ import annotations

import json
import time
import uuid

from . import openai_adapter as adapter


class ResponseStore:
    """In-memory chain memory for ``previous_response_id``."""

    def __init__(self) -> None:
        self._items: dict[str, list[dict]] = {}

    def save(self, response_id: str, items: list[dict]) -> None:
        self._items[response_id] = list(items)

    def chain(self, previous_response_id: str | None) -> list[dict]:
        if not previous_response_id:
            return []
        return list(self._items.get(previous_response_id, []))


def _item_text(item: dict) -> str:
    content = item.get("content", "")
    if isinstance(content, str):
        return content
    parts = []
    for part in content if isinstance(content, list) else []:
        if not isinstance(part, dict):
            parts.append(str(part))
        elif part.get("type") in ("input_text", "output_text", "text"):
            parts.append(part.get("text", ""))
        elif part.get("type") in ("input_image", "image_url", "image"):
            parts.append("[attached image]")
        else:
            parts.append(adapter.extract_text(part))
    return "".join(parts)


def to_openai_messages(instructions, input_items, chained: list[dict]
                       ) -> list[dict]:
    """Flatten instructions + chained history + input items to messages."""
    messages: list[dict] = []
    if instructions:
        messages.append({"role": "system", "content": str(instructions)})
    for item in chained + list(input_items or []):
        if not isinstance(item, dict):
            messages.append({"role": "user", "content": str(item)})
            continue
        kind = item.get("type", "message")
        if kind == "message":
            role = item.get("role", "user")
            messages.append({"role": role, "content": _item_text(item)})
        elif kind == "function_call_output":
            messages.append({
                "role": "tool",
                "tool_call_id": item.get("call_id", ""),
                "content": str(item.get("output", "")),
            })
        elif kind == "function_call":
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": item.get("call_id", item.get("id", "")),
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", "{}"),
                    },
                }],
            })
        elif kind == "reasoning":
            continue
        else:
            messages.append({"role": "user", "content": _item_text(item)})
    return messages


def to_openai_tools(tools: list[dict] | None) -> list[dict] | None:
    if not tools:
        return None
    converted = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        converted.append({
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {"type": "object"}),
            },
        })
    return converted or None


def extract_images(input_items: list[dict]) -> list[str]:
    """Collect image URLs/data URIs from Responses API input items."""
    from .images import extract_responses_images

    return extract_responses_images(input_items)


def to_openai_tool_choice(tool_choice) -> object:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        return tool_choice if tool_choice in (
            "auto", "none", "required") else None
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") in ("function", "function_call"):
            return {"type": "function",
                    "function": {"name": tool_choice.get("name", "")}}
        return None
    return None


def build_response(response_id: str, model: str, created_at: int,
                   thinking: str, content: str | None, calls: list[dict],
                   prompt_tokens: int, completion_tokens: int) -> dict:
    output: list[dict] = []
    if thinking:
        output.append({"type": "reasoning", "id": f"rs_{uuid.uuid4().hex[:12]}",
                       "summary": [{"type": "summary_text", "text": thinking}]})
    output.append({"type": "message", "id": f"msg_{uuid.uuid4().hex[:12]}",
                   "status": "completed", "role": "assistant",
                   "content": [{"type": "output_text", "text": content or "",
                                "annotations": []}]})
    for call in calls:
        output.append({"type": "function_call",
                       "id": f"fc_{uuid.uuid4().hex[:12]}",
                       "call_id": call["id"], "name": call["name"],
                       "arguments": json.dumps(call["arguments"],
                                               ensure_ascii=False),
                       "status": "completed"})
    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "model": model,
        "status": "completed",
        "output": output,
        "usage": {"input_tokens": prompt_tokens,
                  "output_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens},
    }


def stream_event(payload: dict) -> str:
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


def now_epoch() -> int:
    return int(time.time())


def new_response_id() -> str:
    return f"resp_{uuid.uuid4().hex[:12]}"
