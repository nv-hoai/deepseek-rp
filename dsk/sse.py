"""SSE parsing for ``POST /chat/completion`` (``text/event-stream``).

Wire format (reverse-engineered from ``main.*.js``):
- ``event: ready`` + ``data: {request_message_id, response_message_id, model_type}``
- bare ``data:`` lines are deltas: ``{o, p, v}`` (op/path/value) where ``o``/``p``
  inherit the previous values when omitted
- ``{v: {response: {message_id, fragments: [...]}}}`` replaces the fragment list
- ``{p: response/fragments, o: APPEND, v: [...]}`` appends fragments
- ``{p: response/fragments/<i>/content, v: str}`` streams token text
  (``-1`` = last fragment); ``o: BATCH`` carries status/counters, no text
- ``event: title|close|finish|hint|toast|...`` are metadata/terminal events

Fragment ``type`` mapping: ``RESPONSE`` -> ``text``; ``THINK`` -> ``thinking``;
``SEARCH``/``TOOL_*`` -> ``search``.
"""

from __future__ import annotations

import json
from typing import Any

from .exceptions import APIError
from .models import Chunk, Fragment


def fragment_to_chunk_type(fragment_type: str | None) -> str:
    if fragment_type in ("RESPONSE", "TEMPLATE_RESPONSE"):
        return "text"
    if fragment_type == "THINK":
        return "thinking"
    if fragment_type and "SEARCH" in fragment_type:
        return "search"
    if fragment_type in ("TOOL_OPEN", "TOOL_FIND", "TIP"):
        return "thinking"
    return "text"


class SseParser:
    """Stateful parser: feed one ``event:``/``data:`` line, get 0+ chunks."""

    def __init__(self) -> None:
        self.pending_event: str | None = None
        self.current_op = "SET"
        self.current_path = ""
        self.fragments: list[Fragment] = []

    def feed(self, raw_line: bytes | str) -> list[Chunk]:
        if not raw_line:
            return []
        line = (raw_line.decode("utf-8", "ignore")
                if isinstance(raw_line, bytes) else raw_line).strip()
        if not line:
            return []
        if line.startswith("event:"):
            self.pending_event = line[len("event:"):].strip()
            return []
        if not line.startswith("data:"):
            return []
        payload = line[len("data:"):].strip()
        if not payload:
            return []
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as e:
            raise APIError("Invalid JSON in response chunk") from e

        event = self.pending_event or "delta"
        self.pending_event = None

        if event == "ready":
            return [Chunk(
                type="ready", event="ready",
                request_message_id=data.get("request_message_id"),
                response_message_id=data.get("response_message_id"),
                message_id=data.get("response_message_id"),
                model_type=data.get("model_type"),
            )]
        if event in ("finish", "close"):
            return [Chunk(type="finish", event=event, finish_reason="stop")]
        if event == "title":
            return [Chunk(type="title", event="title",
                          content=data.get("content", "") or "")]
        if event == "update_session":
            return []
        if event in ("hint", "toast", "update_parent_message", "update_file"):
            content = data.get("content", "") if isinstance(data, dict) else ""
            return [Chunk(type=event, event=event, content=content or "")]
        return self._parse_delta(data)

    def _parse_delta(self, data: Any) -> list[Chunk]:
        if not isinstance(data, dict):
            return []
        op = data.get("o", self.current_op)
        path = data.get("p", self.current_path)
        value = data.get("v")
        self.current_op, self.current_path = op, path

        if isinstance(value, dict) and "response" in value and not path:
            return self._set_full_response(value["response"] or {})
        if path == "response/fragments" and op == "APPEND" and isinstance(value, list):
            return self._append_fragments(value)
        if op == "BATCH":
            return []
        if (isinstance(path, str) and path.endswith("/content")
                and isinstance(value, (str, int, float))):
            return [self._append_fragment_text(path, op, str(value))]
        return []

    def _set_full_response(self, response: dict[str, Any]) -> list[Chunk]:
        frags = response.get("fragments") or []
        self.fragments = [Fragment.from_dict(f) for f in frags
                          if isinstance(f, dict)]
        out = []
        for frag in self.fragments:
            if frag.content:
                out.append(Chunk(
                    type=fragment_to_chunk_type(frag.type),
                    content=frag.content, event="delta",
                    message_id=response.get("message_id"),
                    fragment_type=frag.type,
                ))
        return out

    def _append_fragments(self, values: list[Any]) -> list[Chunk]:
        out = []
        for item in values:
            if not isinstance(item, dict):
                continue
            frag = Fragment.from_dict(item)
            self.fragments.append(frag)
            if frag.content:
                out.append(Chunk(
                    type=fragment_to_chunk_type(frag.type),
                    content=frag.content, event="delta",
                    fragment_type=frag.type,
                ))
        return out

    def _append_fragment_text(self, path: str, op: str, text: str) -> Chunk:
        fragment_type: str | None = None
        try:
            parts = path.split("/")
            token = parts[2] if len(parts) >= 4 else "-1"
            index = int(token)
            if index == -1:
                index = len(self.fragments) - 1
            if 0 <= index < len(self.fragments):
                current = self.fragments[index]
                fragment_type = current.type
                updated = (current.content + text) if op == "APPEND" else text
                self.fragments[index] = Fragment(
                    type=current.type, content=updated, id=current.id)
            elif self.fragments:
                fragment_type = self.fragments[-1].type
        except Exception:
            if self.fragments:
                fragment_type = self.fragments[-1].type
        return Chunk(
            type=fragment_to_chunk_type(fragment_type),
            content=text, event="delta", fragment_type=fragment_type,
        )


def parse_single_line(chunk: bytes | str) -> Chunk | None:
    """Best-effort stateless parse (legacy ``choices/delta`` + new format)."""
    if not chunk:
        return None
    line = (chunk.decode("utf-8", "ignore")
            if isinstance(chunk, bytes) else chunk).strip()
    if line.startswith("event:"):
        return None
    if not line.startswith("data:"):
        return None
    try:
        data = json.loads(line[5:].strip())
    except json.JSONDecodeError as e:
        raise APIError("Invalid JSON in response chunk") from e
    if "choices" in data and data["choices"]:
        choice = data["choices"][0]
        delta = choice.get("delta", {})
        return Chunk(content=delta.get("content", ""),
                     type=delta.get("type", ""),
                     finish_reason=choice.get("finish_reason"))
    if isinstance(data, dict) and "v" in data:
        value = data["v"]
        if isinstance(value, dict) and "response" in value:
            frags = (value["response"] or {}).get("fragments") or []
            texts = [f.get("content", "") for f in frags if f.get("content")]
            if texts:
                first = frags[0].get("type") if frags else None
                return Chunk(
                    content="".join(texts),
                    type="text" if first == "RESPONSE" else "thinking",
                )
        if isinstance(value, str):
            return Chunk(content=value, type="thinking")
    return None
