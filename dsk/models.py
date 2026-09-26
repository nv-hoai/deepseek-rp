"""Typed domain models. Raw frontend JSON is converted here, nowhere else."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class PowChallenge:
    algorithm: str
    challenge: str
    salt: str
    signature: str
    difficulty: int
    expire_at: int
    target_path: str
    expire_after: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PowChallenge":
        return cls(
            algorithm=data["algorithm"],
            challenge=data["challenge"],
            salt=data["salt"],
            signature=data["signature"],
            difficulty=int(data["difficulty"]),
            expire_at=int(data["expire_at"]),
            target_path=data["target_path"],
            expire_after=data.get("expire_after"),
        )


ChunkType = Literal[
    "thinking", "text", "search",
    "ready", "title", "finish",
    "hint", "toast", "update_parent_message", "update_file",
]

FragmentType = Literal[
    "RESPONSE", "TEMPLATE_RESPONSE", "THINK",
    "SEARCH", "TOOL_SEARCH", "TOOL_OPEN", "TOOL_FIND", "TIP",
]


@dataclass(frozen=True)
class Fragment:
    type: str
    content: str = ""
    id: Any = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Fragment":
        return cls(
            type=data.get("type", ""),
            content=data.get("content") or "",
            id=data.get("id"),
        )


@dataclass(frozen=True)
class Chunk:
    """One streaming unit. Body text is in ``content``; metadata in the rest."""

    type: str
    content: str = ""
    event: str = "delta"
    finish_reason: str | None = None
    message_id: int | None = None
    fragment_type: str | None = None
    request_message_id: int | None = None
    response_message_id: int | None = None
    model_type: str | None = None

    def is_body(self) -> bool:
        return self.type in ("thinking", "text", "search") and bool(self.content)


@dataclass(frozen=True)
class ChatRequest:
    chat_session_id: str
    prompt: str
    parent_message_id: int | None = None
    model_type: str | None = "default"
    thinking_enabled: bool = True
    search_enabled: bool = False
    source: str | None = None
    action: str | None = None
    preempt: bool = False
    ref_file_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.prompt or not isinstance(self.prompt, str):
            raise ValueError("Prompt must be a non-empty string")
        if not self.chat_session_id or not isinstance(self.chat_session_id, str):
            raise ValueError("Chat session ID must be a non-empty string")

    def to_json(self) -> dict[str, Any]:
        return {
            "chat_session_id": self.chat_session_id,
            "parent_message_id": self.parent_message_id,
            "model_type": self.model_type,
            "prompt": self.prompt,
            "ref_file_ids": self.ref_file_ids,
            "thinking_enabled": self.thinking_enabled,
            "search_enabled": self.search_enabled,
            "source": self.source,
            "action": self.action,
            "preempt": self.preempt,
        }
