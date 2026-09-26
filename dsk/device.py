"""Device-ID persistence (mirrors frontend localStorage ``deepseek-device-id:chat``)."""

from __future__ import annotations

import uuid
from pathlib import Path

PACKAGE_DEVICE_FILE = Path(__file__).parent / ".device_id"
HOME_DEVICE_FILE = Path.home() / ".deepseek_device_id"


def load_or_create_device_id(explicit: str | None = None) -> str:
    if explicit:
        uuid.UUID(explicit)
        return explicit
    for candidate in (PACKAGE_DEVICE_FILE, HOME_DEVICE_FILE):
        try:
            if candidate.exists():
                value = candidate.read_text().strip()
                uuid.UUID(value)
                return value
        except Exception:
            continue
    new_id = str(uuid.uuid4())
    for target in (PACKAGE_DEVICE_FILE, HOME_DEVICE_FILE):
        try:
            target.write_text(new_id)
            break
        except Exception:
            continue
    return new_id
