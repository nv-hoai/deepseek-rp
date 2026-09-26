"""Image handling for vision completions (pure extraction + fetching).

DeepSeek web chat supports images through a vision model: files are uploaded
via ``POST /api/v0/file/upload_file``, polled until ``SUCCESS``, then
referenced by id with ``model_type="vision"`` on completion. This module
extracts image references from the three agent protocols and resolves them
to upload-ready bytes. Network fetching (remote URLs) lives here too so the
server stays orchestration-only.
"""

from __future__ import annotations

import base64
import mimetypes
from typing import Any

from . import config

_EXTENSION_BY_MIME = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
    "image/gif": "gif",
    "image/heic": "heic",
    "image/heif": "heif",
}

_MIME_BY_EXTENSION = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "gif": "image/gif",
    "heic": "image/heic",
    "heif": "image/heif",
}


def is_data_uri(value: str) -> bool:
    return isinstance(value, str) and value.startswith("data:")


def decode_data_uri(uri: str) -> tuple[bytes, str]:
    """Split a ``data:<mime>;base64,<payload>`` URI into bytes + mime."""
    try:
        header, payload = uri.split(",", 1)
    except ValueError as e:
        raise ValueError("Malformed data URI (missing comma)") from e
    if ";base64" not in header:
        raise ValueError("Only base64 data URIs are supported")
    mime = header[len("data:"):].split(";")[0] or "image/png"
    if not mime.startswith("image/"):
        raise ValueError(f"Not an image data URI: {mime}")
    try:
        return base64.b64decode(payload, validate=True), mime
    except Exception as e:
        raise ValueError("Invalid base64 in data URI") from e


def extension_for(mime: str, fallback: str = "png") -> str:
    if mime in _EXTENSION_BY_MIME:
        return _EXTENSION_BY_MIME[mime]
    guessed = mimetypes.guess_extension(mime or "")
    if guessed:
        return guessed.lstrip(".")
    return fallback


def mime_for_extension(extension: str) -> str:
    return _MIME_BY_EXTENSION.get(
        (extension or "").lower().lstrip("."), "image/png")


def download_bytes(url: str,
                   limit: int = config.MAX_IMAGE_DOWNLOAD_BYTES
                   ) -> tuple[bytes, str | None]:
    """Fetch a remote image; returns ``(bytes, content_type|None)``."""
    from curl_cffi import requests as _requests

    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        raise ValueError(f"Unsupported image URL: {url[:80]!r}")
    try:
        response = _requests.get(
            url, impersonate=config.IMPERSONATE, timeout=30, stream=True,
            headers={"user-agent": config.USER_AGENT},
        )
    except Exception as e:
        raise ValueError(f"Could not download image: {e}") from e
    if response.status_code != 200:
        raise ValueError(
            f"Image download failed with HTTP {response.status_code}")
    content_type = (response.headers.get("content-type", "") or "").split(
        ";")[0].strip() or None
    chunks: list[bytes] = []
    total = 0
    try:
        for piece in response.iter_content(chunk_size=65536):
            if not piece:
                continue
            total += len(piece)
            if total > limit:
                raise ValueError(
                    f"Image exceeds download limit of {limit} bytes")
            chunks.append(piece if isinstance(piece, bytes) else bytes(piece))
    finally:
        try:
            response.close()
        except Exception:
            pass
    data = b"".join(chunks)
    if not data:
        raise ValueError("Downloaded image is empty")
    return data, content_type


def resolve_image(url: str, index: int = 0) -> tuple[bytes, str, str]:
    """Resolve a URL or data URI to ``(bytes, filename, content_type)``."""
    if is_data_uri(url):
        data, mime = decode_data_uri(url)
        ext = extension_for(mime)
        return data, f"image-{index + 1}.{ext}", mime
    data, content_type = download_bytes(url)
    mime = (content_type or "").split(";")[0].strip() or "image/png"
    if not mime.startswith("image/"):
        raise ValueError(f"URL did not return an image: {mime}")
    return data, f"image-{index + 1}.{extension_for(mime)}", mime


def extract_openai_images(messages: list[dict]) -> list[str]:
    """Collect image URLs/data URIs from OpenAI-style content parts."""
    found: list[str] = []
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            kind = part.get("type")
            if kind == "image_url":
                url = (part.get("image_url") or {}).get("url") \
                    if isinstance(part.get("image_url"), dict) \
                    else part.get("image_url")
                if isinstance(url, str) and url:
                    found.append(url)
            elif kind == "image":
                url = part.get("url") or (
                    part.get("source") or {}).get("url") \
                    if isinstance(part.get("source"), dict) else None
                if isinstance(url, str) and url:
                    found.append(url)
    return found


def extract_anthropic_images(system: Any,
                             messages: list[dict]) -> list[str]:
    """Collect image URLs/data URIs from Anthropic blocks."""
    found: list[str] = []
    blocks: list[Any] = []
    if isinstance(system, list):
        blocks.extend(system)
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, list):
            blocks.extend(content)
    for block in blocks:
        if not isinstance(block, dict) or block.get("type") != "image":
            continue
        source = block.get("source") or {}
        if not isinstance(source, dict):
            continue
        if source.get("type") == "base64":
            data = source.get("data", "")
            media = source.get("media_type", "image/png")
            if data:
                found.append(f"data:{media};base64,{data}")
        elif source.get("type") == "url":
            url = source.get("url", "")
            if url:
                found.append(url)
    return found


def extract_responses_images(input_items: list[dict]) -> list[str]:
    """Collect image URLs from Responses API input items."""
    found: list[str] = []
    for item in input_items or []:
        if not isinstance(item, dict):
            continue
        content = item.get("content", [])
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") not in ("input_image", "image_url", "image"):
                continue
            url = part.get("image_url", part.get("url"))
            if isinstance(url, dict):
                url = url.get("url")
            if isinstance(url, str) and url:
                found.append(url)
    return found
