"""Public client facade. Thin orchestration over transport/PoW/SSE."""

from __future__ import annotations

import itertools
import json
from collections.abc import Generator

from curl_cffi import requests

from . import config
from .cookies import CookieStore
from .device import load_or_create_device_id
from .exceptions import APIError, AuthenticationError, NetworkError, WafError
from .headers import build_headers
from .models import ChatRequest, Chunk, PowChallenge
from .pow import DeepSeekPOW
from .sse import SseParser
from .transport import (
    HttpTransport,
    check_biz_error,
    check_http_error,
    is_waf_challenge,
    warn_version_once,
)


class DeepSeekClient:
    """Entry point: sessions + streaming chat completion."""

    def __init__(
        self,
        auth_token: str,
        device_id: str | None = None,
        transport: HttpTransport | None = None,
        pow_solver: DeepSeekPOW | None = None,
        cookies: dict[str, str] | None = None,
    ) -> None:
        if not auth_token or not isinstance(auth_token, str):
            raise AuthenticationError("Invalid auth token provided")
        warn_version_once()
        self.auth_token = auth_token
        self.device_id = load_or_create_device_id(device_id)
        self.cookie_store = CookieStore()
        if cookies is not None:
            self.cookie_store.cookies = dict(cookies)
        self.transport = transport or HttpTransport(cookies=self.cookie_store.cookies)
        # Keep transport cookies in sync with the store.
        self.transport.cookies = self.cookie_store.cookies
        self.pow_solver = pow_solver or DeepSeekPOW()

    def refresh_cookies(self) -> None:
        """Re-run the cookie helper and reload the store (WAF recovery)."""
        from .bypass import refresh_cookies_via_server

        cookies, _ = refresh_cookies_via_server()
        self.cookie_store.save(cookies)
        self.transport.cookies = self.cookie_store.cookies

    def get_pow_challenge(
        self,
        target_path: str | None = None,
    ) -> PowChallenge:
        headers = build_headers(self.auth_token, self.device_id)
        payload = self.transport.request(
            "POST", config.ENDPOINT_POW_CHALLENGE, headers,
            {"target_path": target_path or config.POW_TARGET_PATH},
        )
        try:
            return PowChallenge.from_dict(
                payload["data"]["biz_data"]["challenge"])
        except KeyError as e:
            raise APIError(
                "Invalid challenge response format from server") from e

    def _solve_for(self, challenge: PowChallenge) -> str:
        return self.pow_solver.solve_challenge({
            "algorithm": challenge.algorithm,
            "challenge": challenge.challenge,
            "salt": challenge.salt,
            "signature": challenge.signature,
            "difficulty": challenge.difficulty,
            "expire_at": challenge.expire_at,
            "target_path": challenge.target_path,
        })

    def upload_file(self, file_bytes: bytes, filename: str,
                    content_type: str = "image/png") -> dict:
        """Upload a file; returns the ``biz_data`` record (with ``id``).

        The record starts as ``PENDING``; use :meth:`wait_for_files` before
        referencing the id in a completion.
        """
        if not file_bytes:
            raise ValueError("Cannot upload empty file")
        if len(file_bytes) > config.MAX_UPLOAD_BYTES:
            raise ValueError(
                f"File too large: {len(file_bytes)} bytes "
                f"(limit {config.MAX_UPLOAD_BYTES})")
        challenge = self.get_pow_challenge(config.POW_UPLOAD_PATH)
        headers = build_headers(
            self.auth_token, self.device_id,
            pow_response=self._solve_for(challenge))
        headers.pop("content-type", None)  # multipart sets its own boundary
        try:
            payload = self.transport.upload(
                config.ENDPOINT_UPLOAD, headers,
                file_bytes, filename, content_type)
        except WafError:
            self.refresh_cookies()
            payload = self.transport.upload(
                config.ENDPOINT_UPLOAD, headers,
                file_bytes, filename, content_type)
        try:
            return payload["data"]["biz_data"]
        except KeyError as e:
            raise APIError(
                "Invalid upload response format from server") from e

    def fetch_files(self, file_ids: list[str]) -> list[dict]:
        """Return file records (status, dimensions, audit) for ids."""
        if not file_ids:
            return []
        headers = build_headers(self.auth_token, self.device_id)
        payload = self.transport.get(
            config.ENDPOINT_FETCH_FILES, headers,
            {"file_ids": list(file_ids)})
        try:
            return payload["data"]["biz_data"]["files"]
        except KeyError as e:
            raise APIError(
                "Invalid fetch_files response format from server") from e

    def wait_for_files(self, file_ids: list[str],
                       timeout: float = config.FILE_POLL_TIMEOUT) -> list[dict]:
        """Poll until every file reports ``SUCCESS``; raise on failure."""
        import time as _time

        deadline = _time.monotonic() + timeout
        last: list[dict] = []
        while True:
            last = self.fetch_files(file_ids)
            by_id = {f.get("id"): f for f in last if isinstance(f, dict)}
            pending = [fid for fid in file_ids if by_id.get(fid, {}).get(
                "status") not in ("SUCCESS", "FAILED", "ERROR")]
            failed = [fid for fid in file_ids if by_id.get(fid, {}).get(
                "status") in ("FAILED", "ERROR")]
            if failed:
                raise APIError(f"File processing failed: {failed}")
            if not pending and len(by_id) == len(file_ids):
                return last
            if _time.monotonic() >= deadline:
                raise APIError(
                    f"Timed out waiting for files to process: {pending}")
            _time.sleep(config.FILE_POLL_INTERVAL)

    def create_chat_session(self) -> str:
        """Create a session; frontend sends ``{}`` and reads nested id."""
        headers = build_headers(self.auth_token, self.device_id)
        try:
            payload = self.transport.request(
                "POST", config.ENDPOINT_SESSION_CREATE, headers, {})
        except WafError:
            self.refresh_cookies()
            payload = self.transport.request(
                "POST", config.ENDPOINT_SESSION_CREATE, headers, {})
        try:
            biz = payload["data"]["biz_data"]
            if isinstance(biz, dict) and "chat_session" in biz:
                return biz["chat_session"]["id"]
            return biz["id"]
        except KeyError as e:
            raise APIError(
                "Invalid session creation response format from server") from e

    def chat_completion(
        self,
        chat_session_id: str,
        prompt: str,
        parent_message_id: int | None = None,
        thinking_enabled: bool = True,
        search_enabled: bool = False,
        model_type: str | None = "default",
        source: str | None = None,
        action: str | None = None,
        preempt: bool = False,
        ref_file_ids: list[str] | None = None,
    ) -> Generator[Chunk, None, None]:
        """Stream a completion. History requires ``parent_message_id`` (int).

        Pass ``model_type="vision"`` with ``ref_file_ids`` to ask about
        uploaded images (see :meth:`upload_file`).
        """
        request = ChatRequest(
            chat_session_id=chat_session_id,
            prompt=prompt,
            parent_message_id=parent_message_id,
            model_type=model_type,
            thinking_enabled=thinking_enabled,
            search_enabled=search_enabled,
            source=source,
            action=action,
            preempt=preempt,
            ref_file_ids=list(ref_file_ids or []),
        )
        challenge = self.get_pow_challenge()
        headers = build_headers(
            self.auth_token, self.device_id,
            pow_response=self._solve_for(challenge),
            sse=True,
        )
        try:
            response = self.transport.stream_post(
                config.ENDPOINT_COMPLETION, headers, request.to_json())
        except WafError:
            self.refresh_cookies()
            response = self.transport.stream_post(
                config.ENDPOINT_COMPLETION, headers, request.to_json())

        if is_waf_challenge(response.status_code, response.headers,
                            getattr(response, "text", "")):
            raise WafError(
                "Blocked by AWS WAF during streaming. Refresh cookies and retry.")
        if response.status_code != 200:
            first = next(response.iter_lines(), b"")
            error_text = (first.decode("utf-8", "ignore")
                          if isinstance(first, bytes) else str(first))
            check_http_error(response.status_code, response.headers, error_text)

        parser = SseParser()
        lines = response.iter_lines()
        # A muted/banned account gets a bare JSON error envelope instead of
        # SSE (HTTP 200). Surface it instead of streaming empty silence.
        first = next(lines, b"")
        text = (first.decode("utf-8", "ignore")
                if isinstance(first, bytes) else str(first))
        if text and not text.startswith(("event:", "data:")):
            try:
                check_biz_error(json.loads(text),
                                f"POST {config.ENDPOINT_COMPLETION}")
            except json.JSONDecodeError:
                pass
        try:
            for raw in itertools.chain([first], lines):
                for chunk in parser.feed(raw):
                    yield chunk
                    if chunk.type == "finish":
                        return
        except requests.exceptions.RequestException as e:
            raise NetworkError(
                f"Network error occurred during streaming: {e}") from e

    def collect_text(self, chunks: Generator[Chunk, None, None]) -> dict[str, str]:
        """Convenience: accumulate a stream into thinking/search/text/title."""
        thinking, search, text, title = [], [], [], ""
        for chunk in chunks:
            if chunk.type == "thinking":
                thinking.append(chunk.content)
            elif chunk.type == "search":
                search.append(chunk.content)
            elif chunk.type == "text":
                text.append(chunk.content)
            elif chunk.type == "title":
                title = chunk.content
        return {"thinking": "".join(thinking), "search": "".join(search),
                "text": "".join(text), "title": title}
