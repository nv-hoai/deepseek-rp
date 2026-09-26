"""Public client facade. Thin orchestration over transport/PoW/SSE."""

from __future__ import annotations

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
from .transport import HttpTransport, check_http_error, is_waf_challenge, warn_version_once


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

    def get_pow_challenge(self) -> PowChallenge:
        headers = build_headers(self.auth_token, self.device_id)
        payload = self.transport.request(
            "POST", config.ENDPOINT_POW_CHALLENGE, headers,
            {"target_path": config.POW_TARGET_PATH},
        )
        try:
            return PowChallenge.from_dict(
                payload["data"]["biz_data"]["challenge"])
        except KeyError as e:
            raise APIError(
                "Invalid challenge response format from server") from e

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
    ) -> Generator[Chunk, None, None]:
        """Stream a completion. History requires ``parent_message_id`` (int)."""
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
        )
        challenge = self.get_pow_challenge()
        headers = build_headers(
            self.auth_token, self.device_id,
            pow_response=self.pow_solver.solve_challenge({
                "algorithm": challenge.algorithm,
                "challenge": challenge.challenge,
                "salt": challenge.salt,
                "signature": challenge.signature,
                "difficulty": challenge.difficulty,
                "expire_at": challenge.expire_at,
                "target_path": challenge.target_path,
            }),
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
        try:
            for raw in response.iter_lines():
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
