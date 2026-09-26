"""HTTP transport over curl-cffi. Owns status/WAF/biz-code mapping."""

from __future__ import annotations

import json
import sys
from typing import Any

from curl_cffi import requests

from . import config
from .exceptions import (
    APIError,
    AuthenticationError,
    NetworkError,
    RateLimitError,
    WafError,
)

AUTH_CODES = (40002, 40003)


def is_waf_challenge(status_code: int, headers: Any, body_text: str) -> bool:
    headers = headers or {}
    if status_code == 202 and headers.get("x-amzn-waf-action") == "challenge":
        return True
    if status_code == 405 and headers.get("x-amzn-waf-action") == "captcha":
        return True
    if "<!DOCTYPE html>" in body_text and (
        "Just a moment" in body_text
        or "awsWaf" in body_text
        or "AwsWafIntegration" in body_text
    ):
        return True
    return False


def check_biz_error(payload: Any, context: str = "") -> None:
    if not isinstance(payload, dict):
        return
    code = payload.get("code")
    if code not in (None, 0):
        msg = payload.get("msg", "")
        if code in AUTH_CODES:
            raise AuthenticationError(f"Invalid or expired token: {msg} ({code})")
        raise APIError(f"{context} failed: {msg} (code={code})")
    data = payload.get("data")
    if isinstance(data, dict):
        biz_code = data.get("biz_code", 0)
        if biz_code not in (None, 0):
            msg = data.get("biz_msg", "")
            if biz_code in AUTH_CODES:
                raise AuthenticationError(
                    f"Invalid or expired token: {msg} ({biz_code})")
            if biz_code == 40029:
                raise APIError(f"IP restricted: {msg} ({biz_code})")
            raise APIError(f"{context} failed: {msg} (biz_code={biz_code})")


def check_http_error(status_code: int, headers: Any, body_text: str) -> None:
    if status_code == 401:
        raise AuthenticationError("Invalid or expired authentication token")
    if status_code == 429:
        raise RateLimitError("API rate limit exceeded")
    if status_code == 403 and "cf-mitigated" in (headers or {}):
        raise WafError(f"Cloudflare challenge: {body_text[:300]}")
    if status_code >= 500:
        raise APIError(f"Server error occurred: {body_text}", status_code)
    if status_code != 200:
        raise APIError(f"API request failed: {body_text}", status_code)


def warn_version_once() -> None:
    try:
        from importlib.metadata import version, PackageNotFoundError
        found = version("curl-cffi")
        if found != config.CURL_CFFI_TESTED_VERSION:
            print(
                f"Warning: tested with curl-cffi {config.CURL_CFFI_TESTED_VERSION}, "
                f"found {found}",
                file=sys.stderr,
            )
    except Exception:
        print("Warning: curl-cffi not found. Install requirements.txt",
              file=sys.stderr)


class HttpTransport:
    """Thin wrapper so the client stays testable without network."""

    def __init__(self, cookies: dict[str, str] | None = None):
        self.cookies = cookies or {}

    def request(self, method: str, endpoint: str,
                headers: dict[str, str], json_data: dict[str, Any]) -> dict[str, Any]:
        url = f"{config.BASE_URL}{endpoint}"
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                json=json_data,
                cookies=self.cookies,
                impersonate=config.IMPERSONATE,
                timeout=None,
            )
        except requests.exceptions.RequestException as e:
            raise NetworkError(f"Network error occurred: {e}") from e

        body = response.text
        if is_waf_challenge(response.status_code, response.headers, body):
            raise WafError("Blocked by WAF protection.")
        check_http_error(response.status_code, response.headers, body)
        try:
            payload = response.json()
        except json.JSONDecodeError as e:
            raise APIError("Invalid JSON response from server") from e
        check_biz_error(payload, f"{method} {endpoint}")
        return payload

    def stream_post(self, endpoint: str, headers: dict[str, str],
                    json_data: dict[str, Any]):
        url = f"{config.BASE_URL}{endpoint}"
        try:
            return requests.post(
                url,
                headers=headers,
                json=json_data,
                cookies=self.cookies,
                impersonate=config.IMPERSONATE,
                stream=True,
                timeout=None,
            )
        except requests.exceptions.RequestException as e:
            raise NetworkError(
                f"Network error occurred during streaming: {e}") from e

    def get(self, endpoint: str, headers: dict[str, str],
            params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{config.BASE_URL}{endpoint}"
        try:
            response = requests.get(
                url,
                headers=headers,
                params=params,
                cookies=self.cookies,
                impersonate=config.IMPERSONATE,
                timeout=30,
            )
        except requests.exceptions.RequestException as e:
            raise NetworkError(f"Network error occurred: {e}") from e

        body = response.text
        if is_waf_challenge(response.status_code, response.headers, body):
            raise WafError("Blocked by WAF protection.")
        check_http_error(response.status_code, response.headers, body)
        try:
            payload = response.json()
        except json.JSONDecodeError as e:
            raise APIError("Invalid JSON response from server") from e
        check_biz_error(payload, f"GET {endpoint}")
        return payload

    def upload(self, endpoint: str, headers: dict[str, str],
               file_bytes: bytes, filename: str,
               content_type: str) -> dict[str, Any]:
        """Multipart file upload via curl mime (curl-cffi has no files=)."""
        from curl_cffi.curl import CurlMime

        url = f"{config.BASE_URL}{endpoint}"
        mime = CurlMime.from_list([{
            "name": "file",
            "filename": filename,
            "content_type": content_type,
            "data": file_bytes,
        }])
        try:
            response = requests.post(
                url,
                headers=headers,
                multipart=mime,
                cookies=self.cookies,
                impersonate=config.IMPERSONATE,
                timeout=120,
            )
        except requests.exceptions.RequestException as e:
            raise NetworkError(f"Network error occurred: {e}") from e
        finally:
            mime.close()

        body = response.text
        if is_waf_challenge(response.status_code, response.headers, body):
            raise WafError("Blocked by WAF protection.")
        check_http_error(response.status_code, response.headers, body)
        try:
            payload = response.json()
        except json.JSONDecodeError as e:
            raise APIError("Invalid JSON response from server") from e
        check_biz_error(payload, f"POST {endpoint}")
        return payload
