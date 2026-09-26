"""Header construction. All frontend-mirroring header values come from config."""

from __future__ import annotations

import time

from . import config


def local_timezone_offset_seconds() -> int:
    # Browser sends seconds east of UTC (e.g. 25200 for UTC+7).
    return -time.timezone if time.daylight == 0 else -time.altzone


def build_headers(
    auth_token: str,
    device_id: str,
    pow_response: str | None = None,
    sse: bool = False,
) -> dict[str, str]:
    headers = {
        "accept": "text/event-stream" if sse else "*/*",
        "accept-language": config.ACCEPT_LANGUAGE,
        "authorization": f"Bearer {auth_token}",
        "content-type": "application/json",
        "origin": config.ORIGIN,
        "referer": config.REFERER,
        "user-agent": config.USER_AGENT,
        "x-client-locale": config.CLIENT_LOCALE,
        "x-client-platform": config.CLIENT_PLATFORM,
        "x-client-version": config.CLIENT_VERSION,
        "x-client-bundle-id": config.CLIENT_BUNDLE_ID,
        "x-client-timezone-offset": str(local_timezone_offset_seconds()),
        "x-device-id": device_id,
        "x-device-model": "",
    }
    if pow_response:
        headers[config.POW_HEADER] = pow_response
    return headers
