"""Cookie-capture server. Loads a URL in Chromium (WAF-aware) and returns cookies."""

from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import time
from typing import Optional
from urllib.parse import urlparse

from DrissionPage import ChromiumPage
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from typing import Dict
import uvicorn

try:
    from pyvirtualdisplay import Display
except ImportError:
    Display = None

from .browser import make_chromium_options
from .waf import WafBypasser

DOCKER_MODE = os.getenv("DOCKERMODE", "false").lower() == "true"
SERVER_PORT = int(os.getenv("SERVER_PORT", 8000))

app = FastAPI()


class CookieResponse(BaseModel):
    cookies: Dict[str, str]
    user_agent: str


def is_safe_url(url: str) -> bool:
    parsed = urlparse(url)
    blocked = re.compile(
        r"^(127\.0\.0\.1|localhost|0\.0\.0\.0|::1|10\.\d+\.\d+\.\d+|"
        r"172\.1[6-9]\.\d+\.\d+|172\.2[0-9]\.\d+\.\d+|"
        r"172\.3[0-1]\.\d+\.\d+|192\.168\.\d+\.\d+)$"
    )
    if ((parsed.hostname and blocked.match(parsed.hostname))
            or parsed.scheme == "file"):
        return False
    return True


def load_page(url: str, retries: int = 5, proxy: Optional[str] = None,
              headless: bool = True) -> ChromiumPage:
    from . import config

    for attempt in range(3):
        driver = ChromiumPage(addr_or_opts=make_chromium_options(
            headless=headless, proxy=proxy))
        try:
            driver.get(url)
            time.sleep(5)
            try:
                body = driver.ele("tag:body", timeout=10)
                if len(body.html) <= 100:
                    raise RuntimeError("empty page")
            except Exception as e:
                raise RuntimeError(f"page did not load: {e}") from e
            WafBypasser(driver, retries, log=True).bypass()
            return driver
        except Exception:
            try:
                driver.quit()
            except Exception:
                pass
            if attempt >= 2:
                raise
            time.sleep(3)
    raise RuntimeError("unreachable")


def _cookies_dict(driver: ChromiumPage) -> dict[str, str]:
    return {c.get("name", ""): c.get("value", " ")
            for c in driver.cookies()}


@app.get("/cookies", response_model=CookieResponse)
async def get_cookies(url: str, retries: int = 5, proxy: str | None = None):
    from . import config

    if not is_safe_url(url):
        raise HTTPException(status_code=400, detail="Invalid URL")
    driver = None
    try:
        driver = load_page(url, retries, proxy)
        try:
            user_agent = driver.user_agent
        except Exception:
            user_agent = config.USER_AGENT
        return CookieResponse(cookies=_cookies_dict(driver),
                              user_agent=user_agent)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        try:
            if driver:
                driver.quit()
        except Exception:
            pass


@app.get("/html")
async def get_html(url: str, retries: int = 5, proxy: str | None = None):
    from . import config

    if not is_safe_url(url):
        raise HTTPException(status_code=400, detail="Invalid URL")
    driver = None
    try:
        driver = load_page(url, retries, proxy)
        response = Response(content=driver.html, media_type="text/html")
        response.headers["cookies"] = json.dumps(_cookies_dict(driver))
        try:
            response.headers["user_agent"] = driver.user_agent
        except Exception:
            response.headers["user_agent"] = config.USER_AGENT
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        try:
            if driver:
                driver.quit()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WAF-aware cookie server")
    parser.add_argument("--nolog", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--port", type=int, default=SERVER_PORT)
    args = parser.parse_args()

    if (args.headless or DOCKER_MODE) and Display is not None:
        display = Display(visible=0, size=(1920, 1080))
        display.start()
        atexit.register(display.stop)

    uvicorn.run(app, host="0.0.0.0", port=args.port)
