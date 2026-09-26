"""Email/password login. Returns the ``userToken`` value.

Quirks (reverse-engineered):
- Spoof Chrome 132 UA (CloudFront 403 on new Chromium).
- Use a fresh browser profile per login.
- Patch ``SMSdk.ready`` (late registration never fires; without it the login
  POST sends ``device_id: null`` and fails validation).
"""

from __future__ import annotations

import json
import tempfile
import time

from .browser import make_chromium_options, patch_shumei_ready, wait_for_shumei


def login(email: str, password: str, timeout: int = 60,
          browser_path: str | None = None, headless: bool = True) -> str:
    if not email or not password:
        raise ValueError("email and password are required")
    try:
        from DrissionPage import ChromiumPage
    except ImportError as e:
        raise RuntimeError(
            "DrissionPage is required for login (pip install -r requirements.txt)"
        ) from e

    del browser_path  # resolved inside make_chromium_options via env/defaults
    profile = tempfile.mkdtemp(prefix="deepseek-auth-")
    page = ChromiumPage(addr_or_opts=make_chromium_options(
        headless=headless, user_data_dir=profile))
    try:
        page.listen.start("/api/v0/users/login")
        page.get("https://chat.deepseek.com/sign_in")
        wait_for_shumei(page)
        patch_shumei_ready(page)
        time.sleep(1)

        inputs = page.eles("tag:input")
        if len(inputs) < 2:
            raise RuntimeError("Login form not found")
        inputs[0].clear()
        inputs[0].input(email)
        time.sleep(0.5)
        inputs[1].clear()
        inputs[1].input(password)
        time.sleep(0.5)
        button = next((b for b in page.eles('css:[role="button"]')
                       if (b.text or "").strip() == "Log in"), None)
        if button is None:
            raise RuntimeError("Log in button not found")
        button.click()

        deadline = time.time() + timeout
        last_response: str | None = None
        while time.time() < deadline:
            time.sleep(2)
            for packet in page.listen.steps(timeout=0.5):
                try:
                    body = packet.response.body if packet.response else None
                    if body is not None:
                        last_response = str(body)
                except Exception:
                    pass
            token = _read_user_token(page)
            if token:
                return token
        raise RuntimeError(
            f"Login did not complete (last response: {(last_response or '')[:500]})")
    finally:
        try:
            page.quit()
        except Exception:
            pass


def _read_user_token(page) -> str | None:
    try:
        raw = page.run_js("return localStorage.getItem('userToken')")
        if raw:
            value = json.loads(raw).get("value")
            if value:
                return value
    except Exception:
        pass
    try:
        if "/sign_in" not in (page.url or ""):
            raw = page.run_js("return localStorage.getItem('userToken')")
            if raw:
                return json.loads(raw).get("value") or None
    except Exception:
        pass
    return None


if __name__ == "__main__":
    import os
    import sys

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    _email = os.getenv("DEEPSEEK_EMAIL") or (sys.argv[1] if len(sys.argv) > 1 else "")
    _password = os.getenv("DEEPSEEK_PASSWORD") or (sys.argv[2] if len(sys.argv) > 2 else "")
    if not _email or not _password:
        print("Usage: python -m dsk.auth <email> <password>")
        sys.exit(2)
    print(login(_email, _password))
