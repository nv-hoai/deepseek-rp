"""Shared Chromium helpers (UA pin, binary resolution, WAF/Shumei waits)."""

from __future__ import annotations

import os
import shutil
import time
from typing import Any

from . import config

CHROMIUM_ARGS = [
    "-no-first-run",
    "-force-color-profile=srgb",
    "-metrics-recording-only",
    "-password-store=basic",
    "-use-mock-keychain",
    "-export-tagged-pdf",
    "-no-default-browser-check",
    "-disable-background-mode",
    "-enable-features=NetworkService,NetworkServiceInProcess,"
    "LoadCryptoTokenExtension,PermuteTLSExtensions",
    "-disable-features=FlashDeprecationWarning,EnablePasswordsAccountStorage",
    "-deny-permission-prompts",
    "-disable-gpu",
    "-accept-lang=en-US",
]

SMSDK_READY_PATCH = """
if(window.SMSdk){
  try{ window.SMSdk.ready = (cb)=>{ try{cb&&cb()}catch(e){} }; }catch(e){}
}
return 'patched';
"""


def resolve_browser_path(explicit: str | None = None) -> str | None:
    candidates = [
        explicit or "",
        os.getenv("CHROME_PATH") or "",
        os.getenv("BROWSER_PATH") or "",
        "/usr/bin/google-chrome",
        "/usr/bin/chromium-browser",
        "/usr/bin/chromium",
        "/opt/google/chrome/chrome",
    ]
    for cand in candidates:
        if cand and os.path.exists(cand):
            return cand
    return (shutil.which("google-chrome")
            or shutil.which("chromium-browser")
            or shutil.which("chromium"))


def make_chromium_options(headless: bool = True, proxy: str | None = None,
                          user_data_dir: str | None = None):
    from DrissionPage import ChromiumOptions

    options = ChromiumOptions().auto_port()
    binary = resolve_browser_path()
    if binary:
        try:
            options.set_paths(browser_path=binary)
        except Exception:
            pass
    if user_data_dir:
        try:
            options.set_user_data_path(user_data_dir)
        except Exception:
            pass
    options.headless(headless)
    options.set_argument("--no-sandbox")
    options.set_argument("--disable-gpu")
    options.set_argument(f"--user-agent={config.USER_AGENT}")
    for arg in CHROMIUM_ARGS:
        try:
            options.set_argument(arg)
        except Exception:
            pass
    if proxy:
        options.set_proxy(proxy)
    return options


def wait_for_shumei(page: Any, timeout: int = 25) -> str:
    """Wait until ``SMSdk.getDeviceId()`` returns a usable fingerprint."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(2)
        try:
            value = page.run_js(
                "return (window.SMSdk && window.SMSdk.getDeviceId() || '')")
            if value and len(str(value)) > 100:
                return str(value)
        except Exception:
            pass
    raise RuntimeError("Shumei device SDK did not initialise")


def patch_shumei_ready(page: Any) -> None:
    """Fix late ``SMSdk.ready()`` registration (login needs this)."""
    try:
        page.run_js(SMSDK_READY_PATCH)
    except Exception:
        pass
