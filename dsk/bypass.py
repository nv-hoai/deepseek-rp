"""Cookie CLI: serve -> capture ``aws-waf-token`` -> save to cookies.json.

Usage:
    python -m dsk.bypass [--server-port 8000] [--url https://chat.deepseek.com]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import requests

from .cookies import COOKIE_FILE, CookieStore


def run_server_background(port: int):
    server_script = Path(__file__).parent / "server.py"
    try:
        return subprocess.Popen(
            [sys.executable, str(server_script), "--port", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(server_script.parent),
            start_new_session=True,
        )
    except Exception:
        return None


def fetch_cookies(server_url: str, timeout: int = 60) -> dict:
    response = requests.get(server_url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def refresh_cookies_via_server(port: int = 8000,
                               url: str = "https://chat.deepseek.com",
                               retries: int = 5) -> tuple[dict[str, str], str]:
    """Start the server, capture cookies, stop it. Returns (cookies, user_agent)."""
    process = run_server_background(port)
    if process is None:
        raise RuntimeError("Failed to start bypass server")
    try:
        time.sleep(10)
        last_error: str = ""
        for attempt in range(retries):
            try:
                data = fetch_cookies(
                    f"http://localhost:{port}/cookies?url={url}")
            except requests.ConnectionError as e:
                last_error = str(e)
                time.sleep(5)
                continue
            cookies = data.get("cookies", {})
            if CookieStore.has_waf_token(cookies):
                return cookies, data.get("user_agent", "")
            last_error = "aws-waf-token cookie not found"
            time.sleep(5)
        raise RuntimeError(f"Failed to obtain cookies: {last_error}")
    finally:
        try:
            process.terminate()
        except Exception:
            pass


# Legacy names used by client.py.
def validate_cookies(cookies_data: dict) -> bool:
    return CookieStore.has_waf_token(cookies_data.get("cookies", {}))


def get_and_save_cookies(server_url: str, cookie_file_path: str,
                         max_retries: int = 3) -> bool:
    for _ in range(max_retries):
        try:
            data = fetch_cookies(server_url)
        except requests.ConnectionError:
            time.sleep(5)
            continue
        if not validate_cookies(data):
            time.sleep(5)
            continue
        CookieStore(Path(cookie_file_path)).save(
            data.get("cookies", {}), data.get("user_agent", ""))
        print("Successfully obtained and saved cookies with aws-waf-token!")
        return True
    print("Failed to obtain valid aws-waf-token cookie after all attempts")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--url", default="https://chat.deepseek.com")
    parser.add_argument("--out", default=str(COOKIE_FILE))
    args = parser.parse_args()
    print("Getting the cookies...")
    cookies, user_agent = refresh_cookies_via_server(
        port=args.server_port, url=args.url)
    CookieStore(Path(args.out)).save(cookies, user_agent)
    print(f"Saved {len(cookies)} cookies to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
