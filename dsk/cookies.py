"""Cookie persistence (AWS WAF ``aws-waf-token``; legacy ``cf_clearance``)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

COOKIE_FILE = Path(__file__).parent / "cookies.json"


class CookieStore:
    def __init__(self, path: Path = COOKIE_FILE):
        self.path = path
        self.cookies: dict[str, str] = self._load()

    def _load(self) -> dict[str, str]:
        try:
            data = json.loads(self.path.read_text())
            cookies = data.get("cookies", {})
            return dict(cookies) if isinstance(cookies, dict) else {}
        except FileNotFoundError:
            return {}
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: Could not load cookies from {self.path}: {e}",
                  file=sys.stderr)
            return {}

    def save(self, cookies: dict[str, str], user_agent: str = "") -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"cookies": cookies, "user_agent": user_agent},
                       indent=4, ensure_ascii=False)
        )
        self.cookies = dict(cookies)

    @staticmethod
    def has_waf_token(cookies: dict[str, str]) -> bool:
        return bool(cookies.get("aws-waf-token", "").strip()
                    or cookies.get("cf_clearance", "").strip())
