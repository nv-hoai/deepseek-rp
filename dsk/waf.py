"""WAF/turnstile bypass. AWS WAF solves itself; Turnstile may need a click."""

from __future__ import annotations

import time
from typing import Any


class WafBypasser:
    def __init__(self, driver: Any, max_retries: int = -1, log: bool = True):
        self.driver = driver
        self.max_retries = max_retries
        self.log = log

    def _log(self, message: str) -> None:
        if self.log:
            print(message)

    def has_waf_token(self) -> bool:
        try:
            return any(
                c.get("name") == "aws-waf-token" and (c.get("value") or "").strip()
                for c in self.driver.cookies()
            )
        except Exception:
            return False

    def is_bypassed(self) -> bool:
        try:
            title = (self.driver.title or "").lower()
            if "just a moment" in title:
                return False
            try:
                html = self.driver.html[:4000]
            except Exception:
                html = ""
            if "AwsWafIntegration" in html and "challenge-container" in html:
                return False
            if self.has_waf_token():
                return True
            return "just a moment" not in title
        except Exception as e:
            self._log(f"Error checking page state: {e}")
            return False

    def _find_turnstile_frame(self) -> Any | None:
        try:
            root = self.driver.ele("tag:body")
        except Exception:
            return None
        return self._search_frame(root)

    def _search_frame(self, element: Any) -> Any | None:
        try:
            if element.shadow_root:
                if element.shadow_root.child().tag == "iframe":
                    return element.shadow_root.child()
                return None
            for child in element.children():
                found = self._search_frame(child)
                if found:
                    return found
        except Exception:
            pass
        return None

    def click_turnstile_if_present(self) -> None:
        try:
            for element in self.driver.eles("tag:input"):
                attrs = element.attrs or {}
                if ("turnstile" in attrs.get("name", "")
                        and attrs.get("type") == "hidden"):
                    element.parent().shadow_root.child()(
                        "tag:body").shadow_root("tag:input").click()
                    self._log("Clicked Turnstile checkbox.")
                    return
            frame = self._find_turnstile_frame()
            if frame is not None:
                try:
                    frame("tag:body").shadow_root("tag:input").click()
                    self._log("Clicked Turnstile checkbox (recursive).")
                except Exception:
                    self._log("Turnstile iframe found but click failed.")
        except Exception as e:
            self._log(f"Error clicking verification button: {e}")

    def bypass(self) -> bool:
        tries = 0
        while not self.is_bypassed():
            if 0 < self.max_retries + 1 <= tries:
                self._log("Exceeded maximum retries. Bypass failed.")
                return False
            self._log(f"Attempt {tries + 1}: challenge detected, waiting/clicking...")
            self.click_turnstile_if_present()
            tries += 1
            time.sleep(2)
        self._log("Bypass successful.")
        return True
