"""Shared constants. Single source of truth for frontend-mirroring values.

When chat.deepseek.com changes, update here first (see README "Reverse-engineering notes").
"""

BASE_URL = "https://chat.deepseek.com/api/v0"
ORIGIN = "https://chat.deepseek.com"
REFERER = "https://chat.deepseek.com/"

# CloudFront returns 403 for new Chromium (e.g. 153) headless UA (2026-09).
# Chrome 132 UA is the known-good fingerprint; also used with curl-cffi
# impersonate="chrome120".
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36"
)
IMPERSONATE = "chrome120"

CLIENT_PLATFORM = "web"
CLIENT_VERSION = "2.5.0"  # frontend appVersion; was 1.0.0-always
CLIENT_BUNDLE_ID = "com.deepseek.chat"
CLIENT_LOCALE = "en_US"

ACCEPT_LANGUAGE = (
    "en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,"
    "am;q=0.4,de;q=0.3"
)

POW_HEADER = "X-DS-PoW-Response"
POW_TARGET_PATH = "/api/v0/chat/completion"

ENDPOINT_POW_CHALLENGE = "/chat/create_pow_challenge"
ENDPOINT_COMPLETION = "/chat/completion"
ENDPOINT_SESSION_CREATE = "/chat_session/create"
ENDPOINT_LOGIN = "/users/login"

WASM_FILENAME = "sha3_wasm_bg.7b9ca65ddd.wasm"

CURL_CFFI_TESTED_VERSION = "0.8.1b9"

BROWSER_SIGN_IN_URL = "https://chat.deepseek.com/sign_in"
BROWSER_HOME_URL = "https://chat.deepseek.com"
