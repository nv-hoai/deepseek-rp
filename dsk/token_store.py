"""DeepSeek auth token resolution for long-running servers.

Priority: explicit ``DEEPSEEK_AUTH_TOKEN`` env, then a cached token file
(``DEEPSEEK_TOKEN_FILE`` or ``~/.deepseek_token``), then a browser login with
``DEEPSEEK_EMAIL``/``DEEPSEEK_PASSWORD`` whose token is cached for next time.
"""

from __future__ import annotations

import os
from pathlib import Path

from .exceptions import AuthenticationError

TOKEN_ENV = "DEEPSEEK_AUTH_TOKEN"
TOKEN_FILE_ENV = "DEEPSEEK_TOKEN_FILE"
EMAIL_ENV = "DEEPSEEK_EMAIL"
PASSWORD_ENV = "DEEPSEEK_PASSWORD"


def default_token_path() -> Path:
    override = os.getenv(TOKEN_FILE_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".deepseek_token"


def load_cached_token(path: Path | None = None) -> str | None:
    try:
        value = (path or default_token_path()).read_text().strip()
        return value or None
    except OSError:
        return None


def save_cached_token(token: str, path: Path | None = None) -> Path:
    target = path or default_token_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(token.strip() + "\n")
    try:
        os.chmod(target, 0o600)
    except OSError:
        pass
    return target


def invalidate_cached_token(path: Path | None = None) -> None:
    try:
        (path or default_token_path()).unlink()
    except OSError:
        pass


def credentials_available() -> bool:
    return bool(os.getenv(EMAIL_ENV) and os.getenv(PASSWORD_ENV))


def auto_login_enabled() -> bool:
    """True when expiry recovery via credential re-login is possible."""
    return not os.getenv(TOKEN_ENV) and credentials_available()


def resolve_auth_token(login_fn=None) -> str:
    """Return a usable token, logging in and caching if configured."""
    explicit = os.getenv(TOKEN_ENV, "").strip()
    if explicit:
        return explicit
    cached = load_cached_token()
    if cached:
        return cached
    if not credentials_available():
        raise AuthenticationError(
            "Server misconfigured: set DEEPSEEK_AUTH_TOKEN, or "
            "DEEPSEEK_EMAIL and DEEPSEEK_PASSWORD for auto-login")
    login = login_fn or _browser_login
    token = login(os.getenv(EMAIL_ENV, ""), os.getenv(PASSWORD_ENV, ""))
    if not token or not token.strip():
        raise AuthenticationError("Email login returned an empty token")
    save_cached_token(token)
    return token.strip()


def _browser_login(email: str, password: str) -> str:
    from .auth import login

    return login(email, password)
