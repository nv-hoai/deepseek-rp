"""Unit tests for token resolution/caching and 401 re-login (no network)."""

import os
import stat

import pytest
from fastapi.testclient import TestClient

from dsk import token_store
from dsk.exceptions import AuthenticationError
from dsk.openai_server import build_app


@pytest.fixture
def isolated_env(monkeypatch, tmp_path):
    monkeypatch.delenv("DEEPSEEK_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("DEEPSEEK_EMAIL", raising=False)
    monkeypatch.delenv("DEEPSEEK_PASSWORD", raising=False)
    monkeypatch.setenv("DEEPSEEK_TOKEN_FILE", str(tmp_path / "token"))
    return tmp_path


def test_explicit_token_wins(isolated_env):
    os.environ["DEEPSEEK_AUTH_TOKEN"] = "explicit"
    (isolated_env / "token").write_text("cached\n")
    assert token_store.resolve_auth_token(
        login_fn=lambda e, p: (_ for _ in ()).throw(AssertionError())) == \
        "explicit"


def test_cached_token_used(isolated_env):
    (isolated_env / "token").write_text("cached-token\n")
    assert token_store.resolve_auth_token() == "cached-token"


def test_login_fallback_saves_restricted_cache(isolated_env, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_EMAIL", "user@example.com")
    monkeypatch.setenv("DEEPSEEK_PASSWORD", "secret")
    token = token_store.resolve_auth_token(login_fn=lambda e, p: "fresh-token")
    assert token == "fresh-token"
    saved = isolated_env / "token"
    assert saved.read_text() == "fresh-token\n"
    assert stat.S_IMODE(saved.stat().st_mode) == 0o600
    # Second call uses the cache without logging in again.
    assert token_store.resolve_auth_token(
        login_fn=lambda e, p: (_ for _ in ()).throw(AssertionError())) == \
        "fresh-token"


def test_login_empty_token_rejected(isolated_env, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_EMAIL", "user@example.com")
    monkeypatch.setenv("DEEPSEEK_PASSWORD", "secret")
    with pytest.raises(AuthenticationError):
        token_store.resolve_auth_token(login_fn=lambda e, p: "  ")


def test_nothing_configured_raises_helpful_error(isolated_env):
    with pytest.raises(AuthenticationError) as exc:
        token_store.resolve_auth_token()
    assert "DEEPSEEK_AUTH_TOKEN" in str(exc.value)
    assert "DEEPSEEK_EMAIL" in str(exc.value)


def test_invalidate_removes_cache(isolated_env):
    (isolated_env / "token").write_text("x\n")
    token_store.invalidate_cached_token()
    assert not (isolated_env / "token").exists()
    token_store.invalidate_cached_token()  # missing file is fine


def test_auto_login_enabled_matrix(isolated_env, monkeypatch):
    assert token_store.auto_login_enabled() is False
    monkeypatch.setenv("DEEPSEEK_EMAIL", "u")
    monkeypatch.setenv("DEEPSEEK_PASSWORD", "p")
    assert token_store.auto_login_enabled() is True
    monkeypatch.setenv("DEEPSEEK_AUTH_TOKEN", "explicit")
    assert token_store.auto_login_enabled() is False


class _FlakyChunk:
    def __init__(self, content="recovered"):
        self.type = "text"
        self.content = content


class _FlakyClient:
    """Fails with 401 until constructed with the refreshed token."""

    def __init__(self, token):
        self._token = token

    def create_chat_session(self):
        return "sess-1"

    def chat_completion(self, session_id, prompt, thinking_enabled=True,
                        search_enabled=False, model_type="default",
                        ref_file_ids=None):
        if self._token != "fresh-token":
            raise AuthenticationError("Invalid or expired token")
        yield _FlakyChunk()


def test_non_streaming_relogin_recovers(isolated_env, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_EMAIL", "user@example.com")
    monkeypatch.setenv("DEEPSEEK_PASSWORD", "secret")
    monkeypatch.setattr(token_store, "_browser_login",
                        lambda e, p: "fresh-token")
    (isolated_env / "token").write_text("stale-token\n")
    made = []

    def factory():
        made.append(True)
        return _FlakyClient(token_store.resolve_auth_token())

    response = TestClient(build_app(factory)).post(
        "/v1/chat/completions", json={
            "model": "deepseek",
            "messages": [{"role": "user", "content": "hi"}],
        })
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "recovered"
    assert len(made) == 2  # stale client + one post-refresh retry
    assert (isolated_env / "token").read_text() == "fresh-token\n"


def test_explicit_token_does_not_retry(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_AUTH_TOKEN", "explicit")

    def factory():
        return _FlakyClient("explicit")

    response = TestClient(build_app(factory)).post(
        "/v1/chat/completions", json={
            "model": "deepseek",
            "messages": [{"role": "user", "content": "hi"}],
        })
    assert response.status_code == 401
