"""Unit tests for waste reduction (no network; transport is faked)."""

import pytest
from fastapi.testclient import TestClient

from dsk import client as client_module
from dsk.client import DeepSeekClient
from dsk.openai_server import build_app
from dsk.pow import WASM_PATH, DeepSeekPOW, _compiled_module


@pytest.fixture(autouse=True)
def clean_upload_cache():
    client_module._UPLOAD_CACHE.clear()
    yield
    client_module._UPLOAD_CACHE.clear()


def test_wasm_compiled_once_per_process():
    assert _compiled_module(WASM_PATH) is _compiled_module(WASM_PATH)
    first = DeepSeekPOW()
    second = DeepSeekPOW()
    # Module (and engine) shared; instances stay per-hasher (own Store).
    assert first.hasher.instance is not second.hasher.instance


class _FakePow:
    def solve_challenge(self, challenge):
        return "pow"


class _FakeTransport:
    def __init__(self, status="SUCCESS"):
        self.cookies = {}
        self.uploads = 0
        self.gets = 0
        self._status = status

    def request(self, method, endpoint, headers, json_data):
        return {"data": {"biz_data": {"challenge": {
            "algorithm": "DeepSeekHashV1", "challenge": "c", "salt": "s",
            "signature": "sig", "difficulty": 1, "expire_at": 1,
            "target_path": "/api/v0/file/upload_file"}}}}

    def upload(self, endpoint, headers, file_bytes, filename, content_type):
        self.uploads += 1
        return {"data": {"biz_data": {
            "id": f"file-{self.uploads}", "status": "PENDING"}}}

    def get(self, endpoint, headers, params):
        self.gets += 1
        return {"data": {"biz_data": {"files": [
            {"id": fid, "status": self._status}
            for fid in params["file_ids"]]}}}


def _client(transport):
    return DeepSeekClient(
        "token", device_id="12345678-1234-5678-1234-567812345678",
        transport=transport, pow_solver=_FakePow())


def test_identical_bytes_upload_once():
    transport = _FakeTransport()
    client = _client(transport)
    first = client.resolve_image_file(b"img-bytes", "image-1.png", "image/png")
    second = client.resolve_image_file(b"img-bytes", "image-1.png", "image/png")
    assert first == second == "file-1"
    assert transport.uploads == 1
    assert transport.gets == 1  # one cheap revalidation, no PoW


def test_different_bytes_upload_separately():
    transport = _FakeTransport()
    client = _client(transport)
    assert client.resolve_image_file(b"aaa", "a.png") == "file-1"
    assert client.resolve_image_file(b"bbb", "b.png") == "file-2"
    assert transport.uploads == 2


def test_failed_file_reuploads():
    transport = _FakeTransport(status="FAILED")
    client = _client(transport)
    assert client.resolve_image_file(b"img", "i.png") == "file-1"
    assert client.resolve_image_file(b"img", "i.png") == "file-2"
    assert transport.uploads == 2


class _PowCountingClient:
    pow_calls = 0

    def get_pow_challenge(self):
        type(self).pow_calls += 1
        return object()


def test_deep_health_cached_per_token(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_AUTH_TOKEN", "dummy-token")
    _PowCountingClient.pow_calls = 0
    made = []

    def factory():
        made.append(True)
        return _PowCountingClient()

    client = TestClient(build_app(factory))
    assert client.get("/health/deep").status_code == 200
    assert client.get("/health/deep").status_code == 200
    assert made == [True]  # second call served from cache
    assert _PowCountingClient.pow_calls == 1

    monkeypatch.setenv("DEEPSEEK_AUTH_TOKEN", "rotated-token")
    assert client.get("/health/deep").status_code == 200
    assert len(made) == 2  # token change revalidates
    assert _PowCountingClient.pow_calls == 2
