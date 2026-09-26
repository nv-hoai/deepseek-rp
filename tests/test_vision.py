"""Unit tests for vision support (no network; transport is faked)."""

import pytest
from fastapi.testclient import TestClient

from dsk import images
from dsk import openai_adapter as adapter
from dsk.client import DeepSeekClient
from dsk.openai_server import build_app

TINY_PNG_URI = ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAA"
                "AfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")


def test_decode_data_uri_round_trip():
    data, mime = images.decode_data_uri(TINY_PNG_URI)
    assert mime == "image/png"
    assert data[:8] == b"\x89PNG\r\n\x1a\n"


def test_decode_data_uri_rejects_non_image():
    with pytest.raises(ValueError):
        images.decode_data_uri("data:text/plain;base64,aGk=")
    with pytest.raises(ValueError):
        images.decode_data_uri("not-a-uri")


def test_extract_openai_images():
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "see?"},
        {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
        {"type": "image_url", "image_url": {"url": TINY_PNG_URI}},
    ]}]
    assert images.extract_openai_images(messages) == [
        "https://x/y.png", TINY_PNG_URI]
    assert images.extract_openai_images(
        [{"role": "user", "content": "plain"}]) == []


def test_extract_anthropic_images():
    system = [{"type": "text", "text": "sys"}]
    messages = [{"role": "user", "content": [
        {"type": "image", "source": {
            "type": "base64", "media_type": "image/png", "data": "eA=="}},
        {"type": "image", "source": {
            "type": "url", "url": "https://x/y.jpg"}},
    ]}]
    assert images.extract_anthropic_images(system, messages) == [
        "data:image/png;base64,eA==", "https://x/y.jpg"]


def test_extract_responses_images():
    items = [{"type": "message", "role": "user", "content": [
        {"type": "input_text", "text": "hi"},
        {"type": "input_image", "image_url": "https://x/y.png"},
    ]}]
    assert images.extract_responses_images(items) == ["https://x/y.png"]


def test_resolve_model_type():
    assert adapter.resolve_model_type("deepseek-chat", False, {}) == "default"
    assert adapter.resolve_model_type(
        "deepseek-chat", True, {}) == "vision"
    assert adapter.resolve_model_type("deepseek-vision", False, {}) == "vision"
    assert adapter.resolve_model_type(
        "deepseek-chat", False, {"model_type": "vision"}) == "vision"


def test_image_placeholder_in_prompt():
    prompt = adapter.messages_to_prompt([{
        "role": "user",
        "content": [{"type": "text", "text": "look"},
                    {"type": "image_url",
                     "image_url": {"url": "https://x/y.png"}}]}])
    assert "[attached image]" in prompt


class _FakePow:
    def solve_challenge(self, challenge):
        assert challenge["target_path"] == "/api/v0/file/upload_file"
        return "pow-response"


class _FakeTransport:
    def __init__(self):
        self.cookies = {}
        self.uploaded = []

    def request(self, method, endpoint, headers, json_data):
        assert json_data["target_path"] == "/api/v0/file/upload_file"
        return {"data": {"biz_data": {"challenge": {
            "algorithm": "DeepSeekHashV1", "challenge": "c", "salt": "s",
            "signature": "sig", "difficulty": 1, "expire_at": 1,
            "target_path": "/api/v0/file/upload_file"}}}}

    def upload(self, endpoint, headers, file_bytes, filename, content_type):
        assert endpoint == "/file/upload_file"
        assert headers["X-DS-PoW-Response"] == "pow-response"
        assert "content-type" not in {k.lower(): v for k, v in headers.items()}
        self.uploaded.append((filename, content_type, file_bytes))
        return {"data": {"biz_data": {"id": "file-1", "status": "PENDING"}}}

    def get(self, endpoint, headers, params):
        assert endpoint == "/file/fetch_files"
        assert params == {"file_ids": ["file-1"]}
        return {"data": {"biz_data": {"files": [
            {"id": "file-1", "status": "SUCCESS"}]}}}


def _client():
    transport = _FakeTransport()
    client = DeepSeekClient(
        "token", device_id="12345678-1234-5678-1234-567812345678",
        transport=transport, pow_solver=_FakePow())
    return client, transport


def test_upload_file_sends_pow_and_multipart():
    client, transport = _client()
    info = client.upload_file(b"\x89PNG", "image-1.png", "image/png")
    assert info["id"] == "file-1"
    filename, content_type, payload = transport.uploaded[0]
    assert (filename, content_type, payload) == (
        "image-1.png", "image/png", b"\x89PNG")


def test_upload_file_rejects_empty_and_oversize():
    import dsk.config as config

    client, _ = _client()
    with pytest.raises(ValueError):
        client.upload_file(b"", "e.png")
    with pytest.raises(ValueError):
        client.upload_file(b"x" * (config.MAX_UPLOAD_BYTES + 1), "big.png")


def test_wait_for_files_success():
    client, _ = _client()
    files = client.wait_for_files(["file-1"], timeout=5)
    assert files[0]["status"] == "SUCCESS"


class _FakeChunk:
    def __init__(self, type, content=""):
        self.type = type
        self.content = content


class _VisionClient:
    def __init__(self):
        self.last_model_type = None
        self.last_ref_file_ids = None
        self.uploaded = []

    def create_chat_session(self):
        return "sess-1"

    def chat_completion(self, session_id, prompt, thinking_enabled=True,
                        search_enabled=False, model_type="default",
                        ref_file_ids=None):
        self.last_model_type = model_type
        self.last_ref_file_ids = ref_file_ids
        yield _FakeChunk("text", "a pink square")

    def upload_file(self, file_bytes, filename, content_type="image/png"):
        self.uploaded.append((filename, content_type))
        return {"id": "file-9", "status": "PENDING"}

    def wait_for_files(self, file_ids, timeout=60):
        return [{"id": fid, "status": "SUCCESS"} for fid in file_ids]


def test_chat_vision_end_to_end():
    fake = _VisionClient()
    client = TestClient(build_app(lambda: fake))
    response = client.post("/v1/chat/completions", json={
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "What is this?"},
            {"type": "image_url", "image_url": {"url": TINY_PNG_URI}},
        ]}],
    })
    assert response.status_code == 200
    assert fake.last_model_type == "vision"
    assert fake.last_ref_file_ids == ["file-9"]
    assert fake.uploaded == [("image-1.png", "image/png")]
    assert response.json()["choices"][0]["message"]["content"] == \
        "a pink square"


def test_chat_vision_model_name_without_images():
    fake = _VisionClient()
    client = TestClient(build_app(lambda: fake))
    response = client.post("/v1/chat/completions", json={
        "model": "deepseek-vision",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert response.status_code == 200
    assert fake.last_model_type == "vision"
    assert fake.last_ref_file_ids == []
    assert fake.uploaded == []


def test_chat_bad_image_is_400():
    fake = _VisionClient()
    client = TestClient(build_app(lambda: fake))
    response = client.post("/v1/chat/completions", json={
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:xyz"}},
        ]}],
    })
    assert response.status_code == 400
    assert fake.uploaded == []


def test_models_include_vision():
    client = TestClient(build_app(lambda: _VisionClient()))
    ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
    assert "deepseek-vision" in ids
