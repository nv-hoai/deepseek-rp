"""Unit tests for Anthropic Messages and Responses adapters (no network)."""

import json

from fastapi.testclient import TestClient

from dsk import anthropic_adapter as anthropic
from dsk import responses_adapter as responses
from dsk.openai_server import build_app


def test_anthropic_blocks_to_openai_messages():
    messages = anthropic.to_openai_messages(None, [
        {"role": "user", "content": "What time is it?"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Checking"},
            {"type": "tool_use", "id": "tu_1", "name": "get_time",
             "input": {}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu_1",
             "content": "noon"}]},
    ])
    assert messages[0] == {"role": "user", "content": "What time is it?"}
    assert messages[1]["role"] == "assistant"
    assert messages[1]["tool_calls"][0]["id"] == "tu_1"
    assert messages[2]["role"] == "tool"
    assert messages[2]["tool_call_id"] == "tu_1"


def test_anthropic_system_string_and_blocks():
    assert anthropic.to_openai_messages("Be brief.", []) == [
        {"role": "system", "content": "Be brief."}]
    assert anthropic.to_openai_messages(
        [{"type": "text", "text": "Sys"}], [])[0]["content"] == "Sys"


def test_anthropic_tool_shapes():
    tools = [{"name": "get_time", "description": "d",
              "input_schema": {"type": "object"}}]
    converted = anthropic.to_openai_tools(tools)
    assert converted[0]["function"]["parameters"] == {"type": "object"}
    assert anthropic.to_openai_tools(None) is None
    assert anthropic.to_openai_tool_choice({"type": "any"}) == "required"
    assert anthropic.to_openai_tool_choice({"type": "none"}) == "none"
    assert anthropic.to_openai_tool_choice(
        {"type": "tool", "name": "x"}) == {
            "type": "function", "function": {"name": "x"}}


def test_anthropic_response_shape():
    body = anthropic.build_response("m", "think", "hi",
                                    [{"id": "tu_1", "name": "t",
                                      "arguments": {"a": 1}}], 10, 5)
    assert [b["type"] for b in body["content"]] == [
        "thinking", "text", "tool_use"]
    assert body["stop_reason"] == "tool_use"
    assert body["content"][2]["input"] == {"a": 1}
    plain = anthropic.build_response("m", "", "hi", [], 10, 5)
    assert plain["stop_reason"] == "end_turn"
    assert [b["type"] for b in plain["content"]] == ["text"]


def test_responses_store_chains():
    store = responses.ResponseStore()
    assert store.chain("missing") == []
    store.save("resp_1", [{"type": "message"}])
    assert store.chain("resp_1") == [{"type": "message"}]


def test_responses_items_to_messages():
    messages = responses.to_openai_messages("Sys", [
        {"type": "message", "role": "user",
         "content": [{"type": "input_text", "text": "hi"}]},
        {"type": "function_call_output", "call_id": "c1",
         "output": "noon"},
    ], [{"type": "message", "role": "user",
         "content": [{"type": "input_text", "text": "old"}]}])
    assert messages[0] == {"role": "system", "content": "Sys"}
    assert messages[1]["content"] == "old"
    assert messages[3]["role"] == "tool"
    assert messages[3]["tool_call_id"] == "c1"


class _FakeChunk:
    def __init__(self, type, content=""):
        self.type = type
        self.content = content


class _FakeClient:
    def __init__(self, thinking="", text="hi"):
        self._thinking = thinking
        self._text = text

    def create_chat_session(self):
        return "sess-1"

    def chat_completion(self, session_id, prompt, thinking_enabled=True,
                        search_enabled=False, model_type="default",
                        ref_file_ids=None):
        self.last_model_type = model_type
        self.last_ref_file_ids = ref_file_ids
        if self._thinking:
            yield _FakeChunk("thinking", self._thinking)
        yield _FakeChunk("text", self._text)


def test_anthropic_endpoint():
    client = TestClient(build_app(lambda: _FakeClient(
        thinking="reason", text="hello")))
    response = client.post("/v1/messages", json={
        "model": "deepseek",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert response.status_code == 200
    body = response.json()
    assert [b["type"] for b in body["content"]] == ["thinking", "text"]
    assert body["content"][1]["text"] == "hello"
    assert body["stop_reason"] == "end_turn"


def test_anthropic_endpoint_tool_use():
    tools = [{"name": "get_time", "description": "t",
              "input_schema": {"type": "object"}}]
    client = TestClient(build_app(lambda: _FakeClient(
        text='<tool_call>{"name": "get_time", "arguments": {}}</tool_call>')))
    response = client.post("/v1/messages", json={
        "model": "deepseek",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "time?"}],
        "tools": tools,
    })
    assert response.status_code == 200
    body = response.json()
    assert body["stop_reason"] == "tool_use"
    assert body["content"][-1] == {"type": "tool_use", "id": body["content"][-1]["id"],
                                   "name": "get_time", "input": {}}


class _VisionFakeClient(_FakeClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.uploaded = []

    def upload_file(self, file_bytes, filename, content_type="image/png"):
        self.uploaded.append((filename, content_type, len(file_bytes)))
        return {"id": "file-test-1", "status": "PENDING"}

    def resolve_image_file(self, file_bytes, filename,
                           content_type="image/png"):
        info = self.upload_file(file_bytes, filename, content_type)
        self.wait_for_files([info["id"]])
        return info["id"]

    def wait_for_files(self, file_ids, timeout=60):
        assert file_ids == ["file-test-1"]
        return [{"id": "file-test-1", "status": "SUCCESS"}]


_TINY_PNG = ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAA"
             "AfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")


def test_anthropic_endpoint_vision_uploads_images():
    fake = _VisionFakeClient(text="a pink square")
    client = TestClient(build_app(lambda: fake))
    response = client.post("/v1/messages", json={
        "model": "deepseek",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "What is this?"},
            {"type": "image",
             "source": {"type": "base64", "media_type": "image/png",
                        "data": _TINY_PNG.split(",", 1)[1]}}]}],
    })
    assert response.status_code == 200
    assert fake.last_model_type == "vision"
    assert fake.last_ref_file_ids == ["file-test-1"]
    assert len(fake.uploaded) == 1
    assert response.json()["content"][-1]["text"] == "a pink square"


def test_anthropic_stream_events():
    client = TestClient(build_app(lambda: _FakeClient(text="hi")))
    with client.stream("POST", "/v1/messages", json={
            "model": "deepseek",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
    }) as response:
        assert response.status_code == 200
        raw = response.read().decode()
    assert "event: message_start" in raw
    assert "event: content_block_delta" in raw
    assert "event: message_stop" in raw


def test_responses_endpoint_and_chain():
    tools = [{"type": "function", "name": "get_time", "description": "t",
              "parameters": {"type": "object"}}]
    client = TestClient(build_app(lambda: _FakeClient(
        text='<tool_call>{"name": "get_time", "arguments": {}}</tool_call>')))
    first = client.post("/v1/responses", json={
        "model": "deepseek",
        "input": "What time is it?",
        "tools": tools,
    })
    assert first.status_code == 200
    body = first.json()
    assert body["object"] == "response"
    kinds = [item["type"] for item in body["output"]]
    assert "function_call" in kinds
    response_id = body["id"]

    followup = client.post("/v1/responses", json={
        "model": "deepseek",
        "previous_response_id": response_id,
        "input": [{"type": "function_call_output", "call_id": "c1",
                   "output": "noon"}],
    })
    assert followup.status_code == 200

    fetched = client.get(f"/v1/responses/{response_id}")
    assert fetched.status_code == 200
    assert fetched.json()["id"] == response_id


def test_responses_stream_events():
    client = TestClient(build_app(lambda: _FakeClient(text="hi")))
    with client.stream("POST", "/v1/responses", json={
            "model": "deepseek",
            "input": "hi",
            "stream": True,
    }) as response:
        assert response.status_code == 200
        raw = response.read().decode()
    assert "event: response.created" in raw
    assert "event: response.output_text.delta" in raw
    assert "event: response.completed" in raw
