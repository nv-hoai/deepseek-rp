"""Unit tests for the OpenAI-compat layer (no network)."""

import json

from fastapi.testclient import TestClient

from dsk import openai_adapter as adapter
from dsk.openai_server import build_app


def test_messages_to_prompt_covers_roles():
    prompt = adapter.messages_to_prompt([
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "tool", "name": "search", "tool_call_id": "call_1",
         "content": "result"},
    ])
    assert "System: Be brief." in prompt
    assert "User: Hi" in prompt
    assert "Assistant: Hello" in prompt
    assert "Tool result (search): result" in prompt


def test_messages_content_parts():
    prompt = adapter.messages_to_prompt([
        {"role": "user",
         "content": [{"type": "text", "text": "a"},
                     {"type": "image_url", "image_url": {"url": "x"}}]},
    ])
    assert "User: a" in prompt


def test_preamble_none_choice():
    tools = [{"type": "function",
              "function": {"name": "get_time", "description": "d",
                           "parameters": {"type": "object"}}}]
    assert adapter.build_tool_preamble(tools, "none") == ""
    assert adapter.build_tool_preamble(None, "auto") == ""
    auto = adapter.build_tool_preamble(tools, "auto")
    assert "get_time" in auto and "<tool_call>" in auto
    required = adapter.build_tool_preamble(tools, "required")
    assert "MUST consist of at least one <tool_call>" in required
    specific = adapter.build_tool_preamble(
        tools, {"type": "function", "function": {"name": "get_time"}})
    assert "get_time" in specific and "MUST consist" in specific


def test_preamble_has_example_and_rules():
    tools = [{"type": "function",
              "function": {"name": "get_time",
                           "description": "Current time",
                           "parameters": {"type": "object",
                                          "properties": {},
                                          "required": []}}}]
    preamble = adapter.build_tool_preamble(tools, "auto")
    assert "<tool_call>" in preamble
    assert "get_time" in preamble
    assert "DSML" in preamble


def test_parse_tool_calls_round_trip():
    tools = [{"type": "function",
              "function": {"name": "search", "parameters": {"type": "object"}}}]
    text = ('Thinking out loud\n'
            '<tool_call>{"name": "search", '
            '"arguments": {"q": "x"}}</tool_call>\nDone')
    content, calls = adapter.parse_tool_calls(text, tools)
    assert content == "Thinking out loud\n\nDone"
    assert len(calls) == 1
    assert calls[0]["name"] == "search"
    assert calls[0]["arguments"] == {"q": "x"}
    assert calls[0]["id"].startswith("call_")


def test_parse_tool_calls_drops_unknown_and_bad_json():
    tools = [{"type": "function",
              "function": {"name": "known", "parameters": {"type": "object"}}}]
    text = ('<tool_call>{"name": "known", "arguments": {}}</tool_call>'
            '<tool_call>{"name": "nope", "arguments": {}}</tool_call>'
            '<tool_call>not json</tool_call>tail')
    content, calls = adapter.parse_tool_calls(text, tools)
    assert [c["name"] for c in calls] == ["known"]
    assert content == "tail"


def test_resolve_flags():
    assert adapter.resolve_flags("deepseek-chat", {}) == (False, False)
    assert adapter.resolve_flags("deepseek-reasoner", {}) == (True, False)
    assert adapter.resolve_flags("x", {"thinking_enabled": False,
                                       "search_enabled": True}) == (False, True)


class _FakeChunk:
    def __init__(self, type, content=""):
        self.type = type
        self.content = content


class _FakeClient:
    def __init__(self, thinking="", text="4"):
        self._thinking = thinking
        self._text = text
        self.last_prompt = ""

    def create_chat_session(self):
        return "sess-1"

    def chat_completion(self, session_id, prompt, thinking_enabled=True,
                        search_enabled=False, model_type="default",
                        ref_file_ids=None):
        self.last_prompt = prompt
        self.last_model_type = model_type
        self.last_ref_file_ids = ref_file_ids
        if self._thinking:
            yield _FakeChunk("thinking", self._thinking)
        yield _FakeChunk("text", self._text)


def _client(messages_text="4"):
    fake = _FakeClient(text=messages_text)
    return TestClient(build_app(lambda: fake)), fake


def test_models_endpoint():
    client, _ = _client()
    response = client.get("/v1/models")
    assert response.status_code == 200
    ids = [m["id"] for m in response.json()["data"]]
    assert "deepseek-chat" in ids and "deepseek-reasoner" in ids


def test_chat_non_stream_no_tools():
    client, _ = _client("2+2=4")
    response = client.post("/v1/chat/completions", json={
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
    })
    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "2+2=4"
    assert body["choices"][0]["finish_reason"] == "stop"


def test_chat_returns_tool_calls():
    tools = [{"type": "function",
              "function": {"name": "get_time",
                           "description": "Current time",
                           "parameters": {"type": "object",
                                          "properties": {}}}}]
    fake_text = ('<tool_call>{"name": "get_time", '
                 '"arguments": {}}</tool_call>')
    client = TestClient(build_app(
        lambda: _FakeClient(text=fake_text)))
    response = client.post("/v1/chat/completions", json={
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "What time is it?"}],
        "tools": tools,
        "tool_choice": "auto",
    })
    assert response.status_code == 200
    message = response.json()["choices"][0]["message"]
    assert message["content"] is None
    assert message["tool_calls"][0]["function"]["name"] == "get_time"
    assert json.loads(
        message["tool_calls"][0]["function"]["arguments"]) == {}


def test_chat_stream_tool_calls_ends_with_done():
    tools = [{"type": "function",
              "function": {"name": "get_time",
                           "parameters": {"type": "object"}}}]
    client = TestClient(build_app(lambda: _FakeClient(
        text='<tool_call>{"name": "get_time", "arguments": {}}</tool_call>')))
    with client.stream("POST", "/v1/chat/completions", json={
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "time?"}],
            "tools": tools,
            "stream": True,
    }) as response:
        assert response.status_code == 200
        raw = response.read().decode()
    assert "tool_calls" in raw
    assert raw.strip().endswith("data: [DONE]")


def test_chat_empty_messages_rejected():
    client, _ = _client()
    response = client.post("/v1/chat/completions", json={
        "model": "deepseek-chat", "messages": []})
    assert response.status_code == 400


def test_chat_stream_text_streams_content():
    client, _ = _client("hello world")
    with client.stream("POST", "/v1/chat/completions", json={
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
    }) as response:
        assert response.status_code == 200
        raw = response.read().decode()
    assert '"content": "hello world"' in raw
    assert raw.strip().endswith("data: [DONE]")


def test_auth_error_maps_to_401():
    from dsk.exceptions import AuthenticationError

    class _BadClient:
        def create_chat_session(self):
            raise AuthenticationError("bad token")

    client = TestClient(build_app(lambda: _BadClient()))
    response = client.post("/v1/chat/completions", json={
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 401


DSML_SAMPLE = (
    "I'll create the files.\n\n"
    "<｜｜DSML｜｜ calls>\n"
    "<｜｜DSML｜｜ invoke name=\"write\">\n"
    "<｜｜DSML｜｜ parameter name=\"path\" string=\"true\">/tmp/x.py</｜｜DSML｜｜ parameter>\n"
    "<｜｜DSML｜｜ parameter name=\"content\" string=\"true\">a = 1\n</｜｜DSML｜｜ parameter>\n"
    "</｜｜DSML｜｜ invoke>\n"
    "</｜｜DSML｜｜ calls>"
)


def test_parse_dsml_fallback():
    tools = [{"type": "function",
              "function": {"name": "write",
                           "parameters": {"type": "object"}}}]
    content, calls = adapter.parse_tool_calls(DSML_SAMPLE, tools)
    assert content == "I'll create the files."
    assert len(calls) == 1
    assert calls[0]["name"] == "write"
    assert calls[0]["arguments"] == {"path": "/tmp/x.py", "content": "a = 1"}


def test_parse_dsml_drops_unknown_tool():
    tools = [{"type": "function",
              "function": {"name": "write",
                           "parameters": {"type": "object"}}}]
    text = DSML_SAMPLE.replace('name="write"', 'name="nope"')
    content, calls = adapter.parse_tool_calls(text, tools)
    assert calls == []
    assert "nope" not in content


def test_tool_call_format_wins_over_dsml():
    tools = [{"type": "function",
              "function": {"name": "write",
                           "parameters": {"type": "object"}}}]
    text = ('<tool_call>{"name": "write", "arguments": {"path": "a"}}</tool_call>\n'
            + DSML_SAMPLE)
    _, calls = adapter.parse_tool_calls(text, tools)
    assert [c["name"] for c in calls] == ["write"]
    assert calls[0]["arguments"] == {"path": "a"}


def test_parse_hybrid_tool_call_with_dsml_tail():
    tools = [{"type": "function",
              "function": {"name": "shell",
                           "parameters": {"type": "object"}}}]
    text = ('<tool_call>{"name": "shell", "arguments": {"command": "ls"}}</｜｜DSML｜｜ parameter>\n'
            '</｜｜DSML｜｜ invoke>\n</｜｜DSML｜｜ calls>')
    content, calls = adapter.parse_tool_calls(text, tools)
    assert [c["name"] for c in calls] == ["shell"]
    assert calls[0]["arguments"] == {"command": "ls"}
    assert "DSML" not in content


def test_parse_multiple_tool_call_blocks():
    tools = [{"type": "function",
              "function": {"name": "edit",
                           "parameters": {"type": "object"}}}]
    text = ('<tool_call>{"name": "edit", "arguments": {"a": 1}}</tool_call>\n'
            '<tool_call>{"name": "edit", "arguments": {"a": 2}}</tool_call>\n'
            '<tool_call>{"name": "edit", "arguments": {"a": 3}}</tool_call>')
    _, calls = adapter.parse_tool_calls(text, tools)
    assert [c["arguments"] for c in calls] == [{"a": 1}, {"a": 2}, {"a": 3}]
    assert len({c["id"] for c in calls}) == 3
