"""Unit tests for streaming error envelopes (no network)."""

import pytest

from dsk.client import DeepSeekClient
from dsk.exceptions import APIError

MUTED_ENVELOPE = ('{"code":0,"msg":"","data":{"biz_code":5,'
                  '"biz_msg":"user is muted",'
                  '"biz_data":{"is_muted":1}}}')


class _FakePow:
    def solve_challenge(self, challenge):
        return "pow"


class _FakeStream:
    def __init__(self, lines):
        self.status_code = 200
        self.headers = {}
        self._lines = lines

    def iter_lines(self):
        return iter(self._lines)


class _FakeTransport:
    def __init__(self, lines):
        self.cookies = {}
        self._lines = lines

    def request(self, method, endpoint, headers, json_data):
        return {"data": {"biz_data": {"challenge": {
            "algorithm": "DeepSeekHashV1", "challenge": "c", "salt": "s",
            "signature": "sig", "difficulty": 1, "expire_at": 1,
            "target_path": "/api/v0/chat/completion"}}}}

    def stream_post(self, endpoint, headers, json_data):
        return _FakeStream(self._lines)


def _client(lines):
    return DeepSeekClient(
        "token", device_id="12345678-1234-5678-1234-567812345678",
        transport=_FakeTransport(lines), pow_solver=_FakePow())


def test_muted_account_raises_instead_of_empty_stream():
    client = _client([MUTED_ENVELOPE.encode()])
    with pytest.raises(APIError) as exc:
        list(client.chat_completion("sess-1", "hi"))
    assert "muted" in str(exc.value)


def test_normal_stream_still_parses_first_line():
    line = ('data: {"o":"APPEND","p":"response/fragments",'
            '"v":[{"type":"RESPONSE","content":"hi"}]}')
    client = _client([line.encode()])
    chunks = [c for c in client.chat_completion("sess-1", "hi")
              if c.is_body()]
    assert "".join(c.content for c in chunks) == "hi"
