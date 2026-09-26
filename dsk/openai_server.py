"""OpenAI-compatible API over DeepSeek web chat.

Run:
    DEEPSEEK_AUTH_TOKEN=... uvicorn dsk.openai_server:app --port 8080

Use with any OpenAI client:
    base_url="http://localhost:8080/v1", api_key="anything"

Endpoints: ``GET /v1/models``, ``POST /v1/chat/completions``
(streaming and non-streaming), ``GET /health``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from collections.abc import Callable, Generator
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from . import openai_adapter as adapter
from .exceptions import (
    APIError,
    AuthenticationError,
    DeepSeekError,
    NetworkError,
    RateLimitError,
    WafError,
)

MODELS = [
    {"id": "deepseek-chat", "object": "model", "owned_by": "deepseek",
     "description": "DeepSeek web chat without thinking"},
    {"id": "deepseek-reasoner", "object": "model", "owned_by": "deepseek",
     "description": "DeepSeek web chat with thinking (reasoning_content)"},
]


class ChatCompletionRequest(BaseModel):
    model: str = "deepseek-chat"
    messages: list[dict[str, Any]]
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    temperature: float | None = None
    max_tokens: int | None = None
    stop: Any = None
    response_format: dict[str, Any] | None = None

    model_config = {"extra": "allow"}

    @property
    def extra_flags(self) -> dict[str, Any]:
        extra = self.model_extra or {}
        return {k: extra[k] for k in ("thinking_enabled", "search_enabled")
                if k in extra}


def _error(status: int, message: str, kind: str = "api_error"):
    return JSONResponse(status_code=status,
                        content={"error": {"message": message, "type": kind}})


def _map_error(e: Exception):
    if isinstance(e, AuthenticationError):
        return _error(401, str(e), "authentication_error")
    if isinstance(e, RateLimitError):
        return _error(429, str(e), "rate_limit_error")
    if isinstance(e, WafError):
        return _error(503, str(e), "server_error")
    if isinstance(e, (APIError, NetworkError)):
        return _error(500, str(e), "server_error")
    if isinstance(e, (ValueError, KeyError)):
        return _error(400, str(e), "invalid_request_error")
    return _error(500, f"{type(e).__name__}: {e}", "server_error")


def _check_api_key(request: Request) -> JSONResponse | None:
    required = os.getenv("OPENAI_API_KEY")
    if not required:
        return None
    auth = request.headers.get("authorization", "")
    if auth != f"Bearer {required}":
        return _error(401, "Invalid API key", "authentication_error")
    return None


def _default_client_factory():
    from .client import DeepSeekClient

    token = os.getenv("DEEPSEEK_AUTH_TOKEN", "")
    if not token:
        raise AuthenticationError(
            "Server misconfigured: set DEEPSEEK_AUTH_TOKEN")
    return DeepSeekClient(token)


def _run_deepseek(client, prompt: str, thinking: bool, search: bool,
                  ) -> tuple[str, str]:
    session_id = client.create_chat_session()
    thinking_parts, text_parts = [], []
    for chunk in client.chat_completion(
            session_id, prompt,
            thinking_enabled=thinking, search_enabled=search):
        if chunk.type == "thinking":
            thinking_parts.append(chunk.content)
        elif chunk.type == "text":
            text_parts.append(chunk.content)
    return "".join(thinking_parts), "".join(text_parts)


def _collect_live(client, prompt: str, thinking: bool, search: bool,
                  ) -> Generator[tuple[str, str], None, None]:
    """Yield ``(kind, token)`` live: kind is thinking|text."""
    session_id = client.create_chat_session()
    for chunk in client.chat_completion(
            session_id, prompt,
            thinking_enabled=thinking, search_enabled=search):
        if chunk.type == "thinking" and chunk.content:
            yield ("thinking", chunk.content)
        elif chunk.type == "text" and chunk.content:
            yield ("text", chunk.content)


def _debug_dump(request_id: str, body, prompt: str) -> None:
    import sys

    tools = body.tools or []
    print(f"[debug {request_id}] model={body.model} "
          f"tool_choice={body.tool_choice!r} tools={[t.get('function', {}).get('name') for t in tools if isinstance(t, dict)]}",
          file=sys.stderr)
    print(f"[debug {request_id}] PROMPT chars={len(prompt)}",
          file=sys.stderr)
    try:
        Path(f"/tmp/opencode/prompt-{request_id}.txt").write_text(prompt)
    except Exception:
        pass
    for message in body.messages:
        msg = message if isinstance(message, dict) else message.model_dump()
        role = msg.get("role")
        text = adapter.extract_text(msg.get("content"))
        print(f"[debug {request_id}] msg role={role} chars={len(text)} "
              f"head={text[:300]!r}", file=sys.stderr)


def build_app(client_factory: Callable[[], Any] | None = None) -> FastAPI:
    factory = client_factory or _default_client_factory
    app = FastAPI(title="DeepSeek OpenAI-compat API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models():
        return {"object": "list", "data": MODELS}

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest, request: Request):
        denied = _check_api_key(request)
        if denied is not None:
            return denied
        if not body.messages:
            return _error(400, "messages must not be empty",
                          "invalid_request_error")
        try:
            client = factory()
        except DeepSeekError as e:
            return _map_error(e)
        except Exception as e:
            return _map_error(e)

        prompt = adapter.messages_to_prompt(
            [m if isinstance(m, dict) else m.model_dump() for m in body.messages])
        if (body.response_format or {}).get("type") == "json_object":
            prompt += "\n\nRespond with valid JSON only."
        preamble = adapter.build_tool_preamble(body.tools, body.tool_choice)
        if preamble:
            prompt += "\n\n" + preamble
        thinking, search = adapter.resolve_flags(body.model, body.extra_flags)
        request_id = adapter.new_request_id()
        created = adapter.now_epoch()
        if os.getenv("OPENAI_DEBUG"):
            _debug_dump(request_id, body, prompt)

        try:
            if body.stream:
                return _stream_response(
                    factory_client=client, request_id=request_id,
                    created=created, model=body.model, prompt=prompt,
                    thinking=thinking, search=search,
                    tools=body.tools, tool_choice=body.tool_choice,
                    prompt_tokens=adapter.estimate_tokens(prompt))
            thinking_text, text = _run_deepseek(client, prompt, thinking, search)
            if os.getenv("OPENAI_DEBUG"):
                import sys as _sys
                print(f"[debug {request_id}] RAW thinking chars={len(thinking_text)} "
                      f"text chars={len(text)}", file=_sys.stderr)
                print(f"[debug {request_id}] RAW text head={text[:800]!r}",
                      file=_sys.stderr)
            content, calls = adapter.parse_tool_calls(text, body.tools) \
                if preamble else (text, [])
            if calls and not content:
                content_out: str | None = None
            else:
                content_out = content
            completion_tokens = adapter.estimate_tokens(thinking_text + text)
            return adapter.completion_response(
                request_id, created, body.model, thinking_text, content_out,
                calls, adapter.estimate_tokens(prompt), completion_tokens)
        except DeepSeekError as e:
            return _map_error(e)
        except Exception as e:
            return _map_error(e)

    return app


def _stream_response(factory_client, request_id: str, created: int,
                     model: str, prompt: str, thinking: bool, search: bool,
                     tools, tool_choice, prompt_tokens: int):
    use_tools = bool(tools) and tool_choice != "none"

    def event_stream():
        if not use_tools:
            yield adapter.sse_chunk(request_id, created, model,
                                    {"role": "assistant"})
            completion_len = 0
            try:
                for kind, token in _collect_live(
                        factory_client, prompt, thinking, search):
                    completion_len += len(token)
                    if kind == "thinking":
                        yield adapter.sse_chunk(
                            request_id, created, model,
                            {"reasoning_content": token})
                    else:
                        yield adapter.sse_chunk(
                            request_id, created, model, {"content": token})
            except DeepSeekError as e:
                yield adapter.sse_chunk(
                    request_id, created, model, {}, finish_reason="error")
                _ = e
                return
            yield adapter.sse_chunk(request_id, created, model, {},
                                    finish_reason="stop")
            yield "data: [DONE]\n\n"
            return

        thinking_text, text = _run_deepseek(
            factory_client, prompt, thinking, search)
        if os.getenv("OPENAI_DEBUG"):
            import sys as _sys
            print(f"[debug {request_id}] STREAM-RAW thinking chars={len(thinking_text)} "
                  f"text chars={len(text)}", file=_sys.stderr)
            try:
                Path(f"/tmp/opencode/reply-{request_id}.txt").write_text(
                    f"THINKING:\n{thinking_text}\n\nTEXT:\n{text}")
            except Exception:
                pass
        content, calls = adapter.parse_tool_calls(text, tools)
        yield adapter.sse_chunk(request_id, created, model,
                                {"role": "assistant"})
        for i in range(0, len(thinking_text), 200):
            yield adapter.sse_chunk(request_id, created, model, {
                "reasoning_content": thinking_text[i:i + 200]})
        if content:
            yield adapter.sse_chunk(request_id, created, model,
                                    {"content": content})
        for index, call in enumerate(calls):
            yield adapter.sse_chunk(request_id, created, model, {
                "tool_calls": [{
                    "index": index,
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": json.dumps(
                            call["arguments"], ensure_ascii=False),
                    },
                }],
            })
        yield adapter.sse_chunk(
            request_id, created, model, {},
            finish_reason="tool_calls" if calls else "stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


app = build_app()

if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int,
                        default=int(os.getenv("PORT", "8080")))
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
