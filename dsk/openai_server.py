"""OpenAI-compatible API over DeepSeek web chat.

Run:
    DEEPSEEK_AUTH_TOKEN=... uvicorn dsk.openai_server:app --port 8080
    # or with auto-login (token cached in ~/.deepseek_token):
    DEEPSEEK_EMAIL=... DEEPSEEK_PASSWORD=... uvicorn dsk.openai_server:app

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
from . import anthropic_adapter as anthropic
from . import responses_adapter as responses
from . import token_store
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
    {"id": "deepseek-vision", "object": "model", "owned_by": "deepseek",
     "description": "DeepSeek web chat vision model (image input)"},
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


class AnthropicMessagesRequest(BaseModel):
    model: str = "deepseek-chat"
    max_tokens: int = 1024
    messages: list[dict[str, Any]]
    system: Any = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    stream: bool = False
    thinking: dict[str, Any] | None = None
    temperature: float | None = None
    stop_sequences: list[str] | None = None

    model_config = {"extra": "allow"}

    @property
    def extra_flags(self) -> dict[str, Any]:
        extra = self.model_extra or {}
        return {k: extra[k] for k in ("thinking_enabled", "search_enabled")
                if k in extra}


class ResponsesRequest(BaseModel):
    model: str = "deepseek-chat"
    input: Any = None
    instructions: str | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    stream: bool = False
    max_output_tokens: int | None = None
    previous_response_id: str | None = None

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

    return DeepSeekClient(token_store.resolve_auth_token())


async def _resilient_turn(factory, client, image_urls, prompt, thinking,
                          search, tools, tool_choice, stop, model_type):
    """Upload vision files + run one turn; re-login once on 401.

    Retry only applies when the token came from credential auto-login
    (explicit ``DEEPSEEK_AUTH_TOKEN`` setups fail fast instead).
    """
    try:
        file_ids = await _run_blocking(
            _prepare_vision_files, client, image_urls) if image_urls else []
        return await _run_blocking(
            _complete_turn, client, prompt, thinking, search,
            tools, tool_choice, stop, model_type, file_ids)
    except AuthenticationError:
        if not token_store.auto_login_enabled():
            raise
        token_store.invalidate_cached_token()
        fresh = await _run_blocking(factory)
        file_ids = await _run_blocking(
            _prepare_vision_files, fresh, image_urls) if image_urls else []
        return await _run_blocking(
            _complete_turn, fresh, prompt, thinking, search,
            tools, tool_choice, stop, model_type, file_ids)


async def _run_blocking(func, *args):
    """Run blocking DeepSeek I/O off the event loop."""
    import anyio as _anyio

    return await _anyio.to_thread.run_sync(func, *args)


def _run_deepseek(client, prompt: str, thinking: bool, search: bool,
                  model_type: str = "default",
                  ref_file_ids: list[str] | None = None,
                  ) -> tuple[str, str]:
    session_id = client.create_chat_session()
    thinking_parts, text_parts = [], []
    for chunk in client.chat_completion(
            session_id, prompt,
            thinking_enabled=thinking, search_enabled=search,
            model_type=model_type, ref_file_ids=ref_file_ids):
        if chunk.type == "thinking":
            thinking_parts.append(chunk.content)
        elif chunk.type == "text":
            text_parts.append(chunk.content)
    return "".join(thinking_parts), "".join(text_parts)


def _collect_live(client, prompt: str, thinking: bool, search: bool,
                  model_type: str = "default",
                  ref_file_ids: list[str] | None = None,
                  ) -> Generator[tuple[str, str], None, None]:
    """Yield ``(kind, token)`` live: kind is thinking|text."""
    session_id = client.create_chat_session()
    for chunk in client.chat_completion(
            session_id, prompt,
            thinking_enabled=thinking, search_enabled=search,
            model_type=model_type, ref_file_ids=ref_file_ids):
        if chunk.type == "thinking" and chunk.content:
            yield ("thinking", chunk.content)
        elif chunk.type == "text" and chunk.content:
            yield ("text", chunk.content)


def _prepare_vision_files(client, image_urls: list[str]) -> list[str]:
    """Resolve images to bytes, upload, await processing; return file ids.

    Blocking: call via :func:`_run_blocking`. Raises ``ValueError`` for bad
    references (mapped to 400) and ``APIError`` for upload failures.
    """
    from .images import resolve_image

    file_ids = []
    for index, url in enumerate(image_urls):
        data, filename, content_type = resolve_image(url, index)
        info = client.upload_file(data, filename, content_type)
        file_id = info.get("id") if isinstance(info, dict) else None
        if not file_id:
            raise APIError(f"Upload returned no file id for {filename}")
        file_ids.append(file_id)
    if file_ids:
        client.wait_for_files(file_ids)
    return file_ids


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


def _complete_turn(client, prompt: str, thinking: bool, search: bool,
                   tools_openai, tool_choice_openai, stop=None,
                   model_type: str = "default",
                   ref_file_ids: list[str] | None = None):
    """Run one DeepSeek turn; return ``(thinking, content, calls)``."""
    thinking_text, text = _run_deepseek(
        client, prompt, thinking, search, model_type, ref_file_ids)
    if tools_openai and tool_choice_openai != "none":
        content, calls = adapter.parse_tool_calls(text, tools_openai)
    else:
        content, calls = text, []
    return thinking_text, adapter.truncate_at_stop(content, stop), calls


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

    @app.get("/health/deep")
    async def health_deep():
        """Validate the DeepSeek token via a side-effect-free PoW challenge."""
        from .exceptions import AuthenticationError as _AuthError

        try:
            client = await _run_blocking(factory)
        except DeepSeekError as e:
            return _map_error(e)
        try:
            import anyio as _anyio

            await _anyio.to_thread.run_sync(client.get_pow_challenge)
            return {"status": "ok", "deepseek": "reachable"}
        except _AuthError as e:
            return _error(503, f"DeepSeek token invalid: {e}", "server_error")
        except DeepSeekError as e:
            return _map_error(e)

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
        messages = [m if isinstance(m, dict) else m.model_dump()
                    for m in body.messages]
        image_urls = adapter.extract_images(messages)
        model_type = adapter.resolve_model_type(
            body.model, bool(image_urls), body.extra_flags)
        try:
            client = await _run_blocking(factory)
        except DeepSeekError as e:
            return _map_error(e)
        except Exception as e:
            return _map_error(e)

        prompt = adapter.messages_to_prompt(messages)
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
                if image_urls:
                    file_ids = await _run_blocking(
                        _prepare_vision_files, client, image_urls)
                else:
                    file_ids = []
                return _stream_response(
                    factory_client=client, request_id=request_id,
                    created=created, model=body.model, prompt=prompt,
                    thinking=thinking, search=search,
                    tools=body.tools, tool_choice=body.tool_choice,
                    stop=body.stop, model_type=model_type,
                    ref_file_ids=file_ids)
            thinking_text, content, calls = await _resilient_turn(
                factory, client, image_urls, prompt, thinking, search,
                body.tools if preamble else None, body.tool_choice, body.stop,
                model_type)
            if os.getenv("OPENAI_DEBUG"):
                import sys as _sys
                print(f"[debug {request_id}] RAW thinking chars={len(thinking_text)} "
                      f"content chars={len(content)} calls={len(calls)}",
                      file=_sys.stderr)
                print(f"[debug {request_id}] RAW text head={content[:800]!r}",
                      file=_sys.stderr)
            if calls and not content:
                content_out: str | None = None
            else:
                content_out = content
            completion_tokens = adapter.estimate_tokens(thinking_text + content)
            return adapter.completion_response(
                request_id, created, body.model, thinking_text, content_out,
                calls, adapter.estimate_tokens(prompt), completion_tokens)
        except DeepSeekError as e:
            return _map_error(e)
        except Exception as e:
            return _map_error(e)

    store = responses.ResponseStore()

    @app.post("/v1/messages")
    async def anthropic_messages(body: AnthropicMessagesRequest,
                                 request: Request):
        denied = _check_api_key(request)
        if denied is not None:
            return denied
        if not body.messages:
            return _anthropic_error(400, "messages must not be empty")
        image_urls = anthropic.extract_images(body.system, body.messages)
        try:
            client = await _run_blocking(factory)
        except DeepSeekError as e:
            return _map_anthropic_error(e)
        except Exception as e:
            return _map_anthropic_error(e)

        openai_messages = anthropic.to_openai_messages(
            body.system, body.messages)
        tools = anthropic.to_openai_tools(body.tools)
        choice = anthropic.to_openai_tool_choice(body.tool_choice)
        prompt = adapter.messages_to_prompt(openai_messages)
        preamble = adapter.build_tool_preamble(tools, choice)
        if preamble:
            prompt += "\n\n" + preamble
        thinking, search = adapter.resolve_flags(body.model, body.extra_flags)
        model_type = adapter.resolve_model_type(
            body.model, bool(image_urls), body.extra_flags)
        if (body.thinking or {}).get("type") == "enabled":
            thinking = True
        elif (body.thinking or {}).get("type") == "disabled":
            thinking = False
        message_id = f"msg_{adapter.new_request_id()[9:]}"
        if os.getenv("OPENAI_DEBUG"):
            _debug_dump(request_id=message_id, body=body, prompt=prompt)

        try:
            if body.stream:
                if image_urls:
                    file_ids = await _run_blocking(
                        _prepare_vision_files, client, image_urls)
                else:
                    file_ids = []
                thinking_text, content, calls = await _run_blocking(
                    _complete_turn, client, prompt, thinking, search,
                    tools if preamble else None, choice,
                    None, model_type, file_ids)
                content, truncated = adapter.truncate_to_token_budget(
                    content, body.max_tokens)
                prompt_tokens = adapter.estimate_tokens(prompt)
                completion_tokens = adapter.estimate_tokens(
                    thinking_text + content)

                def anthropic_events():
                    yield from _anthropic_stream(
                        message_id, body.model, thinking_text, content,
                        calls, prompt_tokens, completion_tokens, truncated)

                return StreamingResponse(anthropic_events(),
                                         media_type="text/event-stream")
            thinking_text, content, calls = await _resilient_turn(
                factory, client, image_urls, prompt, thinking, search,
                tools if preamble else None, choice,
                None, model_type)
            content, truncated = adapter.truncate_to_token_budget(
                content, body.max_tokens)
            prompt_tokens = adapter.estimate_tokens(prompt)
            completion_tokens = adapter.estimate_tokens(
                thinking_text + content)
            return anthropic.build_response(
                body.model, thinking_text, content, calls,
                prompt_tokens, completion_tokens, truncated)
        except DeepSeekError as e:
            return _map_anthropic_error(e)
        except Exception as e:
            return _map_anthropic_error(e)

    @app.post("/v1/responses")
    async def create_response(body: ResponsesRequest, request: Request):
        denied = _check_api_key(request)
        if denied is not None:
            return denied
        raw_input = body.input
        input_items = raw_input if isinstance(raw_input, list) \
            else ([{"type": "message", "role": "user",
                    "content": [{"type": "input_text",
                                 "text": str(raw_input)}]}]
                  if raw_input else [])
        if not input_items:
            return _error(400, "input must not be empty",
                          "invalid_request_error")
        try:
            client = await _run_blocking(factory)
        except DeepSeekError as e:
            return _map_error(e)
        except Exception as e:
            return _map_error(e)

        chained = store.chain(body.previous_response_id)
        openai_messages = responses.to_openai_messages(
            body.instructions, input_items, chained)
        image_urls = responses.extract_images(input_items) \
            + adapter.extract_images(openai_messages)
        # Deduplicate (input items are also flattened into messages).
        image_urls = list(dict.fromkeys(image_urls))
        model_type = adapter.resolve_model_type(
            body.model, bool(image_urls), body.extra_flags)
        tools = responses.to_openai_tools(body.tools)
        choice = responses.to_openai_tool_choice(body.tool_choice)
        prompt = adapter.messages_to_prompt(openai_messages)
        preamble = adapter.build_tool_preamble(tools, choice)
        if preamble:
            prompt += "\n\n" + preamble
        thinking, search = adapter.resolve_flags(body.model, body.extra_flags)
        response_id = responses.new_response_id()
        created_at = responses.now_epoch()

        try:
            if body.stream:
                if image_urls:
                    file_ids = await _run_blocking(
                        _prepare_vision_files, client, image_urls)
                else:
                    file_ids = []
                thinking_text, content, calls = await _run_blocking(
                    _complete_turn, client, prompt, thinking, search,
                    tools if preamble else None, choice,
                    None, model_type, file_ids)
                content, _ = adapter.truncate_to_token_budget(
                    content, body.max_output_tokens)
                payload = responses.build_response(
                    response_id, body.model, created_at, thinking_text,
                    content, calls, adapter.estimate_tokens(prompt),
                    adapter.estimate_tokens(thinking_text + content))
                store.save(response_id, input_items + payload["output"])
                return StreamingResponse(_responses_stream(payload),
                                         media_type="text/event-stream")
            thinking_text, content, calls = await _resilient_turn(
                factory, client, image_urls, prompt, thinking, search,
                tools if preamble else None, choice,
                None, model_type)
            content, _ = adapter.truncate_to_token_budget(
                content, body.max_output_tokens)
            payload = responses.build_response(
                response_id, body.model, created_at, thinking_text,
                content, calls, adapter.estimate_tokens(prompt),
                adapter.estimate_tokens(thinking_text + content))
            store.save(response_id, input_items + payload["output"])
            return payload
        except DeepSeekError as e:
            return _map_error(e)
        except Exception as e:
            return _map_error(e)

    @app.get("/v1/responses/{response_id}")
    async def get_response(response_id: str):
        items = store.chain(response_id)
        if not items:
            return _error(404, "response not found", "invalid_request_error")
        return {"id": response_id, "object": "response", "output": items}

    return app


def _anthropic_error(status: int, message: str, kind: str = "invalid_request_error"):
    return JSONResponse(status_code=status, content={
        "type": "error",
        "error": {"type": kind, "message": message},
    })


def _map_anthropic_error(e: Exception):
    if isinstance(e, AuthenticationError):
        return _anthropic_error(401, str(e), "authentication_error")
    if isinstance(e, RateLimitError):
        return _anthropic_error(429, str(e), "rate_limit_error")
    if isinstance(e, (APIError, NetworkError, WafError)):
        return _anthropic_error(500, str(e), "api_error")
    if isinstance(e, (ValueError, KeyError)):
        return _anthropic_error(400, str(e), "invalid_request_error")
    return _anthropic_error(500, f"{type(e).__name__}: {e}", "api_error")


def _anthropic_stream(message_id: str, model: str, thinking: str,
                      content: str, calls: list[dict],
                      prompt_tokens: int, completion_tokens: int,
                      truncated: bool):
    def event(kind: str, payload: dict) -> str:
        return f"event: {kind}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    blocks: list[dict] = []
    if thinking:
        blocks.append({"type": "thinking", "thinking": thinking})
    blocks.append({"type": "text", "text": content})
    for call in calls:
        blocks.append({"type": "tool_use", "id": call["id"],
                       "name": call["name"], "input": call["arguments"]})
    stop_reason = "tool_use" if calls else "end_turn"
    if truncated:
        stop_reason = "max_tokens"
    yield event("message_start", {"type": "message_start", "message": {
        "id": message_id, "type": "message", "role": "assistant",
        "model": model, "content": [], "stop_reason": None, "usage": {
            "input_tokens": prompt_tokens, "output_tokens": 0}}})
    for index, block in enumerate(blocks):
        yield event("content_block_start", {
            "type": "content_block_start", "index": index,
            "content_block": {k: v for k, v in block.items()}})
        delta: dict
        if block["type"] == "thinking":
            delta = {"type": "thinking_delta", "thinking": block["thinking"]}
        elif block["type"] == "text":
            delta = {"type": "text_delta", "text": block["text"]}
        else:
            delta = {"type": "input_json_delta",
                     "partial_json": json.dumps(block["input"],
                                                ensure_ascii=False)}
        yield event("content_block_delta", {
            "type": "content_block_delta", "index": index, "delta": delta})
        yield event("content_block_stop", {
            "type": "content_block_stop", "index": index})
    yield event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": completion_tokens}})
    yield event("message_stop", {"type": "message_stop"})


def _responses_stream(response: dict):
    def event(kind: str, payload: dict) -> str:
        return f"event: {kind}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    yield event("response.created", {"type": "response.created",
                                    "response": response})
    yield event("response.in_progress", {"type": "response.in_progress",
                                        "response": response})
    for index, item in enumerate(response["output"]):
        yield event("response.output_item.added", {
            "type": "response.output_item.added",
            "output_index": index, "item": item})
        if item["type"] == "message":
            text = item["content"][0]["text"]
            for i in range(0, len(text), 200):
                yield event("response.output_text.delta", {
                    "type": "response.output_text.delta",
                    "item_id": item["id"], "output_index": index,
                    "content_index": 0, "delta": text[i:i + 200]})
        elif item["type"] == "function_call":
            yield event("response.function_call_arguments.delta", {
                "type": "response.function_call_arguments.delta",
                "item_id": item["id"], "output_index": index,
                "delta": item["arguments"]})
        yield event("response.output_item.done", {
            "type": "response.output_item.done",
            "output_index": index, "item": item})
    yield event("response.completed", {"type": "response.completed",
                                       "response": response})


def _stream_response(factory_client, request_id: str, created: int,
                     model: str, prompt: str, thinking: bool, search: bool,
                     tools, tool_choice, stop,
                     model_type: str = "default",
                     ref_file_ids: list[str] | None = None):
    use_tools = bool(tools) and tool_choice != "none"

    def event_stream():
        if not use_tools:
            yield adapter.sse_chunk(request_id, created, model,
                                    {"role": "assistant"})
            completion_len = 0
            try:
                for kind, token in _collect_live(
                        factory_client, prompt, thinking, search,
                        model_type, ref_file_ids):
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
            factory_client, prompt, thinking, search, model_type, ref_file_ids)
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
        content = adapter.truncate_at_stop(content, stop)
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


def _preflight_auth() -> None:
    """Fail fast when no auth is configured; report the active mode."""
    import sys as _sys

    if os.getenv(token_store.TOKEN_ENV):
        print("Auth: explicit DEEPSEEK_AUTH_TOKEN", file=_sys.stderr)
    elif token_store.load_cached_token():
        print(f"Auth: cached token ({token_store.default_token_path()})",
              file=_sys.stderr)
    elif token_store.credentials_available():
        print("Auth: no token yet; will auto-login with DEEPSEEK_EMAIL "
              "on first request", file=_sys.stderr)
    else:
        print("Error: set DEEPSEEK_AUTH_TOKEN, or DEEPSEEK_EMAIL and "
              "DEEPSEEK_PASSWORD for auto-login", file=_sys.stderr)
        _sys.exit(2)


app = build_app()

if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int,
                        default=int(os.getenv("PORT", "8080")))
    parser.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    args = parser.parse_args()
    if not os.getenv("OPENAI_API_KEY"):
        import sys as _sys

        print("Warning: OPENAI_API_KEY is not set; any client can use this "
              "server and spend the DeepSeek account quota. Set OPENAI_API_KEY "
              "for any non-local exposure.", file=_sys.stderr)
    _preflight_auth()
    uvicorn.run(app, host=args.host, port=args.port)
