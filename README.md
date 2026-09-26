# DeepSeek4Free

A Python package for interacting with the DeepSeek AI chat API. This package provides a clean interface to interact with DeepSeek's chat model, with support for streaming responses, thinking process visibility, and web search capabilities.

### Learn how to reverse engineer private api's !!
- and reverse wasm like it was required here
- [whop.com/reverser-academy](https://whop.com/reverser-academy/) (beta)


> **Service Notice**: DeepSeek API is currently experiencing high load. Work is in progress to integrate additional API providers. Please expect intermittent errors.

> **Note**: If you encounter any errors, please ensure you are using the latest version of this library. The DeepSeek API may change frequently, and updates are released to maintain compatibility.

## Features

- **Streaming Responses**: Real-time interaction with token-by-token output
- **Thinking Process**: Optional visibility into the model's reasoning steps
- **Web Search**: Optional integration for up-to-date information
- **Session Management**: Persistent chat sessions with conversation history
- **Efficient PoW**: WebAssembly-based proof of work implementation
- **Error Handling**: Comprehensive error handling with specific exceptions
- **No Timeouts**: Designed for long-running conversations without timeouts
- **Thread Support**: Parent message tracking for threaded conversations

## Project layout

```
dsk/
  __init__.py    public exports (DeepSeekClient, Chunk, errors)
  client.py      orchestration only (sessions, chat, PoW wiring)
  config.py      single source of truth: URLs, endpoints, UA, versions
  models.py      typed requests/chunks (ChatRequest, Chunk, PowChallenge)
  transport.py   curl-cffi wrapper, WAF detection, biz-code mapping
  sse.py         stateful SSE parser (covered by tests/test_sse.py)
  pow.py         WASM PoW solver
  headers.py     header builder | device.py  device-id store
  cookies.py     cookie store   | browser.py  shared Chromium helpers
  auth.py        email login    | waf.py       Turnstile/WAF bypass
  server.py      cookie server  | bypass.py    cookie CLI (python -m dsk.bypass)
  openai_adapter.py OpenAI translation (pure) |
  anthropic_adapter.py Anthropic translation (pure) |
  responses_adapter.py Responses translation + chain store |
  images.py      image extraction + download/decode for vision |
  openai_server.py  agent APIs: chat, messages, responses
tests/test_sse.py  fixtures for the volatile streaming format
tests/test_openai_adapter.py  OpenAI translation + endpoint tests
tests/test_protocols.py  Anthropic/Responses translation + endpoint tests
tests/test_vision.py  image extraction, upload, vision endpoint tests
```

Rules: frontend-mirroring literals live in `config.py`; raw JSON is converted
in `models.py`/`sse.py` only; `client.py` contains no parsing.

## Updating after frontend changes

1. Fetch `https://fe-static.deepseek.com/chat/static/main.<hash>.js`
   (URL is in the HTML fallback served by `/api/v0/auth/login`).
2. Search it for: `/api/v0/` (endpoints), `x-client-version` / `appVersion`
   (bump `config.CLIENT_VERSION`), `X-DS-PoW-Response` (PoW shape),
   `chat_session_id` payload (new fields go to `models.ChatRequest`),
   `NewSSEEventName` + `response/fragments` (update `sse.py` + fixtures).
3. Run `pytest tests/ -q`.

## OpenAI-compatible server (agent integration)

Expose DeepSeek web chat as agent APIs so harnesses can use it:

```bash
DEEPSEEK_AUTH_TOKEN=... uvicorn dsk.openai_server:app --port 8080
```

Endpoints: `GET /v1/models`, `POST /v1/chat/completions`
(`stream: true/false`), `POST /v1/messages` (Anthropic),
`POST /v1/responses` + `GET /v1/responses/{id}` (Responses),
`GET /health`, `GET /health/deep` (validates the DeepSeek token).

Models: `deepseek-chat` (no thinking), `deepseek-reasoner` (thinking,
returned as `reasoning_content`), `deepseek-vision` (vision model for image
input). Per-request overrides are accepted as
extra body fields: `thinking_enabled`, `search_enabled`, `model_type`.

Vision (image input): pass standard image parts (`image_url` with http(s)
URL or `data:` URI, Anthropic `image` blocks, Responses `input_image`).
The server downloads/decodes each image, uploads it via
`POST /api/v0/file/upload_file`, waits for processing, and sends the
completion with `model_type="vision"` + `ref_file_ids`. Any model switches
to vision automatically when images are present; `deepseek-vision` selects
it with no images attached (text-only vision turn).

Tool calling (agent executes tools): pass standard OpenAI `tools` +
`tool_choice` (`auto`/`none`/`required` or `{"function": {"name": ...}}`).
The server injects the schemas into the prompt and parses
`<tool_call>{"name": ..., "arguments": {...}}</tool_call>` blocks back into
`tool_calls` (`finish_reason: "tool_calls"`). The prompt uses numbered rules
plus a concrete example, and the parser tolerates missing closers, mixed
DSML closers, and harness-native DSML `<invoke>` blocks as fallback.
Multiple calls in one turn are returned with distinct stream indices.
forwards DeepSeek tokens live; with tools it buffers, then emits
`reasoning_content`, `content`, and `tool_calls` chunks plus `data: [DONE]`.

Set `OPENAI_API_KEY` to require a client key; otherwise any key is accepted
and the server's `DEEPSEEK_AUTH_TOKEN` is used for all requests. The server
binds `127.0.0.1` by default; a startup warning is printed when no client
key is set.

Harness matrix (verified = passing nested run, not just unit tests):

| Harness | Protocol | Status |
| --- | --- | --- |
| OpenCode | Chat Completions | verified |
| Claude Code | Anthropic Messages | unit-tested, live run pending |
| Codex CLI | Responses API | unit-tested, live run pending |

Known limits: `temperature` is
accepted but has no effect on this backend, `max_tokens` truncates text
output, usage tokens are length estimates, each request opens a fresh
DeepSeek session, and `stop` is enforced on buffered (non-streaming tool
and protocol) paths. Vision notes: remote downloads are capped at 20MB,
uploads at 100MB; malformed image references return 400; the server waits
up to 60s for file processing before completing.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/deepseek4free.git
cd deepseek4free
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Authentication

To use this package, you need a DeepSeek auth token. Here's how to obtain it:

If you know how to use chrome devtools, simply run this snipped in the console:

```js
JSON.parse(localStorage.getItem("userToken")).value
```

### Method 1: From LocalStorage (Recommended)

<img width="1150" alt="image" src="https://github.com/user-attachments/assets/b4e11650-3d1b-4638-956a-c67889a9f37e" />

1. Visit [chat.deepseek.com](https://chat.deepseek.com)
2. Log in to your account
3. Open browser developer tools (F12 or right-click > Inspect)
4. Go to Application tab (if not visible, click >> to see more tabs)
5. In the left sidebar, expand "Local Storage"
6. Click on "https://chat.deepseek.com"
7. Find the key named `userToken`
8. Copy `"value"` - this is your authentication token

### Method 2: From Network Tab

Alternatively, you can get the token from network requests:

1. Visit [chat.deepseek.com](https://chat.deepseek.com)
2. Log in to your account
3. Open browser developer tools (F12)
4. Go to Network tab
5. Make any request in the chat
6. Find the request headers
7. Copy the `authorization` token (without 'Bearer ' prefix)

### Handling WAF Challenges

If login or the web app hits AWS WAF (`aws-waf-token`), capture cookies:

```bash
python -m dsk.bypass
```

This opens Chromium with the pinned Chrome 132 UA, waits for the WAF token,
and saves it to `dsk/cookies.json` (auto-loaded by the client). The chat
completion API itself normally works without cookies.

## Usage

### Basic Example

```python
from dsk import DeepSeekClient

# Initialize with your auth token
api = DeepSeekClient("YOUR_AUTH_TOKEN")

# Create a new chat session
chat_id = api.create_chat_session()

# Simple chat completion
prompt = "What is Python?"
for chunk in api.chat_completion(chat_id, prompt):
    if chunk.type == 'text':
        print(chunk.content, end='', flush=True)
```

### Advanced Features

#### Thinking Process Visibility

The thinking process shows the model's reasoning steps:

```python
# With thinking process enabled
for chunk in api.chat_completion(
    chat_id,
    "Explain quantum computing",
    thinking_enabled=True
):
    if chunk.type == 'thinking':
        print(f"Thinking: {chunk.content}")
    elif chunk.type == 'text':
        print(chunk.content, end='', flush=True)
```

#### Web Search Integration

Enable web search for up-to-date information:

```python
# With web search enabled
for chunk in api.chat_completion(
    chat_id,
    "What are the latest developments in AI?",
    thinking_enabled=True,
    search_enabled=True
):
    if chunk.type == 'search':
        print(f"Searching: {chunk.content}")
    elif chunk.type == 'text':
        print(chunk.content, end='', flush=True)
```

#### Threaded Conversations

Create threaded conversations by tracking parent messages (int IDs from `ready` chunks).
Without `parent_message_id`, follow-ups start a new branch with no history.

```python
# Start a conversation
chat_id = api.create_chat_session()

# Send initial message
parent_id = None
for chunk in api.chat_completion(chat_id, "Tell me about neural networks"):
    if chunk.type == 'text':
        print(chunk.content, end='', flush=True)
    elif chunk.type == 'ready':
        parent_id = chunk.response_message_id  # int, e.g. 2

# Send follow-up question in the thread
for chunk in api.chat_completion(
    chat_id,
    "How do they compare to other ML models?",
    parent_message_id=parent_id
):
    if chunk.type == 'text':
        print(chunk.content, end='', flush=True)
```

#### Getting an Auth Token

```bash
# Option 1: automated login (email + password)
python -m dsk.auth you@example.com yourpassword
# prints userToken value -> use as DEEPSEEK_AUTH_TOKEN

# Option 2: from browser LocalStorage
# chat.deepseek.com -> DevTools -> Application -> Local Storage -> userToken -> value
```

### Error Handling

The package provides specific exceptions for different error scenarios:

```python
from dsk import (
    DeepSeekClient,
    AuthenticationError,
    RateLimitError,
    NetworkError,
    WafError,
    APIError
)

try:
    api = DeepSeekClient("YOUR_AUTH_TOKEN")
    chat_id = api.create_chat_session()
    
    for chunk in api.chat_completion(chat_id, "Your prompt here"):
        if chunk.type == 'text':
            print(chunk.content, end='', flush=True)
            
except AuthenticationError:
    print("Authentication failed. Please check your token.")
except RateLimitError:
    print("Rate limit exceeded. Please wait before making more requests.")
except WafError as e:
    print(f"WAF protection encountered: {str(e)}")
except NetworkError:
    print("Network error occurred. Check your internet connection.")
except APIError as e:
    print(f"API error occurred: {str(e)}")
