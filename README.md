# DeepSeek4Free (OpenAI-compatible fork)

**Free access to DeepSeek through its own web API — exposed as a standard OpenAI-compatible server, packaged in Docker, and built for agent coding.**

This project talks directly to `chat.deepseek.com` with your own account token (`userToken`) instead of the official paid API, then re-exposes it behind the familiar OpenAI endpoints (`/v1/chat/completions`, `/v1/models`). That means any tool that speaks the OpenAI API — [aider](https://aider.chat) / **AiderDesk agent mode**, OpenWebUI, LiteLLM, LibreChat, the `openai` SDK, anything else — can use DeepSeek for free.

```
┌──────────────┐   OpenAI API    ┌────────────────────────────┐   web API   ┌───────────────────┐
│ aider /      │ ──────────────► │  dsk/openai_server.py      │ ──────────► │ chat.deepseek.com │
│ AiderDesk /  │  /v1/chat/...   │  FastAPI + SSE + tool-call │  token+PoW  │  (Cloudflare in   │
│ any OpenAI   │ ◄────────────── │  emulation                 │ ◄────────── │   front)          │
│ client       │  SSE chunks     └────────────────────────────┘  SSE chunks └───────────────────┘
└──────────────┘
```

## 🍴 Fork notice & credits

> This repository is a **fork of [xtekky/deepseek4free](https://github.com/xtekky/deepseek4free)**.
> All credit for the original reverse-engineered DeepSeek client — the `dsk` library, the WASM proof-of-work implementation, and the Cloudflare bypass — goes to **[@xtekky](https://github.com/xtekky)** and the original project's contributors. Huge thanks! 🙏
>
> **What this fork adds on top of the original:**
> - a full **OpenAI-compatible API server** (`dsk/openai_server.py`): `/v1/chat/completions` (streaming + non-streaming), `/v1/models`, `/health`
> - **tool-calling emulation** so function-calling clients (aider / AiderDesk agent mode) work end-to-end
> - `reasoning_content` streaming for the thinking model
> - complete **Docker / docker-compose packaging** with persistent Cloudflare-cookie storage
> - `.env.example` configuration and a `userToken`-as-API-key auth mode

---

## 📖 How it works

This section explains the whole pipeline, from your token to the model's answer.

### 1. Authentication via `userToken`

DeepSeek's web app authenticates every request with a bearer token stored in the browser's local storage under the `userToken` key. There is no official API key — we simply reuse that token:

```js
// run in the browser console on chat.deepseek.com (while logged in)
JSON.parse(localStorage.getItem("userToken")).value
```

The resulting string is the **only credential** this project needs. You can provide it either as the `DEEPSEEK_AUTH_TOKEN` environment variable, or directly as the OpenAI API key of the client (the server accepts your `userToken` in place of an API key).

### 2. Cloudflare bypass (`dsk/bypass.py`, `dsk/CloudflareBypasser.py`)

`chat.deepseek.com` sits behind Cloudflare. If requests start being challenged ("Just a moment…"), the bypass module spins up a real (undetected) Chromium via [DrissionPage](https://github.com/g1879/DrissionPage), visits the site, clicks through the challenge with `CloudflareBypasser`, and captures the resulting `cf_clearance` cookie. The cookie is saved to `cookies.json` (in Docker: `/data/cookies.json`, bind-mounted to the project's `./data` directory) and silently attached to every subsequent API request until it expires.

You normally don't have to do anything — but if you ever see Cloudflare errors, run the helper once:

```bash
python -m dsk.bypass
```

### 3. Proof-of-Work via WASM (`dsk/pow.py`)

Before certain calls, DeepSeek's web API requires a **proof-of-work**: it hands out a challenge (algorithm, challenge string, salt, difficulty) and expects a hash answer back. The site itself solves this in WebAssembly — so this project does the same: `dsk/pow.py` loads the actual `sha3_wasm_bg.*.wasm` binary shipped from DeepSeek's frontend, writes the challenge into WASM memory, and executes DeepSeek's own hashing code (`DeepSeekHash`) to produce a valid answer (`DeepSeekPOW.solve_challenge`). Because it's the genuine WASM module, the answers are always accepted and never break when the algorithm changes.

### 4. The reverse-engineered client (`dsk/api.py`)

`DeepSeekAPI` wraps `https://chat.deepseek.com/api/v0`:

- `create_chat_session()` opens a chat session
- `chat_completion(session_id, prompt, thinking_enabled, search_enabled, parent_message_id)` posts the prompt and **streams Server-Sent-Event chunks** back as a generator
- each chunk is a dict like `{'type': 'thinking' | 'text', 'content': ...}` (plus message ids for threading)
- it also handles header building, cookie refreshing, PoW challenge solving, retries, and maps failures to typed exceptions (`AuthenticationError`, `RateLimitError`, `NetworkError`, `CloudflareError`, `APIError`)

### 5. The OpenAI translation layer (`dsk/openai_server.py`)

This is the main addition of this fork. A FastAPI server translates between the OpenAI wire format and the `dsk` client:

- **Prompt flattening** — the DeepSeek web API only accepts a single flat prompt per turn, while OpenAI clients send a full message list. The server therefore renders the whole conversation into one prompt: `[System]`, `[Assistant]`, `[Tool result]` turns, including assistant messages that contain `tool_calls`.
- **Model mapping** — the OpenAI `model` field is mapped to DeepSeek capabilities:

  | OpenAI model name | DeepSeek behaviour |
  |---|---|
  | `deepseek-reasoner` | thinking process **enabled** |
  | `deepseek-chat` | thinking **disabled** |
  | `deepseek-search` | thinking disabled + **web search** enabled |

  (names are overridable via `DSF_MODEL_THINKER` / `DSF_MODEL_FAST` / `DSF_MODEL_SEARCH`)

  The DeepSeek web API does **not** expose per-model metadata, so `/v1/models` advertises DeepSeek's documented limits — 128K context (`context_length` / `max_model_len`) and max output of 64K for `deepseek-reasoner` / 32K otherwise (`max_completion_tokens` / `max_tokens`). Agent tools like AiderDesk and aider read these fields to size the context window and max output tokens; adjust them via `DSF_CONTEXT_LENGTH`, `DSF_MAX_OUTPUT_THINKING` and `DSF_MAX_OUTPUT` if DeepSeek changes its limits.
- **Streaming** — the synchronous DeepSeek generator runs in a worker thread and is bridged into an async SSE response. `thinking` chunks are re-emitted as OpenAI `reasoning_content` deltas; `text` chunks become normal `delta.content` deltas; the stream ends with a proper `finish_reason` chunk and `data: [DONE]`.
- **Tool-calling emulation** — see the next section.
- **Compatibility** — standard OpenAI parameters (`temperature`, `top_p`, `max_tokens`, `stop`, `seed`, `frequency_penalty`, `presence_penalty`, `response_format`, `stream_options.include_usage`, …) are accepted; the ones DeepSeek cannot honour are tolerated and ignored, so exotic clients never get validation errors.

### 6. Tool calling (agent coding)

DeepSeek's web API has **no native function calling**, which normally breaks agent tools. The server emulates it:

1. When a request includes `tools`, the server appends a **tool protocol** to the flattened prompt: the available functions with their JSON schemas, plus the instruction that the model must answer with exactly one line:

   ```
   TOOL_CALL: {"name": "<tool name>", "arguments": {...}}
   ```

2. The streamed answer is buffered and parsed. If it contains a `TOOL_CALL:` directive, the server converts it back into a **genuine OpenAI tool-call response**: `choices[0].message.tool_calls` (with `call_…` ids and JSON `arguments`), `finish_reason: "tool_calls"`, and the matching tool-call delta chunks in streaming mode.
3. When the conversation comes back with the assistant's `tool_calls` history and `[Tool result]` messages, they are rendered into the next prompt so the model can see its own calls and their results.

The result: aider, **AiderDesk agent mode**, and every other function-calling client work end-to-end — the agent believes it is talking to a real tool-calling model. `tool_choice: "none"` disables the protocol; a forced `tool_choice: {"function": {"name": …}}` is honoured too.

### Module map

| File | Role |
|---|---|
| `dsk/api.py` | Reverse-engineered DeepSeek client (sessions, streaming, errors, cookies) |
| `dsk/pow.py` | WASM proof-of-work solver using DeepSeek's own WASM binary |
| `dsk/bypass.py` | Cloudflare `cf_clearance` cookie fetcher/validator |
| `dsk/CloudflareBypasser.py` | Browser automation that clicks through Cloudflare challenges |
| `dsk/run_and_get_cookies.py` | Standalone helper to grab cookies from a running bypass server |
| `dsk/server.py` | Upstream's original bypass HTTP service (unchanged from the original repo) |
| `dsk/openai_server.py` | **This fork:** OpenAI-compatible API server (aider/AiderDesk entry point) |
| `dsk/wasm/` | DeepSeek's SHA3 WASM module used for PoW |

---

## 🚀 Quick start (Docker — recommended)

### 1. Get your token

Visit [chat.deepseek.com](https://chat.deepseek.com), log in, then either:

- run `JSON.parse(localStorage.getItem("userToken")).value` in the browser console (**recommended**), or
- open DevTools → Network tab, send any chat message, and copy the `authorization` header value (without the `Bearer ` prefix).

### 2. Configure

```bash
cp .env.example .env
# edit .env and set DEEPSEEK_AUTH_TOKEN=<paste your token>
```

`.env.example` contains every supported variable (`DEEPSEEK_AUTH_TOKEN`, optional `DSF_API_KEY`, `DSF_PORT`, and the model-name overrides) with comments.

### 3. Run

```bash
docker compose up -d --build
```

The API is now available at **`http://localhost:8000/v1`**.

> ℹ️ The compose file intentionally sets `restart: no` — on this machine the container's lifecycle is managed by the systemd unit `docker-compose@.service`, so Docker itself must not restart it.

### 4. Point your tools at it

**aider / AiderDesk** (`~/.aider.conf.yml` or environment):

```bash
export OPENAI_API_BASE=http://localhost:8000/v1
export OPENAI_API_KEY=anything          # or your DSF_API_KEY value
aider --model openai/deepseek-chat
```

For agent mode, just enable it in AiderDesk — tool calling is emulated transparently (see *Tool calling* above).

**openai Python SDK:**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="anything")
resp = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)
for chunk in resp:
    delta = chunk.choices[0].delta
    print(delta.reasoning_content or delta.content or "", end="", flush=True)
```

### Alternative: token as API key

You can skip `DEEPSEEK_AUTH_TOKEN` entirely and pass your DeepSeek `userToken` **as the OpenAI API key** — the server uses it directly:

```bash
aider --model openai/deepseek-chat \
      --openai-api-base http://localhost:8000/v1 \
      --openai-api-key <your_userToken>
```

---

## 🔌 API reference

| Endpoint | Description |
|---|---|
| `GET /v1/models` | Lists the exposed models (`deepseek-reasoner`, `deepseek-chat`, `deepseek-search`) |
| `POST /v1/chat/completions` | Chat completions, streaming (`stream: true`) and non-streaming; supports `tools` |
| `GET /health` | Liveness probe |

- Requests may include any standard OpenAI field; unsupported ones are ignored.
- `deepseek-reasoner` streams reasoning as `reasoning_content` deltas.
- Errors follow the OpenAI error shape: `{"error": {"message", "type", "param", "code"}}`.
- Authentication: `Authorization: Bearer <DSF_API_KEY>` if you set one, otherwise any key (including your `userToken`) is accepted.

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `DEEPSEEK_AUTH_TOKEN` | — | Your DeepSeek `userToken` (required unless the client sends it as API key) |
| `DSF_API_KEY` | *(none)* | If set, clients must send this as `Authorization: Bearer <key>` |
| `DSF_PORT` | `8000` | Host port binding (compose maps it to container port 8000) |
| `DSF_HOST` | `0.0.0.0` | Server bind host (non-Docker runs) |
| `DSF_MODEL_THINKER` | `deepseek-reasoner` | Name exposed for the thinking-enabled model |
| `DSF_MODEL_FAST` | `deepseek-chat` | Name exposed for the fast model |
| `DSF_MODEL_SEARCH` | `deepseek-search` | Name exposed for the web-search model |
| `DSF_CONTEXT_LENGTH` | `131072` | Context length advertised on `/v1/models` (DeepSeek's documented 128K) |
| `DSF_MAX_OUTPUT_THINKING` | `65536` | Max output tokens advertised for the thinking model (documented 64K) |
| `DSF_MAX_OUTPUT` | `32768` | Max output tokens advertised for the other models (documented 32K) |

---

## ☁️ Cloudflare cookies

In normal operation cookies are fetched and refreshed automatically. If you hit persistent Cloudflare errors:

1. Run `python -m dsk.bypass` (outside Docker, or with `DOCKERMODE=true` which uses Xvfb). It opens a browser, solves the challenge and writes `dsk/cookies.json`.
2. In Docker the `./data` directory (bind-mounted to `/data`) persists cookies at `/data/cookies.json` across restarts — `dsk/api.py` picks them up automatically.

You only need this when you see Cloudflare challenges, your `cf_clearance` cookie expired, or you get "Please wait a few minutes before trying again".

---

## 💻 Local (non-Docker) run

```bash
git clone https://github.com/blastbeng/deepseek4free.git
cd deepseek4free
pip install -r requirements.txt
DEEPSEEK_AUTH_TOKEN=yourtoken python -m dsk.openai_server
```

---

## 📚 Original `dsk` library usage

The underlying library from the original repo can also be used directly:

### Basic example

```python
from dsk.api import DeepSeekAPI

api = DeepSeekAPI("YOUR_AUTH_TOKEN")
chat_id = api.create_chat_session()

for chunk in api.chat_completion(chat_id, "What is Python?"):
    if chunk['type'] == 'text':
        print(chunk['content'], end='', flush=True)
```

### Thinking process & web search

```python
for chunk in api.chat_completion(
    chat_id,
    "What are the latest developments in AI?",
    thinking_enabled=True,
    search_enabled=True,
):
    if chunk['type'] == 'thinking':
        print(f"🔍 Thinking: {chunk['content']}")
    elif chunk['type'] == 'text':
        print(chunk['content'], end='', flush=True)
```

### Threaded conversations

```python
chat_id = api.create_chat_session()
parent_id = None
for chunk in api.chat_completion(chat_id, "Tell me about neural networks"):
    if chunk['type'] == 'text':
        print(chunk['content'], end='', flush=True)
    elif 'message_id' in chunk:
        parent_id = chunk['message_id']

for chunk in api.chat_completion(
    chat_id,
    "How do they compare to other ML models?",
    parent_message_id=parent_id,
):
    if chunk['type'] == 'text':
        print(chunk['content'], end='', flush=True)
```

### Error handling

```python
from dsk.api import (
    DeepSeekAPI,
    AuthenticationError,
    RateLimitError,
    NetworkError,
    CloudflareError,
    APIError,
)

try:
    api = DeepSeekAPI("YOUR_AUTH_TOKEN")
    chat_id = api.create_chat_session()
    for chunk in api.chat_completion(chat_id, "Your prompt here"):
        if chunk['type'] == 'text':
            print(chunk['content'], end='', flush=True)
except AuthenticationError:
    print("Authentication failed. Please check your token.")
except RateLimitError:
    print("Rate limit exceeded. Please wait before making more requests.")
except CloudflareError as e:
    print(f"Cloudflare protection encountered: {e}")
except NetworkError:
    print("Network error occurred. Check your internet connection.")
except APIError as e:
    print(f"API error occurred: {e}")
```

---

## 🛠️ Troubleshooting

| Symptom | Fix |
|---|---|
| `401` / `invalid_token` | Your `userToken` expired or is wrong — grab a fresh one (step 1 of Quick start) |
| Cloudflare / "Just a moment…" | Run `python -m dsk.bypass` once to refresh `cf_clearance` |
| "Please wait a few minutes before trying again" | DeepSeek rate limiting — wait or switch models |
| Tools never get called / agent misbehaves | Ensure the client sends `tools`; tool calling only activates when tools are present |
| API changes break the client | Update to the latest version — DeepSeek's web API changes frequently |

## ⚠️ Disclaimer

This project uses DeepSeek's **web** interface, not an official API. It is intended for personal, educational use. DeepSeek may change its API at any time, may rate-limit or block automated access, and you remain subject to DeepSeek's terms of service.

## 🙏 Credits

- **[xtekky/deepseek4free](https://github.com/xtekky/deepseek4free)** — the original project: reverse-engineered API client, WASM proof-of-work, Cloudflare bypass.
- **[blastbeng/deepseek4free](https://github.com/blastbeng/deepseek4free)** — this fork: OpenAI-compatible server, tool-calling emulation, Docker packaging, documentation.
- [aider](https://aider.chat) / [AiderDesk](https://github.com/hotstepper23/aiderdesk) — the agent coding tools this fork targets.
