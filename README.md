# DeepSeek4Free (OpenAI-compatible fork)

**Free access to DeepSeek, Gemini (Web) and ChatGPT (Web) through their own web APIs — exposed as a standard OpenAI-compatible server, packaged in Docker, and built for agent coding.**

This project talks directly to the chat web apps — `chat.deepseek.com` (account token), `gemini.google.com` (session cookies) and `chatgpt.com` backend-api (session cookies) — instead of any official paid API, then re-exposes them behind the familiar OpenAI endpoints (`/v1/chat/completions`, `/v1/models`). That means any tool that speaks the OpenAI API — [aider](https://aider.chat) / **AiderDesk agent mode**, OpenWebUI, LiteLLM, LibreChat, the `openai` SDK, anything else — can use these models for free.

```
┌──────────────┐   OpenAI API    ┌────────────────────────────┐   web API    ┌──────────────────────────┐
│ aider /      │ ──────────────► │  dsk/openai_server.py      │ ───────────► │ chat.deepseek.com (token)│
│ AiderDesk /  │  /v1/chat/...   │  FastAPI + SSE + tool-call │  reverse-    │ gemini.google.com (cookie│
│ any OpenAI   │ ◄────────────── │  emulation + playground    │  engineered  │ chatgpt.com backend-api  │
│ client       │  SSE chunks     └────────────────────────────┘              └──────────────────────────┘
└──────────────┘
```

**No hardcoded models.** Model lists are discovered *dynamically* from each provider's live web session: DeepSeek exposes its three web-app modes, ChatGPT is discovered via `/backend-api/models`, Gemini via its own web RPC — so new upstream models appear on `/v1/models` automatically (refreshed every `DSF_MODELS_TTL` seconds, with on-demand re-discovery when an unknown model id is requested). Unknown providers are skipped gracefully and previously discovered routes are kept on refresh failures.

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
- **Vision & image generation** — models whose provider supports them accept OpenAI-style multimodal `content` parts (`{"type": "image_url", "image_url": {"url": "data:image/png;base64,…"}}`) and are advertised in `/v1/models` via `vision: true` / `image_gen: true` capability flags. Vision-capable models also power `POST /v1/images/generations` (OpenAI Images API shape, `b64_json` responses). DeepSeek web chat is text-only — sending images to it returns `400 model_does_not_support_vision`; Gemini-web and ChatGPT-web models support both when configured.
- **Per-request direct connection** — non-standard `disable_proxy: true` (top level of the request body, chat and image-generation endpoints) forces the request to skip Tor and the rotating proxy pool and connect directly. Handy for low-latency testing when you don't want a random free proxy in the path.

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
| `dsk/openai_server.py` | **This fork:** OpenAI-compatible API server (aider/AiderDesk entry point) + playground web UI |
| `dsk/providers/base.py` | Provider abstraction, dynamic `Route` model registry, retry/fallback error taxonomy |
| `dsk/providers/deepseek_provider.py` | DeepSeek web provider (PoW + Cloudflare bypass) |
| `dsk/providers/gemini_provider.py` | Gemini web-chat provider (`gemini.google.com` session cookies, no official API) |
| `dsk/providers/chatgpt_provider.py` | ChatGPT web provider (`chatgpt.com/backend-api` session cookies, no official API) |
| `dsk/providers/router.py` | Dynamic model discovery (TTL cache) + retry/fallback orchestration |
| `dsk/static/index.html` | llama.cpp-style playground web UI (served at `/` and `/playground`) |
| `dsk/wasm/` | DeepSeek's SHA3 WASM module used for PoW |

---

## 🔀 Providers (all reverse-engineered — no official API keys)

Three free providers are supported. Each one borrows the credentials of a
normal browser session — no API keys, no payments. Providers without
configured credentials are simply skipped; the server runs with whichever
are available. Configure them in `.env` (see `cp .env.example .env`) and
restart the stack (`sudo systemctl restart docker-compose@deepseek4free`
or `docker compose up -d`).

### 1. DeepSeek (`deepseek-chat`, `deepseek-reasoner`, `deepseek-search`)

Uses your free-account `userToken` from the DeepSeek web app:

1. Log in at [chat.deepseek.com](https://chat.deepseek.com) (a free account is enough).
2. Open DevTools (F12) → **Console** and run:
   `JSON.parse(localStorage.getItem("userToken")).value`
   (alternative: Network tab → send any chat → copy the `authorization`
   header value without the `Bearer ` prefix).
3. Put it in `.env`:
   ```bash
   DEEPSEEK_AUTH_TOKEN=userToken value from step 2
   ```
4. Cloudflare: nothing to configure — the bypass module solves challenges
   with a headless Chromium and caches `cf_clearance` in `./data/cookies.json`
   automatically. If you ever see Cloudflare errors, run `python -m dsk.bypass`
   once (see *Cloudflare cookies* below).

Notes: the `userToken` expires when you log out or rotate sessions — if
requests start returning 401, repeat step 2.

### 2. Gemini web (`gemini.google.com`)

Uses the two session cookies of your Google account:

1. Log in at [gemini.google.com](https://gemini.google.com).
2. DevTools (F12) → **Application** → Cookies → `https://gemini.google.com`.
3. Copy `__Secure-1PSID` and `__Secure-1PSIDTS` into `.env`:
   ```bash
   GEMINI_1PSID=<value of __Secure-1PSID>
   GEMINI_1PSIDTS=<value of __Secure-1PSIDTS>
   ```
4. Models are discovered live from the session (e.g. `gemini-2.5-flash`,
   `gemini-2.5-pro`) — no model list to configure.

Notes: `__Secure-1PSIDTS` rotates periodically; if discovery fails or
requests 401, re-copy **both** cookies. A cookie jar can also be dropped
as a file into the `./data` volume as `gemini_cookies.json`.

### 3. ChatGPT web (`chatgpt.com` backend-api)

Two options — the cookie jar is preferred (the access token is refreshed
automatically):

1. Log in at [chatgpt.com](https://chatgpt.com).
2. Either
   - fetch `https://chatgpt.com/api/auth/session` in the same browser
     (or DevTools → Network) and copy `accessToken` into `.env`:
     ```bash
     CHATGPT_ACCESS_TOKEN=<accessToken>
     ```
   - or export the cookie jar (DevTools → Application → Cookies → export,
     or a JSON array/object of cookies) into `.env`:
     ```bash
     CHATGPT_SESSION_COOKIES=<JSON string>
     ```
3. Alternatively drop the jar as a file into the `./data` volume as
   `chatgpt_cookies.json`.
4. Models are discovered live via `/backend-api/models`.

### Verifying a provider

```bash
# which routes/models are live (discovered from your sessions):
curl -s http://localhost:${DSF_PORT:-8000}/v1/models | python3 -m json.tool

# quick smoke test (streaming):
curl -N -X POST http://localhost:${DSF_PORT:-8000}/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-chat","messages":[{"role":"user","content":"ping"}],"stream":true}'

# per-provider proxy assignments + pool state:
curl -s http://localhost:${DSF_PORT:-8000}/health
```

Unknown model ids are routed by fuzzy match, and per-model fallback
chains (`DSF_FALLBACKS` / `DSF_DEFAULT_FALLBACKS`) kick in automatically
when a provider fails.

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

`.env.example` contains every supported variable (DeepSeek token, optional Gemini/ChatGPT web-session credentials, `DSF_API_KEY`, `DSF_PORT`, model-name overrides, discovery TTL and fallback chains) with comments.

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

## 🖥️ Playground

A llama.cpp-style chat playground is served at `http://localhost:18010/` (and `/playground`): pick any discovered model from the dropdown, stream responses (with a collapsible thinking panel for `reasoning_content`), tweak system message / temperature / max tokens / web-search toggle, and stop generations mid-stream. It talks to the same OpenAI endpoints your agent tools use, so it doubles as an end-to-end test harness.

## 🔌 API reference

| Endpoint | Description |
|---|---|
| `GET /v1/models` | Lists every model discovered from all configured providers (dynamic, TTL-cached) |
| `POST /v1/chat/completions` | Chat completions, streaming (`stream: true`) and non-streaming; supports `tools` |
| `GET /` or `/playground` | Chat playground web UI |
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
| `GEMINI_1PSID` | *(none)* | `__Secure-1PSID` cookie from `gemini.google.com` (enables the Gemini web provider) |
| `GEMINI_1PSIDTS` | *(none)* | `__Secure-1PSIDTS` cookie from `gemini.google.com` |
| `CHATGPT_ACCESS_TOKEN` | *(none)* | ChatGPT web `accessToken` (from `/api/auth/session`) |
| `CHATGPT_SESSION_COOKIES` | *(none)* | ChatGPT session cookie jar as JSON (preferred over the raw token; auto-refresh) |
| `DSF_MODELS_TTL` | `300` | Seconds between dynamic model re-discovery across providers |
| `DSF_FALLBACKS` | *(none)* | JSON map of per-model fallback chains, e.g. `{"deepseek-chat": ["deepseek-reasoner"]}` |
| `DSF_DEFAULT_FALLBACKS` | *(none)* | Comma-separated fallbacks applied to every route |
| `COOKIES_DIR` | *(none)* (Docker: `/data`) | Directory where provider cookie files are persisted |
| `DSF_PROXY_TOR` | `false` | Route provider traffic through the local Tor SOCKS5 proxy |
| `DSF_PROXY_TOR_URL` | `socks5h://torproxy:9050` | Tor proxy URL (docker-network service name) |
| `DSF_PROXY` / `DSF_PROXIES` | *(none)* | Single / comma-separated proxy URLs |
| `DSF_PROXY_LIST_URL` | *(none)* | URL fetching a dynamic proxy list (text/JSON), TTL-refreshed |
| `DSF_PROXY_LIST_TTL` | `3600` | Seconds between dynamic proxy list refreshes |
| `DSF_PROXY_MODE` | `random` | Proxy selection: `random`, `round`, or `single` |
| `DSF_PROXY_EXCLUDE` | *(none)* | Providers that always go direct (e.g. `deepseek`) |
| `DSF_PROXY_COOLDOWN` | `120` | Seconds a failing proxy is skipped |
| `DSF_PROXY_ROTATE_TTL` | `300` | Seconds a provider keeps its assigned proxy before re-randomizing |
| `DSF_PROXY_AUTO` | `false` | Aggregate public free-proxy lists from the web automatically |
| `DSF_PROXY_SOURCES` | *(built-in)* | Override the auto source list (`socks5=<url>`, ... ) |
| `DSF_PROXY_MAX_POOL` | `250` | Random sample cap for the aggregated pool |
| `DSF_PROXY_LIST_URLS` | *(none)* | Extra list URL(s) (text/JSON) fetched with `DSF_PROXY_LIST_TTL` |
| `DSF_PROXY_CHECK` | `false` | Background health probing; traffic only uses alive proxies |
| `DSF_PROXY_CHECK_TTL` | `1800` | Healthy-lease duration / re-check interval |
| `DSF_PROXY_CHECK_TIMEOUT` | `8` | Per-probe timeout in seconds |
| `DSF_PROXY_CHECK_CONCURRENCY` | `24` | Concurrent health probes |

---

## ☁️ Cloudflare cookies

In normal operation cookies are fetched and refreshed automatically. If you hit persistent Cloudflare errors:

1. Run `python -m dsk.bypass` (outside Docker, or with `DOCKERMODE=true` which uses Xvfb). It opens a browser, solves the challenge and writes `dsk/cookies.json`.
2. In Docker the `./data` directory (bind-mounted to `/data`) persists cookies at `/data/cookies.json` across restarts — `dsk/api.py` picks them up automatically.

You only need this when you see Cloudflare challenges, your `cf_clearance` cookie expired, or you get "Please wait a few minutes before trying again".

---

## 🔀 Proxy rotation (rate-limit friendly)

All outbound provider traffic can be routed through one or more **outbound proxies** to spread requests across exit IPs and soften per-IP rate limiting. Configure in `.env`:

```bash
# Use the local Tor proxy (the compose file attaches the container to the
# shared "ai-mcp" docker network where the torproxy container runs):
DSF_PROXY_TOR=true

# Or any single proxy / comma-separated list:
DSF_PROXY=socks5h://torproxy:9050
DSF_PROXIES=socks5://1.2.3.4:1080,http://5.6.7.8:8080

# FULLY DYNAMIC: automatically aggregate public free-proxy lists from the web
# (TheSpeedX, monosans, proxifly, proxyscrape, roosterkid, geonode — verified
# working sources), refreshed every 30 min and randomly sampled to 250:
DSF_PROXY_AUTO=true
DSF_PROXY_MAX_POOL=250
# Optional custom sources ("socks5=<url>" sets the scheme):
DSF_PROXY_SOURCES=socks5=https://example.com/socks5.txt
# Optional extra list URL(s) (plain text or JSON):
DSF_PROXY_LIST_URL=https://example.com/proxy-list.txt
```

**Health checking** (`DSF_PROXY_CHECK=true`) — strongly recommended with free lists, where typically only ~10% of published proxies are alive at any moment: a background worker probes every pooled proxy concurrently and traffic only uses the ones that answer. Combined with `DSF_PROXY_COOLDOWN`, a proxy that dies mid-session is skipped and traffic falls back to direct, so a dead pool never breaks the service.

Selection is `random` by default (`DSF_PROXY_MODE=round|single` also available). A proxy that fails is put on cooldown (`DSF_PROXY_COOLDOWN`, 120s) and traffic falls back to direct, so a dead proxy never breaks the service. Providers that misbehave behind proxies (e.g. bot-protection false positives) can be pinned to direct with `DSF_PROXY_EXCLUDE=deepseek`.

**Per-provider randomization:** every provider (`deepseek`, `gemini`, `chatgpt`, …) gets its *own* proxy, chosen randomly and — while the pool is large enough — distinct from the proxies already used by the other providers, so concurrent providers are spread across different exit IPs instead of hammering one shared proxy. Each assignment is sticky for `DSF_PROXY_ROTATE_TTL` seconds (default 300), then the provider is re-randomized; a runtime failure releases the assignment immediately so the next request picks a fresh random proxy. `GET /health` reports current assignments (`assigned=deepseek->1.2.3.4:8080,...`).

Sanity-check your setup from the host: `DSF_PROXY_TOR=true DSF_PROXY_TOR_URL=socks5h://127.0.0.1:9050 python -m dsk.proxies` — it prints each proxy and the exit IP it reaches.

> Note: DeepSeek rate limits are mostly **per-account**, so Tor helps most for the Gemini/ChatGPT web providers and for IP-level throttling. Tor exit nodes are also sometimes blocked by bot protection — use `DSF_PROXY_EXCLUDE` to tune per provider.

---

## 💻 Local (non-Docker) run

```bash
git clone https://github.com/blastbeng/deepseek4free.git
cd deepseek4free
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt "setuptools<81"
DEEPSEEK_AUTH_TOKEN=yourtoken .venv/bin/python -m dsk.openai_server
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
