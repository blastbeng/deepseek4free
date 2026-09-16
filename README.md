# DeepSeek4Free (OpenAI-compatible fork)

**Free access to DeepSeek, Gemini (Web) and ChatGPT (Web) through their own web APIs — exposed as a standard OpenAI-compatible server, packaged in Docker, and built for agent coding.**

This project talks directly to the chat web apps — `chat.deepseek.com`, `gemini.google.com`, `chatgpt.com`, `claude.ai`, `grok.com`, `chat.mistral.ai`, `chat.qwen.ai`, `kimi.com`, `copilot.microsoft.com`, `perplexity.ai` and `chat.z.ai` (GLM) — instead of any official paid API, then re-exposes them behind the familiar OpenAI endpoints (`/v1/chat/completions`, `/v1/models`). That means any tool that speaks the OpenAI API — [aider](https://aider.chat) / **AiderDesk agent mode**, OpenWebUI, LiteLLM, LibreChat, the `openai` SDK, anything else — can use these models for free.

**Zero-credential operation:** a background credential bot *creates every account itself* (auto-signup with auto-generated mailboxes, verification OTPs read automatically), stores the sessions in `./data/` and renews them automatically. Nothing to paste, nothing to maintain.

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
> - **autonomous credential bot** (`dsk/refresher.py`): creates accounts with auto-generated
>   mailboxes (catch-all IMAP or mail.tm throwaways), reads the verification OTPs, stores the
>   sessions in `./data/` and renews them automatically (refresh → re-login → fresh signup)
> - **11-provider router** with dynamic model discovery, fallback chains and a
>   `DSF_PROVIDERS` allowlist; **self-healing** provider patches (`dsk/selfheal.py`)
> - **llmtrim** request stage (`dsk/llmtrim.py`): LLM CALL → llmtrim → proxy rotator → LLM response
> - **fast-only proxy rotation** with the no-proxy route as a first-class candidate
> - a built-in **ChatGPT-style playground** (`/`) with saved chats (24 h expiry)

---

## 📖 How it works

This section explains the whole pipeline, from account creation to the model's answer.

### 1. Credentials are created and renewed automatically

DeepSeek's web app authenticates every request with a bearer `userToken`; the other providers use session cookies. **You never paste any of them**: the credential bot (`dsk/refresher.py`) runs a renewal ladder on its own —

1. **HTTP refresh** of the bot-managed session jars in `./data/` (`deepseek_token`, `gemini_cookies.json`, `chatgpt_cookies.json`, …);
2. **headless-Chromium re-login** with the accounts the bot created itself (`data/accounts.json`);
3. **fully automatic signup**: a throwaway mailbox is generated (catch-all IMAP domain via `DSF_MAIL_DOMAIN`, or a mail.tm temp account with zero configuration), the signup form is filled in a real browser, the verification OTP is read from that mailbox and the fresh account is stored.

Credentials therefore live exclusively in the `./data` volume — the `.env` file contains **no LLM credentials at all** (the server still honours a `userToken` sent as the client's API key, and legacy env vars if present, but nothing requires them). Providers whose web apps have no signup flow (claude, grok, qwen, kimi, mistral) stay dormant until one of their tokens exists in `data/`; they never block the other providers, and `DSF_PROVIDERS` can hide them entirely.

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
- **Per-request direct connection** — non-standard `disable_proxy: true` (top level of the request body, chat and image-generation endpoints) forces the request to skip the rotating proxy pool and connect directly. Handy for low-latency testing when you don't want a random free proxy in the path.

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
| `dsk/providers/jar.py` | Shared bot-managed cookie-jar helpers (`<name>_cookies.json` + `<NAME>_COOKIES` env fallback) |
| `dsk/providers/claude_provider.py` | Claude web provider (`claude.ai` temporary conversations, `sessionKey` cookie) |
| `dsk/providers/grok_provider.py` | Grok web provider (`grok.com` REST app-chat, `sso` cookie, auto image-gen via Aurora) |
| `dsk/providers/mistral_provider.py` | Mistral Le Chat provider (`chat.mistral.ai`, single create-mode stream call; session-token gated) |
| `dsk/providers/qwen_provider.py` | Qwen web provider (`chat.qwen.ai` api/v2, bundled `bx-ua` anti-bot fingerprint) |
| `dsk/providers/kimi_provider.py` | Kimi web provider (`kimi.com` connect+json gRPC-web frames) |
| `dsk/providers/copilot_provider.py` | Microsoft Copilot provider (`copilot.microsoft.com` websocket, anonymous by default) |
| `dsk/providers/perplexity_provider.py` | Perplexity provider (`perplexity.ai` SSE ask, Sonar-backed routes) |
| `dsk/providers/glm_provider.py` | GLM provider (dual backend: anonymous `chat.z.ai` + `chatglm.cn` signed stream) |
| `dsk/providers/router.py` | Dynamic model discovery (TTL cache) + retry/fallback orchestration |
| `dsk/static/index.html` | llama.cpp-style playground web UI (served at `/` and `/playground`) |
| `dsk/wasm/` | DeepSeek's SHA3 WASM module used for PoW |

---

## 🔀 Providers (all reverse-engineered — no official API keys)

Eleven free providers are supported. Each one borrows the credentials of a
normal browser session (or runs fully anonymous) — no API keys, no payments.
Providers without configured credentials are simply skipped; the server runs
with whichever are available. Configure them in `.env` (see
`cp .env.example .env`) and restart the stack
(`sudo systemctl restart docker-compose@deepseek4free` or `docker compose up -d`).

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

### 4. Claude web (`claude.ai`)

Borrows the `sessionKey` cookie of a logged-in browser session. Every request
runs in an ephemeral *temporary* conversation that is never persisted to your
account history.

1. Log in at [claude.ai](https://claude.ai).
2. DevTools (F12) → **Application** → Cookies → `https://claude.ai` → copy `sessionKey`.
3. Put it in `.env`: `CLAUDE_SESSION_KEY=sk-ant-sid01-...` (or
   `CLAUDE_COOKIES={"sessionKey": "..."}`).
4. Models: `claude-sonnet-4-6`, `claude-opus-4-6`, `claude-haiku-4-5` (thinking included).

### 5. Grok web (`grok.com`)

Borrows the `sso` cookie of a logged-in X/Grok session.

1. Log in at [grok.com](https://grok.com).
2. DevTools → **Application** → Cookies → `https://grok.com` → copy `sso`.
3. Put it in `.env`: `GROK_SSO=<value>` (or `GROK_COOKIES={"sso": "..."}`).
4. Models: `grok-4`, `grok-4-reasoning`, `grok-4-heavy`, `grok-3`, `grok-3-mini`,
   `grok-deepsearch`; image generation is available via Aurora.

### 6. Mistral Le Chat (`chat.mistral.ai`) — session token required

Anonymous access is now **account-gated upstream** (it returns an "An account
is now required" upsell instead of model output), so a Le Chat session token
is **required**: DevTools (F12) → Application → Cookies →
`https://chat.mistral.ai` → `session_token` → set
`MISTRAL_SESSION_TOKEN=<value>`. The provider makes a single `POST /api/chat`
`mode: 'create'` call that both starts the conversation and streams the answer
(upstream model `mistral-large-2411`, exposed as route `mistral-large`).

### 7. Qwen web (`chat.qwen.ai`)

Borrows the Bearer token of a logged-in session:

1. Log in at [chat.qwen.ai](https://chat.qwen.ai).
2. DevTools → **Network** → send any message → open an `api/v2/*` request →
   copy the `Authorization` header value without `Bearer `.
3. Put it in `.env`: `QWEN_TOKEN=<value>`.
4. Models: `qwen3.7-max`, `qwen3.6-plus`, `qwen3-coder-plus`, `qwen3.6-35b-a3b`,
   `qwen3.6-27b`. The bundled `bx-ua` anti-bot fingerprint is configurable via
   `QWEN_BX_UA`/`QWEN_UMID` if Alibaba changes it.

### 8. Kimi web (`kimi.com`)

Borrows the `token` cookie of a logged-in session:

1. Log in at [kimi.com](https://www.kimi.com).
2. DevTools → **Application** → Cookies → `https://www.kimi.com` → copy `token`.
3. Put it in `.env`: `KIMI_TOKEN=<value>` (or `KIMI_COOKIES={"token": "..."}`).
4. Models: `kimi-k2.6`, `kimi-k2.5` (thinking included).

### 9. Microsoft Copilot (`copilot.microsoft.com`) — anonymous

Works with **zero credentials** (anonymous tier with synthetic device
cookies, generated and persisted automatically). Paste a whole cookie jar
JSON for authenticated quality: `COPILOT_COOKIES=<JSON>`. Models:
`copilot-chat`, `copilot-think`, `copilot-smart` (streams over websocket).

### 10. Perplexity (`perplexity.ai`) — anonymous

Works with **zero credentials** (anonymous search-backed answers). Paste a
logged-in cookie jar JSON for pro-tier models: `PERPLEXITY_COOKIES=<JSON>`.
Models: `perplexity-turbo`, `perplexity-pro`, `perplexity-reasoning`,
`perplexity-gpt5`, `perplexity-claude-4.5-sonnet`, `perplexity-o3`.

### 11. GLM — Z.ai / ChatGLM (`chat.z.ai`, `chatglm.cn`)

Dual backend, anonymous by default: z.ai serves `glm-4.5` and
`glm-4.5-thinking` with no setup (the access token is fetched automatically).
Setting `GLM_REFRESH_TOKEN` (chatglm.cn → DevTools → Application → Local
Storage → `refresh_token`) unlocks `glm-4.6` and `glm-4.6-thinking` routes.

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
| `CLAUDE_SESSION_KEY` | *(none)* | `sessionKey` cookie from `claude.ai` (enables the Claude web provider) |
| `GROK_SSO` | *(none)* | `sso` cookie from `grok.com` (enables the Grok web provider) |
| `MISTRAL_SESSION_TOKEN` | *(none)* | `session_token` cookie from `chat.mistral.ai` (**required** — anonymous access is account-gated) |
| `QWEN_TOKEN` | *(none)* | Bearer token from `chat.qwen.ai` (enables the Qwen web provider) |
| `QWEN_UMID` / `QWEN_BX_UA` | *(bundled)* | Override Qwen's anti-bot fingerprint headers if upstream changes them |
| `KIMI_TOKEN` | *(none)* | `token` cookie from `kimi.com` (enables the Kimi web provider) |
| `COPILOT_COOKIES` | *(none)* | Cookie jar JSON for `copilot.microsoft.com` (anonymous works without it) |
| `PERPLEXITY_COOKIES` | *(none)* | Cookie jar JSON for `perplexity.ai` (anonymous works without it) |
| `GLM_REFRESH_TOKEN` | *(none)* | `refresh_token` from `chatglm.cn` (unlocks glm-4.6 routes; z.ai anonymous always on) |
| `<PROVIDER>_COOKIES` | *(none)* | Generic JSON cookie-jar fallback for every provider (e.g. `CLAUDE_COOKIES`) |
| `DSF_<PROVIDER>_CONTEXT_LENGTH` / `DSF_<PROVIDER>_MAX_OUTPUT` | *(per-provider defaults)* | Advertised limits for claude/grok/mistral/qwen/kimi/copilot/perplexity/glm routes |
| `DSF_MODELS_TTL` | `300` | Seconds between dynamic model re-discovery across providers |
| `DSF_FALLBACKS` | *(none)* | JSON map of per-model fallback chains, e.g. `{"deepseek-chat": ["deepseek-reasoner"]}` |
| `DSF_DEFAULT_FALLBACKS` | *(none)* | Comma-separated fallbacks applied to every route |
| `COOKIES_DIR` | *(none)* (Docker: `/data`) | Directory where provider cookie files are persisted |
| `DSF_PROXY` / `DSF_PROXIES` | *(none)* | Single / comma-separated proxy URLs (always in the pool) |
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
| `DSF_PROXY_MAX_LATENCY` | `1200` | ms — only proxies answering this fast get traffic (fast-only rotation) |
| `DSF_PROXY_DIRECT` | `true` | Include the no-proxy route in the rotation (`round` mode: direct first) |
| `DSF_PROXY_ENSURE_TIMEOUT` | `90` | Seconds CLI one-shots wait for the warm-up health pass |
| `DSF_ZAI_CONTEXT_LENGTH` | `10000` | Prompt budget (tokens) for the z.ai web transport — its browser input silently fails beyond ~40k characters |

---

## 🔄 Credential refresher (default ON)

The service keeps its web sessions alive **autonomously** — no human re-copying cookies. `GET /health` exposes the live state under `refresher`.

**Renewal ladder** (run when a self-heal probe classifies a provider as `auth`, or proactively every `DSF_REFRESHER_TTL`):

1. **HTTP cookie refresh** (always) — Gemini/ChatGPT cookie jars are rotated over plain HTTP; DeepSeek is verified with a live probe (its `userToken` only changes on login).
2. **Headless-browser re-login** (default ON) — re-signs-in with the per-provider login credentials below inside the container's Chromium and exports the fresh token/cookies.
3. **Auto-signup** (default ON, **all providers**) — when even the login is dead — or a provider has **no credentials at all** — a brand-new free account is **created** (the refresher daemon bootstraps every missing provider on its first cycle). If no `<PROVIDER>_LOGIN_EMAIL` is configured, the address is **auto-generated**:
   - your own **catch-all IMAP domain** (`DSF_MAIL_DOMAIN` + IMAP settings) — random local parts, OTP read from your mailbox; or
   - a **mail.tm throwaway mailbox** (public temp-mail, zero configuration) as fallback.

   Created accounts are persisted to `data/accounts.json` so later renewals can re-login with them. Google/OpenAI may still hit captcha or phone-verification walls — those rungs are **best effort** and their failures surface in the history log.

All rungs respect per-provider cooldowns and daily attempt caps; every action is logged to `data/refresher/history.jsonl` (`python -m dsk.refresher status`, `python -m dsk.refresher bootstrap` to force-create missing credentials now). Everything can be disabled: `DSF_REFRESHER=false`, `DSF_REFRESHER_LOGIN=false`, `DSF_REFRESHER_AUTOSIGNUP=false`, `DSF_MAIL_AUTOGEN=false`.

Bot-written credential files (`data/deepseek_token`, `data/gemini_cookies.json`, `data/chatgpt_cookies.json`) **win over** the env vars — delete a file to hand control back to the environment.

```bash
# environment variables (all shown with their defaults)
DSF_REFRESHER=true
DSF_REFRESHER_TTL=21600        # proactive refresh cycle, seconds
DSF_REFRESHER_COOLDOWN=1800    # per-provider cooldown after an attempt
DSF_REFRESHER_MAX_RENEWS=6     # daily attempt budget per provider
DSF_REFRESHER_LOGIN=true       # rung 2: headless re-login
DSF_REFRESHER_AUTOSIGNUP=true  # rung 3: create fresh accounts
DSF_MAIL_AUTOGEN=true          # auto-create throwaway mailboxes
DSF_MAIL_DOMAIN=               # your catch-all domain (optional; else mail.tm)
DSF_MAIL_IMAP_HOST=            # IMAP mailbox for OTP delivery
DSF_MAIL_IMAP_PORT=993
DSF_MAIL_IMAP_USER=
DSF_MAIL_IMAP_PASS=
DSF_MAIL_OTP_SENDER=deepseek   # sender substring filter
DSF_MAIL_OTP_MAX_AGE=30        # ignore older mail, minutes
DSF_MAIL_OTP_REGEX=\b(\d{6})\b # code extraction

DEEPSEEK_LOGIN_EMAIL=          # optional; autogen e-mail is used when empty
DEEPSEEK_LOGIN_PASSWORD=
GEMINI_LOGIN_EMAIL=
GEMINI_LOGIN_PASSWORD=
CHATGPT_LOGIN_EMAIL=
CHATGPT_LOGIN_PASSWORD=
```

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
# Any single proxy / comma-separated list (always kept in the pool):
DSF_PROXY=socks5://1.2.3.4:1080
DSF_PROXIES=socks5://1.2.3.4:1080,http://5.6.7.8:8080

# FULLY DYNAMIC: automatically aggregate public free-proxy lists from the web
# (TheSpeedX, monosans, proxifly, proxyscrape, roosterkid, geonode — verified
# working sources), refreshed every 30 min and randomly sampled to 400:
DSF_PROXY_AUTO=true
DSF_PROXY_MAX_POOL=400
# Optional custom sources ("socks5=<url>" sets the scheme):
DSF_PROXY_SOURCES=socks5=https://example.com/socks5.txt
# Optional extra list URL(s) (plain text or JSON):
DSF_PROXY_LIST_URL=https://example.com/proxy-list.txt
```

**Health checking** (`DSF_PROXY_CHECK=true`) — strongly recommended with free lists, where typically only ~10% of published proxies are alive at any moment: a background worker probes every pooled proxy concurrently and **only proxies that answer within `DSF_PROXY_MAX_LATENCY` (1200 ms) receive traffic** — slow exits never slow down responses. Combined with `DSF_PROXY_COOLDOWN`, a proxy that dies mid-session is skipped and traffic falls back to direct, so a dead pool never breaks the service.

**Selection is `random` by default** (`DSF_PROXY_MODE=round|single` also available). The no-proxy route is a first-class rotation candidate (`DSF_PROXY_DIRECT=true`); in `round` mode the direct route comes first. A proxy that fails is put on cooldown (`DSF_PROXY_COOLDOWN`, 120s). Providers that misbehave behind proxies (e.g. bot-protection false positives) can be pinned to direct with `DSF_PROXY_EXCLUDE=deepseek`.

**Per-provider randomization:** every provider (`deepseek`, `gemini`, `chatgpt`, …) gets its *own* proxy, chosen randomly and — while the pool is large enough — distinct from the proxies already used by the other providers, so concurrent providers are spread across different exit IPs instead of hammering one shared proxy. Each assignment is sticky for `DSF_PROXY_ROTATE_TTL` seconds (default 300), then the provider is re-randomized; a runtime failure releases the assignment immediately so the next request picks a fresh random proxy. `GET /health` reports current assignments (`assigned=deepseek->1.2.3.4:8080,...`).

Short-lived CLI one-shots (`python -m dsk.refresher …`) start with an empty pool; `ensure_pool()` refreshes the sources and awaits one bounded health pass (`DSF_PROXY_ENSURE_TIMEOUT`, 90 s) so the first ladder already draws fast, validated exits.

Sanity-check your setup from the host: `python -m dsk.proxies` — it prints each pooled proxy and the exit IP it reaches.

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
