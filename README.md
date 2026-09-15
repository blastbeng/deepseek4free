# DeepSeek4Free

A Python package for interacting with the DeepSeek AI chat API. This package provides a clean interface to interact with DeepSeek's chat model, with support for streaming responses, thinking process visibility, and web search capabilities.

## 🍴 Fork Notice

> This repository is a **fork** of [blastbeng/deepseek4free](https://github.com/blastbeng/deepseek4free).
> All credit for the original reverse-engineered DeepSeek API client, the WASM proof-of-work implementation, and the Cloudflare bypass goes to [@blastbeng](https://github.com/blastbeng) — huge thanks! 🙏
>
> This fork extends the original library with an **OpenAI-compatible API server** (`/v1/chat/completions`, `/v1/models`) suitable for agent coding tools (e.g. [aider](https://aider.chat) / aiderdesk), plus full **Docker / docker-compose** packaging.

### Learn how to reverse engineer private api's !!
- and reverse wasm like it was required here
- [whop.com/reverser-academy](https://whop.com/reverser-academy/) (beta)


> ⚠️ **Service Notice**: DeepSeek API is currently experiencing high load. Work is in progress to integrate additional API providers. Please expect intermittent errors.

> 📝 **Note**: If you encounter any errors, please ensure you are using the latest version of this library. The DeepSeek API may change frequently, and updates are released to maintain compatibility.

## ✨ Features

- 🔄 **Streaming Responses**: Real-time interaction with token-by-token output
- 🤔 **Thinking Process**: Optional visibility into the model's reasoning steps
- 🔍 **Web Search**: Optional integration for up-to-date information
- 💬 **Session Management**: Persistent chat sessions with conversation history
- ⚡ **Efficient PoW**: WebAssembly-based proof of work implementation
- 🛡️ **Error Handling**: Comprehensive error handling with specific exceptions
- ⏱️ **No Timeouts**: Designed for long-running conversations without timeouts
- 🧵 **Thread Support**: Parent message tracking for threaded conversations

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/deepseek4free.git
cd deepseek4free
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🔑 Authentication

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

### Handling Cloudflare Challenges

If you encounter Cloudflare challenges ("Just a moment..." page), you'll need to get a `cf_clearance` cookie. Run this command:

```bash
python -m dsk.bypass
```

This will:
1. Open an undetected browser
2. Visit DeepSeek and solve the Cloudflare challenge
3. Capture and save the `cf_clearance` cookie
4. The cookie will be automatically used in future requests

You only need to run this when:
- You get Cloudflare challenges in your requests
- Your existing cf_clearance cookie expires
- You see the error "Please wait a few minutes before trying again"

The captured cookie will be stored in `dsk/cookies.json` and automatically used by the API.

## 🐳 Docker (OpenAI-compatible server)

The easiest way to run DeepSeek4Free is as an OpenAI-compatible server in Docker. It exposes the standard OpenAI endpoints (`/v1/chat/completions`, `/v1/models`) so it works with **aider, aiderdesk, OpenWebUI, LiteLLM, LibreChat, the `openai` SDK**, and any other tool that speaks the OpenAI API.

### 1. Configure your token

```bash
cp .env.example .env
# then edit .env and set DEEPSEEK_AUTH_TOKEN
```

### 2. Start the server

```bash
docker compose up -d --build
```

The API is now available at `http://localhost:8000/v1`.

### 3. Point your tools at it

**aider / aiderdesk** (`~/.aider.conf.yml` or env):
```bash
export OPENAI_API_BASE=http://localhost:8000/v1
export OPENAI_API_KEY=anything          # or your DSF_API_KEY value
aider --model openai/deepseek-chat
```

**openai python sdk**
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="anything")
resp = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)
for chunk in resp:
    print(chunk.choices[0].delta.content or "", end="")
```

### Exposed models

| Model | Maps to |
|---|---|
| `deepseek-reasoner` | DeepSeek with thinking process enabled |
| `deepseek-chat` | DeepSeek without thinking process |
| `deepseek-search` | DeepSeek with web search enabled |

Streaming responses include the model's reasoning as `reasoning_content` deltas for `deepseek-reasoner`.

### Configuration (env vars)

| Variable | Default | Description |
|---|---|---|
| `DEEPSEEK_AUTH_TOKEN` | — | Your DeepSeek userToken (required unless provided per-request as API key) |
| `DSF_API_KEY` | *(none)* | If set, clients must send this as `Authorization: Bearer <key>` |
| `DSF_PORT` | `8000` | Host port binding |
| `DSF_HOST` | `0.0.0.0` | Server bind host (non-Docker runs) |
| `DSF_MODEL_THINKER` | `deepseek-reasoner` | Name of the thinking-enabled model |
| `DSF_MODEL_FAST` | `deepseek-chat` | Name of the fast model |
| `DSF_MODEL_SEARCH` | `deepseek-search` | Name of the web-search model |

### Alternative: pass your token as the API key

You can skip `DEEPSEEK_AUTH_TOKEN` entirely and pass your DeepSeek userToken **as the OpenAI API key** — the server uses it directly:

```bash
aider --model openai/deepseek-chat --openai-api-base http://localhost:8000/v1 \
      --openai-api-key <your_userToken>
```

### Cloudflare cookies in Docker

If requests get blocked by Cloudflare, obtain a `cf_clearance` cookie once (needs a real display, run outside Docker or with `DOCKERMODE=true` which uses Xvfb):

```bash
python -m dsk.bypass    # writes dsk/cookies.json
```

The `dsf-data` Docker volume persists cookies at `/data/cookies.json` between restarts; `dsk/api.py` reads them automatically.

### Local (non-Docker) run

```bash
pip install -r requirements.txt
DEEPSEEK_AUTH_TOKEN=yourtoken python -m dsk.openai_server
```

## 📚 Original Library Usage

### Basic Example

```python
from dsk.api import DeepSeekAPI

# Initialize with your auth token
api = DeepSeekAPI("YOUR_AUTH_TOKEN")

# Create a new chat session
chat_id = api.create_chat_session()

# Simple chat completion
prompt = "What is Python?"
for chunk in api.chat_completion(chat_id, prompt):
    if chunk['type'] == 'text':
        print(chunk['content'], end='', flush=True)
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
    if chunk['type'] == 'thinking':
        print(f"🤔 Thinking: {chunk['content']}")
    elif chunk['type'] == 'text':
        print(chunk['content'], end='', flush=True)
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
    if chunk['type'] == 'thinking':
        print(f"🔍 Searching: {chunk['content']}")
    elif chunk['type'] == 'text':
        print(chunk['content'], end='', flush=True)
```

#### Threaded Conversations

Create threaded conversations by tracking parent messages:

```python
# Start a conversation
chat_id = api.create_chat_session()

# Send initial message
parent_id = None
for chunk in api.chat_completion(chat_id, "Tell me about neural networks"):
    if chunk['type'] == 'text':
        print(chunk['content'], end='', flush=True)
    elif 'message_id' in chunk:
        parent_id = chunk['message_id']

# Send follow-up question in the thread
for chunk in api.chat_completion(
    chat_id,
    "How do they compare to other ML models?",
    parent_message_id=parent_id
):
    if chunk['type'] == 'text':
        print(chunk['content'], end='', flush=True)
```

### Error Handling

The package provides specific exceptions for different error scenarios:

```python
from dsk.api import (
    DeepSeekAPI, 
    AuthenticationError,
    RateLimitError,
    NetworkError,
    CloudflareError,
    APIError
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
    print(f"Cloudflare protection encountered: {str(e)}")
except NetworkError:
    print("Network error occurred. Check your internet connection.")
except APIError as e:
    print(f"API error occurred: {str(e)}")
