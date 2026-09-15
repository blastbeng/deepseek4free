"""
OpenAI-compatible API server for DeepSeek (reverse-engineered chat API).

Exposes standard OpenAI endpoints so any OpenAI client / agent tooling
(aider, aiderdesk, openai SDK, LiteLLM, ...) can use DeepSeek for free:

    POST /v1/chat/completions   (streaming + non-streaming)
    GET  /v1/models
    GET  /health

Configuration (env):
    DSF_API_KEY       optional API key clients must send as Bearer token
                      (default: none, all keys accepted)
    DSF_HOST          bind host (default 0.0.0.0)
    DSF_PORT          bind port (default 8000)
    DEEPSEEK_AUTH_TOKEN  userToken from chat.deepseek.com localStorage
                         (or an existing dsk/cookies.json is reused)

Run:  python -m dsk.openai_server
"""

import json
import os
import time
import uuid
import secrets
import threading
import asyncio
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from .api import (
    DeepSeekAPI,
    AuthenticationError,
    RateLimitError,
    NetworkError,
    APIError,
)

HOST = os.getenv("DSF_HOST", "0.0.0.0")
PORT = int(os.getenv("DSF_PORT", "8000"))

MODEL_THINKER = os.getenv("DSF_MODEL_THINKER", "deepseek-reasoner")
MODEL_FAST = os.getenv("DSF_MODEL_FAST", "deepseek-chat")
MODEL_SEARCH = os.getenv("DSF_MODEL_SEARCH", "deepseek-search")

API_KEY = os.getenv("DSF_API_KEY", "")

MODELS = [
    {
        "id": MODEL_THINKER,
        "object": "model",
        "created": 1700000000,
        "owned_by": "deepseek4free",
    },
    {
        "id": MODEL_FAST,
        "object": "model",
        "created": 1700000000,
        "owned_by": "deepseek4free",
    },
    {
        "id": MODEL_SEARCH,
        "object": "model",
        "created": 1700000000,
        "owned_by": "deepseek4free",
    },
]

app = FastAPI(title="DeepSeek4Free OpenAI-compatible API")

# One DeepSeekAPI per auth token, guarded by a lock (session creation is not
# thread-safe and DeepSeek rate-limits aggressive parallel requests).
_api_lock = threading.Lock()
_api_instance: Optional[DeepSeekAPI] = None


def _resolve_auth_token(provided_key: Optional[str]) -> Optional[str]:
    """Auth token resolution order:
    1. DEEPSEEK_AUTH_TOKEN env var
    2. key sent by the client (Authorization Bearer) — allows users to pass
       their DeepSeek userToken straight from the OpenAI client config
    3. existing dsk/cookies.json (contains cookies; token may still come from client)
    """
    env_token = os.getenv("DEEPSEEK_AUTH_TOKEN", "").strip()
    if env_token:
        return env_token
    if provided_key and provided_key.strip():
        return provided_key.strip()
    return None


def _check_api_key(request: Request) -> Optional[str]:
    """Validate the OpenAI-style Bearer key if DSF_API_KEY is set.
    Returns the client-provided key (may be the DeepSeek token itself)."""
    auth = request.headers.get("authorization", "")
    provided = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    if API_KEY:
        if not provided or not secrets.compare_digest(provided, API_KEY):
            raise HTTPException(
                status_code=401,
                detail={
                    "error": {
                        "message": "Invalid API key",
                        "type": "invalid_request_error",
                        "code": "invalid_api_key",
                    }
                },
            )
    return provided or None


def _get_api(token: Optional[str] = None) -> DeepSeekAPI:
    global _api_instance
    with _api_lock:
        if _api_instance is None or (token and token != _api_instance.auth_token):
            if not token:
                raise HTTPException(
                    status_code=401,
                    detail={
                        "error": {
                            "message": "No DeepSeek auth token. Set DEEPSEEK_AUTH_TOKEN "
                            "or send your userToken as the API key.",
                            "type": "invalid_request_error",
                            "code": "missing_token",
                        }
                    },
                )
            _api_instance = DeepSeekAPI(token)
        return _api_instance


class ChatMessage(BaseModel):
    role: str
    content: Any  # str or list of content parts


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_FAST
    messages: List[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    # DeepSeek-specific extras (ignored by standard clients)
    search_enabled: Optional[bool] = None


def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return "" if content is None else str(content)


def _build_prompt(messages: List[ChatMessage]) -> str:
    """DeepSeek chat API takes a single flat prompt per request, so we render
    the full OpenAI message list into one prompt, marking system/assistant turns."""
    rendered = []
    for msg in messages:
        role = msg.role if msg.role in ("system", "user", "assistant", "tool") else "user"
        text = _flatten_content(msg.content)
        if role == "system":
            rendered.append(f"[System]\n{text}")
        elif role == "assistant":
            rendered.append(f"[Assistant]\n{text}")
        elif role == "tool":
            rendered.append(f"[Tool result]\n{text}")
        else:
            rendered.append(text)
    return "\n\n".join(rendered).strip()


def _chunk_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def _sse(data: Dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _error_response(message: str, err_type: str, code: str, status: int) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "message": message,
                "type": err_type,
                "param": None,
                "code": code,
            }
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models(request: Request):
    _check_api_key(request)
    return {"object": "list", "data": MODELS}


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request):
    client_key = _check_api_key(request)
    token = _resolve_auth_token(client_key)
    api = _get_api(token)

    model = body.model if body.model in (MODEL_THINKER, MODEL_FAST, MODEL_SEARCH) else MODEL_FAST
    thinking_enabled = model == MODEL_THINKER
    search_enabled = bool(body.search_enabled) or model == MODEL_SEARCH

    prompt = _build_prompt(body.messages)
    if not prompt:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": "messages must not be empty",
                    "type": "invalid_request_error",
                    "code": "empty_messages",
                }
            },
        )

    created = int(time.time())
    model_name = model

    # ---- Streaming ----
    if body.stream:
        return StreamingResponse(
            _stream_completion(api, prompt, thinking_enabled, search_enabled,
                               created, model_name),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    # ---- Non-streaming ----
    content_parts: List[str] = []
    try:
        session_id = api.create_chat_session()
        for chunk in api.chat_completion(
            session_id, prompt,
            thinking_enabled=thinking_enabled,
            search_enabled=search_enabled,
        ):
            if chunk.get("type") == "text" and chunk.get("content"):
                content_parts.append(chunk["content"])
    except AuthenticationError as e:
        return _error_response(str(e), "invalid_request_error", "invalid_token", 401)
    except RateLimitError as e:
        return _error_response(str(e), "rate_limit_error", "rate_limit", 429)
    except NetworkError as e:
        return _error_response(str(e), "api_error", "network_error", 502)
    except (APIError, ValueError) as e:
        return _error_response(str(e), "api_error", "upstream_error", 502)

    return {
        "id": _chunk_id(),
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "".join(content_parts)},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt) // 4,
            "completion_tokens": sum(len(p) for p in content_parts) // 4,
            "total_tokens": (len(prompt) + sum(len(p) for p in content_parts)) // 4,
        },
    }
async def _stream_completion(
    api: DeepSeekAPI,
    prompt: str,
    thinking_enabled: bool,
    search_enabled: bool,
    created: int,
    model: str,
):
    """Streams the blocking DeepSeek generator into OpenAI-style SSE chunks.

    The DeepSeek client is synchronous and blocking, so it runs in a worker
    thread and formatted SSE payloads are pushed through an asyncio queue."""
    cid = _chunk_id()
    first = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None,
            }
        ],
    }
    yield _sse(first)

    def _error_sse(message: str, err_type: str = "api_error", code: str = "upstream_error") -> str:
        return _sse(
            {
                "error": {
                    "message": message,
                    "type": err_type,
                    "param": None,
                    "code": code,
                }
            }
        )

    queue: "asyncio.Queue[Optional[str]]" = asyncio.Queue()

    def _worker():
        try:
            session_id = api.create_chat_session()
            for chunk in api.chat_completion(
                session_id,
                prompt,
                thinking_enabled=thinking_enabled,
                search_enabled=search_enabled,
            ):
                ctype = chunk.get("type", "")
                content = chunk.get("content", "") or ""
                if not content:
                    continue
                if ctype == "thinking":
                    payload = {"reasoning_content": content}
                else:
                    payload = {"content": content}
                data = {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {"index": 0, "delta": payload, "finish_reason": None}
                    ],
                }
                queue.put_nowait(_sse(data))
            queue.put_nowait(None)
        except AuthenticationError as e:
            queue.put_nowait(_error_sse(str(e), "invalid_request_error", "invalid_token"))
            queue.put_nowait(None)
        except RateLimitError as e:
            queue.put_nowait(_error_sse(f"Rate limit: {e}", "rate_limit_error", "rate_limit"))
            queue.put_nowait(None)
        except (NetworkError, APIError, ValueError) as e:
            queue.put_nowait(_error_sse(str(e)))
            queue.put_nowait(None)

    threading.Thread(target=_worker, daemon=True).start()

    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item
        done = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {"index": 0, "delta": {}, "finish_reason": "stop"}
            ],
        }
        yield _sse(done)
    finally:
        yield "data: [DONE]\n\n"


def main():
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
