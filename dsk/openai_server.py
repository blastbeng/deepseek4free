"""
OpenAI-compatible API server for DeepSeek4Free (multi-provider).

Exposes standard OpenAI endpoints so any OpenAI client / agent tooling
(aider, aiderdesk, openai SDK, LiteLLM, ...) can use DeepSeek, Gemini and
ChatGPT for free, plus a built-in llama.cpp-style playground UI:

    GET  /                     playground web UI (also /playground)
    POST /v1/chat/completions  streaming + non-streaming
    GET  /v1/models
    GET  /health

Configuration (env):
    DSF_API_KEY       optional API key clients must send as Bearer token
                      (default: none, all keys accepted)
    DSF_HOST          bind host (default 0.0.0.0)
    DSF_PORT          bind port (default 8000)
    DEEPSEEK_AUTH_TOKEN  userToken from chat.deepseek.com localStorage
                         (or an existing dsk/cookies.json is reused)
    GEMINI_1PSID / GEMINI_1PSIDTS  __Secure-1PSID cookies of a logged-in
                         gemini.google.com session (or gemini_cookies.json)
    CHATGPT_ACCESS_TOKEN  accessToken from chatgpt.com/api/auth/session, or
                         session cookies via CHATGPT_SESSION_COOKIES /
                         chatgpt_cookies.json

Models are discovered dynamically from each provider's web session — nothing
is hardcoded (see dsk/providers/router.py). Fallback chains:
    DSF_FALLBACKS     JSON {model_id: [fallback_id, ...]} fallback chains
    DSF_DEFAULT_FALLBACKS  comma list for routes without explicit fallbacks

See dsk/providers/router.py for the full model-registry configuration.

Run:  python -m dsk.openai_server
"""

import json
import os
import queue
import time
import uuid
import secrets
import threading
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from pydantic import BaseModel

from .providers.base import (
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderUnavailableError,
)
from .providers.router import Router

HOST = os.getenv("DSF_HOST", "0.0.0.0")
PORT = int(os.getenv("DSF_PORT", "8000"))

API_KEY = os.getenv("DSF_API_KEY", "")

# Static playground UI (llama.cpp-style chat) served from dsk/static.
STATIC_DIR = Path(__file__).resolve().parent / "static"

ROUTER = Router()
DEFAULT_MODEL = ROUTER.routes[os.getenv("DSF_MODEL_FAST", "deepseek-chat").strip()].model_id \
    if os.getenv("DSF_MODEL_FAST", "deepseek-chat").strip() in ROUTER.routes \
    else next(iter(ROUTER.routes))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Kick off dynamic model discovery in a background thread (never blocks
    startup; failures are tolerated and logged by the router)."""
    threading.Thread(
        target=ROUTER.refresh_models, kwargs={"force": True},
        name="model-discovery", daemon=True,
    ).start()
    yield


app = FastAPI(title="DeepSeek4Free OpenAI-compatible API", lifespan=lifespan)


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


class ChatMessage(BaseModel):
    role: str
    content: Any  # str or list of content parts
    tool_calls: Optional[Any] = None  # assistant tool_calls (OpenAI format)
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = DEFAULT_MODEL
    messages: List[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    # DeepSeek-specific extras (ignored by standard clients)
    search_enabled: Optional[bool] = None
    # OpenAI params accepted for compatibility. Unknown extra fields are
    # silently ignored by pydantic, so exotic clients never get 422s.
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    stop: Optional[Any] = None                        # ignored (no upstream support)
    stream_options: Optional[Dict[str, Any]] = None   # {'include_usage': true}
    response_format: Optional[Dict[str, Any]] = None  # ignored
    frequency_penalty: Optional[float] = None         # ignored
    presence_penalty: Optional[float] = None          # ignored
    seed: Optional[int] = None                        # ignored
    user: Optional[str] = None                        # ignored
    n: Optional[int] = None                           # ignored


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
            if msg.tool_calls:
                calls = []
                raw_calls = msg.tool_calls if isinstance(msg.tool_calls, list) else [msg.tool_calls]
                for tc in raw_calls:
                    fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                    calls.append(f"{fn.get('name', '?')}({fn.get('arguments', '')})")
                suffix = f"\nTool calls: " + "; ".join(calls) if calls else ""
                rendered.append(f"[Assistant]\n{text}{suffix}")
            else:
                rendered.append(f"[Assistant]\n{text}")
        elif role == "tool":
            # Explicit continuation cue: the model otherwise tends to re-issue
            # the same call instead of consuming the result.
            rendered.append(f"[Tool result]\n{text}\n(Tool call completed successfully. "
                            f"Use this result to continue the task. Do NOT repeat the call.)")
        else:
            rendered.append(text)
    return "\n\n".join(rendered).strip()


_TOOL_MARKER = "TOOL_CALL: "


def _render_tool_instructions(tools: List[Dict[str, Any]], tool_choice: Any) -> str:
    """Builds the system block that teaches the DeepSeek model the tool-call
    protocol. DeepSeek's web API has no native function calling, so we emulate
    it: the model answers with a single TOOL_CALL: {json} line which is parsed
    back into OpenAI tool_calls."""
    lines = [
        "# Tool calling protocol",
        "You can use tools to help complete the task. Available tools:",
    ]
    for i, tool in enumerate(tools or [], start=1):
        fn = tool.get("function", {}) if isinstance(tool, dict) else {}
        name = fn.get("name", "unknown")
        desc = fn.get("description", "")
        params = fn.get("parameters", {})
        lines.append(f"{i}. name: {name}")
        if desc:
            lines.append(f"   description: {desc}")
        if params:
            lines.append(f"   parameters (JSON Schema): {json.dumps(params, ensure_ascii=False)}")
    lines += [
        "",
        "To call a tool, your ENTIRE response must be exactly one line in this format",
        "and nothing else (no markdown fences, no extra text):",
        f'{_TOOL_MARKER.strip()} {{"name": "<tool name>", "arguments": {{}}}}',
        "Call at most ONE tool per response; the result is provided afterwards as a [Tool result] message.",
        "After you receive [Tool result] messages, use them to continue or complete the task.",
        "Never repeat a tool call that was already made with the same arguments.",
        "If the task is complete or you have all the information you need, respond with plain text instead of calling a tool.",
        "If you do not need a tool, respond with plain text.",
    ]
    if isinstance(tool_choice, dict) and isinstance(tool_choice.get("function"), dict):
        forced = tool_choice["function"].get("name")
        if forced:
            lines.append(f"You MUST call the tool '{forced}' in your next response.")
    return "\n".join(lines)


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Find and parse the first balanced JSON object in *text*."""
    depth = 0
    start: Optional[int] = None
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        obj = json.loads(text[start:i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except ValueError:
                        pass
                    start = None
    return None


def _parse_tool_call(text: str):
    """Returns (pre_text, tool_name, arguments_json_str) or (text, None, None)."""
    idx = text.find(_TOOL_MARKER)
    if idx == -1:
        return text, None, None
    pre = text[:idx].strip()
    payload = _extract_json_object(text[idx + len(_TOOL_MARKER):])
    if payload is None:
        return text, None, None
    name = payload.get("name") or payload.get("tool") or payload.get("tool_name")
    if not name:
        return text, None, None
    args = payload.get("arguments", payload.get("args", {}))
    if not isinstance(args, str):
        args = json.dumps(args if isinstance(args, dict) else {}, ensure_ascii=False)
    return pre, str(name), args


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
    try:
        from . import proxies as _proxies
        proxy_summary = _proxies.active_summary()
    except Exception:
        proxy_summary = 'unavailable'
    return {"status": "ok", "proxy": proxy_summary}


@app.get("/")
@app.get("/playground")
async def playground():
    """llama.cpp-style chat playground (static, self-contained, no CDN)."""
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


@app.get("/v1/models")
async def list_models(request: Request):
    _check_api_key(request)
    # Best-effort TTL-cached re-discovery so newly available upstream models
    # show up without a restart.
    await asyncio.get_running_loop().run_in_executor(None, ROUTER.refresh_models)
    return {"object": "list", "data": ROUTER.list_models()}


def _error_status(err: ProviderError) -> tuple:
    """Map provider errors to (type, code, HTTP status)."""
    if isinstance(err, ProviderAuthError):
        return "invalid_request_error", "invalid_token", 401
    if isinstance(err, ProviderRateLimitError):
        return "rate_limit_error", "rate_limit", 429
    if isinstance(err, ProviderUnavailableError):
        return "api_error", "provider_unavailable", 502
    return "api_error", "upstream_error", 502


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request):
    client_key = _check_api_key(request)
    # resolve() may trigger a network re-discovery for unknown ids — keep it
    # off the event loop.
    route = await asyncio.get_running_loop().run_in_executor(
        None, lambda: ROUTER.resolve(body.model, auth_key=client_key))

    # Body-level overrides: search flag stays opt-in via the extra field.
    thinking_override = route.thinking_enabled
    search_override = True if body.search_enabled else None

    prompt = _build_prompt(body.messages)
    use_tools = bool(body.tools) and body.tool_choice != "none"
    if use_tools:
        # The protocol block is PREPENDED (not appended): if placed at the end
        # it becomes the most recent text the model reads and its "to call a
        # tool..." phrasing biases the model into re-issuing calls even after
        # a tool result was returned (agent-loop repeat-call bug).
        prompt = f"[System]\n{_render_tool_instructions(body.tools, body.tool_choice)}\n\n{prompt}"
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
    model_name = route.model_id
    prompt_len = len(prompt)

    chunk_gen = ROUTER.stream(
        route, prompt,
        temperature=body.temperature, max_tokens=body.max_tokens,
        auth_key=client_key,
        thinking_override=thinking_override,
        search_override=search_override,
    )

    # ---- Streaming ----
    if body.stream:
        include_usage = bool((body.stream_options or {}).get("include_usage"))
        return StreamingResponse(
            _stream_completion(chunk_gen, created, model_name,
                               use_tools, include_usage, prompt_len),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    # ---- Non-streaming ----
    content_parts: List[str] = []
    reasoning_parts: List[str] = []
    try:
        for chunk in chunk_gen:
            if chunk.get("type") == "thinking" and chunk.get("content"):
                reasoning_parts.append(chunk["content"])
            elif chunk.get("type") == "text" and chunk.get("content"):
                content_parts.append(chunk["content"])
    except ProviderError as e:
        err_type, code, status = _error_status(e)
        return _error_response(str(e), err_type, code, status)

    full_text = "".join(content_parts)
    pre_text, tool_name, tool_args = (full_text, None, None)
    if use_tools:
        pre_text, tool_name, tool_args = _parse_tool_call(full_text)

    message: Dict[str, Any] = {
        "role": "assistant",
        "content": pre_text if tool_name else full_text,
    }
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts)
    finish_reason = "stop"
    if tool_name:
        message["tool_calls"] = [{
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": tool_name, "arguments": tool_args},
        }]
        finish_reason = "tool_calls"

    return {
        "id": _chunk_id(),
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_len // 4,
            "completion_tokens": sum(len(p) for p in content_parts) // 4,
            "total_tokens": (prompt_len + sum(len(p) for p in content_parts)) // 4,
        },
    }


async def _stream_completion(
    chunk_gen: Generator[Dict[str, Any], None, None],
    created: int,
    model: str,
    use_tools: bool = False,
    include_usage: bool = False,
    prompt_len: int = 0,
):
    """Streams a provider-agnostic chunk generator into OpenAI-style SSE chunks.

    The underlying provider generators are synchronous and blocking, so they
    run in a worker thread and formatted SSE payloads are pushed through an
    asyncio queue."""
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

    # Thread-safe bridge: the worker runs in a plain thread, the consumer
    # awaits items via run_in_executor (asyncio.Queue is NOT thread-safe
    # for cross-thread put_nowait and can deadlock the event loop).
    q: "queue.Queue[Optional[str]]" = queue.Queue()
    finish_holder = {"reason": "stop"}

    def _worker():
        buffered_text: List[str] = []
        try:
            for chunk in chunk_gen:
                ctype = chunk.get("type", "")
                content = chunk.get("content", "") or ""
                if not content:
                    continue
                if ctype == "thinking":
                    payload = {"reasoning_content": content}
                elif use_tools:
                    # Tool-call responses must be parsed as a whole, so the
                    # text is buffered and emitted after the stream ends.
                    buffered_text.append(content)
                    continue
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
                q.put(_sse(data))

            if use_tools:
                pre_text, tool_name, tool_args = _parse_tool_call("".join(buffered_text))
                if tool_name:
                    if pre_text:
                        q.put(_sse({
                            "id": cid, "object": "chat.completion.chunk",
                            "created": created, "model": model,
                            "choices": [{"index": 0,
                                         "delta": {"content": pre_text},
                                         "finish_reason": None}],
                        }))
                    q.put(_sse({
                        "id": cid, "object": "chat.completion.chunk",
                        "created": created, "model": model,
                        "choices": [{"index": 0,
                                     "delta": {"tool_calls": [{
                                         "index": 0,
                                         "id": f"call_{uuid.uuid4().hex[:24]}",
                                         "type": "function",
                                         "function": {"name": tool_name,
                                                      "arguments": tool_args},
                                     }]},
                                     "finish_reason": None}],
                    }))
                    finish_holder["reason"] = "tool_calls"
                elif buffered_text:
                    q.put(_sse({
                        "id": cid, "object": "chat.completion.chunk",
                        "created": created, "model": model,
                        "choices": [{"index": 0,
                                     "delta": {"content": "".join(buffered_text)},
                                     "finish_reason": None}],
                    }))
            q.put(None)
        except ProviderAuthError as e:
            q.put(_error_sse(str(e), "invalid_request_error", "invalid_token"))
            q.put(None)
        except ProviderRateLimitError as e:
            q.put(_error_sse(f"Rate limit: {e}", "rate_limit_error", "rate_limit"))
            q.put(None)
        except ProviderError as e:
            q.put(_error_sse(str(e)))
            q.put(None)

    threading.Thread(target=_worker, daemon=True).start()
    loop = asyncio.get_running_loop()

    try:
        while True:
            item = await loop.run_in_executor(None, q.get)
            if item is None:
                break
            yield item
        done = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {"index": 0, "delta": {}, "finish_reason": finish_holder["reason"]}
            ],
        }
        yield _sse(done)
        if include_usage:
            yield _sse({
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [],
                "usage": {
                    "prompt_tokens": prompt_len // 4,
                    "completion_tokens": 0,
                    "total_tokens": prompt_len // 4,
                },
            })
    finally:
        yield "data: [DONE]\n\n"


def main():
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
