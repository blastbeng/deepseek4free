"""
OpenAI-compatible API server for DeepSeek4Free (multi-provider).

Exposes standard OpenAI endpoints so any OpenAI client / agent tooling
(aider, aiderdesk, openai SDK, LiteLLM, ...) can use DeepSeek, Gemini and
ChatGPT for free, plus a built-in llama.cpp-style playground UI:

    GET  /                     playground web UI (also /playground)
    POST /v1/chat/completions  streaming + non-streaming (vision via image_url parts)
    POST /v1/images/generations  image generation (capable models only)
    GET  /v1/models
    GET  /health
    GET  /selfheal/status      self-maintenance (selfheal + refresher) status
    POST /selfheal/probe       force a probe cycle (heal/renew on failure)
    POST /selfheal/refresh     force a credential refresh cycle

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

import base64
import json
import logging
import os
import queue
import re
import time
import uuid
import secrets
import threading
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from pydantic import BaseModel, ConfigDict

from .providers.base import (
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderUnavailableError,
    Route,
    fetch_image_bytes,
    parse_data_uri,
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
    """Kick off dynamic model discovery and the self-maintenance daemons in
    background threads (never blocks startup; failures are tolerated)."""
    threading.Thread(
        target=ROUTER.refresh_models, kwargs={"force": True},
        name="model-discovery", daemon=True,
    ).start()
    try:
        from . import selfheal as _selfheal
        _selfheal.start_daemon()
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[selfheal] daemon unavailable: {exc}")
    try:
        from . import refresher as _refresher
        _refresher.start_daemon()
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[refresher] daemon unavailable: {exc}")
    yield


app = FastAPI(title="DeepSeek4Free OpenAI-compatible API", lifespan=lifespan)
logger = logging.getLogger('dsk.openai_server')


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
    # extra='allow' keeps the common non-standard top-level ``image_url``
    # field (normalized in _normalize_image_fields) instead of dropping it.
    model_config = ConfigDict(extra='allow')
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
    # Per-request proxy control (non-standard extension): force a DIRECT
    # connection for this request — skips Tor and the rotating free-proxy
    # pool entirely. Useful for fast, low-latency testing.
    disable_proxy: bool = False
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


class ImageGenerationRequest(BaseModel):
    model: str = DEFAULT_MODEL
    prompt: str
    # Accepted-for-compat params the web apps cannot honor per-image.
    n: Optional[int] = 1      # web apps generate a single image per request
    size: Optional[str] = None
    quality: Optional[str] = None
    style: Optional[str] = None
    response_format: Optional[str] = 'url'  # 'url' | 'b64_json'
    user: Optional[str] = None
    # Per-request proxy control (non-standard extension): force a DIRECT
    # connection for this request — skips Tor and the rotating free-proxy
    # pool entirely. Useful for fast, low-latency testing.
    disable_proxy: bool = False


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


def _extract_images(messages: List[ChatMessage]) -> List[Dict[str, Any]]:
    """Collect OpenAI ``image_url`` content parts from user messages.

    Returns provider-agnostic attachments ``{'mime': str, 'data': bytes}``:
    ``data:`` URIs are base64-decoded, remote URLs are downloaded. Non-image
    or unknown-scheme parts are skipped silently so exotic clients never 422.
    """
    images: List[Dict[str, Any]] = []
    for msg in messages:
        if msg.role != "user" or not isinstance(msg.content, list):
            continue
        for part in msg.content:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            image_url = part.get("image_url")
            url = image_url.get("url") if isinstance(image_url, dict) else image_url
            if not isinstance(url, str) or not url:
                continue
            if url.startswith("data:"):
                mime, data = parse_data_uri(url)
            elif url.startswith(("http://", "https://")):
                mime, data = fetch_image_bytes(url)
            else:
                continue
            if not mime.startswith("image/"):
                mime = "image/png"
            images.append({"mime": mime, "data": data})
    return images


def _normalize_image_fields(messages: List[ChatMessage]) -> None:
    """Accept the common non-standard top-level ``image_url`` message field.

    Some clients put ``image_url`` on the message object instead of inside a
    content part; without normalization the image is silently dropped and
    the request degrades to plain text. Mutates messages in place."""
    for msg in messages:
        if msg.role != 'user' or not isinstance(msg.content, str):
            continue
        img = getattr(msg, 'image_url', None)  # extra field (extra='allow')
        if not img:
            continue
        url = img.get('url') if isinstance(img, dict) else img
        if isinstance(url, str) and url.strip():
            msg.content = [{'type': 'text', 'text': msg.content},
                           {'type': 'image_url', 'image_url': {'url': url}}]


def _has_image_parts(messages: List[ChatMessage]) -> bool:
    """Cheap sync check: does any user message carry image_url parts?"""
    for msg in messages:
        if msg.role == "user" and isinstance(msg.content, list):
            for part in msg.content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return True
    return False


def _chain_capability(route, key: str) -> Optional[bool]:
    """Whether any target in the route's fallback chain supports a capability.

    Returns False only when every fallback is known and none has it; None when
    a fallback target has not been discovered yet (the router decides at
    serve time, it skips incapable targets)."""
    fallbacks = [f for f in (route.fallbacks or []) if f != route.model_id]
    if any(f not in ROUTER.routes for f in fallbacks):
        return None
    if getattr(route, key, False):
        return True
    return any(getattr(ROUTER.routes[f], key, False) for f in fallbacks)


def _llmtrim_stage(messages: List[ChatMessage], route: Route) -> List[ChatMessage]:
    """LLM CALL -> llmtrim -> proxy rotator -> LLM response.

    Always-on payload trim before the request enters the router/provider
    layer (whose HTTP calls go through the proxy rotator). Stale history is
    dropped and oversized messages middle-out truncated so every egress
    route carries the smallest sufficient payload.
    """
    try:
        from dsk.llmtrim import trim_messages
        trimmed, stats = trim_messages(messages, route.context_length,
                                       route.max_output_tokens)
        if stats.get('trimmed'):
            logger.info('llmtrim: %s -> %s chars (dropped %s, truncated %s)',
                        stats.get('in_chars'), stats.get('out_chars'),
                        stats.get('dropped'), stats.get('truncated'))
        return trimmed
    except Exception as e:  # noqa: BLE001 — trimming must never break a call
        logger.warning('llmtrim stage skipped: %s', e)
        return messages


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
                # Re-render past calls in the exact DSML protocol the model
                # was taught (and natively knows) so it recognizes its own
                # behavior instead of learning a second, inconsistent format.
                raw_calls = (msg.tool_calls if isinstance(msg.tool_calls, list)
                             else [msg.tool_calls])
                invokes = []
                for tc in raw_calls:
                    fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                    raw_args = fn.get("arguments", "{}")
                    try:
                        args_obj = (json.loads(raw_args)
                                    if isinstance(raw_args, str) else raw_args)
                    except ValueError:
                        args_obj = raw_args
                    if not isinstance(args_obj, dict):
                        args_obj = {"value": args_obj}
                    parts = [f'<｜DSML｜ invoke name="{fn.get("name", "?")}">']
                    for k, v in args_obj.items():
                        if isinstance(v, str):
                            parts.append(f'<｜DSML｜ parameter name="{k}" '
                                         f'string="true">{v}</｜DSML｜ parameter>')
                        else:
                            parts.append(f'<｜DSML｜ parameter name="{k}" '
                                         f'string="false">'
                                         f'{json.dumps(v, ensure_ascii=False)}'
                                         f'</｜DSML｜ parameter>')
                    parts.append("</｜DSML｜ invoke>")
                    invokes.append("\n".join(parts))
                suffix = (("\n<｜DSML｜ calls>\n" + "\n".join(invokes)
                           + "\n</｜DSML｜ calls>") if invokes else "")
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
# LLMs routinely mangle the exact marker (bold, backticks, missing space,
# lowercase, full-width colon). Accept all of these variants:
_TOOL_CALL_RE = re.compile(
    r"[`*]{0,3}(?:TOOL[\s_\-]?CALL)[`*]{0,3}\s*(?::|：|=)\s*[`*]{0,3}",
    re.IGNORECASE,
)

# --- DeepSeek native DSML tool-call markup ----------------------------------
# The chat.deepseek.com agent models natively emit their internal DSML markup
# when they want to call tools, e.g.:
#   <｜DSML｜ calls>
#   <｜DSML｜ invoke name="power---bash">
#   <｜DSML｜ parameter name="command" string="true">ls -la</｜DSML｜ parameter>
#   <｜DSML｜ parameter name="timeout" string="false">120000</｜DSML｜ parameter>
#   </｜DSML｜ invoke>
#   </｜DSML｜ calls>
# ('｜' is U+FF5C FULLWIDTH VERTICAL LINE; models vary the bar count.) When
# this native markup was used instead of the taught protocol it leaked
# verbatim into content and the tool calls never executed on the client.
_DSML_TAG_RE = re.compile(
    r"<(/?)(｜+)\s*DSML\s*(｜+)\s*(\w+)\s*([^>]*)>",
    re.IGNORECASE,
)
_DSML_ATTR_RE = re.compile(r'([\w:-]+)\s*=\s*"([^"]*)"')
# Opening of a tool-call section, for early boundary freezing while streaming.
_DSML_OPEN_RE = re.compile(r"<｜+\s*DSML\s*｜+\s*(?:calls?|invoke)\b", re.IGNORECASE)


def _parse_dsml_calls(text: str) -> Tuple[str, List[Dict[str, str]]]:
    """Parse native DSML tool-call markup.

    Returns (pre_text, calls); calls is empty when the text contains no
    usable DSML block. Tolerates missing <calls> wrappers, unclosed invokes
    and template placeholders echoed from the instructions."""
    tags = list(_DSML_TAG_RE.finditer(text))
    if not tags:
        return text, []
    calls: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None
    param: Optional[Tuple[str, bool, int]] = None
    pre_end: Optional[int] = None
    for m in tags:
        closing = m.group(1) == "/"
        tag = m.group(4).lower()
        attrs = dict(_DSML_ATTR_RE.findall(m.group(5) or ""))
        if tag == "calls":
            if not closing and pre_end is None:
                pre_end = m.start()
            continue
        if tag == "invoke":
            if not closing:
                if cur is not None:            # tolerate unclosed invoke
                    calls.append(cur)
                cur = {"name": (attrs.get("name") or "").strip(), "args": {}}
                if pre_end is None:
                    pre_end = m.start()
            elif cur is not None:
                calls.append(cur)
                cur = None
            continue
        if tag == "parameter" and cur is not None:
            if not closing:
                param = (attrs.get("name", ""),
                         attrs.get("string", "").strip().lower() == "true",
                         m.end())
            elif param is not None:
                pname, as_str, start = param
                content = text[start:m.start()]
                if as_str:
                    value: Any = content
                else:
                    parsed = _loads_repaired(content.strip())
                    value = content if parsed is None else parsed
                if pname:
                    cur["args"][pname] = value
                param = None
            continue
    if cur is not None:
        calls.append(cur)
    if not calls:
        return text, []
    out: List[Dict[str, str]] = []
    for c in calls:
        name = c["name"]
        # Skip template placeholders the model may echo from the instructions.
        if not name or "<" in name:
            continue
        args = c["args"]
        # Native-style emission wraps the whole OpenAI arguments object in a
        # single <parameter name="arguments" string="false">{...}</parameter>.
        # Unwrap it so the tool receives its own flat parameters instead of
        # a bogus "arguments" key.
        if set(args) == {"arguments"}:
            v = args["arguments"]
            if isinstance(v, str):
                v = _loads_repaired(v.strip())
            if isinstance(v, dict):
                args = v
        out.append({"name": name,
                    "arguments": json.dumps(args, ensure_ascii=False)})
        if len(out) >= 8:                      # parallel-call cap
            break
    if not out:
        return text, []
    pre = text[:pre_end].strip() if pre_end is not None else ""
    return pre, out


def _loads_repaired(raw: str):
    """json.loads with pragmatic repairs for typical LLM quirks (smart
    quotes, trailing commas). Returns None when unrecoverable."""
    try:
        return json.loads(raw)
    except ValueError:
        pass
    fixed = (raw.replace("\u201c", '"').replace("\u201d", '"')
             .replace("\u2018", "'").replace("\u2019", "'"))
    try:
        return json.loads(fixed)
    except ValueError:
        pass
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
    try:
        return json.loads(fixed)
    except ValueError:
        pass
    # Invalid escape sequences from shell/regex content (e.g. \| \( \d) —
    # models often emit them inside "string=false" JSON. Double the
    # backslash on any \X that JSON does not define.
    fixed = re.sub(
        r'\\(u[0-9a-fA-F]{4}|["\\/bfnrt])|\\(.)',
        lambda m: m.group(0) if m.group(1) else "\\\\" + m.group(2),
        fixed)
    try:
        return json.loads(fixed)
    except ValueError:
        return None


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
                    obj = _loads_repaired(text[start:i + 1])
                    if isinstance(obj, dict):
                        return obj
                    start = None
    return None


def _parse_tool_calls(text: str) -> Tuple[str, List[Dict[str, str]]]:
    """Extract (pre_text, [{name, arguments}, ...]) from model output.

    Accepts the native DeepSeek DSML markup first, then the taught
    TOOL_CALL: {...} protocol. Tolerates marker variants, junk around the
    JSON payload and malformed JSON (best-effort repair). Returns all calls
    found (parallel tool calls are supported); pre_text is whatever the
    model wrote before the first tool-call block."""
    pre, calls = _parse_dsml_calls(text)
    if calls:
        return pre, calls
    calls: List[Dict[str, str]] = []
    pre_end = len(text)
    for match in _TOOL_CALL_RE.finditer(text):
        payload = _extract_json_object(text[match.end():])
        if payload is None:
            break
        name = payload.get("name") or payload.get("tool") or payload.get("tool_name")
        if not name:
            break
        args = payload.get("arguments",
                           payload.get("args", payload.get("parameters", {})))
        if not isinstance(args, str):
            args = json.dumps(args if isinstance(args, dict) else {},
                              ensure_ascii=False)
        calls.append({"name": str(name), "arguments": args})
        pre_end = min(pre_end, match.start())
        if len(calls) >= 8:  # hard cap on parallel calls
            break
    if not calls:
        return text, []
    return text[:pre_end].strip(), calls


def _parse_tool_call(text: str):
    """Returns (pre_text, tool_name, arguments_json_str) or (text, None, None)."""
    pre, calls = _parse_tool_calls(text)
    if calls:
        return pre, calls[0]["name"], calls[0]["arguments"]
    return text, None, None


def _render_tool_instructions(tools: List[Dict[str, Any]], tool_choice: Any) -> str:
    """Builds the system block that teaches the DeepSeek model the tool-call
    protocol. DeepSeek's web API has no native function calling; its agent
    models natively know DSML markup, so we teach exactly that (matching
    their internal format maximizes adherence) and still parse the legacy
    TOOL_CALL: {json} shorthand as a fallback."""
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
        "To call tool(s), end your response with a DSML tool-call block using",
        "exactly this markup (keep the special fullwidth bars ｜ intact):",
        "",
        "<｜DSML｜ calls>",
        '<｜DSML｜ invoke name="<tool name>">',
        '<｜DSML｜ parameter name="<param name>" string="true">plain text value</｜DSML｜ parameter>',
        '<｜DSML｜ parameter name="<param name>" string="false">{"json": "value"}</｜DSML｜ parameter>',
        "</｜DSML｜ invoke>",
        "</｜DSML｜ calls>",
        "",
        "Rules:",
        '- string="true" = the value is a plain string; string="false" = the value is raw JSON',
        "  (objects, arrays, numbers, booleans). Omit optional parameters you do not need.",
        "- Use one <DSML invoke> per tool; multiple invokes may share one block to run in",
        "  parallel. Any text before the block is delivered as your prose answer.",
        "- The tool result arrives as a [Tool result] message; then continue the task.",
        "- Never repeat a tool call that was already made with the same arguments.",
        "- If the task is complete or you have all the information you need, respond with",
        "  plain text and NO tool-call block.",
        "- As a shorthand you may instead answer with a single line:",
        '  TOOL_CALL: {"name": "<tool name>", "arguments": {}}',
    ]
    if isinstance(tool_choice, dict) and isinstance(tool_choice.get("function"), dict):
        forced = tool_choice["function"].get("name")
        if forced:
            lines.append(f"You MUST call the tool '{forced}' in your next response.")
    return "\n".join(lines)


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
    try:
        from . import selfheal as _selfheal
        selfheal_summary = _selfheal.status()
    except Exception:
        selfheal_summary = 'unavailable'
    try:
        from . import refresher as _refresher
        refresher_summary = _refresher.status()
    except Exception:
        refresher_summary = 'unavailable'
    return {"status": "ok", "proxy": proxy_summary,
            "selfheal": selfheal_summary, "refresher": refresher_summary}


@app.get("/selfheal/status")
async def selfheal_status():
    """Self-maintenance status: upstream health probes, auto-patch incidents
    and credential-renewal bookkeeping."""
    out: dict = {}
    try:
        from . import selfheal as _selfheal
        out["selfheal"] = _selfheal.status()
    except Exception as exc:
        out["selfheal"] = {"error": str(exc)}
    try:
        from . import refresher as _refresher
        out["refresher"] = _refresher.status()
    except Exception as exc:
        out["refresher"] = {"error": str(exc)}
    return out


@app.post("/selfheal/probe")
async def selfheal_probe():
    """Force a probe cycle now: structural failures trigger LLM auto-patching,
    auth failures trigger credential renewal (on-demand, daemon-independent)."""
    try:
        from . import selfheal as _selfheal
        results = await asyncio.get_running_loop().run_in_executor(
            None, _selfheal.probe_cycle)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"probed": results}


@app.post("/selfheal/refresh")
async def selfheal_refresh():
    """Force a credential refresh cycle now (HTTP cookie/token refresh rung)."""
    try:
        from . import refresher as _refresher
        results = await asyncio.get_running_loop().run_in_executor(
            None, _refresher.refresh_cycle)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"refreshed": results}


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

    _normalize_image_fields(body.messages)
    trimmed_messages = _llmtrim_stage(body.messages, route)
    prompt = _build_prompt(trimmed_messages)
    # Image parts may require downloading remote URLs — keep that off the
    # event loop (no-op scan when the request carries no image parts).
    if _has_image_parts(trimmed_messages):
        images = await asyncio.get_running_loop().run_in_executor(
            None, lambda: _extract_images(trimmed_messages))
    else:
        images: List[Dict[str, Any]] = []
    if images:
        gate = _chain_capability(route, "vision")
        if gate is False:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": (f"Model {route.model_id} does not support vision "
                                    "input and none of its fallbacks do"),
                        "type": "invalid_request_error",
                        "code": "model_does_not_support_vision",
                    }
                },
            )
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
        images=images or None,
        no_proxy=body.disable_proxy,
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
    def _collect():
        c_parts: List[str] = []
        r_parts: List[str] = []
        n_chunks = 0
        for chunk in chunk_gen:
            n_chunks += 1
            if chunk.get("type") == "thinking" and chunk.get("content"):
                r_parts.append(chunk["content"])
            elif chunk.get("type") == "image" and chunk.get("content"):
                c_parts.append(chunk["content"])
            elif chunk.get("type") == "text" and chunk.get("content"):
                c_parts.append(chunk["content"])
        return c_parts, r_parts, n_chunks

    content_parts: List[str] = []
    reasoning_parts: List[str] = []
    try:
        # Consume the blocking provider stream in a worker thread — pulling
        # it on the event loop would serialize ALL requests behind this one.
        content_parts, reasoning_parts, n_chunks = (
            await asyncio.get_running_loop().run_in_executor(
                None, _collect))
    except ProviderError as e:
        err_type, code, status = _error_status(e)
        return _error_response(str(e), err_type, code, status)
    if n_chunks == 0:
        # A completed-but-empty stream would surface as a 200 with an empty
        # message; surface it as an upstream failure instead.
        return _error_response(
            'upstream returned no content', 'api_error',
            'upstream_error', 502)

    full_text = "".join(content_parts)
    pre_text, calls = full_text, []
    if use_tools:
        pre_text, calls = _parse_tool_calls(full_text)

    message: Dict[str, Any] = {
        "role": "assistant",
        "content": pre_text if calls else full_text,
    }
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts)
    finish_reason = "stop"
    if calls:
        message["tool_calls"] = [{
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": c["name"], "arguments": c["arguments"]},
        } for c in calls]
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
    errored = {"flag": False}
    stop = threading.Event()

    def _guard():
        """Provider chunks with a client-disconnect circuit breaker.

        Without this the worker keeps draining the provider (and, for browser
        transports like z.ai, holds the singleton session lock) long after
        the client gave up — subsequent requests then queue behind a wedged
        session. Checking ``stop`` between chunks releases the provider
        within one poll interval of a disconnect.
        """
        for chunk in chunk_gen:
            if stop.is_set():
                break
            yield chunk

    def _worker():
        # Tool-call responses are parsed authoritatively at stream end, but
        # text is emitted incrementally: a holdback window keeps partially-
        # arrived markers (the TOOL_CALL: line or DSML <｜DSML｜ invoke> tags)
        # from leaking into content deltas. Once a marker is detected the
        # boundary is frozen and everything after it is buffered until the
        # final parse decides tool_calls vs plain text.
        holdback = 64   # chars withheld while no marker has been seen
        rescan = 32     # marker re-scan overlap before the last emit point
        text = ""
        emitted = 0                      # chars already sent as content deltas
        scanned = 0                      # marker-search cursor
        boundary: Optional[int] = None   # start of the tool-call section

        def _emit_upto(upto: int) -> None:
            nonlocal emitted
            if upto <= emitted:
                return
            data = {
                "id": cid, "object": "chat.completion.chunk",
                "created": created, "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": text[emitted:upto]},
                    "finish_reason": None,
                }],
            }
            q.put(_sse(data))
            emitted = upto

        def _tool_deltas(calls: List[Dict[str, str]]) -> None:
            for i, call in enumerate(calls):
                q.put(_sse({
                    "id": cid, "object": "chat.completion.chunk",
                    "created": created, "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {"tool_calls": [{
                            "index": i,
                            "id": f"call_{uuid.uuid4().hex[:24]}",
                            "type": "function",
                            "function": {"name": call["name"],
                                         "arguments": call["arguments"]},
                        }]},
                        "finish_reason": None,
                    }],
                }))

        def _flush_prose() -> None:
            # On upstream failure hand back whatever prose was received so
            # the client loses nothing; raw tool-call sections stay hidden.
            if boundary is None:
                _emit_upto(len(text))

        try:
            for chunk in _guard():
                ctype = chunk.get("type", "")
                content = chunk.get("content", "") or ""
                if not content:
                    continue
                if ctype == "image":
                    # Generated image (markdown): emitted immediately — URLs
                    # never contain the tool-call marker, so no holdback.
                    q.put(_sse({
                        "id": cid, "object": "chat.completion.chunk",
                        "created": created, "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": content},
                            "finish_reason": None,
                        }],
                    }))
                    continue
                if ctype == "thinking":
                    q.put(_sse({
                        "id": cid, "object": "chat.completion.chunk",
                        "created": created, "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {"reasoning_content": content},
                            "finish_reason": None,
                        }],
                    }))
                    continue
                text += content
                if not use_tools:
                    _emit_upto(len(text))
                    continue
                if boundary is not None:
                    continue  # inside the tool-call section: buffer only
                match = _TOOL_CALL_RE.search(text, scanned)
                if match is None:
                    match = _DSML_OPEN_RE.search(text, scanned)
                if match:
                    boundary = match.start()
                    _emit_upto(boundary)
                else:
                    safe = max(0, len(text) - holdback)
                    scanned = max(0, safe - rescan)
                    _emit_upto(safe)

            if use_tools:
                _pre, calls = _parse_tool_calls(text)
                if calls:
                    if boundary is not None:
                        _emit_upto(boundary)  # release any prose tail
                    _tool_deltas(calls)
                    finish_holder["reason"] = "tool_calls"
                else:
                    # No valid call: flush everything, including any raw
                    # marker the model produced, as plain content.
                    _emit_upto(len(text))
            q.put(None)
        except ProviderAuthError as e:
            errored["flag"] = True
            _flush_prose()
            q.put(_error_sse(str(e), "invalid_request_error", "invalid_token"))
            q.put(None)
        except ProviderRateLimitError as e:
            errored["flag"] = True
            _flush_prose()
            q.put(_error_sse(f"Rate limit: {e}", "rate_limit_error", "rate_limit"))
            q.put(None)
        except ProviderError as e:
            errored["flag"] = True
            _flush_prose()
            q.put(_error_sse(str(e)))
            q.put(None)
        finally:
            # Close the provider generator so transports with session locks
            # (z.ai browser) release them immediately on disconnect/finish.
            try:
                chunk_gen.close()
            except Exception:  # noqa: BLE001 — already closing
                pass

    threading.Thread(target=_worker, daemon=True).start()
    loop = asyncio.get_running_loop()

    try:
        while True:
            item = await loop.run_in_executor(None, q.get)
            if item is None:
                break
            yield item
        if not errored["flag"]:
            # Suppress the finish chunk on upstream failure: an error event
            # followed by a normal stop would let clients treat the stream
            # as a successful (truncated) completion.
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
        stop.set()   # client gone: stop draining the provider promptly
        yield "data: [DONE]\n\n"


@app.post("/v1/images/generations")
async def images_generations(body: ImageGenerationRequest, request: Request):
    """OpenAI image generation: POST /v1/images/generations.

    Routed to an image-gen capable model (per /v1/models ``image_gen`` flag);
    the resolved route's fallback chain is filtered to capable targets. The
    provider streams ``image`` chunks; URLs are returned directly or base64-
    encoded for ``response_format: 'b64_json'``.
    """
    client_key = _check_api_key(request)
    if not (body.prompt or "").strip():
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": "prompt must not be empty",
                    "type": "invalid_request_error",
                    "code": "empty_prompt",
                }
            },
        )
    route = await asyncio.get_running_loop().run_in_executor(
        None, lambda: ROUTER.resolve(body.model, auth_key=client_key))
    gate = _chain_capability(route, "image_gen")
    if gate is False:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": (f"Model {route.model_id} does not support image "
                                "generation and none of its fallbacks do"),
                    "type": "invalid_request_error",
                    "code": "model_does_not_support_image_generation",
                }
            },
        )

    created = int(time.time())

    def _generate():
        """Collect image URLs from the blocking provider stream (worker thread)."""
        found: List[str] = []
        try:
            for chunk in ROUTER.stream(route, body.prompt, auth_key=client_key,
                                       image_generation=True,
                                       no_proxy=body.disable_proxy):
                if chunk.get("type") == "image" and chunk.get("url"):
                    found.append(chunk["url"])
        except ProviderError as e:
            return e, found
        return None, found

    error, urls = await asyncio.get_running_loop().run_in_executor(None, _generate)
    if error is not None:
        err_type, code, status = _error_status(error)
        return _error_response(str(error), err_type, code, status)
    if not urls:
        return _error_response("the model did not return any image",
                               "api_error", "no_image_returned", 502)

    response_format = (body.response_format or "url").lower()

    def _encode() -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for url in urls:
            if response_format == "b64_json":
                _mime, raw = fetch_image_bytes(url)
                out.append({"b64_json": base64.b64encode(raw).decode()})
            else:
                out.append({"url": url})
        return out

    try:
        data = await asyncio.get_running_loop().run_in_executor(None, _encode)
    except ProviderError as e:
        err_type, code, status = _error_status(e)
        return _error_response(str(e), err_type, code, status)
    return {"created": created, "data": data}


def main():
    import uvicorn

    # Surface dsk.* logger.info lines (llmtrim stats, registry updates,
    # refresher/copyist activity) on stderr alongside uvicorn's own logs.
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s:%(name)s: %(message)s')
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
