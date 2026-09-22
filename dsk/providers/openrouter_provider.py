"""OpenRouter provider — free-model gateway (openrouter.ai, official API).

OpenRouter aggregates hundreds of upstream models behind one OpenAI-
compatible endpoint. A free tier exists without any payment method: model
ids carrying the ``:free`` suffix are requestable with a plain API key
(free keys: ~20 req/min, 50 req/day — the bot renews credentials and the
router falls back across providers, so both ceilings are workable).

How it works
------------
1. Credentials: ``OPENROUTER_API_KEY`` env, the bot-managed
   ``openrouter_cookies.json`` jar (``api_key``), or the request key.
2. Discovery: ``GET /api/v1/models`` is PUBLIC (no key) — every id ending
   in ``:free`` becomes a route. Context lengths and vision capability
   come from the catalog (``architecture.modality``).
3. Stream: ``POST /api/v1/chat/completions`` with ``stream: true``; SSE
   deltas map to the unified chunk shape (``reasoning`` deltas become
   'thinking' chunks for reasoning-capable free models).

Renewal: the refresher verifies the key via ``GET /api/v1/auth/key`` and
the signup rung creates accounts end-to-end (email OTP → session → key
via the page's own session-authenticated API).
"""

import base64
import json
import logging
import os
from typing import Any, Dict, Generator, List, Optional

from .base import (
    Provider,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderUnavailableError,
    classify_http_error,
    http_get,
    http_post_stream,
    parse_sse_data,
)
from .jar import load_jar

logger = logging.getLogger('dsk.providers.openrouter')

OPENROUTER_BASE = 'https://openrouter.ai'
MODELS_URL = f'{OPENROUTER_BASE}/api/v1/models'
CHAT_URL = f'{OPENROUTER_BASE}/api/v1/chat/completions'
KEY_URL = f'{OPENROUTER_BASE}/api/v1/auth/key'

_USER_AGENT = ('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
               '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

# free-tier ceilings are per-key daily/minutely, so keep default caps tight
DEFAULT_MAX_OUTPUT = int(os.getenv('DSF_OPENROUTER_MAX_OUTPUT', '4096'))


def _key(auth_key: Optional[str] = None) -> str:
    """API key: request key → env → credential jar."""
    if auth_key:
        raw = auth_key.strip()
        if raw.lower().startswith('bearer '):
            raw = raw[7:].strip()
        if raw:
            return raw
    env = (os.getenv('OPENROUTER_API_KEY', '') or '').strip()
    if env:
        return env
    jar = load_jar('openrouter') or {}
    for field in ('api_key', 'key', 'token'):
        value = (jar.get(field) or '').strip()
        if value:
            return value
    return ''


def _headers(key: str) -> Dict[str, str]:
    # HTTP-Referer/X-Title: OpenRouter's recommended attribution headers
    return {
        'Authorization': f'Bearer {key}',
        'Content-Type': 'application/json',
        'HTTP-Referer': 'https://github.com/blastbeng/deepseek4free',
        'X-Title': 'deepseek4free',
        'User-Agent': _USER_AGENT,
    }


def _catalog() -> List[Dict[str, Any]]:
    """Public model catalog (no key needed). Raises on transport failure."""
    r = http_get(MODELS_URL, headers={'User-Agent': _USER_AGENT}, timeout=30)
    if r.status_code != 200:
        raise ProviderUnavailableError(
            f'openrouter catalog unavailable (HTTP {r.status_code})')
    try:
        data = r.json().get('data') or []
    except ValueError as exc:
        raise ProviderUnavailableError(
            f'openrouter catalog unparsable: {exc}') from exc
    return [m for m in data if isinstance(m, dict) and m.get('id')]


def _vision_of(entry: Dict[str, Any]) -> bool:
    arch = entry.get('architecture') or {}
    modality = str(arch.get('input_modalities') or arch.get('modality') or '')
    return 'image' in modality.lower()


class OpenRouterProvider(Provider):
    name = 'openrouter'

    def available(self, auth_key: Optional[str] = None) -> bool:
        # the catalog is public but chat REQUIRES a key — a keyless
        # provider would just burn fallback time on every request
        return bool(_key(auth_key))

    def list_models(self, auth_key: Optional[str] = None
                    ) -> List[Dict[str, Any]]:
        key = _key(auth_key)
        if not key:
            raise ProviderAuthError('no openrouter API key configured')
        models: List[Dict[str, Any]] = []
        for entry in _catalog():
            model_id = entry.get('id') or ''
            if not model_id.endswith(':free'):
                continue  # paid models stay out — this is the free provider
            slug = model_id[:-len(':free')]
            ctx = int(entry.get('context_length') or 32768)
            models.append({
                'id': f'openrouter-{slug}',
                'upstream_model': model_id,
                'thinking_enabled': True,   # reasoning deltas are skipped
                                            # when absent — harmless to allow
                'search_enabled': False,
                'vision': _vision_of(entry),
                'image_gen': False,
                'context_length': min(ctx, 131072),
                'max_output_tokens': DEFAULT_MAX_OUTPUT,
                'extra': {
                    'openrouter_model': model_id,
                    'name': entry.get('name') or slug,
                    'created': entry.get('created') or 0,
                },
            })
        if not models:
            raise ProviderUnavailableError(
                'openrouter catalog has no :free models right now')
        return models

    # -- streaming -------------------------------------------------------

    def stream(self, prompt: str, *, model: str,
               thinking_enabled: bool = False, search_enabled: bool = False,
               temperature: Optional[float] = None,
               max_tokens: Optional[int] = None,
               images: Optional[List[Dict[str, Any]]] = None,
               image_generation: bool = False,
               auth_key: Optional[str] = None,
               no_proxy: bool = False
               ) -> Generator[Dict[str, Any], None, None]:
        key = _key(auth_key)
        if not key:
            raise ProviderAuthError('no openrouter API key configured')
        if images:
            content: List[Dict[str, Any]] = []
            for img in images:
                data = img.get('data') or b''
                mime = img.get('mime') or 'image/jpeg'
                if data:
                    b64 = base64.b64encode(data).decode()
                    content.append({'type': 'image_url', 'image_url': {
                        'url': f'data:{mime};base64,{b64}'}})
            content.append({'type': 'text', 'text': prompt})
            messages = [{'role': 'user', 'content': content}]
        else:
            messages = [{'role': 'user', 'content': prompt}]
        body: Dict[str, Any] = {
            'model': model,
            'messages': messages,
            'stream': True,
        }
        if temperature is not None:
            body['temperature'] = max(0.0, min(2.0, float(temperature)))
        if max_tokens:
            body['max_tokens'] = int(max_tokens)
        resp = http_post_stream(CHAT_URL, headers=_headers(key),
                                json_body=body, no_proxy=no_proxy)
        if resp.status_code != 200:
            text = ''
            try:
                text = resp.text[:400]
            except Exception:  # noqa: BLE001
                pass
            raise classify_http_error(resp.status_code, text)
        saw_finish = False
        for line in resp.iter_lines():
            if not line:
                continue
            if isinstance(line, bytes) and line[:6] == b': OPEN':
                continue  # OpenRouter's keep-alive comment
            payload = parse_sse_data(line if isinstance(line, bytes)
                                     else str(line).encode())
            if not payload:
                continue
            err = payload.get('error')
            if err:
                code = int((err or {}).get('code') or 0)
                msg = str((err or {}).get('message') or err)[:300]
                if code == 429:
                    raise ProviderRateLimitError(msg)
                if code in (401, 403):
                    raise ProviderAuthError(msg)
                raise ProviderError(f'openrouter: {msg}')
            for choice in payload.get('choices') or []:
                delta = choice.get('delta') or {}
                reasoning = (delta.get('reasoning')
                             or (delta.get('reasoning_content') or ''))
                if reasoning:
                    yield {'type': 'thinking', 'content': str(reasoning),
                           'finish_reason': None}
                text = delta.get('content')
                if text:
                    yield {'type': 'text', 'content': str(text),
                           'finish_reason': None}
                if choice.get('finish_reason'):
                    saw_finish = True
                    yield {'type': 'text', 'content': '',
                           'finish_reason': 'stop'}
        if not saw_finish:
            # stream died without a finish — treat as transport failure so
            # the router can retry/fall back instead of ending mid-answer
            raise ProviderUnavailableError(
                'openrouter stream ended without finish_reason')
