"""Groq provider — free-tier LPU inference API (api.groq.com, official API).

Groq's free tier needs no credit card: a console.groq.com API key unlocks
current-production open models (llama, qwen, gpt-oss, …) at high speed
with per-model rate ceilings (the router's fallback chain absorbs them).

How it works
------------
1. Credentials: ``GROQ_API_KEY`` env, the bot-managed ``groq_cookies.json"
   jar (``api_key``), or the request key. The web-session ``sso`` cookie
   (existing grok-style jar) is NOT sufficient — chat needs the API key.
2. Discovery: ``GET /openai/v1/models`` with the key returns the exact
   live model list (ids, context windows, owned_by).
3. Stream: ``POST /openai/v1/chat/completions`` with ``stream: true``;
   SSE deltas map to the unified chunk shape (``reasoning`` deltas of
   reasoning models become 'thinking' chunks).

Renewal: the refresher verifies the key via the models endpoint; the
signup rung attempts an autonomous console account (email) and stores
the key. Keys can always be supplied manually via GROQ_API_KEY.
"""

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

logger = logging.getLogger('dsk.providers.groq')

GROQ_BASE = 'https://api.groq.com/openai/v1'
MODELS_URL = f'{GROQ_BASE}/models'
CHAT_URL = f'{GROQ_BASE}/chat/completions'

_USER_AGENT = ('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
               '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

DEFAULT_MAX_OUTPUT = int(os.getenv('DSF_GROQ_MAX_OUTPUT', '4096'))

# model ids that never make sense as a chat route on this proxy
_MODEL_EXCLUDE = ('whisper', 'tts', 'playai', 'distil-whisper',
                  'guard', 'prompt-guard')


def _key(auth_key: Optional[str] = None) -> str:
    """API key: request key → env → credential jar."""
    if auth_key:
        raw = auth_key.strip()
        if raw.lower().startswith('bearer '):
            raw = raw[7:].strip()
        if raw:
            return raw
    env = (os.getenv('GROQ_API_KEY', '') or '').strip()
    if env:
        return env
    jar = load_jar('groq') or {}
    for field in ('api_key', 'key', 'gsk_key'):
        value = (jar.get(field) or '').strip()
        if value:
            return value
    return ''


def _headers(key: str) -> Dict[str, str]:
    return {
        'Authorization': f'Bearer {key}',
        'Content-Type': 'application/json',
        'User-Agent': _USER_AGENT,
    }


class GroqProvider(Provider):
    name = 'groq'

    def available(self, auth_key: Optional[str] = None) -> bool:
        return bool(_key(auth_key))

    def list_models(self, auth_key: Optional[str] = None
                    ) -> List[Dict[str, Any]]:
        key = _key(auth_key)
        if not key:
            raise ProviderAuthError('no groq API key configured')
        r = http_get(MODELS_URL, headers=_headers(key), timeout=30)
        if r.status_code in (401, 403):
            raise ProviderAuthError('groq API key rejected')
        if r.status_code == 429:
            raise ProviderRateLimitError('groq models rate limited')
        if r.status_code != 200:
            raise ProviderUnavailableError(
                f'groq model list unavailable (HTTP {r.status_code})')
        try:
            data = r.json().get('data') or []
        except ValueError as exc:
            raise ProviderUnavailableError(
                f'groq model list unparsable: {exc}') from exc
        models: List[Dict[str, Any]] = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            model_id = entry.get('id') or ''
            if not model_id or any(x in model_id.lower()
                                   for x in _MODEL_EXCLUDE):
                continue
            ctx = int(entry.get('context_window') or 8192)
            models.append({
                'id': f'groq-{model_id}',
                'upstream_model': model_id,
                'thinking_enabled': True,   # reasoning deltas are skipped
                                            # when absent — harmless to allow
                'search_enabled': False,
                'vision': 'vision' in model_id.lower()
                          or 'scout' in model_id.lower(),
                'image_gen': False,
                'context_length': min(ctx, 131072),
                'max_output_tokens': DEFAULT_MAX_OUTPUT,
                'extra': {
                    'groq_model': model_id,
                    'owned_by': entry.get('owned_by') or '',
                },
            })
        if not models:
            raise ProviderUnavailableError('groq returned no chat models')
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
            raise ProviderAuthError('no groq API key configured')
        if images:
            # Groq vision models accept OpenAI-style image_url parts
            content: List[Dict[str, Any]] = []
            for img in images:
                url = img.get('url') or ''
                if url:
                    content.append({'type': 'image_url', 'image_url':
                                    {'url': url}})
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
                raise ProviderError(f'groq: {msg}')
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
            raise ProviderUnavailableError(
                'groq stream ended without finish_reason')
