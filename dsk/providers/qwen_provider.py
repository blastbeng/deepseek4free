"""Qwen provider via chat.qwen.ai (reverse-engineered).

The Qwen web app exposes OpenAI-shaped SSE on private v2 endpoints guarded by
Alibaba's anti-bot headers (bx-v / bx-umidtoken / bx-ua). This provider talks
to those endpoints directly with the web session Bearer token — no official
DashScope API, no paid keys.

How it works
------------
1. Credentials: the ``token`` of a logged-in chat.qwen.ai session
   (``QWEN_TOKEN`` env var, bot-managed ``qwen_cookies.json`` jar, or
   ``QWEN_COOKIES`` env JSON as fallback).
2. Create chat: POST ``/api/v2/chats/new`` → ``data.id``.
3. Stream: POST ``/api/v2/chat/completions?chat_id=...`` with
   ``incremental_output: true``; SSE deltas carry ``phase``
   (think / thinking_summary / answer) which we map to thinking/text chunks.

The bx-ua fingerprint below is the static value observed in the web client
(overridable via ``QWEN_BX_UA`` / ``QWEN_UMID`` env vars).

WAF reality check (measured 2026-09): Aliyun punishes EVERY non-browser POST
to ``/api/v2/chat/completions`` — 200 + text/html interstitial even with SPA
cookies/headers/TLS fingerprints — and stalls signed-in SPA sessions. Only
the real qwen SPA in GUEST mode streams. So ``stream()`` tries the HTTP path
first (it still works for ``/api/v2/chats/new`` + model discovery) and falls
back to the browser relay in ``dsk/qwen_relay.py`` when completions is
punished, the token is dead, or no token exists at all. Disable the relay
with ``DSF_QWEN_RELAY=0``.
"""

import json
import logging
import os
import re
import time
import uuid
from typing import Any, Dict, Generator, List, Optional

from .base import (
    Provider,
    ProviderError,
    ProviderAuthError,
    ProviderUnavailableError,
    classify_http_error,
    http_get,
    http_post_raw,
    http_post_stream,
    parse_sse_data,
)
from .jar import env_cookies, load_jar

logger = logging.getLogger('dsk.providers.qwen')

QWEN_BASE_URL = 'https://chat.qwen.ai'
QWEN_NEWCHAT_URL = f'{QWEN_BASE_URL}/api/v2/chats/new'
QWEN_COMPLETIONS_URL = f'{QWEN_BASE_URL}/api/v2/chat/completions'
QWEN_MODELS_URL = f'{QWEN_BASE_URL}/api/models'

QWEN_CONTEXT_LENGTH = int(os.getenv('DSF_QWEN_CONTEXT_LENGTH', '131072'))
QWEN_MAX_OUTPUT = int(os.getenv('DSF_QWEN_MAX_OUTPUT', '8192'))

_USER_AGENT = (
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) '
    'Chrome/120.0.0.0 Safari/537.36'
)

# Static anti-bot headers observed in the web client (Chat2API qwen-ai.ts).
QWEN_BX_V = '2.5.36'
QWEN_UMID_DEFAULT = ('T2gAr9z8byN8sNOmfQ3X9j61MNTNmSqDO5L1rs2jMcQCVhOKgZICcBN-'
                     'UdTuJGig-NM=')
QWEN_BXUA_DEFAULT = '231!lWD36kmUe5E+joKDK5gBZ48FEl2ZWfPwIPF92lBLek2KxVW/XJ2EwruCiDOX5Px4EXNhmh6EfS9eDwQGRwijIK64A4nPqeLysJcDjUACje/H3J4ZgGZpicG6K8AkiGGaEKC830+QSiSUsLRlL/EyhXTmLcJc/5iDkMuOpUhNz0e0Q/nTqjVJ3ko00Q/oyE+jauHhUHfb1GxGHkE+++3+qCS4+ItkaA6tiItCo+romzElfLFD6RIj7oHt9vffs98nLwpHnaqKjufnLFMejSlAUGiQvTofIiGhIvftAMcoFV4mrUHsqyQ/ncQihmJHkbxXjvM57FCb6b9dEIRZl7jgj0+QLNLRs0NZ4azdZ6rzbGTSO8KA5I3Aq/3gBr87X16Mj0oJtaPKmFGaP2zghfOVhxQht8YjRd50lJa+Ue4PAuPSdu2O69DKLH8VOhrsB+psaBIRxnRi5POUQ6w8s8qlb9vxvExjHNOAKWXV1by1Nz+6FPWdyTeAgcmonjCcV0dCtPj/KyeVDkeSrDkKZjnDzHEqeCdfmJ65kve+Vy3YS0vagzyHfVEnzN0ULUZtkGfJXFNm6+bIa55wmGBhUeXbHL0EdlQXMu1YXxmcwBgTaq7tlQcfv7AefanbfjGE8R1IFnNyg2/jXLbnLg5Z6l1oKqgnxZQg0DE9BJuw6s0XjGwTdSxybWxp+WFD/RsXt76uwvCBk7z+YmSFLtFj2UlTsoq+vl0DTmsVItDKf9SZ94NcuJ7mxJYI02S/2kQBfbbHG0d4hXevDrEC0cb86EvzN2ud+v6bAunNRGNFz/RH0KLusoBVeo+puCFKeeIJWEo0t1UicX5YxJwMAoV7+g0gK93y4W9sMQtso8/wY5wsBzis9dwfLvIwXpaAM1g0MZp/YIRq8T/Qc+U/8x99tam4er0IWizvrkjqhIzCWBKpJ4Y4gj3bOmiS3VCMEaoVfKCwUWENwYKuP3H5VI0n+O2vVVRrekUrwvkm6URRhVhN4eEFTCjB9nSQu++qKyDH8HPpkS3YfwF8/OQtrZo7hQXxvNmP2HcH/K7zcweD00BaoOLiYUtXRItGYbl06sVSbm04soRf1Jqpyo3XiRqBWD9rmJfr4w8NOEGVGUCKXLDLsXy+8JC4Iqf0FsIjWxjMVdraTUtCbwXRbYUownQVm6bt7LYD1SNPoWNPqUJgsLMwP33ugrb1UbHCs24roOch6Go5QHIPA8E15SZE9pkr1SkmqrNs/+KRomFJ9HyFnWUYhZIV9MRLqlOAt6XBBTash3WJnCjhx/PZGhXVvdn2jX4+0Pm55LsiNugA8vaAUJQBxD/8a1u/RvTgbj35+b7I7m8tG0hMhClNZF+tpsOmZZhUGuXH9uVbkJMlMuAmMVCHwn3O31GlLeXXzzep2WS3xN2U+p5J0I7GySnuZUkuGs1ZTVqGUvR2g4q+7ljU55Ak78yPZiQXeUeqS74azszvZvCqWxXn2eePj+gcpliOjrYKpglUP19rQrMt8PqLt8L0ghIqVCmMwl3Hgr/VUcqDpXdpPTR='
QWEN_VERSION = '0.2.7'

# Web-app model ids (upstream aliases as exposed in the chat.qwen.ai picker;
# refreshed 2026-09-22 from live discovery — offline fallback only).
QWEN_MODELS: List[Dict[str, Any]] = [
    {'id': 'qwen3.7-plus', 'thinking': True},
    {'id': 'qwen3.8-max', 'thinking': True},
    {'id': 'qwen3.8-omni-flash', 'thinking': False},
    {'id': 'qwen3.7-max', 'thinking': True},
    {'id': 'qwen3.6-plus', 'thinking': True},
    {'id': 'qwen3.5-plus', 'thinking': True},
    {'id': 'qwen3.5-omni-plus', 'thinking': False},
]

# Live model discovery: chat.qwen.ai exposes its picker anonymously on
# ``GET /api/models`` (OpenWebUI-style), so the route list tracks whatever
# the web app currently offers instead of a frozen client-side list.
_QWEN_MODELS_TTL = float(os.getenv('DSF_QWEN_MODELS_TTL', '1800'))
_discover_cache: Dict[str, Any] = {'at': 0.0, 'models': []}


def _model_entry(model_id: str, title: str, caps: Dict[str, Any],
                 context_length: int) -> Dict[str, Any]:
    return {
        'id': model_id,
        'upstream_model': model_id,
        'thinking_enabled': bool(caps.get('thinking')),
        'search_enabled': bool(caps.get('search')),
        # chat.qwen.ai advertises vision, but this provider only forwards
        # text prompts, so image routing stays disabled.
        'vision': False,
        'image_gen': False,
        'context_length': int(context_length or 0) or QWEN_CONTEXT_LENGTH,
        'max_output_tokens': QWEN_MAX_OUTPUT,
        'extra': {'title': title or model_id},
    }


def _static_models() -> List[Dict[str, Any]]:
    """Offline fallback used only when live discovery is unreachable."""
    return [_model_entry(entry['id'], entry['id'],
                         {'thinking': entry['thinking']},
                         QWEN_CONTEXT_LENGTH)
            for entry in QWEN_MODELS]


def _discover_models(force: bool = False, anon: bool = False) -> List[Dict[str, Any]]:
    """Fetch the live model picker from chat.qwen.ai (``GET /api/models``).

    The endpoint serves the public picker anonymously and reflects the
    session entitlement: signed-in sees the full catalog, anonymous sees
    exactly what the guest UI offers. ``anon=True`` skips the bearer so
    relay-only deployments list what the relay can actually serve. Results
    are cached for ``DSF_QWEN_MODELS_TTL`` seconds. Raises on transport/HTTP
    failure so the caller can fall back to the static list.
    """
    now = time.time()
    if (not force and _discover_cache['models']
            and now - _discover_cache['at'] < _QWEN_MODELS_TTL):
        return list(_discover_cache['models'])
    headers = _headers()
    if anon:
        headers = {k: v for k, v in headers.items()
                   if k.lower() != 'authorization'}
    resp = http_get(QWEN_MODELS_URL, headers=headers, timeout=20)
    if not anon and resp.status_code in (401, 403):
        anon = {k: v for k, v in headers.items()
                if k.lower() != 'authorization'}
        resp = http_get(QWEN_MODELS_URL, headers=anon, timeout=20)
    if resp.status_code != 200:
        raise classify_http_error(resp.status_code,
                                  getattr(resp, 'text', '') or '',
                                  resp.headers)
    try:
        payload = resp.json() or {}
    except ValueError as e:
        raise ProviderError(f'qwen /api/models returned non-JSON: {e}') from e
    raw = payload.get('data') if isinstance(payload, dict) else None
    if isinstance(raw, dict):  # /api/v2/models nests it once more
        raw = raw.get('data')
    if not isinstance(raw, list):
        raise ProviderError('qwen /api/models payload has no model list')
    models: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get('id') or '').strip()
        if not model_id:
            continue
        info = item.get('info') if isinstance(item.get('info'), dict) else {}
        meta = info.get('meta') if isinstance(info.get('meta'), dict) else {}
        caps = meta.get('capabilities') or meta.get('abilities') or {}
        if not isinstance(caps, dict):
            caps = {}
        try:
            ctx = int(meta.get('max_context_length') or 0)
        except (TypeError, ValueError):
            ctx = 0
        models.append(_model_entry(
            model_id, str(item.get('name') or info.get('name') or ''),
            caps, ctx))
    if not models:
        raise ProviderError('qwen /api/models returned no usable models')
    _discover_cache['at'] = now
    _discover_cache['models'] = models
    logger.info('qwen: discovered %d models live', len(models))
    return list(models)


def _token() -> str:
    raw = (os.getenv('QWEN_TOKEN', '') or '').strip()
    if raw:
        return raw
    jar = load_jar('qwen') or env_cookies('QWEN')
    return (jar.get('token') or '').strip()


def _relay_enabled() -> bool:
    """Browser-relay fallback switch (``DSF_QWEN_RELAY``, default on).

    Import is lazy: the relay pulls in DrissionPage/Xvfb, which must not load
    with every provider module import.
    """
    if os.getenv('DSF_QWEN_RELAY', '1').strip().lower() in \
            ('0', 'false', 'no', 'off'):
        return False
    try:
        from dsk.qwen_relay import get_relay
        return get_relay().enabled()
    except Exception as e:  # noqa: BLE001 — degraded, not fatal
        logger.warning('qwen relay unavailable (%s); HTTP transport only', e)
        return False


def _relay_offered() -> List[str]:
    """Normalized ids the guest picker actually serves (empty until the
    relay's first dropdown open)."""
    try:
        from dsk.qwen_relay import get_relay, norm_model
        return sorted({norm_model(t) for t in get_relay().offered()})
    except Exception:  # noqa: BLE001
        return []


def _norm(model_id: str) -> str:
    try:
        from dsk.qwen_relay import norm_model
        return norm_model(model_id)
    except Exception:  # noqa: BLE001
        return re.sub(r'[^a-z0-9]', '', (model_id or '').lower())


# After a WAF punish/auth failure on the HTTP completions path, stop paying
# the punish roundtrip tax and go straight to the relay for a while.
_HTTP_COOLDOWN_S = float(os.getenv('DSF_QWEN_HTTP_COOLDOWN', '300'))
_http_cooldown: Dict[str, float] = {'until': 0.0}


def _http_skipped() -> bool:
    return time.time() < _http_cooldown['until']


def _skip_http_for(seconds: float) -> None:
    _http_cooldown['until'] = time.time() + seconds


def _iter_events(events: Any) -> Generator[Dict[str, Any], None, None]:
    """Map upstream qwen SSE dicts (HTTP or relay capture) to provider chunks.

    ``phase`` drives chunk type: ``thinking_summary`` deltas carry an
    accumulated ``summary_thought.content`` string list (diffed here),
    ``think``/answer content streams as thinking/text, ``finished`` stops.
    """
    prev_summary = ''
    for data in events:
        if not isinstance(data, dict):
            continue
        if data.get('error'):
            raise ProviderError(f"qwen stream error: {data.get('error')}")
        choices = data.get('choices') or []
        if not choices:
            continue
        delta = (choices[0] or {}).get('delta') or {}
        phase = delta.get('phase') or ''
        status = delta.get('status') or ''
        if phase == 'thinking_summary':
            # summary_thought.content is an accumulated list of strings
            extra = delta.get('extra') or {}
            parts = ((extra.get('summary_thought') or {}).get('content')
                     or [])
            joined = ''.join(p for p in parts if isinstance(p, str))
            if joined.startswith(prev_summary):
                piece = joined[len(prev_summary):]
            else:
                piece = joined
                prev_summary = ''
            if piece:
                prev_summary += piece
                yield {'content': piece, 'type': 'thinking',
                       'finish_reason': None}
        elif delta.get('content'):
            yield {'content': delta['content'],
                   'type': 'thinking' if phase == 'think' else 'text',
                   'finish_reason': None}
        if status == 'finished' and phase in ('', 'answer', None):
            # 'finished' also terminates the thinking_summary phase; only the
            # answer phase's finish ends the generation (thinking streams
            # first when enabled).
            break
    yield {'content': '', 'type': 'text', 'finish_reason': 'stop'}


def validate_token(token: Optional[str] = None,
                   no_proxy: bool = False) -> str:
    """Live-check a session JWT against ``GET /api/v1/auths``.

    The model picker is served anonymously, so it cannot tell a working
    session from an expired one. This probe can: it returns ``'ok'``,
    ``'unauth'`` (token missing/rejected) or ``'httpNNN'`` /
    ``'unreachable: <Exc>'`` so selfheal, the renewal ladder and the operator
    panel report the truth instead of a static list's optimism.
    """
    tok = (token if token is not None else _token()) or ''
    tok = tok.strip()
    if not tok:
        return 'unauth'
    try:
        resp = http_get(f'{QWEN_BASE_URL}/api/v1/auths',
                        headers={**_headers(),
                                 'Authorization': f'Bearer {tok}'},
                        timeout=20, no_proxy=no_proxy)
    except Exception as e:  # noqa: BLE001 — classification is the point
        return f'unreachable: {type(e).__name__}'
    if resp.status_code == 200:
        return 'ok'
    if resp.status_code in (401, 403):
        return 'unauth'
    return f'http{resp.status_code}'


def _headers(chat_id: str = '', accept: str = 'application/json') -> Dict[str, str]:
    headers = {
        'Authorization': f'Bearer {_token()}',
        'User-Agent': _USER_AGENT,
        'Content-Type': 'application/json',
        'Accept': accept,
        'Origin': QWEN_BASE_URL,
        'Referer': f'{QWEN_BASE_URL}/c/{chat_id}' if chat_id
                   else f'{QWEN_BASE_URL}/',
        'bx-v': os.getenv('QWEN_BX_V', QWEN_BX_V),
        'bx-umidtoken': os.getenv('QWEN_UMID', QWEN_UMID_DEFAULT),
        'bx-ua': os.getenv('QWEN_BX_UA', QWEN_BXUA_DEFAULT),
        'Version': QWEN_VERSION,
        'X-Request-Id': str(uuid.uuid4()),
    }
    return headers


class _QwenPunish(ProviderUnavailableError):
    """The WAF served a punish/interstitial page instead of the SSE stream."""


class QwenProvider(Provider):
    name = 'qwen'

    # ---------------------------------------------------------------- provider
    def available(self, auth_key: Optional[str] = None) -> bool:
        # Guest relay mode needs no credentials: qwen models stay listed even
        # without a session token (streams then ride the browser relay).
        return bool(_token()) or _relay_enabled()

    def list_models(self, auth_key: Optional[str] = None) -> List[Dict[str, Any]]:
        """Live picker from chat.qwen.ai; static list only as last resort.

        With the relay enabled, listing is done ANONYMOUSLY: the anon picker
        mirrors the guest entitlement, which is exactly what the browser
        relay can serve (the signed-in HTTP completions path is WAF-punished
        regardless of entitlement). The relay's observed picker titles
        narrow the list further once known.
        """
        token = _token()
        relay = _relay_enabled()
        if not token and not relay:
            raise ProviderAuthError('no chat.qwen.ai token configured')
        try:
            models = _discover_models(anon=relay)
        except ProviderAuthError:
            raise
        except Exception as e:  # noqa: BLE001 — never lose the picker
            logger.warning('qwen live model discovery failed (%s); '
                           'falling back to the static list', e)
            models = _static_models()
        if relay:
            offered = _relay_offered()
            if offered:
                models = ([m for m in models if _norm(m['id']) in offered]
                          or models)
        return models

    def stream(self, prompt: str, *, model: str, thinking_enabled: bool = False,
               search_enabled: bool = False, temperature: Optional[float] = None,
               max_tokens: Optional[int] = None,
               images: Optional[List[Dict[str, Any]]] = None,
               image_generation: bool = False,
               no_proxy: bool = False,
               auth_key: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
        if images:
            raise ProviderError('qwen image input is not supported yet')
        relay = _relay_enabled()
        token = _token()
        if relay and (not token or _http_skipped()):
            # No session token, or HTTP is on punish cooldown: the guest
            # browser relay is the only working transport.
            yield from self._relay_stream(prompt, model, thinking_enabled,
                                          search_enabled)
            return
        if not relay:
            yield from self._http_stream(prompt, model, thinking_enabled,
                                         search_enabled, no_proxy)
            return
        emitted = False
        try:
            for chunk in self._http_stream(prompt, model, thinking_enabled,
                                           search_enabled, no_proxy):
                emitted = True
                yield chunk
            return
        except ProviderError as e:
            # Never re-send a half-streamed answer through the relay; only a
            # failure before the first chunk may fall back.
            if emitted:
                raise
            _skip_http_for(_HTTP_COOLDOWN_S)
            logger.warning('qwen http transport failed (%s); falling back '
                           'to the browser relay for %ss', e,
                           _HTTP_COOLDOWN_S)
        yield from self._relay_stream(prompt, model, thinking_enabled,
                                      search_enabled)

    def _http_stream(self, prompt: str, model: str, thinking_enabled: bool,
                     search_enabled: bool,
                     no_proxy: bool) -> Generator[Dict[str, Any], None, None]:
        chat_id = self._create_chat(model, no_proxy=no_proxy)
        now_ms = int(__import__('time').time() * 1000)
        child_id = str(uuid.uuid4())
        body = {
            'stream': True,
            'version': '2.1',
            'incremental_output': True,
            'chat_id': chat_id,
            'chat_mode': 'normal',
            'model': model,
            'parent_id': None,
            'messages': [{
                'fid': str(uuid.uuid4()),
                'parentId': None,
                'childrenIds': [child_id],
                'role': 'user',
                'content': prompt,
                'user_action': 'chat',
                'files': [],
                'timestamp': now_ms // 1000,
                'models': [model],
                'chat_type': 't2t',
                'feature_config': {
                    'thinking_enabled': bool(thinking_enabled),
                    'output_schema': 'phase',
                    'research_mode': 'normal',
                    'auto_thinking': False,
                    'thinking_format': 'summary',
                    'auto_search': bool(search_enabled),
                },
                'extra': {'meta': {'subChatType': 't2t'}},
                'sub_chat_type': 't2t',
                'parent_id': None,
            }],
            'timestamp': now_ms // 1000 + 1,
        }
        url = f'{QWEN_COMPLETIONS_URL}?chat_id={chat_id}'
        response = http_post_stream(url, headers=_headers(chat_id,
                                                          'text/event-stream'),
                                    json_body=body, no_proxy=no_proxy)
        if response.status_code != 200:
            try:
                error_text = response.text or ''
            except Exception:  # pragma: no cover
                error_text = f'HTTP {response.status_code}'
            raise classify_http_error(response.status_code, error_text,
                                      response.headers)
        try:
            ctype = (response.headers.get('content-type') or '').lower()
        except Exception:  # pragma: no cover
            ctype = ''
        if 'text/html' in ctype:
            # Aliyun WAF punish page: HTTP 200 but HTML, not SSE.
            raise _QwenPunish(
                'completions served an HTML interstitial instead of SSE')
        return self._iter_chunks(response)

    def _relay_stream(self, prompt: str, model: str, thinking_enabled: bool,
                      search_enabled: bool) -> Generator[Dict[str, Any], None, None]:
        """Drive the real qwen UI in a guest browser (dsk/qwen_relay.py)."""
        from dsk.qwen_relay import RelayPunish, get_relay
        relay = get_relay()
        try:
            events = relay.stream(_token(), model, prompt,
                                  thinking_enabled=thinking_enabled,
                                  search_enabled=search_enabled)
            yield from _iter_events(events)
        except RelayPunish as e:
            raise ProviderUnavailableError(
                f'qwen browser relay blocked by WAF: {e}') from e
        except RuntimeError as e:
            # Relay-specific: model not offered in the guest picker, etc.
            raise ProviderError(f'qwen browser relay: {e}') from e

    def _create_chat(self, model: str, no_proxy: bool = False) -> str:
        body = {
            'title': 'New Chat',
            'models': [model],
            'chat_mode': 'normal',
            'chat_type': 't2t',
            'timestamp': int(__import__('time').time() * 1000),
            'project_id': '',
        }
        response = http_post_raw(QWEN_NEWCHAT_URL, json.dumps(body).encode(),
                                 headers=_headers(), no_proxy=no_proxy)
        if response.status_code != 200:
            try:
                error_text = response.text or ''
            except Exception:  # pragma: no cover
                error_text = f'HTTP {response.status_code}'
            raise classify_http_error(response.status_code, error_text,
                                      response.headers)
        try:
            data = response.json().get('data') or {}
            chat_id = (data.get('data') or data).get('id')
        except ValueError as e:
            raise ProviderError(f'qwen chats/new returned non-JSON: {e}') from e
        if not chat_id:
            raise ProviderError('qwen chats/new returned no chat id')
        return str(chat_id)

    def _iter_chunks(self, response) -> Generator[Dict[str, Any], None, None]:
        def events() -> Generator[Dict[str, Any], None, None]:
            for line in response.iter_lines():
                data = parse_sse_data(line)
                if data:
                    yield data
        yield from _iter_events(events())
