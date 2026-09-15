"""Kimi provider via www.kimi.com (Moonshot Kimi web app, reverse-engineered).

The Kimi web app talks to a gRPC-web (connect protocol) gateway with the JWT
of the logged-in session. This provider speaks that protocol directly — no
official Moonshot API, no paid keys.

How it works
------------
1. Credentials: the session JWT (``KIMI_TOKEN`` env var, bot-managed
   ``kimi_cookies.json`` jar, or ``KIMI_COOKIES`` env JSON as fallback).
2. Stream: POST ``/apiv2/kimi.gateway.chat.v1.ChatService/Chat`` with
   ``Content-Type: application/connect+json``; the request body is a single
   connect frame (``0x00 + uint32be(len) + json``) and the response is a
   sequence of the same frames.
3. Frame events: ``block.think`` masks carry reasoning, ``block.text`` masks
   carry the answer, ``multiStage`` announces phase switches, ``done`` ends
   the stream.
"""

import json
import logging
import os
import struct
import uuid
from typing import Any, Dict, Generator, List, Optional, Tuple

from .base import (
    Provider,
    ProviderAuthError,
    ProviderError,
    classify_http_error,
    http_post_raw,
)
from .jar import env_cookies, load_jar

logger = logging.getLogger('dsk.providers.kimi')

KIMI_BASE_URL = 'https://www.kimi.com'
KIMI_CHAT_URL = (f'{KIMI_BASE_URL}/apiv2/kimi.gateway.chat.v1.ChatService'
                 '/Chat')

KIMI_CONTEXT_LENGTH = int(os.getenv('DSF_KIMI_CONTEXT_LENGTH', '256000'))
KIMI_MAX_OUTPUT = int(os.getenv('DSF_KIMI_MAX_OUTPUT', '8192'))

STAGE_NAME_THINKING = 'STAGE_NAME_THINKING'

_USER_AGENT = (
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) '
    'Chrome/120.0.0.0 Safari/537.36'
)

# Web-app scenarios (K2 model generations exposed by kimi.com).
KIMI_MODELS: List[Dict[str, Any]] = [
    {'id': 'kimi-k2.6', 'scenario': 'SCENARIO_K2D6', 'thinking': True},
    {'id': 'kimi-k2.5', 'scenario': 'SCENARIO_K2D5', 'thinking': True},
]


def _token() -> str:
    raw = (os.getenv('KIMI_TOKEN', '') or '').strip()
    if raw:
        return raw
    jar = load_jar('kimi') or env_cookies('KIMI')
    return (jar.get('token') or jar.get('jwt') or '').strip()


def _encode_frame(payload: Dict[str, Any]) -> bytes:
    body = json.dumps(payload).encode()
    return b'\x00' + struct.pack('>I', len(body)) + body


def _headers() -> Dict[str, str]:
    return {
        'Authorization': f'Bearer {_token()}',
        'Content-Type': 'application/connect+json',
        'Accept': 'application/connect+json',
        'User-Agent': _USER_AGENT,
        'Origin': KIMI_BASE_URL,
        'Referer': f'{KIMI_BASE_URL}/',
        'X-Request-Id': str(uuid.uuid4()),
    }


def _split_frames(data: bytes) -> Generator[Dict[str, Any], None, None]:
    """Iterate connect+json frames: 1 flag byte + 4-byte BE length + JSON."""
    offset = 0
    while offset + 5 <= len(data):
        length = struct.unpack('>I', data[offset + 1:offset + 5])[0]
        if offset + 5 + length > len(data):
            break
        payload = data[offset + 5:offset + 5 + length]
        offset += 5 + length
        try:
            obj = json.loads(payload.decode('utf-8'))
        except (ValueError, UnicodeDecodeError):
            continue
        if isinstance(obj, dict):
            yield obj


class KimiProvider(Provider):
    name = 'kimi'

    # ---------------------------------------------------------------- provider
    def available(self, auth_key: Optional[str] = None) -> bool:
        return bool(_token())

    def list_models(self, auth_key: Optional[str] = None) -> List[Dict[str, Any]]:
        """kimi.com exposes a fixed K2 picker (no discovery endpoint)."""
        if not _token():
            raise ProviderAuthError('no kimi.com JWT configured')
        return [{
            'id': entry['id'],
            'upstream_model': entry['scenario'],
            'thinking_enabled': entry['thinking'],
            'search_enabled': False,
            'vision': False,
            'image_gen': False,
            'context_length': KIMI_CONTEXT_LENGTH,
            'max_output_tokens': KIMI_MAX_OUTPUT,
            'extra': {'scenario': entry['scenario']},
        } for entry in KIMI_MODELS]

    def stream(self, prompt: str, *, model: str, thinking_enabled: bool = False,
               search_enabled: bool = False, temperature: Optional[float] = None,
               max_tokens: Optional[int] = None,
               images: Optional[List[Dict[str, Any]]] = None,
               image_generation: bool = False,
               no_proxy: bool = False,
               auth_key: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
        if images:
            raise ProviderError('kimi image input is not supported yet')
        scenario = model if model.startswith('SCENARIO_') else 'SCENARIO_K2D5'
        for entry in KIMI_MODELS:
            if entry['id'] == model or entry['scenario'] == model:
                scenario = entry['scenario']
                break
        payload = {
            'scenario': scenario,
            'chat_id': '',
            'tools': ([{'type': 'TOOL_TYPE_SEARCH', 'search': {}}]
                      if search_enabled else []),
            'message': {
                'parent_id': '',
                'role': 'user',
                'blocks': [{'message_id': '', 'text': {'content': prompt}}],
                'scenario': scenario,
            },
            'options': {'thinking': bool(thinking_enabled)},
        }
        response = http_post_raw(KIMI_CHAT_URL, _encode_frame(payload),
                                 headers=_headers(), no_proxy=no_proxy)
        if response.status_code != 200:
            try:
                error_text = response.text or ''
            except Exception:  # pragma: no cover
                error_text = f'HTTP {response.status_code}'
            raise classify_http_error(response.status_code, error_text,
                                      response.headers)
        return self._iter_chunks(response.content or b'')

    def _iter_chunks(self, data: bytes) -> Generator[Dict[str, Any], None, None]:
        phase = 'answer'
        done = False
        for obj in _split_frames(data):
            if done:
                break
            if obj.get('heartbeat'):
                continue
            if obj.get('error'):
                error = obj['error']
                message = (error.get('message') if isinstance(error, dict)
                           else str(error)) or json.dumps(error)[:300]
                raise ProviderError(f'kimi stream error: {message}')
            # multiStage announces phase switches (thinking vs answer)
            stages = ((obj.get('block') or {}).get('multiStage') or {}) \
                .get('stages') or []
            if stages and stages[0].get('name') == STAGE_NAME_THINKING:
                phase = 'answer' if stages[0].get('status') == 'completed' \
                    else 'thinking'
            flags = (obj.get('block') or {}).get('text') or {}
            if flags.get('flags') == 'thinking':
                phase = 'thinking'
            elif flags.get('flags') == 'answer':
                phase = 'answer'
            op = obj.get('op')
            if op in ('set', 'append'):
                mask = str(obj.get('mask') or '')
                think = (obj.get('block') or {}).get('think') or {}
                text = (obj.get('block') or {}).get('text') or {}
                if 'block.think' in mask and think.get('content'):
                    yield {'content': think['content'], 'type': 'thinking',
                           'finish_reason': None}
                elif 'block.text' in mask and text.get('content'):
                    yield {'content': text['content'],
                           'type': 'text' if phase != 'thinking' else 'thinking',
                           'finish_reason': None}
                elif text.get('content'):
                    yield {'content': text['content'],
                           'type': 'text' if phase != 'thinking' else 'thinking',
                           'finish_reason': None}
            if obj.get('done') is not None:
                done = True
        yield {'content': '', 'type': 'text', 'finish_reason': 'stop'}
