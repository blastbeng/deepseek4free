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
"""

import json
import logging
import os
import uuid
from typing import Any, Dict, Generator, List, Optional

from .base import (
    Provider,
    ProviderError,
    ProviderAuthError,
    classify_http_error,
    http_post_raw,
    http_post_stream,
    parse_sse_data,
)
from .jar import env_cookies, load_jar

logger = logging.getLogger('dsk.providers.qwen')

QWEN_BASE_URL = 'https://chat.qwen.ai'
QWEN_NEWCHAT_URL = f'{QWEN_BASE_URL}/api/v2/chats/new'
QWEN_COMPLETIONS_URL = f'{QWEN_BASE_URL}/api/v2/chat/completions'

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

# Web-app model ids (upstream aliases as exposed in the chat.qwen.ai picker).
QWEN_MODELS: List[Dict[str, Any]] = [
    {'id': 'qwen3.7-max', 'thinking': True},
    {'id': 'qwen3.6-plus', 'thinking': True},
    {'id': 'qwen3-coder-plus', 'thinking': False},
    {'id': 'qwen3.6-35b-a3b', 'thinking': True},
    {'id': 'qwen3.6-27b', 'thinking': False},
]


def _token() -> str:
    raw = (os.getenv('QWEN_TOKEN', '') or '').strip()
    if raw:
        return raw
    jar = load_jar('qwen') or env_cookies('QWEN')
    return (jar.get('token') or '').strip()


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


class QwenProvider(Provider):
    name = 'qwen'

    # ---------------------------------------------------------------- provider
    def available(self, auth_key: Optional[str] = None) -> bool:
        return bool(_token())

    def list_models(self, auth_key: Optional[str] = None) -> List[Dict[str, Any]]:
        """Web-app model picker (validated live on first chat creation)."""
        if not _token():
            raise ProviderAuthError('no chat.qwen.ai token configured')
        return [{
            'id': entry['id'],
            'upstream_model': entry['id'],
            'thinking_enabled': entry['thinking'],
            'search_enabled': False,
            'vision': False,  # image input not implemented
            'image_gen': False,
            'context_length': QWEN_CONTEXT_LENGTH,
            'max_output_tokens': QWEN_MAX_OUTPUT,
            'extra': {'title': entry['id']},
        } for entry in QWEN_MODELS]

    def stream(self, prompt: str, *, model: str, thinking_enabled: bool = False,
               search_enabled: bool = False, temperature: Optional[float] = None,
               max_tokens: Optional[int] = None,
               images: Optional[List[Dict[str, Any]]] = None,
               image_generation: bool = False,
               no_proxy: bool = False,
               auth_key: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
        if images:
            raise ProviderError('qwen image input is not supported yet')
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
        return self._iter_chunks(response)

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
        prev_summary = ''
        for line in response.iter_lines():
            data = parse_sse_data(line)
            if not data:
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
            if status == 'finished':
                break
        yield {'content': '', 'type': 'text', 'finish_reason': 'stop'}
