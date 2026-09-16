"""Grok provider via the grok.com web app (reverse-engineered).

Talks to grok.com's own app-chat endpoints with the browser session cookies
(``sso`` / ``sso-rw``) — no official xAI API, no paid keys.

How it works
------------
1. Credentials: the ``sso`` cookie of a logged-in grok.com session
   (``GROK_SSO`` env var, bot-managed ``grok_cookies.json`` jar, or
   ``GROK_COOKIES`` env JSON as fallback).
2. Generation: POST ``https://grok.com/rest/app-chat/conversations/new`` with
   a ``temporary: true`` conversation so nothing is persisted server-side.
   The mode (auto/fast/heavy/reasoning/deepsearch) is derived from the
   exposed model id, exactly like the web UI's mode picker.
3. Stream: newline-delimited JSON — ``result.response.token`` carries the
   text, ``isThinking`` marks reasoning tokens, and Aurora image generations
   arrive on ``streamingImageGenerationResponse.imageUrl`` (assets.grok.com).
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
    http_post_stream,
)
from .jar import env_cookies, load_jar

logger = logging.getLogger('dsk.providers.grok')

GROK_BASE_URL = 'https://grok.com'
GROK_NEW_URL = f'{GROK_BASE_URL}/rest/app-chat/conversations/new'
GROK_ASSETS_URL = 'https://assets.grok.com'

GROK_CONTEXT_LENGTH = int(os.getenv('DSF_GROK_CONTEXT_LENGTH', '131072'))
GROK_MAX_OUTPUT = int(os.getenv('DSF_GROK_MAX_OUTPUT', '8192'))

_USER_AGENT = (
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) '
    'Chrome/120.0.0.0 Safari/537.36'
)

# Exposed routes; the upstream mode is derived from the id like the web picker.
GROK_MODELS: List[Dict[str, Any]] = [
    {'id': 'grok-4', 'mode': 'auto', 'thinking': True},
    {'id': 'grok-4-reasoning', 'mode': 'reasoning', 'thinking': True},
    {'id': 'grok-4-heavy', 'mode': 'heavy', 'thinking': True},
    {'id': 'grok-3', 'mode': 'fast', 'thinking': False},
    {'id': 'grok-3-mini', 'mode': 'fast', 'thinking': False},
    {'id': 'grok-deepsearch', 'mode': 'deepsearch', 'thinking': True},
]


def _sso() -> str:
    raw = (os.getenv('GROK_SSO', '') or '').strip()
    if raw:
        return raw
    jar = load_jar('grok') or env_cookies('GROK')
    return (jar.get('sso') or jar.get('sso-rw') or '').strip()


def _mode_for(model: str) -> str:
    lowered = model.lower()
    if 'heavy' in lowered or 'big-brain' in lowered:
        return 'heavy'
    if 'expert' in lowered:
        return 'expert'
    if 'reasoning' in lowered or 'thinking' in lowered or 'r1' in lowered:
        return 'reasoning'
    if 'deepsearch' in lowered or 'deepersearch' in lowered:
        return 'deepsearch'
    return 'auto' if 'auto' in lowered else 'fast'


class GrokProvider(Provider):
    name = 'grok'

    # ------------------------------------------------------------- credentials
    def _cookies(self) -> str:
        sso = _sso()
        jar = load_jar('grok') or env_cookies('GROK')
        extra = {k: v for k, v in jar.items() if k.startswith('sso')}
        extra.setdefault('sso', sso)
        extra.setdefault('sso-rw', sso)
        return '; '.join(f'{k}={v}' for k, v in extra.items() if v)

    def _headers(self, accept: str = 'application/json') -> Dict[str, str]:
        return {
            'Cookie': self._cookies(),
            'User-Agent': _USER_AGENT,
            'Content-Type': 'application/json',
            'Accept': accept,
            'Origin': GROK_BASE_URL,
            'Referer': f'{GROK_BASE_URL}/',
            'x-xai-request-id': str(uuid.uuid4()),
        }

    # ---------------------------------------------------------------- provider
    def available(self, auth_key: Optional[str] = None) -> bool:
        return bool(_sso())

    def list_models(self, auth_key: Optional[str] = None) -> List[Dict[str, Any]]:
        """Web-app mode picker (grok.com has no public discovery endpoint)."""
        if not _sso():
            raise ProviderAuthError('no grok.com sso cookie configured')
        return [{
            'id': entry['id'],
            'upstream_model': entry['id'],
            'thinking_enabled': entry['thinking'],
            'search_enabled': entry['mode'] == 'deepsearch',
            'vision': False,  # web attachment upload not implemented
            'image_gen': True,   # Aurora image generation is built into chat
            'context_length': GROK_CONTEXT_LENGTH,
            'max_output_tokens': GROK_MAX_OUTPUT,
            'extra': {'modeId': entry['mode']},
        } for entry in GROK_MODELS]

    def stream(self, prompt: str, *, model: str, thinking_enabled: bool = False,
               search_enabled: bool = False, temperature: Optional[float] = None,
               max_tokens: Optional[int] = None,
               images: Optional[List[Dict[str, Any]]] = None,
               image_generation: bool = False,
               no_proxy: bool = False,
               auth_key: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
        if images:
            raise ProviderError('grok.com web attachments are not supported yet')
        mode = _mode_for(model)
        payload: Dict[str, Any] = {
            'temporary': True,
            'message': prompt,
            'fileAttachments': [],
            'imageAttachments': [],
            'disableSearch': not search_enabled,
            'enableImageGeneration': bool(image_generation),
            'returnImageBytes': False,
            'returnRawGrokInXaiRequest': False,
            'enableImageStreaming': True,
            'imageGenerationCount': 2,
            'forceConcise': False,
            'enableSideBySide': False,
            'sendFinalMetadata': True,
            'disableTextFollowUps': True,
            'responseMetadata': {},
            'disableMemory': True,
            'forceSideBySide': False,
            'isAsyncChat': False,
            'disableSelfHarmShortCircuit': False,
            'collectionIds': [],
            'disabledConnectorIds': [],
            'deviceEnvInfo': {
                'darkModeEnabled': True,
                'devicePixelRatio': 1.0,
                'screenWidth': 1920,
                'screenHeight': 1080,
                'viewportWidth': 1920,
                'viewportHeight': 1080,
            },
            'modeId': mode,
            'linkQuery': False,
        }
        response = http_post_stream(GROK_NEW_URL,
                                    headers=self._headers('text/event-stream'),
                                    json_body=payload, no_proxy=no_proxy)
        if response.status_code != 200:
            try:
                error_text = response.text or ''
            except Exception:  # pragma: no cover
                error_text = f'HTTP {response.status_code}'
            raise classify_http_error(response.status_code, error_text,
                                      response.headers)
        return self._iter_chunks(response)

    def _iter_chunks(self, response) -> Generator[Dict[str, Any], None, None]:
        """Yield unified chunks from grok.com's newline-delimited JSON stream."""
        for line in response.iter_lines():
            text = (line or b'').decode('utf-8', errors='replace').strip() \
                if isinstance(line, bytes) else str(line or '').strip()
            if not text or not text.startswith('{'):
                continue
            try:
                data = json.loads(text)
            except ValueError:
                continue
            result = data.get('result') or {}
            if result.get('errorCode'):
                raise ProviderError(f"grok.com stream error: "
                                    f"{result.get('errorCode')}: "
                                    f"{result.get('errorMessage', '')}")
            upstream = result.get('response') or {}
            token = upstream.get('token', result.get('token'))
            if token:
                is_thinking = bool(upstream.get('isThinking',
                                                result.get('isThinking')))
                yield {'content': token,
                       'type': 'thinking' if is_thinking else 'text',
                       'finish_reason': None}
            image = upstream.get('streamingImageGenerationResponse')
            if isinstance(image, dict) and image.get('imageUrl'):
                url = image['imageUrl']
                if url.startswith('/'):
                    url = f'{GROK_ASSETS_URL}{url}'
                yield {'content': f'![image]({url})', 'type': 'image',
                       'url': url, 'finish_reason': None}
        yield {'content': '', 'type': 'text', 'finish_reason': 'stop'}
