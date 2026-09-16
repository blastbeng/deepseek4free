"""Claude provider via the claude.ai web app (reverse-engineered).

Same approach as the DeepSeek/Gemini/ChatGPT web clients: it talks to the
web app's own private endpoints with browser credentials — no official API,
no paid keys.

How it works
------------
1. Credentials: the ``sessionKey`` cookie of a logged-in claude.ai session
   (``CLAUDE_SESSION_KEY`` env var, bot-managed ``claude_cookies.json`` jar,
   or ``CLAUDE_COOKIES`` env JSON as fallback).
2. Organization: GET ``/api/organizations`` returns the workspaces the
   session belongs to; the first one is used (cached for 10 minutes).
3. Generation: each request creates a throwaway temporary conversation with
   a client-generated uuid (POST ``/api/organizations/{org}/chat_conversations``)
   and then streams from POST ``.../chat_conversations/{uuid}/completion``.
   The response is an anthropic-style SSE stream (content_block_delta events).

Tool calling is emulated by the shared TOOL_CALL protocol in the
OpenAI-compatible server; thinking arrives as ``thinking_delta`` events.
"""

import json
import logging
import os
import threading
import time
import uuid
from typing import Any, Dict, Generator, List, Optional

from .base import (
    Provider,
    ProviderError,
    ProviderAuthError,
    ProviderRateLimitError,
    classify_http_error,
    http_get,
    http_post_stream,
    http_post_raw,
    parse_sse_data,
)
from .jar import env_cookies, load_jar

logger = logging.getLogger('dsk.providers.claude')

CLAUDE_BASE_URL = 'https://claude.ai'
CLAUDE_ORGS_URL = f'{CLAUDE_BASE_URL}/api/organizations'
CLAUDE_CONV_URL = (f'{CLAUDE_BASE_URL}/api/organizations/'
                   '{org}/chat_conversations')

CLAUDE_CONTEXT_LENGTH = int(os.getenv('DSF_CLAUDE_CONTEXT_LENGTH', '200000'))
CLAUDE_MAX_OUTPUT = int(os.getenv('DSF_CLAUDE_MAX_OUTPUT', '8192'))
ORG_TTL = 600.0

_USER_AGENT = (
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) '
    'Chrome/120.0.0.0 Safari/537.36'
)

# The claude.ai web app exposes a fixed model picker (no discovery endpoint);
# these are the ids the completion endpoint accepts.
CLAUDE_MODELS: List[Dict[str, Any]] = [
    {'id': 'claude-sonnet-4-6', 'thinking': True, 'vision': False},
    {'id': 'claude-opus-4-6', 'thinking': True, 'vision': False},
    {'id': 'claude-haiku-4-5', 'thinking': False, 'vision': False},
]


def _session_key() -> str:
    raw = (os.getenv('CLAUDE_SESSION_KEY', '') or '').strip()
    if raw:
        return raw
    jar = load_jar('claude') or env_cookies('CLAUDE')
    return (jar.get('sessionKey') or '').strip()


class ClaudeWebProvider(Provider):
    name = 'claude'

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._device_id = str(uuid.uuid4())
        self._org_id: Optional[str] = None
        self._org_at = 0.0

    # ------------------------------------------------------------- credentials
    def _headers(self, accept: str = 'application/json') -> Dict[str, str]:
        return {
            'Cookie': f'sessionKey={_session_key()}',
            'User-Agent': _USER_AGENT,
            'anthropic-device-id': self._device_id,
            'Content-Type': 'application/json',
            'Accept': accept,
            'Origin': CLAUDE_BASE_URL,
            'Referer': f'{CLAUDE_BASE_URL}/new',
        }

    def _organization_id(self, no_proxy: bool = False) -> str:
        with self._lock:
            if self._org_id and time.time() - self._org_at < ORG_TTL:
                return self._org_id
        response = http_get(CLAUDE_ORGS_URL, headers=self._headers(),
                            no_proxy=no_proxy)
        if response.status_code != 200:
            try:
                error_text = response.text or ''
            except Exception:  # pragma: no cover
                error_text = f'HTTP {response.status_code}'
            raise classify_http_error(response.status_code, error_text,
                                      response.headers)
        try:
            orgs = response.json()
        except ValueError:
            raise ProviderError('claude.ai /api/organizations returned non-JSON')
        if not isinstance(orgs, list):
            raise ProviderError('Unexpected /api/organizations payload shape')
        for org in orgs:
            if isinstance(org, dict) and org.get('uuid'):
                with self._lock:
                    self._org_id = str(org['uuid'])
                    self._org_at = time.time()
                return self._org_id
        raise ProviderAuthError('claude.ai session has no organizations — '
                                'sessionKey expired?')

    # ---------------------------------------------------------------- provider
    def available(self, auth_key: Optional[str] = None) -> bool:
        return bool(_session_key())

    def list_models(self, auth_key: Optional[str] = None) -> List[Dict[str, Any]]:
        """Web-app model picker (fixed set; validated live via the org lookup)."""
        if not _session_key():
            raise ProviderAuthError('no claude.ai sessionKey configured')
        self._organization_id()  # validates the session cheaply
        return [{
            'id': entry['id'],
            'upstream_model': entry['id'],
            'thinking_enabled': entry['thinking'],
            'search_enabled': False,
            'vision': entry['vision'],
            'image_gen': False,
            'context_length': CLAUDE_CONTEXT_LENGTH,
            'max_output_tokens': CLAUDE_MAX_OUTPUT,
            'extra': {'title': entry['id']},
        } for entry in CLAUDE_MODELS]

    def stream(self, prompt: str, *, model: str, thinking_enabled: bool = False,
               search_enabled: bool = False, temperature: Optional[float] = None,
               max_tokens: Optional[int] = None,
               images: Optional[List[Dict[str, Any]]] = None,
               image_generation: bool = False,
               no_proxy: bool = False,
               auth_key: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
        if images:
            raise ProviderError('claude.ai web attachments are not supported yet')
        org = self._organization_id(no_proxy=no_proxy)
        conv_id = str(uuid.uuid4())

        # Throwaway temporary conversation (client-generated uuid); 409/400
        # "already exists" collisions are tolerated like the web client does.
        create_body = json.dumps({
            'include_conversation_preferences': True,
            'is_temporary': True,
            'name': '',
            'uuid': conv_id,
        }).encode()
        create = http_post_raw(CLAUDE_CONV_URL.format(org=org), create_body,
                               headers=self._headers(), no_proxy=no_proxy)
        if create.status_code not in (200, 201, 400, 409):
            try:
                error_text = create.text or ''
            except Exception:  # pragma: no cover
                error_text = f'HTTP {create.status_code}'
            raise classify_http_error(create.status_code, error_text,
                                      create.headers)

        payload = {
            'attachments': [],
            'files': [],
            'locale': 'en-US',
            'model': model,
            'personalized_styles': [{}],
            'prompt': prompt,
            'rendering_mode': 'messages',
            'sync_sources': [],
            'timezone': 'UTC',
            'tools': [],
            'turn_message_uuids': {
                'human_message_uuid': str(uuid.uuid4()),
                'assistant_message_uuid': str(uuid.uuid4()),
            },
            'create_conversation_params': {
                'name': '',
                'model': model,
                'include_conversation_preferences': True,
                'is_temporary': True,
                'enabled_imagine': bool(image_generation),
            },
        }
        response = http_post_stream(
            f'{CLAUDE_CONV_URL.format(org=org)}/{conv_id}/completion',
            headers=self._headers('text/event-stream'), json_body=payload,
            no_proxy=no_proxy)
        if response.status_code != 200:
            try:
                error_text = response.text or ''
            except Exception:  # pragma: no cover
                error_text = f'HTTP {response.status_code}'
            raise classify_http_error(response.status_code, error_text,
                                      response.headers)
        return self._iter_chunks(response)

    def _iter_chunks(self, response) -> Generator[Dict[str, Any], None, None]:
        """Yield unified chunks from the anthropic-style completion stream."""
        for line in response.iter_lines():
            data = parse_sse_data(line)
            if not data:
                continue
            etype = data.get('type')
            if etype == 'content_block_delta':
                delta = data.get('delta') or {}
                if delta.get('type') == 'text_delta' and delta.get('text'):
                    yield {'content': delta['text'], 'type': 'text',
                           'finish_reason': None}
                elif delta.get('type') == 'thinking_delta' and delta.get('thinking'):
                    yield {'content': delta['thinking'], 'type': 'thinking',
                           'finish_reason': None}
            elif etype == 'message_limit':
                if data.get('is_hard_limit'):
                    raise ProviderRateLimitError(
                        'claude.ai usage limit reached for this session')
            elif etype == 'error':
                error = data.get('error') or data
                raise ProviderError(f'claude.ai stream error: {error}')
        yield {'content': '', 'type': 'text', 'finish_reason': 'stop'}
