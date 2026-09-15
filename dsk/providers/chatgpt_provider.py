"""ChatGPT provider via the chatgpt.com web backend-api (reverse-engineered).

Same approach as the DeepSeek reverse client: it talks to the web app's own
private endpoints with browser credentials — no official API, no paid keys.

How it works
------------
1. Credentials: either a directly provided access token (``CHATGPT_ACCESS_TOKEN``
   env var) or the ``__Secure-next-auth.session-token`` cookies of a logged-in
   chatgpt.com session (``CHATGPT_SESSION_COOKIES`` env JSON or a
   ``chatgpt_cookies.json`` file).
2. Access token: GET ``https://chatgpt.com/api/auth/session`` with the session
   cookies returns ``{"accessToken": ...}``; the token is cached and refreshed
   when it expires.
3. Model discovery (dynamic): GET ``https://chatgpt.com/backend-api/models``
   returns the models the account currently has access to. Nothing is
   hardcoded — whatever the web app offers is exposed.
4. Generation: POST ``https://chatgpt.com/backend-api/conversation`` with
   ``action: 'next'``. Each SSE event carries the full assistant message so
   far, which we diff into deltas.

Tool calling and web search hints are limited on this endpoint; tool calls are
emulated by the shared TOOL_CALL protocol in the OpenAI-compatible server.
"""

import json
import logging
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from .base import (
    Provider,
    ProviderAuthError,
    ProviderError,
    classify_http_error,
    http_get,
    http_post_stream,
    parse_sse_data,
)

logger = logging.getLogger('dsk.providers.chatgpt')

CHATGPT_BASE_URL = 'https://chatgpt.com'
CHATGPT_SESSION_URL = f'{CHATGPT_BASE_URL}/api/auth/session'
CHATGPT_MODELS_URL = f'{CHATGPT_BASE_URL}/backend-api/models'
CHATGPT_CONVERSATION_URL = f'{CHATGPT_BASE_URL}/backend-api/conversation'

# Estimated capability metadata advertised on /v1/models for agent tooling
# (upstream reports a per-model context size; this is the fallback).
CHATGPT_CONTEXT_LENGTH = int(os.getenv('DSF_CHATGPT_CONTEXT_LENGTH', '128000'))
CHATGPT_MAX_OUTPUT = int(os.getenv('DSF_CHATGPT_MAX_OUTPUT', '16384'))

TOKEN_TTL = 3600.0  # re-fetch the accessToken from the session endpoint hourly

_USER_AGENT = (
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) '
    'Chrome/120.0.0.0 Safari/537.36'
)


def _env_token() -> str:
    return (os.getenv('CHATGPT_ACCESS_TOKEN', '') or
            os.getenv('CHATGPT_SESSION_TOKEN', '') or '').strip()


def _cookie_file() -> Path:
    cookies_dir = os.getenv('COOKIES_DIR')
    if cookies_dir and Path(cookies_dir).is_dir():
        return Path(cookies_dir) / 'chatgpt_cookies.json'
    return Path(__file__).resolve().parent.parent / 'chatgpt_cookies.json'


def _is_thinking_model(entry: Dict[str, Any]) -> bool:
    """Heuristic: does this upstream model reason by default?"""
    slug = str(entry.get('slug') or '').lower()
    if 'thinking' in slug or 'reasoning' in slug:
        return True
    if re.match(r'^o[134]($|-)', slug) or slug.startswith('gpt-5'):
        return True
    for tag in entry.get('tags') or []:
        text = str(tag).lower()
        if 'thinking' in text or 'reasoning' in text:
            return True
    return False


class ChatGPTProvider(Provider):
    name = 'chatgpt'

    def __init__(self) -> None:
        # Stable per-process device id; ChatGPT rejects requests without one.
        self._device_id = str(uuid.uuid4())
        self._lock = threading.Lock()
        self._token: Optional[str] = None
        self._token_at = 0.0

    # ------------------------------------------------------------- credentials
    def _session_cookies(self) -> Dict[str, str]:
        """Session cookies from env JSON or a cookie file (dict or list form)."""
        raw = (os.getenv('CHATGPT_SESSION_COOKIES', '') or '').strip()
        if raw:
            try:
                data = json.loads(raw)
            except ValueError:
                logger.warning('CHATGPT_SESSION_COOKIES is not valid JSON, ignoring')
            else:
                cookies = self._normalize_cookies(data)
                if cookies:
                    return cookies
        path = _cookie_file()
        if path.is_file():
            try:
                cookies = self._normalize_cookies(json.loads(path.read_text()))
            except (ValueError, OSError) as e:
                logger.warning('chatgpt_cookies.json unreadable: %s', e)
            else:
                if cookies:
                    return cookies
        return {}

    @staticmethod
    def _normalize_cookies(data: Any) -> Dict[str, str]:
        """Accept {'name': 'value'} or [{'name': ..., 'value': ...}] formats."""
        out: Dict[str, str] = {}
        entries: List[Any]
        if isinstance(data, dict):
            entries = list(data.items())
        elif isinstance(data, list):
            entries = data
        else:
            return out
        for entry in entries:
            if isinstance(entry, dict):
                name = str(entry.get('name') or '').strip()
                value = str(entry.get('value') or '').strip()
            else:
                name, value = str(entry[0]).strip(), str(entry[1]).strip()
            if name and value:
                out[name] = value
        return out

    def _get_access_token(self, refresh: bool = False) -> str:
        """Access token: env-provided, or fetched from /api/auth/session with
        the session cookies (cached for TOKEN_TTL)."""
        env_token = _env_token()
        if env_token:
            return env_token
        with self._lock:
            if not refresh and self._token and \
                    time.monotonic() - self._token_at < TOKEN_TTL:
                return self._token
            cookies = self._session_cookies()
            if not cookies:
                raise ProviderAuthError(
                    'No ChatGPT credentials. Set CHATGPT_ACCESS_TOKEN, or provide '
                    'the __Secure-next-auth.session-token cookies of a logged-in '
                    'chatgpt.com session via CHATGPT_SESSION_COOKIES or '
                    'chatgpt_cookies.json.'
                )
            response = http_get(CHATGPT_SESSION_URL,
                                headers={'User-Agent': _USER_AGENT},
                                cookies=cookies)
            if response.status_code != 200:
                try:
                    error_text = response.text or ''
                except Exception:  # pragma: no cover
                    error_text = f'HTTP {response.status_code}'
                raise classify_http_error(response.status_code, error_text,
                                          response.headers)
            try:
                data = response.json()
            except ValueError:
                data = {}
            token = str((data or {}).get('accessToken') or '').strip()
            if not token:
                raise ProviderAuthError(
                    'ChatGPT session cookies are expired or invalid '
                    '(/api/auth/session returned no accessToken).'
                )
            self._token = token
            self._token_at = time.monotonic()
            return token

    def _headers(self, token: str) -> Dict[str, str]:
        return {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json',
            'User-Agent': _USER_AGENT,
            'Oai-Device-Id': self._device_id,
            'Oai-Language': 'en-US',
        }

    # ---------------------------------------------------------------- provider
    def available(self, auth_key: Optional[str] = None) -> bool:
        if _env_token():
            return True
        try:
            return bool(self._session_cookies())
        except OSError:
            return False

    def list_models(self, auth_key: Optional[str] = None) -> List[Dict[str, Any]]:
        """Discover the models the web session currently has access to."""
        token = self._get_access_token()
        response = http_get(CHATGPT_MODELS_URL, headers=self._headers(token))
        if response.status_code != 200:
            try:
                error_text = response.text or ''
            except Exception:  # pragma: no cover
                error_text = f'HTTP {response.status_code}'
            raise classify_http_error(response.status_code, error_text,
                                      response.headers)
        try:
            body = response.json()
        except ValueError:
            raise ProviderError('ChatGPT /backend-api/models returned a non-JSON body')
        entries = body.get('models') if isinstance(body, dict) else None
        if not isinstance(entries, list):
            raise ProviderError('Unexpected /backend-api/models payload shape')

        models: List[Dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            slug = str(entry.get('slug') or '').strip()
            if not slug:
                continue
            max_tokens = entry.get('max_tokens')
            context_length = (int(max_tokens)
                              if isinstance(max_tokens, (int, float))
                              and max_tokens >= 1024 else CHATGPT_CONTEXT_LENGTH)
            models.append({
                'id': slug,
                'upstream_model': slug,
                'thinking_enabled': _is_thinking_model(entry),
                'search_enabled': False,
                'context_length': context_length,
                'max_output_tokens': CHATGPT_MAX_OUTPUT,
                'extra': {'title': entry.get('title'),
                          'description': entry.get('description')},
            })
        if not models:
            raise ProviderError('ChatGPT /backend-api/models returned no models')
        return models

    def stream(self, prompt: str, *, model: str, thinking_enabled: bool = False,
               search_enabled: bool = False, temperature: Optional[float] = None,
               max_tokens: Optional[int] = None,
               auth_key: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
        # Temperature/max_tokens are not honored by the web conversation API.
        token = self._get_access_token()

        body: Dict[str, Any] = {
            'action': 'next',
            'messages': [{
                'id': str(uuid.uuid4()),
                'author': {'role': 'user'},
                'content': {'content_type': 'text', 'parts': [prompt]},
                'metadata': {},
            }],
            'model': model,
            'parent_message_id': str(uuid.uuid4()),
            'conversation_mode': {'kind': 'primary_assistant'},
            'timezone_offset_min': 0,
            'history_and_training_disabled': False,
            'force_paragen': False,
            'force_use_search_plugin': False,
            'system_hints': ['search'] if search_enabled else [],
            'supports_buffering': True,
        }

        response = http_post_stream(CHATGPT_CONVERSATION_URL,
                                    headers=self._headers(token), json_body=body)
        if response.status_code != 200:
            try:
                error_text = response.text or ''
            except Exception:  # pragma: no cover
                error_text = f'HTTP {response.status_code}'
            raise classify_http_error(response.status_code, error_text,
                                      response.headers)
        return self._iter_chunks(response)

    def _iter_chunks(self, response) -> Generator[Dict[str, Any], None, None]:
        """Yield unified chunks from the ChatGPT conversation SSE stream.

        Each event carries the full assistant message so far; we emit deltas by
        diffing against the previously seen text.
        """
        prev = ''
        for line in response.iter_lines():
            data = parse_sse_data(line)
            if not data:
                continue
            if data.get('error'):
                raise ProviderError(f"ChatGPT stream error: {data.get('error')}")
            message = data.get('message')
            if not isinstance(message, dict):
                continue
            author = (message.get('author') or {}).get('role')
            if author not in ('assistant', 'tool'):
                continue
            content = message.get('content') or {}
            if content.get('content_type') not in (None, 'text'):
                # reasoning/thought payloads are not streamed by the web API
                continue
            parts: List[Any] = content.get('parts') or []
            text = ''.join(p for p in parts if isinstance(p, str))
            if text.startswith(prev):
                delta = text[len(prev):]
            else:
                delta = text
                prev = ''
            if delta:
                prev += delta
                yield {'content': delta, 'type': 'text', 'finish_reason': None}
            if message.get('status') == 'finished_successfully':
                break
        yield {'content': '', 'type': 'text', 'finish_reason': 'stop'}
