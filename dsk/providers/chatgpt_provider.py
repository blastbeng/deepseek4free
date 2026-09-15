"""ChatGPT provider via the chatgpt.com web backend-api (reverse-engineered).

Same approach as the DeepSeek reverse client: it talks to the web app's own
private endpoints with browser credentials — no official API, no paid keys.

How it works
------------
1. Credentials: either a directly provided access token (``CHATGPT_ACCESS_TOKEN``
   env var) or the ``__Secure-next-auth.session-token`` cookies of a logged-in
   chatgpt.com session (bot-managed ``chatgpt_cookies.json`` file kept fresh by
   ``dsk.refresher``, or ``CHATGPT_SESSION_COOKIES`` env JSON as fallback).
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
    http_put_raw,
    image_dimensions,
    parse_sse_data,
)

logger = logging.getLogger('dsk.providers.chatgpt')

CHATGPT_BASE_URL = 'https://chatgpt.com'
CHATGPT_SESSION_URL = f'{CHATGPT_BASE_URL}/api/auth/session'
CHATGPT_MODELS_URL = f'{CHATGPT_BASE_URL}/backend-api/models'
CHATGPT_CONVERSATION_URL = f'{CHATGPT_BASE_URL}/backend-api/conversation'
CHATGPT_FILES_URL = f'{CHATGPT_BASE_URL}/backend-api/files'

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
        self._token_sig: str = ''   # detects cookie-jar rotations

    # ------------------------------------------------------------- credentials
    def _session_cookies(self) -> Dict[str, str]:
        """Session cookies: bot-managed jar first, then env JSON fallback."""
        path = _cookie_file()
        if path.is_file():
            try:
                cookies = self._normalize_cookies(json.loads(path.read_text()))
            except (ValueError, OSError) as e:
                logger.warning('chatgpt_cookies.json unreadable: %s', e)
            else:
                if cookies:
                    return cookies
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

    def _get_access_token(self, refresh: bool = False,
                          no_proxy: bool = False) -> str:
        """Access token: env-provided, or fetched from /api/auth/session with
        the session cookies (cached for TOKEN_TTL)."""
        env_token = _env_token()
        if env_token:
            return env_token
        with self._lock:
            cookies = self._session_cookies()
            sig = repr(sorted(cookies.items()))
            if sig != self._token_sig:
                self._token_sig = sig   # cookies rotated by the refresher
                self._token = None      # -> drop the cached access token
                self._token_at = 0.0
            if not refresh and self._token and \
                    time.monotonic() - self._token_at < TOKEN_TTL:
                return self._token
            if not cookies:
                raise ProviderAuthError(
                    'No ChatGPT credentials. Set CHATGPT_ACCESS_TOKEN, or provide '
                    'the __Secure-next-auth.session-token cookies of a logged-in '
                    'chatgpt.com session via CHATGPT_SESSION_COOKIES or '
                    'chatgpt_cookies.json.'
                )
            response = http_get(CHATGPT_SESSION_URL,
                                headers={'User-Agent': _USER_AGENT},
                                cookies=cookies, no_proxy=no_proxy)
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
                'vision': True,
                'image_gen': True,
                'context_length': context_length,
                'max_output_tokens': CHATGPT_MAX_OUTPUT,
                'extra': {'title': entry.get('title'),
                          'description': entry.get('description')},
            })
        if not models:
            raise ProviderError('ChatGPT /backend-api/models returned no models')
        return models

    def _upload_image(self, token: str, mime: str, data: bytes,
                      index: int, no_proxy: bool = False) -> Dict[str, Any]:
        """Upload one image through the web app's 3-step file flow and return
        an ``image_asset_pointer`` part for the conversation message:

        1. POST /backend-api/files  -> {'file_id', 'upload_url'}
        2. PUT upload_url (Azure blob) with the raw bytes (expect 201)
        3. POST /backend-api/files/{id}/uploaded -> registration complete
        """
        ext = {'image/jpeg': 'jpg', 'image/png': 'png', 'image/gif': 'gif',
               'image/webp': 'webp'}.get(mime, (mime.split('/')[-1] or 'png')[:5])
        filename = f'image_{index}.{ext}'

        def _json(response: Any, what: str) -> Dict[str, Any]:
            if response.status_code not in (200, 201):
                raise classify_http_error(response.status_code,
                                          str(getattr(response, 'text', ''))[:300],
                                          getattr(response, 'headers', None))
            try:
                body = response.json()
            except ValueError:
                raise ProviderError(f'ChatGPT file upload ({what}) returned a non-JSON body')
            if not isinstance(body, dict):
                raise ProviderError(f'ChatGPT file upload ({what}) payload is unexpected')
            return body

        step1 = _json(http_post_stream(
            CHATGPT_FILES_URL, headers=self._headers(token),
            json_body={'file_name': filename, 'file_size': len(data),
                       'use_case': 'multimodal'}, no_proxy=no_proxy), 'create')
        file_id = str(step1.get('file_id') or '')
        upload_url = str(step1.get('upload_url') or '')
        if not file_id or not upload_url:
            raise ProviderError('ChatGPT file upload did not return file_id/upload_url')

        step2 = http_put_raw(upload_url, data=data, headers={
            'Content-Type': mime, 'x-ms-blob-type': 'BlockBlob'},
            no_proxy=no_proxy)
        if step2.status_code not in (200, 201):
            raise ProviderUnavailableError(
                f'ChatGPT blob upload failed (HTTP {step2.status_code})')

        step3 = _json(http_post_stream(
            f'{CHATGPT_FILES_URL}/{file_id}/uploaded', headers=self._headers(token),
            json_body={'file_id': file_id}, no_proxy=no_proxy), 'finalize')
        width, height = image_dimensions(data)
        meta = step3.get('file_metadata') if isinstance(step3.get('file_metadata'), dict) else {}
        try:
            width = int(meta.get('width') or width)
            height = int(meta.get('height') or height)
        except (TypeError, ValueError):
            pass
        return {
            'content_type': 'image_asset_pointer',
            'asset_pointer': f'file-service://{file_id}',
            'size_bytes': len(data),
            'width': width,
            'height': height,
            'metadata': {'dalle': None, 'gizmo': None},
        }

    def _download_pointer(self, pointer: str,
                          no_proxy: bool = False) -> Optional[str]:
        """Resolve an ``file-service://`` / ``sediment://`` asset pointer to a
        downloadable URL via /backend-api/files/{id}/download."""
        file_id = pointer.split('://', 1)[-1]
        if not file_id:
            return None
        try:
            token = self._get_access_token(no_proxy=no_proxy)
            response = http_get(
                f'{CHATGPT_FILES_URL}/{file_id}/download', headers=self._headers(token),
                no_proxy=no_proxy)
        except Exception as e:
            logger.warning('ChatGPT image download failed for %s: %s', file_id, e)
            return None
        if response.status_code != 200:
            logger.warning('ChatGPT image download HTTP %s for %s',
                           response.status_code, file_id)
            return None
        try:
            body = response.json()
        except ValueError:
            return None
        url = body.get('download_url') if isinstance(body, dict) else None
        return url if isinstance(url, str) and url.startswith('http') else None

    def stream(self, prompt: str, *, model: str, thinking_enabled: bool = False,
               search_enabled: bool = False, temperature: Optional[float] = None,
               max_tokens: Optional[int] = None,
               images: Optional[List[Dict[str, Any]]] = None,
               image_generation: bool = False,
               no_proxy: bool = False,
               auth_key: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
        # Temperature/max_tokens are not honored by the web conversation API.
        token = self._get_access_token(no_proxy=no_proxy)

        if images:
            pointers = [self._upload_image(token, image.get('mime', 'image/png'),
                                           image.get('data', b''), i,
                                           no_proxy=no_proxy)
                        for i, image in enumerate(images)]
            content: Dict[str, Any] = {
                'content_type': 'multimodal_text',
                'parts': [{'content_type': 'text', 'text': prompt}, *pointers],
            }
        else:
            content = {'content_type': 'text', 'parts': [prompt]}

        body: Dict[str, Any] = {
            'action': 'next',
            'messages': [{
                'id': str(uuid.uuid4()),
                'author': {'role': 'user'},
                'content': content,
                'metadata': {},
            }],
            'model': model,
            'parent_message_id': str(uuid.uuid4()),
            'conversation_mode': {'kind': 'primary_assistant'},
            'timezone_offset_min': 0,
            'history_and_training_disabled': False,
            'force_paragen': bool(image_generation),
            'force_paragen_model_slug': 'dalle' if image_generation else None,
            'force_use_search_plugin': False,
            'system_hints': ['search'] if search_enabled else [],
            'supports_buffering': True,
        }

        response = http_post_stream(CHATGPT_CONVERSATION_URL,
                                    headers=self._headers(token), json_body=body,
                                    no_proxy=no_proxy)
        if response.status_code != 200:
            try:
                error_text = response.text or ''
            except Exception:  # pragma: no cover
                error_text = f'HTTP {response.status_code}'
            raise classify_http_error(response.status_code, error_text,
                                      response.headers)
        return self._iter_chunks(response, no_proxy=no_proxy)

    def _iter_chunks(self, response,
                     no_proxy: bool = False) -> Generator[Dict[str, Any], None, None]:
        """Yield unified chunks from the ChatGPT conversation SSE stream.

        Each event carries the full assistant message so far; we emit deltas by
        diffing against the previously seen text. Generated images arrive as
        ``image_asset_pointer`` parts — each pointer is resolved to a
        downloadable URL once and emitted as an ``image`` chunk.
        """
        prev = ''
        seen_pointers: set = set()
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
            if content.get('content_type') not in (None, 'text', 'multimodal_text'):
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
            for part in parts:
                if not isinstance(part, dict):
                    continue
                pointer = str(part.get('asset_pointer') or '')
                if not pointer or pointer in seen_pointers:
                    continue
                seen_pointers.add(pointer)
                url = self._download_pointer(pointer, no_proxy=no_proxy)
                if url:
                    yield {'content': f'![image]({url})', 'type': 'image',
                           'url': url, 'finish_reason': None}
            if message.get('status') == 'finished_successfully':
                break
        yield {'content': '', 'type': 'text', 'finish_reason': 'stop'}
