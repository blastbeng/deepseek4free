"""Mistral provider via chat.mistral.ai (Le Chat) — reverse-engineered.

Ported from the mistral-proxy APK reverse-engineering: the mobile app talks to
the web app's private tRPC + chat endpoints with either an anonymous
``stableAnonymousIdentifier`` (5 msgs/day, rotated to reset the quota) or an
Ory Kratos session token — no official API, no paid keys.

How it works
------------
1. Bootstrap: GET ``https://auth.mistral.ai/self-service/registration/api``
   warms the Kratos/Cloudflare cookies (non-fatal).
2. Single call: POST ``/api/chat`` with ``mode: 'create'`` — the request both
   starts the conversation and streams the answer (no separate newChat call;
   ``agentId`` must be absent — ``null`` → HTTP 400). The body is NOT SSE —
   it is newline-delimited ``<type_num>:<json>`` frames (15=data patches,
   16=metadata, 6=error, 8=end). Assistant text arrives as JSON-patch ops on
   ``/contentChunks`` (replace = full snapshot, append = delta, including
   ``/contentChunks/N/text`` string deltas).
3. An Ory Kratos session token (``MISTRAL_SESSION_TOKEN``) is required:
   anonymous access now returns an account upsell instead of model output.
4. Quota (error code 6200): rotate the stable UUID and retry once.
"""

import codecs
import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional

from .base import (
    Provider,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    classify_http_error,
    http_get,
    http_post_stream,
)
from .jar import env_cookies, load_jar, save_jar

logger = logging.getLogger('dsk.providers.mistral')

# Mistral's anonymous tier no longer produces model output: the endpoint
# answers HTTP 200 with a chat message that is actually the account upsell
# ("## An account is now required to use Vibe  *Sign in...").  Detect it at
# parse time and surface it as an auth failure so the router falls back and
# selfheal classifies the provider as needing credentials.
_AUTH_WALL_RE = re.compile(r'account is now required', re.IGNORECASE)

MISTRAL_BASE_URL = 'https://chat.mistral.ai'
MISTRAL_AUTH_URL = 'https://auth.mistral.ai'
MISTRAL_CHAT_URL = f'{MISTRAL_BASE_URL}/api/chat'
# Upstream default chat model id observed in the create-mode request.
MISTRAL_MODEL = 'mistral-large-2411'

# Mobile app user-agent — required by the tRPC/chat endpoints.
MISTRAL_APP_UA = (
    'le-chat-mobile/2.8.0 (build:20800191; os_name:android; '
    'device_category:smartphone; device_model:unknown; '
    'device_manufacturer:unknown)'
)

MISTRAL_CONTEXT_LENGTH = int(os.getenv('DSF_MISTRAL_CONTEXT_LENGTH', '131072'))
MISTRAL_MAX_OUTPUT = int(os.getenv('DSF_MISTRAL_MAX_OUTPUT', '8192'))

OTP_CODE_RE = re.compile(r'6200')


def _session_token() -> str:
    raw = (os.getenv('MISTRAL_SESSION_TOKEN', '') or '').strip()
    if raw:
        return raw
    jar = load_jar('mistral') or env_cookies('MISTRAL')
    return (jar.get('session_token') or '').strip()


def _anon_id() -> str:
    jar = load_jar('mistral') or env_cookies('MISTRAL')
    stable = (jar.get('stable_anon_id') or '').strip()
    if not stable:
        stable = str(uuid.uuid4())
        save_jar('mistral', {'stable_anon_id': stable})
    return stable


def _headers(auth: bool = False, accept: str = 'application/json') -> Dict[str, str]:
    headers = {
        'User-Agent': MISTRAL_APP_UA,
        'Accept': accept,
        'Content-Type': 'application/json',
        'Origin': MISTRAL_BASE_URL,
        'Referer': f'{MISTRAL_BASE_URL}/',
    }
    token = _session_token()
    if token:
        headers['Authorization'] = f'Bearer {token}'
    return headers


class MistralProvider(Provider):
    name = 'mistral'

    def __init__(self) -> None:
        self._bootstrapped = False

    # ------------------------------------------------------------- bootstrap
    def _bootstrap(self, no_proxy: bool = False) -> None:
        """Warm Kratos cookies once per process (non-fatal on failure)."""
        if self._bootstrapped:
            return
        try:
            http_get(f'{MISTRAL_AUTH_URL}/self-service/registration/api',
                     headers={'Accept': 'application/json',
                              'User-Agent': MISTRAL_APP_UA},
                     no_proxy=no_proxy)
        except Exception as e:  # noqa: BLE001 - warm-up must never fail requests
            logger.debug('mistral bootstrap failed (non-fatal): %s', e)
        self._bootstrapped = True

    # ---------------------------------------------------------------- provider
    def available(self, auth_key: Optional[str] = None) -> bool:
        # Anonymous access is account-gated upstream (returns an upsell
        # instead of model output) — a Le Chat session token is required.
        return bool(_session_token())

    def list_models(self, auth_key: Optional[str] = None) -> List[Dict[str, Any]]:
        """Le Chat mobile flow has no model discovery; the server picks the
        current default chat model — exposed here as a single route."""
        return [{
            'id': 'mistral-large',
            'upstream_model': MISTRAL_MODEL,
            'thinking_enabled': False,
            'search_enabled': False,
            'vision': False,
            'image_gen': False,
            'context_length': MISTRAL_CONTEXT_LENGTH,
            'max_output_tokens': MISTRAL_MAX_OUTPUT,
            'extra': {'title': 'Mistral Le Chat (web/mobile)'},
        }]

    def stream(self, prompt: str, *, model: str, thinking_enabled: bool = False,
               search_enabled: bool = False, temperature: Optional[float] = None,
               max_tokens: Optional[int] = None,
               images: Optional[List[Dict[str, Any]]] = None,
               image_generation: bool = False,
               no_proxy: bool = False,
               auth_key: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
        if images:
            raise ProviderError('mistral file attachments are not supported yet')
        self._bootstrap(no_proxy=no_proxy)
        anonymous = not _session_token()
        attempts = 2 if anonymous else 1
        last_error: Optional[ProviderError] = None
        for attempt in range(attempts):
            try:
                return self._stream_once(prompt, no_proxy=no_proxy)
            except ProviderRateLimitError as e:
                last_error = e
                if attempt + 1 < attempts:
                    # Fresh anonymous identity resets the 5-msg/day quota.
                    save_jar('mistral', {'stable_anon_id': str(uuid.uuid4())})
                    continue
                raise
        raise last_error or ProviderError('mistral stream failed')

    def _stream_once(self, prompt: str,
                     no_proxy: bool = False) -> Generator[Dict[str, Any], None, None]:
        now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.000Z')
        body = {
            # 'create' starts the conversation and streams the answer in one
            # call; no 'agentId' key (a null value → HTTP 400).
            'mode': 'create',
            'content': [{'type': 'text', 'text': prompt}],
            'files': [],
            'model': MISTRAL_MODEL,
            'stableAnonymousIdentifier': _anon_id(),
            'platform': 'mobile',
            'clientPromptData': {'currentDate': now},
            'supportedTaskCallbacks': [],
            'features': [],
            'libraries': [],
            'integrations': [],
            'disabledFeatures': ['memory-inference'],
        }
        response = http_post_stream(MISTRAL_CHAT_URL,
                                    headers=_headers(accept='text/event-stream'),
                                    json_body=body, no_proxy=no_proxy)
        if response.status_code not in (200, 201):
            try:
                error_text = response.text or ''
            except Exception:  # pragma: no cover
                error_text = f'HTTP {response.status_code}'
            if response.status_code == 429:
                raise ProviderRateLimitError(f'mistral rate limited: {error_text[:200]}')
            raise classify_http_error(response.status_code, error_text,
                                      response.headers)
        return self._iter_chunks(response)

    def _iter_chunks(self, response) -> Generator[Dict[str, Any], None, None]:
        """Parse the ``<type_num>:<json>`` line format with an incremental
        utf-8 decoder (multi-byte characters straddle chunk boundaries).

        The account upsell can arrive split across many tiny append deltas
        ("## ", "An ", "account ", …), so text is HELD until the first 300
        chars have been matched against the login-wall regex — holding (not
        just checking) is required, otherwise the deltas emitted before the
        phrase completes leak to the router and pollute the fallback answer.
        """
        decoder = codecs.getincrementaldecoder('utf-8')('replace')
        buffer = ''
        prefix = ''                       # accumulated text head
        held: List[Dict[str, Any]] = []   # pieces buffered during the window
        for chunk in response.iter_content(chunk_size=None):
            buffer += decoder.decode(chunk or b'')
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                for piece in self._parse_line(line.strip()):
                    if len(prefix) < 300:
                        head = piece.get('content') \
                            if piece.get('type') == 'text' else None
                        if head:
                            prefix += head
                            if _AUTH_WALL_RE.search(prefix):
                                raise ProviderAuthError(
                                    'mistral anonymous access disabled '
                                    f'(account upsell): {prefix[:120]}')
                            held.append(piece)
                            continue
                    while held:               # window closed: drain in order
                        yield held.pop(0)
                    yield piece
        buffer += decoder.decode(b'', final=True)
        if buffer.strip():
            for piece in self._parse_line(buffer.strip()):
                while held:
                    yield held.pop(0)
                yield piece
        while held:
            yield held.pop(0)
        yield {'content': '', 'type': 'text', 'finish_reason': 'stop'}

    def _parse_line(self, line: str) -> List[Dict[str, Any]]:
        if not line or ':' not in line:
            return []
        colon = line.index(':')
        try:
            line_type = int(line[:colon])
        except ValueError:
            return []
        json_str = line[colon + 1:]
        if not json_str or json_str == 'null':
            return []
        try:
            data = json.loads(json_str)
        except ValueError:
            return []
        j = data.get('json', data) if isinstance(data, dict) else data
        if not isinstance(j, dict):
            return []

        if line_type == 6:  # error frame
            code = j.get('internalCode', 0)
            retry = j.get('retryAfterSeconds', 0)
            if code == 6200 or retry:
                raise ProviderRateLimitError(
                    f"mistral anonymous quota reached: {j.get('message', '')}",
                    retry_after=float(retry) if retry else None)
            raise ProviderError(f"mistral stream error {code}: "
                                f"{j.get('message', str(j))[:300]}")
        if line_type != 15:  # only data frames carry text
            return []

        msg_type = j.get('type')
        if msg_type != 'message':
            return []
        out: List[Dict[str, Any]] = []
        for patch in j.get('patches', []):
            op = patch.get('op')
            path = patch.get('path', '')
            value = patch.get('value')
            if path == '/' or '/contentChunks' not in path:
                continue
            if op == 'replace' and isinstance(value, list):
                # Full snapshot of the chunk list — emit only the tail (the
                # web client replaces, we diff to avoid re-emitting text).
                texts = [c.get('text', '') for c in value
                         if isinstance(c, dict) and c.get('type') == 'text']
                if texts:
                    if _AUTH_WALL_RE.search(texts[-1][:300]):
                        raise ProviderAuthError(
                            'mistral anonymous access disabled (account '
                            f'upsell): {texts[-1][:120]}')
                    out.append({'content': texts[-1], 'type': 'text',
                                'finish_reason': None})
            elif op == 'append' and isinstance(value, str) and value:
                # Delta append — e.g. path ``/contentChunks/0/text``.
                if _AUTH_WALL_RE.search(value[:300]):
                    raise ProviderAuthError(
                        'mistral anonymous access disabled (account '
                        f'upsell): {value[:120]}')
                out.append({'content': value, 'type': 'text',
                            'finish_reason': None})
        return out
