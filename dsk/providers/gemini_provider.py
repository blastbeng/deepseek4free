"""Gemini provider via the gemini.google.com web chat (reverse-engineered).

Same approach as the DeepSeek reverse client: it talks to the web app's own
private endpoints with browser credentials (no official API keys, no AI Studio).

How it works
------------
1. Credentials: the ``__Secure-1PSID`` (and optionally ``__Secure-1PSIDTS``)
   cookies of a logged-in gemini.google.com session, provided via a
   bot-managed ``gemini_cookies.json`` file (kept fresh by ``dsk.refresher``)
   or env vars (``GEMINI_1PSID`` / ``GEMINI_1PSIDTS``).
2. Session init: GET ``https://gemini.google.com/app`` and scrape the SNlM0e
   XSRF token (``at``), frontend build label (``bl``), session id (``f.sid``).
3. Model discovery (dynamic): ``batchexecute`` RPC ``otAQ7b`` (user status)
   returns the list of models the account currently has access to, including
   the internal model ids and request header parameters needed to select one.
   Nothing is hardcoded — whatever the web app offers is exposed.
4. Generation: POST to
   ``https://gemini.google.com/_/BardChatUi/data/assistant.lamda.BardFrontendService/StreamGenerate``
   (Google's length-prefixed streaming frame format). Each frame carries the
   full candidate so far; we diff it into deltas. Thinking traces
   (candidate[37]) are emitted as ``thinking`` chunks. Chats run as temporary
   chats so user history is not polluted.
5. Vision: image attachments are uploaded via the single multipart POST to
   ``https://content-push.googleapis.com/upload`` (``X-Tenant-Id: bard-storage``
   header, ``Push-ID`` header) which returns a ``/contrib_service/...`` file
   path; the path is injected at index 3 of the message content array as
   ``[[path], filename]``. Generated images are read from the candidate's
   rich-content block (candidate[12], field 7).
"""

import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from .base import (
    Provider,
    ProviderAuthError,
    ProviderError,
    ProviderUnavailableError,
    classify_http_error,
    http_upload_multipart,
)

try:
    from curl_cffi import requests as cffi_requests
except ImportError:  # pragma: no cover - curl_cffi is in requirements.txt
    cffi_requests = None

logger = logging.getLogger('dsk.providers.gemini')

BASE_URL = 'https://gemini.google.com'
APP_URL = f'{BASE_URL}/app'
GENERATE_URL = (f'{BASE_URL}/_/BardChatUi/data/assistant.lamda.'
                'BardFrontendService/StreamGenerate')
BATCH_EXECUTE_URL = f'{BASE_URL}/_/BardChatUi/data/batchexecute'
UPLOAD_URL = 'https://content-push.googleapis.com/upload'
# Stable per-account push id; the web app sends it with every file upload.
DEFAULT_PUSH_ID = os.getenv('DSF_GEMINI_PUSH_ID', 'feeds/mcudyrk2a4khkz')

GET_USER_STATUS_RPC = 'otAQ7b'

# JSPB model-selection header (same format the web app itself sends).
MODEL_HEADER_KEY = 'x-goog-ext-525001261-jspb'

_UA = ('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
       '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
_BASE_HEADERS = {
    'Content-Type': 'application/x-www-form-urlencoded;charset=utf-8',
    'Origin': BASE_URL,
    'Referer': f'{BASE_URL}/',
    'User-Agent': _UA,
}

# Estimated capability metadata advertised on /v1/models for agent tooling
# (the web app does not expose per-model context windows).
GEMINI_CONTEXT_LENGTH = int(os.getenv('DSF_GEMINI_CONTEXT_LENGTH', '1048576'))
GEMINI_MAX_OUTPUT = int(os.getenv('DSF_GEMINI_MAX_OUTPUT', '65536'))

SESSION_TTL = 900.0  # re-scrape SNlM0e/build label every 15 minutes


def _slugify(name: str) -> str:
    out = ''.join(ch if ch.isalnum() else '-' for ch in name.lower()).strip('-')
    return '-'.join(p for p in out.split('-') if p)


def _compute_capacity(tier_flags: List[Any], capability_flags: List[Any]) -> tuple:
    """Account capacity tuple (capacity, capacity_field) — mirrors the web app."""
    if 115 in capability_flags:
        return 4, 12
    if 16 in tier_flags or 106 in capability_flags:
        return 3, 12
    if 8 in tier_flags or 19 in capability_flags:
        return 2, 12
    return 1, 12


class GeminiWebProvider(Provider):
    """Reverse-engineered gemini.google.com web client."""

    name = 'gemini'

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._session: Optional[Dict[str, Any]] = None   # at/bl/f.sid/language
        self._cookie_sig: str = ''                       # detects jar rotations
        self._models: Optional[List[Dict[str, Any]]] = None
        self._session_at: float = 0.0
        self._reqid = 10000

    # ------------------------------------------------------------ credentials
    def _cookies(self) -> Dict[str, str]:
        """Resolve Gemini web cookies: bot-managed jar first, then env vars."""
        cookies = self._cookies_file()
        psid = str(cookies.get('__Secure-1PSID', '') or '').strip()
        psidts = str(cookies.get('__Secure-1PSIDTS', '') or '').strip()
        if not psid:
            psid = (os.getenv('GEMINI_1PSID', '')
                    or os.getenv('GEMINI_COOKIES_1PSID', '')).strip()
            psidts = psidts or (os.getenv('GEMINI_1PSIDTS', '')
                                or os.getenv('GEMINI_COOKIES_1PSIDTS', '')).strip()
        if not psid:
            return {}
        out = {'__Secure-1PSID': psid}
        if psidts:
            out['__Secure-1PSIDTS'] = psidts
        return out

    @staticmethod
    def _cookies_file() -> Dict[str, Any]:
        for path in (Path(os.getenv('COOKIES_DIR', '')) / 'gemini_cookies.json'
                     if os.getenv('COOKIES_DIR') else None,
                     Path(__file__).resolve().parent.parent / 'gemini_cookies.json'):
            try:
                if path and path.is_file():
                    data = json.loads(path.read_text(encoding='utf-8'))
                    if isinstance(data, dict):
                        return data
            except (OSError, ValueError):
                continue
        return {}

    def available(self, auth_key: Optional[str] = None) -> bool:
        return bool(self._cookies())

    # --------------------------------------------------------------- session
    def _next_reqid(self) -> int:
        self._reqid += 1
        return self._reqid

    def _get_session(self, refresh: bool = False) -> Dict[str, Any]:
        """Scrape SNlM0e (at), build label (bl) and f.sid from the web app."""
        with self._lock:
            cookies = self._cookies()
            sig = repr(sorted(cookies.items()))
            if sig != self._cookie_sig:
                self._cookie_sig = sig        # credentials rotated by the
                self._session = None          # refresher -> force re-init
                self._session_at = 0.0
            if (not refresh and self._session is not None
                    and time.time() - self._session_at < SESSION_TTL):
                return self._session
            if not cookies:
                raise ProviderAuthError(
                    'No Gemini web cookies. Set GEMINI_1PSID (and optionally '
                    'GEMINI_1PSIDTS) from a logged-in gemini.google.com session, '
                    'or provide gemini_cookies.json.'
                )
            requester = cffi_requests or None
            try:
                if requester is not None:
                    response = requester.get(APP_URL, headers={'User-Agent': _UA},
                                             cookies=cookies, impersonate='chrome120',
                                             timeout=60)
                else:
                    import requests as std_requests
                    response = std_requests.get(APP_URL, headers={'User-Agent': _UA},
                                                cookies=cookies, timeout=60)
            except Exception as e:  # network failure
                raise ProviderUnavailableError(f'Gemini web unreachable: {e}') from e
            if response.status_code != 200:
                raise classify_http_error(response.status_code, response.text[:300],
                                          response.headers)
            text = response.text

            def _find(pattern: str) -> Optional[str]:
                import re
                match = re.search(pattern, text)
                return match.group(1) if match else None

            session = {
                'at': _find(r'"SNlM0e":\s*"(.*?)"') or '',
                'bl': _find(r'"cfb2h":\s*"(.*?)"'),
                'f_sid': _find(r'"FdrFJe":\s*"(.*?)"'),
                'language': _find(r'"TuX5cc":\s*"(.*?)"') or 'en',
            }
            if not session['at']:
                raise ProviderAuthError(
                    'Gemini web session rejected (no SNlM0e token). The '
                    '__Secure-1PSID cookies are invalid or expired — re-copy '
                    'them from the browser.'
                )
            self._session = session
            self._session_at = time.time()
            logger.info('Gemini web session initialized (bl=%s)', session['bl'])
            return self._session

    # ------------------------------------------------------------ rpc plumbing
    def _batch_execute(self, rpcid: str, payload: str = '[]') -> Dict[str, Any]:
        """POST a batchexecute RPC; returns the matching part (rpcid-tagged)."""
        session = self._get_session()
        params: Dict[str, Any] = {
            'rpcids': rpcid,
            'hl': session['language'],
            '_reqid': self._next_reqid(),
            'rt': 'c',
            'source-path': '/',
        }
        if session['bl']:
            params['bl'] = session['bl']
        if session['f_sid']:
            params['f.sid'] = session['f_sid']
        headers = {
            **_BASE_HEADERS,
            MODEL_HEADER_KEY: ('[1,null,null,null,null,null,null,null,[4,5,6,8],'
                               'null,null,null,null,null,null,null]'),
            'x-goog-ext-73010989-jspb': '[0]',
            'X-Same-Domain': '1',
        }
        data = {
            'at': session['at'],
            'f.req': json.dumps([[[rpcid, payload, None, 'generic']]]),
        }
        try:
            if cffi_requests is not None:
                response = cffi_requests.post(
                    BATCH_EXECUTE_URL, params=params, headers=headers, data=data,
                    cookies=self._cookies(), impersonate='chrome120', timeout=120)
            else:
                import requests as std_requests
                response = std_requests.post(BATCH_EXECUTE_URL, params=params,
                                             headers=headers, data=data,
                                             cookies=self._cookies(), timeout=120)
        except Exception as e:
            raise ProviderUnavailableError(f'Gemini batchexecute failed: {e}') from e
        if response.status_code != 200:
            raise classify_http_error(response.status_code, response.text[:300],
                                      response.headers)
        for part in parse_google_frames(response.text):
            if isinstance(part, list) and len(part) > 2 and part[1] == rpcid:
                reject = part[5][0] if isinstance(part[5], list) and part[5] else None
                if reject == 7:
                    raise ProviderAuthError('Gemini web: permission denied '
                                            '(cookies invalid or expired).')
                body = part[2]
                if isinstance(body, str):
                    try:
                        return json.loads(body)
                    except ValueError:
                        continue
        raise ProviderError(f'Gemini web: no response part for RPC {rpcid}')

    # --------------------------------------------------------- model discovery
    def list_models(self, auth_key: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch the models the account currently has access to (dynamic)."""
        with self._lock:
            if self._models is not None:
                return self._models
        body = self._batch_execute(GET_USER_STATUS_RPC)
        models = self._parse_model_list(body)
        with self._lock:
            self._models = models
        logger.info('Gemini web: discovered %d models', len(models))
        return models

    @staticmethod
    def _parse_model_list(body: Any) -> List[Dict[str, Any]]:
        def nested(container: Any, path: List[int], default: Any = None) -> Any:
            current = container
            for index in path:
                if not isinstance(current, list) or index >= len(current):
                    return default
                current = current[index]
            return default if current in (None, [], {}) else current

        models_list = nested(body, [15])
        if not isinstance(models_list, list):
            raise ProviderAuthError('Gemini web: model discovery failed '
                                    '(account unavailable or cookies expired).')
        tier_flags = nested(body, [16], []) or []
        capability_flags = nested(body, [17], []) or []
        capacity, capacity_field = _compute_capacity(
            tier_flags if isinstance(tier_flags, list) else [],
            capability_flags if isinstance(capability_flags, list) else [],
        )
        capacity_tail = (f'null,{capacity}' if capacity_field == 13 else str(capacity))

        out: List[Dict[str, Any]] = []
        seen: Dict[str, int] = {}
        for model_data in models_list:
            if not isinstance(model_data, list):
                continue
            internal_id = model_data[0] if model_data and isinstance(model_data[0], str) else ''
            if not internal_id:
                continue
            category = str(nested(model_data, [1], '') or nested(model_data, [10], '') or '')
            display = str(nested(model_data, [11], '') or nested(model_data, [19], '')
                          or category or internal_id)
            raw_number = nested(model_data, [17])
            if not isinstance(raw_number, int):
                raw_number = nested(model_data, [9])
            model_number = raw_number if isinstance(raw_number, int) else 1

            public_id = f'gemini-{_slugify(category)}' if category else f'gemini-{_slugify(display)}'
            count = seen.get(public_id, 0)
            seen[public_id] = count + 1
            if count:
                public_id = f'{public_id}-{count + 1}'

            thinking = 'think' in (category + display).lower()
            out.append({
                'id': public_id,
                'upstream_model': internal_id,
                'thinking_enabled': thinking,
                'search_enabled': False,
                'vision': True,
                'image_gen': True,
                'context_length': GEMINI_CONTEXT_LENGTH,
                'max_output_tokens': GEMINI_MAX_OUTPUT,
                'extra': {
                    'model_number': model_number,
                    'capacity_tail': capacity_tail,
                    'display_name': display,
                },
            })
        return out

    # ---------------------------------------------------------------- serving
    def _upload_image(self, session: Dict[str, Any], mime: str, data: bytes,
                      index: int, no_proxy: bool = False) -> str:
        """Upload one image to Google's push endpoint; return the file path.

        The endpoint is a single multipart POST (no resumable dance) that
        answers with the stored file path, e.g. ``/contrib_service/ttl_1d/...``.
        """
        ext = _image_ext(mime)
        filename = f'image_{index}.{ext}'
        headers = {k: v for k, v in _BASE_HEADERS.items() if k != 'Content-Type'}
        headers['X-Tenant-Id'] = 'bard-storage'
        headers['Push-ID'] = DEFAULT_PUSH_ID
        try:
            response = http_upload_multipart(
                UPLOAD_URL, filename=filename, content_type=mime, data=data,
                headers=headers, proxies=None, no_proxy=no_proxy)
        except Exception as e:
            raise ProviderUnavailableError(f'Gemini image upload failed: {e}') from e
        if response.status_code != 200:
            raise ProviderUnavailableError(
                f'Gemini image upload failed (HTTP {response.status_code}): '
                f'{str(getattr(response, "text", ""))[:200]}')
        file_url = (response.text or '').strip()
        if not file_url:
            raise ProviderError('Gemini image upload returned an empty file path')
        return file_url

    def stream(self, prompt: str, *, model: str, thinking_enabled: bool = False,
               search_enabled: bool = False, temperature: Optional[float] = None,
               max_tokens: Optional[int] = None,
               images: Optional[List[Dict[str, Any]]] = None,
               image_generation: bool = False,
               no_proxy: bool = False,
               auth_key: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
        session = self._get_session()
        internal_id, model_number, capacity_tail = '', 1, '1'
        for entry in self.list_models():
            if entry['upstream_model'] == model or entry['id'] == model:
                internal_id = entry['upstream_model']
                model_number = entry['extra'].get('model_number', 1)
                capacity_tail = entry['extra'].get('capacity_tail', '1')
                thinking_enabled = thinking_enabled or entry['thinking_enabled']
                break

        message_content = [prompt, 0, None, None, None, None, 0]
        if images:
            file_data = []
            for i, image in enumerate(images):
                file_url = self._upload_image(session, image.get('mime', 'image/png'),
                                              image.get('data', b''), i,
                                              no_proxy=no_proxy)
                file_data.append([[file_url], f'image_{i}.{_image_ext(image.get("mime", "image/png"))}'])
            message_content[3] = file_data
        # Default chat metadata: new conversation (temporary chat, not saved).
        inner: List[Any] = [None] * 81
        inner[0] = message_content
        inner[1] = [session['language']]
        inner[2] = ['', '', '', None, None, None, None, None, None, '']
        inner[6] = [1]
        inner[7] = 1        # streaming flag
        inner[10] = 1
        inner[11] = 0
        inner[17] = [[0]]
        inner[18] = 0
        inner[27] = 1
        inner[30] = [4]
        inner[41] = [1]
        inner[45] = 1       # temporary chat — do not save to history
        inner[53] = 0
        inner[59] = str(uuid.uuid4()).upper()
        inner[61] = []
        inner[68] = 1
        inner[79] = model_number
        inner[80] = 2 if thinking_enabled else 1

        request_uuid = str(uuid.uuid4()).upper()
        headers = {
            **_BASE_HEADERS,
            'X-Same-Domain': '1',
            'x-goog-ext-525005358-jspb': f'["{request_uuid}",1]',
        }
        if internal_id:
            headers[MODEL_HEADER_KEY] = (
                f'[1,null,null,null,"{internal_id}",null,null,0,[4,5,6,8],null,null,'
                f'{capacity_tail}, null,null,{model_number}]'
            )
            headers['x-goog-ext-73010989-jspb'] = '[0]'
            headers['x-goog-ext-73010990-jspb'] = '[0,0,0]'

        params: Dict[str, Any] = {
            'hl': session['language'],
            '_reqid': self._next_reqid(),
            'rt': 'c',
        }
        if session['bl']:
            params['bl'] = session['bl']
        if session['f_sid']:
            params['f.sid'] = session['f_sid']

        data = {
            'at': session['at'],
            'f.req': json.dumps([None, json.dumps(inner)]),
        }
        try:
            if cffi_requests is not None:
                response = cffi_requests.post(
                    GENERATE_URL, params=params, headers=headers, data=data,
                    cookies=self._cookies(), impersonate='chrome120', timeout=600)
            else:
                import requests as std_requests
                response = std_requests.post(GENERATE_URL, params=params,
                                             headers=headers, data=data,
                                             cookies=self._cookies(), timeout=600)
        except Exception as e:
            raise ProviderUnavailableError(f'Gemini stream failed: {e}') from e
        if response.status_code != 200:
            raise classify_http_error(response.status_code, response.text[:300],
                                      response.headers)
        return self._iter_chunks(response)

    def _iter_chunks(self, response) -> Generator[Dict[str, Any], None, None]:
        """Diff Gemini's cumulative frames into unified text/thinking chunks."""
        prev_text = ''
        prev_thought = ''
        seen_images: set = set()
        emitted = False
        try:
            for line in response.iter_lines():
                for part in _parse_frame_line(line):
                    error_code = _nested(part, [5, 2, 0, 1, 0])
                    if error_code:
                        raise ProviderError(f'Gemini web error code {error_code}')
                    inner_str = _nested(part, [2])
                    if not isinstance(inner_str, str):
                        continue
                    try:
                        inner = json.loads(inner_str)
                    except ValueError:
                        continue
                    candidates = _nested(inner, [4], [])
                    for candidate in candidates if isinstance(candidates, list) else []:
                        text = _nested(candidate, [1, 0], '') or ''
                        thought = _nested(candidate, [37, 0, 0], '') or ''
                        delta = _diff(prev_text, text)
                        if delta:
                            prev_text = text
                            emitted = True
                            yield {'content': delta, 'type': 'text', 'finish_reason': None}
                        t_delta = _diff(prev_thought, thought)
                        if t_delta:
                            prev_thought = thought
                            emitted = True
                            yield {'content': t_delta, 'type': 'thinking',
                                   'finish_reason': None}
                        for url in _extract_generated_images(candidate):
                            if url in seen_images:
                                continue
                            seen_images.add(url)
                            emitted = True
                            yield {'content': f'![image]({url})', 'type': 'image',
                                   'url': url, 'finish_reason': None}
        except ProviderError:
            raise
        except Exception as e:
            if not emitted:
                raise ProviderUnavailableError(f'Gemini stream interrupted: {e}') from e
            logger.warning('Gemini stream interrupted after output: %s', e)
        if emitted:
            yield {'content': '', 'type': 'text', 'finish_reason': 'stop'}
        else:
            raise ProviderError('Gemini web returned no content (empty response).')


def _nested(container: Any, path: List[int], default: Any = None) -> Any:
    current = container
    for index in path:
        if not isinstance(current, list) or index >= len(current):
            return default
        current = current[index]
    return default if current in (None, [], {}) else current


def _image_ext(mime: str) -> str:
    """File extension for an image mime type."""
    return {'image/jpeg': 'jpg', 'image/png': 'png', 'image/gif': 'gif',
            'image/webp': 'webp'}.get(mime, (mime.split('/')[-1] or 'png')[:5])


def _jspb_field(container: Any, field_number: int) -> Any:
    """Read a JSPB field: positional slot, or sparse trailing dict keyed by
    ``field_number + 1`` (high-numbered fields are collected that way).
    A sparse wrapper dict found in the positional slot itself is unwrapped."""
    key = str(field_number + 1)
    if isinstance(container, list):
        if field_number < len(container):
            value = container[field_number]
            if value is not None and not isinstance(value, dict):
                return value
            if isinstance(value, dict) and key in value:
                return value[key]
        for entry in container:
            if isinstance(entry, dict) and key in entry:
                return entry[key]
    elif isinstance(container, dict):
        return container.get(key)
    return None


def _extract_generated_images(candidate: Any) -> List[str]:
    """URLs of images Gemini generated, from the candidate's rich-content
    block (candidate[12], field 7): each entry exposes its URL at [0, 3, 3].
    """
    rich = _nested(candidate, [12])
    if not rich:
        return []
    block = _jspb_field(rich, 7)  # Field.GENERATED_IMAGES
    entries = _nested(block, [0], []) or []
    urls: List[str] = []
    for entry in entries if isinstance(entries, list) else []:
        url = _nested(entry, [0, 3, 3])
        if isinstance(url, str) and url.startswith('http'):
            urls.append(url)
    return urls


def _diff(previous: str, current: str) -> str:
    """Delta between the cumulative text so far and the new full text."""
    if not current or current == previous:
        return ''
    if current.startswith(previous):
        return current[len(previous):]
    if previous.startswith(current):
        return ''  # candidate shrank/switched; keep what was already sent
    # Non-cumulative (rare): send only what is new after the common prefix.
    common = 0
    for a, b in zip(previous, current):
        if a != b:
            break
        common += 1
    return current[common:]


def _parse_frame_line(line: Any) -> List[Any]:
    """Parse one (or more) Google streaming frames from a text/bytes line."""
    if isinstance(line, bytes):
        line = line.decode('utf-8', 'ignore')
    line = line.strip()
    if not line or line.startswith(")]}'") or line.isdigit():
        return []
    try:
        parsed = json.loads(line)
    except ValueError:
        return []
    if isinstance(parsed, list):
        return parsed
    return [parsed]


def parse_google_frames(text: str) -> List[Any]:
    """Parse a full (non-streaming) batchexecute response into parts."""
    parts: List[Any] = []
    for line in text.splitlines():
        parts.extend(_parse_frame_line(line))
    return parts
