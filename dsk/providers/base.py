"""Shared provider abstractions: errors, routes and HTTP/SSE helpers.

All providers stream the same chunk format as the DeepSeek client:
    {'content': str, 'type': 'text' | 'thinking', 'finish_reason': None | 'stop'}

Error taxonomy (drives the retry/fallback engine in router.py):
    ProviderAuthError        credentials missing/rejected — no retry, fallback
    ProviderRateLimitError   HTTP 429 — retry with backoff, then fallback
    ProviderUnavailableError network/5xx/Cloudflare challenge — retry, then fallback
    ProviderError            anything else (bad model name, upstream 4xx) — fallback
"""

import json
import base64
import binascii
import logging
import os
import struct
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple

try:
    from curl_cffi import requests as cffi_requests
except ImportError:  # pragma: no cover - curl_cffi is in requirements.txt
    cffi_requests = None

import requests as std_requests

try:  # optional outbound proxy rotation (dsk/proxies.py)
    from dsk import proxies as _proxies
except Exception:
    try:
        from .. import proxies as _proxies
    except Exception:
        _proxies = None


def proxy_kwargs_for(url: str, no_proxy: bool = False) -> Dict[str, Any]:
    """Proxy kwargs for the host in ``url`` ({} when proxies are disabled
    or ``no_proxy`` forces a direct connection)."""
    if no_proxy or _proxies is None:
        return {}
    try:
        return _proxies.proxies_kwargs(url=url)
    except Exception:
        return {}


logger = logging.getLogger('dsk.providers.base')

# Connect-phase cap for pooled proxies: free proxies die constantly and the
# OS-level connect can otherwise hang for minutes before the total timeout.
HTTP_CONNECT_TIMEOUT = int(os.getenv('DSF_HTTP_CONNECT_TIMEOUT', '15'))


def _looks_like_network_error(exc: BaseException) -> bool:
    """True for transport-level failures (connect/timeout/proxy/ssl)."""
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    if any(k in name for k in ('requestexception', 'connection', 'timeout',
                               'proxyerror', 'sslerror', 'chunkedencoding')):
        return True
    return any(k in text for k in (
        "couldn't connect", 'failed to connect', 'timed out',
        'connection reset', 'connection refused', 'connection aborted',
        'getaddrinfo failed', 'temporary failure in name resolution'))


def _resilient_request(url: str, extra: Dict[str, Any], do_request,
                       pooled: bool = True):
    """Run ``do_request(extra)``, surviving dead pooled proxies.

    When a pool-assigned proxy fails at the transport level it is put on
    cooldown (``proxies.mark_failure``) and the request is retried once
    DIRECT.  Persistent transport failures are re-raised as
    ProviderUnavailableError so the router can fall back instead of leaking
    an unhandled 500 (curl_cffi RequestException escaped as 500 before).
    """
    proxy = None
    px = extra.get('proxies') or {}
    if px:
        proxy = next(iter(px.values()), None)
    try:
        return do_request(extra)
    except Exception as exc:  # noqa: BLE001 — classification below
        if not _looks_like_network_error(exc):
            raise
        if proxy and pooled:
            try:
                _proxies.mark_failure(proxy)
            except Exception:  # noqa: BLE001 — cooldown is best-effort
                pass
            logger.warning('pool proxy %s failed (%s); retrying direct',
                           proxy, str(exc)[:100])
            try:
                return do_request({})
            except Exception as exc2:  # noqa: BLE001
                if not _looks_like_network_error(exc2):
                    raise
                raise ProviderUnavailableError(
                    f'upstream unreachable via proxy and direct: {exc2}') from exc2
        raise ProviderUnavailableError(f'request failed: {exc}') from exc


class ProviderError(Exception):
    """Base class for provider failures."""


class ProviderAuthError(ProviderError):
    """Credentials missing or rejected. Not retryable; triggers fallback."""


class ProviderRateLimitError(ProviderError):
    """Rate limited. Retryable with backoff, then fallback."""

    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class ProviderUnavailableError(ProviderError):
    """Network or server-side failure. Retryable, then fallback."""


class Provider:
    """Base class every provider must implement.

    ``stream`` returns a generator of unified chunks (same shape as the
    DeepSeek client) so the OpenAI-compatible server can consume all
    providers through a single code path:

        {'content': str, 'type': 'text' | 'thinking' | 'image', 'finish_reason': None | 'stop'}

    ``type: 'image'`` chunks additionally carry ``url`` (and optionally
    ``b64``) for vision/generated-image output; consumers render them as
    markdown images.

    Stream-level failures should raise the typed errors above so the router
    can retry or fall back to another provider/model.
    """

    name: str = ''

    def available(self, auth_key: Optional[str] = None) -> bool:
        """Whether this provider has usable credentials configured."""
        raise NotImplementedError

    def list_models(self, auth_key: Optional[str] = None) -> List[Dict[str, Any]]:
        """Discover the models the web app currently offers.

        Must be dynamic (fetched from the provider's web session) — no
        hardcoded model lists. Returns dicts with keys:
            id, upstream_model, thinking_enabled, search_enabled,
            context_length, max_output_tokens, extra
        Raises a ProviderError subclass when credentials are missing/invalid.
        """
        raise NotImplementedError

    def stream(self, prompt: str, *, model: str, thinking_enabled: bool = False,
               search_enabled: bool = False, temperature: Optional[float] = None,
               max_tokens: Optional[int] = None,
               images: Optional[List[Dict[str, Any]]] = None,
               image_generation: bool = False,
               auth_key: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
        """Yield unified chunks for a single flat prompt.

        ``images`` is a list of ``{'mime': str, 'data': bytes}`` attachments
        for vision-capable models. ``image_generation`` asks an image-gen
        capable model to produce an image for ``prompt``.
        """
        raise NotImplementedError


@dataclass
class Route:
    """An exposed model id mapped to one provider/model combination."""

    model_id: str            # id exposed on /v1/models
    provider_name: str       # 'deepseek' | 'gemini' | 'chatgpt'
    upstream_model: str      # model name/identifier sent to the provider
    thinking_enabled: bool = False
    search_enabled: bool = False
    vision: bool = False           # accepts image attachments in chat
    image_gen: bool = False        # can generate images (/v1/images/generations)
    context_length: int = 131072
    max_output_tokens: int = 32768
    fallbacks: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)  # provider-specific data


def http_post_stream(url: str, headers: Optional[Dict[str, str]] = None,
                     json_body: Optional[Dict[str, Any]] = None,
                     timeout: int = 600, proxies: Optional[Dict[str, str]] = None,
                     no_proxy: bool = False):
    """POST and return a streaming response.

    Pooled-proxy connection failures are retried once DIRECT (the dead
    proxy goes on cooldown); persistent transport errors surface as
    ProviderUnavailableError so the router falls back cleanly.
    """
    extra = ({'proxies': proxies} if proxies
             else proxy_kwargs_for(url, no_proxy=no_proxy))

    def _do(kwargs: Dict[str, Any]):
        px = bool(kwargs.get('proxies'))
        eff = (HTTP_CONNECT_TIMEOUT, timeout) if px else timeout
        if cffi_requests is not None:
            return cffi_requests.post(
                url, headers=headers or {}, json=json_body,
                stream=True, impersonate='chrome120', timeout=eff,
                **kwargs,
            )
        return std_requests.post(
            url, headers=headers or {}, json=json_body, stream=True,
            timeout=eff, **kwargs,
        )

    return _resilient_request(url, extra, _do, pooled=proxies is None)


def http_get(url: str, headers: Optional[Dict[str, str]] = None,
             cookies: Optional[Dict[str, str]] = None, timeout: int = 60,
             proxies: Optional[Dict[str, str]] = None, no_proxy: bool = False):
    """GET a URL and return the raw response.

    Uses curl_cffi with a Chrome TLS fingerprint when available — required for
    Cloudflare-protected hosts (chatgpt.com, gemini.google.com). Callers check
    ``status_code`` and parse the body themselves (JSON or HTML scrape).
    """
    extra = ({'proxies': proxies} if proxies
             else proxy_kwargs_for(url, no_proxy=no_proxy))

    def _do(kwargs: Dict[str, Any]):
        px = bool(kwargs.get('proxies'))
        eff = (HTTP_CONNECT_TIMEOUT, timeout) if px else timeout
        if cffi_requests is not None:
            return cffi_requests.get(
                url, headers=headers or {}, cookies=cookies or None,
                impersonate='chrome120', timeout=eff,
                **kwargs,
            )
        return std_requests.get(
            url, headers=headers or {}, cookies=cookies or None,
            timeout=eff, **kwargs,
        )

    return _resilient_request(url, extra, _do, pooled=proxies is None)


def parse_sse_data(line: bytes) -> Optional[Dict[str, Any]]:
    """Parse one SSE ``data: {...}`` line.

    Returns None for comments, empty lines, ``data: [DONE]`` and unparsable
    payloads so callers can simply skip them.
    """
    try:
        text = line.decode('utf-8', 'ignore').strip()
    except AttributeError:
        text = str(line or '').strip()
    if not text.startswith('data:'):
        return None
    payload = text[5:].strip()
    if not payload or payload == '[DONE]':
        return None
    try:
        obj = json.loads(payload)
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


def classify_http_error(status: int, text: str, headers: Optional[Any] = None) -> ProviderError:
    """Map an upstream HTTP failure to the provider error taxonomy."""
    lowered = (text or '').lower()
    if status == 429:
        retry_after = None
        if headers is not None:
            try:
                ra = headers.get('retry-after')
                if ra:
                    retry_after = float(ra)
            except (ValueError, TypeError, AttributeError):
                retry_after = None
        return ProviderRateLimitError(f'Rate limited (HTTP 429): {text[:300]}', retry_after=retry_after)
    if status in (401, 403):
        if 'just a moment' in lowered or 'cloudflare' in lowered:
            return ProviderUnavailableError(f'Cloudflare challenge (HTTP {status}): {text[:300]}')
        return ProviderAuthError(f'Authentication failed (HTTP {status}): {text[:300]}')
    if status == 404:
        return ProviderError(f'Model or endpoint not found (HTTP 404): {text[:300]}')
    if status >= 500:
        return ProviderUnavailableError(f'Server error (HTTP {status}): {text[:300]}')
    if status == 400 and ('api key' in lowered or 'api_key' in lowered):
        return ProviderAuthError(f'Invalid API key (HTTP 400): {text[:300]}')
    return ProviderError(f'Request failed (HTTP {status}): {text[:300]}')


def gemini_retry_delay(text: str) -> Optional[float]:
    """Extract a Retry-After value from a Gemini error body.

    Gemini reports e.g. {'error': {'details': [{'retryDelay': '7s'}]}}.
    """
    try:
        data = json.loads(text)
        details = (data.get('error') or {}).get('details') or []
        for entry in details:
            if isinstance(entry, dict) and 'retryDelay' in entry:
                raw = str(entry['retryDelay']).strip().rstrip('s')
                return float(raw)
    except (ValueError, TypeError, AttributeError):
        pass
    return None


# ----------------------------------------------------------------- uploads
def http_post_raw(url: str, data: bytes, headers: Optional[Dict[str, str]] = None,
                  timeout: int = 300, proxies: Optional[Dict[str, str]] = None,
                  no_proxy: bool = False):
    """POST a raw body and return the response (Chrome fingerprint)."""
    extra = ({'proxies': proxies} if proxies
             else proxy_kwargs_for(url, no_proxy=no_proxy))

    def _do(kwargs: Dict[str, Any]):
        px = bool(kwargs.get('proxies'))
        eff = (HTTP_CONNECT_TIMEOUT, timeout) if px else timeout
        if cffi_requests is not None:
            return cffi_requests.post(url, headers=headers or {}, data=data,
                                      impersonate='chrome120', timeout=eff,
                                      **kwargs)
        return std_requests.post(url, headers=headers or {}, data=data,
                                 timeout=eff, **kwargs)

    return _resilient_request(url, extra, _do, pooled=proxies is None)


def http_put_raw(url: str, data: bytes, headers: Optional[Dict[str, str]] = None,
                 timeout: int = 300, proxies: Optional[Dict[str, str]] = None,
                 no_proxy: bool = False):
    """PUT a raw body and return the response (used for blob uploads)."""
    extra = ({'proxies': proxies} if proxies
             else proxy_kwargs_for(url, no_proxy=no_proxy))

    def _do(kwargs: Dict[str, Any]):
        px = bool(kwargs.get('proxies'))
        eff = (HTTP_CONNECT_TIMEOUT, timeout) if px else timeout
        if cffi_requests is not None:
            return cffi_requests.put(url, headers=headers or {}, data=data,
                                     impersonate='chrome120', timeout=eff,
                                     **kwargs)
        return std_requests.put(url, headers=headers or {}, data=data,
                                timeout=eff, **kwargs)

    return _resilient_request(url, extra, _do, pooled=proxies is None)


def http_upload_multipart(url: str, *, filename: str, content_type: str,
                          data: bytes, headers: Optional[Dict[str, str]] = None,
                          field: str = 'file', timeout: int = 300,
                          proxies: Optional[Dict[str, str]] = None,
                          no_proxy: bool = False):
    """POST a single-part multipart/form-data body (hand-encoded so it works
    on every curl_cffi version) and return the response."""
    boundary = '----dsk' + uuid.uuid4().hex
    body = (
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
        f'Content-Type: {content_type}\r\n\r\n'
    ).encode() + data + f'\r\n--{boundary}--\r\n'.encode()
    hdrs = dict(headers or {})
    hdrs['Content-Type'] = f'multipart/form-data; boundary={boundary}'
    return http_post_raw(url, body, headers=hdrs, timeout=timeout,
                         proxies=proxies, no_proxy=no_proxy)


# ------------------------------------------------------------------ images
def parse_data_uri(uri: str) -> Tuple[str, bytes]:
    """Decode a ``data:<mime>;base64,<payload>`` URI into (mime, bytes)."""
    header, _, payload = uri.partition(',')
    if not _:
        raise ProviderError('image_url data URI is malformed (missing comma)')
    mime = 'image/png'
    if header.startswith('data:'):
        mime = header[5:].split(';', 1)[0].strip() or 'image/png'
    try:
        return mime, base64.b64decode(payload)
    except (binascii.Error, ValueError) as e:
        raise ProviderError(f'image_url data URI is not valid base64: {e}') from e


def fetch_image_bytes(url: str, *, max_bytes: int = 20 * 1024 * 1024,
                      timeout: int = 60) -> Tuple[str, bytes]:
    """Download an image over HTTP(S) and return (mime, bytes).

    The mime comes from the Content-Type header, falling back to a sniff of
    the magic bytes. Proxies rotate through the shared pool automatically.
    """
    try:
        response = http_get(url, timeout=timeout)
    except Exception as e:
        raise ProviderUnavailableError(f'failed to download image {url}: {e}') from e
    if response.status_code != 200:
        raise ProviderUnavailableError(
            f'failed to download image {url} (HTTP {response.status_code})')
    data = response.content or b''
    if len(data) > max_bytes:
        raise ProviderError(f'image at {url} exceeds {max_bytes // (1024 * 1024)} MB limit')
    if not data:
        raise ProviderError(f'image at {url} is empty')
    mime = (response.headers.get('Content-Type') or '').split(';', 1)[0].strip()
    if not mime.startswith('image/'):
        mime = sniff_image_mime(data)
    return mime or 'image/png', data


def sniff_image_mime(data: bytes) -> str:
    """Identify an image's mime type from its magic bytes ('' if unknown)."""
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        return 'image/png'
    if data[:3] == b'\xff\xd8\xff':
        return 'image/jpeg'
    if data[:6] in (b'GIF87a', b'GIF89a'):
        return 'image/gif'
    if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return 'image/webp'
    return ''


def image_dimensions(data: bytes) -> Tuple[int, int]:
    """Best-effort (width, height) from image headers; (1024, 1024) fallback.

    Pure-python header parsing — PNG IHDR, JPEG SOF markers, GIF logical
    screen descriptor and WebP VP8/VP8L/VP8X chunks.
    """
    try:
        if data[:8] == b'\x89PNG\r\n\x1a\n' and data[12:16] == b'IHDR':
            width, height = struct.unpack('>II', data[16:24])
            return int(width), int(height)
        if data[:6] in (b'GIF87a', b'GIF89a'):
            width, height = struct.unpack('<HH', data[6:10])
            return int(width), int(height)
        if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
            chunk = data[12:16]
            if chunk == b'VP8 ':
                width, height = struct.unpack('<HH', data[26:30])
                return width & 0x3FFF or 1024, height & 0x3FFF or 1024
            if chunk == b'VP8L':
                bits = struct.unpack('<I', data[21:25])[0]
                width = (bits & 0x3FFF) + 1
                height = ((bits >> 14) & 0x3FFF) + 1
                return width, height
            if chunk == b'VP8X':
                width = 1 + int.from_bytes(data[24:27], 'little')
                height = 1 + int.from_bytes(data[27:30], 'little')
                return width, height
        if data[:3] == b'\xff\xd8\xff':
            # JPEG: scan the marker segments for an SOF frame header.
            offset = 2
            while offset + 9 < len(data):
                if data[offset] != 0xFF:
                    offset += 1
                    continue
                marker = data[offset + 1]
                if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                    offset += 2
                    continue
                seg_len = struct.unpack('>H', data[offset + 2:offset + 4])[0]
                if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                    height, width = struct.unpack('>HH', data[offset + 5:offset + 9])
                    return int(width), int(height)
                offset += 2 + seg_len
    except Exception:  # pragma: no cover - defensive
        pass
    return 1024, 1024
