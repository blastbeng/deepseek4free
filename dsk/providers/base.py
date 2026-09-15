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
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional

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


def proxy_kwargs_for(url: str) -> Dict[str, Any]:
    """Proxy kwargs for the host in ``url`` ({} when proxies are disabled)."""
    if _proxies is None:
        return {}
    try:
        return _proxies.proxies_kwargs(url=url)
    except Exception:
        return {}


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

        {'content': str, 'type': 'text' | 'thinking', 'finish_reason': None | 'stop'}

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
               auth_key: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
        """Yield unified chunks for a single flat prompt."""
        raise NotImplementedError


@dataclass
class Route:
    """An exposed model id mapped to one provider/model combination."""

    model_id: str            # id exposed on /v1/models
    provider_name: str       # 'deepseek' | 'gemini' | 'chatgpt'
    upstream_model: str      # model name/identifier sent to the provider
    thinking_enabled: bool = False
    search_enabled: bool = False
    context_length: int = 131072
    max_output_tokens: int = 32768
    fallbacks: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)  # provider-specific data


def http_post_stream(url: str, headers: Optional[Dict[str, str]] = None,
                     json_body: Optional[Dict[str, Any]] = None,
                     timeout: int = 600, proxies: Optional[Dict[str, str]] = None):
    """POST and return a streaming response.

    Uses curl_cffi with a Chrome TLS fingerprint when available — required for
    Cloudflare-protected hosts (chat.deepseek.com, chatgpt.com).
    """
    extra = {'proxies': proxies} if proxies else proxy_kwargs_for(url)
    if cffi_requests is not None:
        return cffi_requests.post(
            url, headers=headers or {}, json=json_body,
            stream=True, impersonate='chrome120', timeout=timeout,
            **extra,
        )
    return std_requests.post(
        url, headers=headers or {}, json=json_body, stream=True, timeout=timeout,
        **extra,
    )


def http_get(url: str, headers: Optional[Dict[str, str]] = None,
             cookies: Optional[Dict[str, str]] = None, timeout: int = 60,
             proxies: Optional[Dict[str, str]] = None):
    """GET a URL and return the raw response.

    Uses curl_cffi with a Chrome TLS fingerprint when available — required for
    Cloudflare-protected hosts (chatgpt.com, gemini.google.com). Callers check
    ``status_code`` and parse the body themselves (JSON or HTML scrape).
    """
    extra = {'proxies': proxies} if proxies else proxy_kwargs_for(url)
    if cffi_requests is not None:
        return cffi_requests.get(
            url, headers=headers or {}, cookies=cookies or None,
            impersonate='chrome120', timeout=timeout,
            **extra,
        )
    return std_requests.get(
        url, headers=headers or {}, cookies=cookies or None, timeout=timeout,
        **extra,
    )


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
