"""Outbound HTTP proxy support with rotation.

Route provider HTTP traffic through one or more proxies to spread requests
across exit IPs and soften per-IP rate limiting.  Configure via environment:

  DSF_PROXY           single proxy URL, e.g. socks5h://torproxy:9050
  DSF_PROXIES         comma-separated list of proxy URLs
  DSF_PROXY_TOR       truthy -> append the Tor SOCKS5 proxy (DSF_PROXY_TOR_URL)
  DSF_PROXY_TOR_URL   Tor SOCKS5 URL (default socks5h://torproxy:9050 — the
                      docker-network service name of a tor proxy container)
  DSF_PROXY_LIST_URL  optional URL returning a dynamic proxy list (plain text,
                      one proxy per line, or a JSON array); refreshed every
                      DSF_PROXY_LIST_TTL seconds in a background thread
  DSF_PROXY_LIST_TTL  list refresh interval in seconds (default 3600)
  DSF_PROXY_MODE      random (default) | round | single
  DSF_PROXY_EXCLUDE   comma-separated provider names that always go direct
                      (deepseek, gemini, chatgpt — e.g. when a proxy breaks a
                      provider's bot protection)
  DSF_PROXY_COOLDOWN  seconds a proxy is skipped after a failure (default 120)

All proxy URLs must be scheme-qualified (http://, https://, socks5://,
socks5h:// — the latter resolves DNS through the proxy, recommended for Tor).
Bare ``host:port`` entries are upgraded to ``http://host:port``.

Call sites splat ``proxies_kwargs(provider)`` into requests/curl_cffi calls;
it returns ``{}`` when no proxy applies so traffic goes direct unchanged.
"""

import json
import os
import random
import threading
import time
from typing import Any, Dict, List, Optional

TOR_DEFAULT_URL = 'socks5h://torproxy:9050'

_PROVIDER_HOSTS = (
    ('chat.deepseek.com', 'deepseek'),
    ('gemini.google.com', 'gemini'),
    ('chatgpt.com', 'chatgpt'),
    ('auth0.openai.com', 'chatgpt'),
)


def _env_bool(name: str, default: str = '') -> bool:
    return os.getenv(name, default).strip().lower() in ('1', 'true', 'yes', 'on')


def _provider_for_url(url: str) -> Optional[str]:
    low = (url or '').lower()
    for host, provider in _PROVIDER_HOSTS:
        if host in low:
            return provider
    return None


def _normalize(entry: str) -> Optional[str]:
    entry = entry.strip()
    if not entry or entry.startswith('#'):
        return None
    if '://' not in entry:
        entry = 'http://' + entry
    return entry


def _parse_list(body: str) -> List[str]:
    """Parse a fetched proxy list: plain text (one per line) or JSON array."""
    out: List[str] = []
    body = (body or '').strip()
    if not body:
        return out
    if body[0] in '[{':
        try:
            data = json.loads(body)
        except ValueError:
            data = None
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    out.append(_normalize(item) or '')
                elif isinstance(item, dict):
                    # Common shapes: {"ip","port"} / {"proxy","protocol"}
                    if 'proxy' in item:
                        out.append(_normalize(str(item['proxy'])) or '')
                    elif 'ip' in item and 'port' in item:
                        proto = str(item.get('protocol') or item.get('proto') or 'http').split(',')[0].strip()
                        out.append(f"{proto}://{item['ip']}:{item['port']}")
            return [p for p in out if p]
        if isinstance(data, dict):
            for key in ('proxies', 'data', 'results', 'list'):
                if isinstance(data.get(key), list):
                    return _parse_list(json.dumps(data[key]))
            return out
    for line in body.splitlines():
        p = _normalize(line)
        if p:
            out.append(p)
    return out


class _State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.static: List[str] = []
        self.dynamic: List[str] = []
        self.fetched_at: float = 0.0
        self.cooldown: Dict[str, float] = {}
        self.rr: int = 0
        self.refresher_started = False


_STATE = _State()


def _static_proxies() -> List[str]:
    proxies: List[str] = []
    single = os.getenv('DSF_PROXY', '').strip()
    if single:
        p = _normalize(single)
        if p:
            proxies.append(p)
    for entry in os.getenv('DSF_PROXIES', '').split(','):
        p = _normalize(entry)
        if p:
            proxies.append(p)
    if _env_bool('DSF_PROXY_TOR'):
        p = _normalize(os.getenv('DSF_PROXY_TOR_URL', '').strip() or TOR_DEFAULT_URL)
        if p:
            proxies.append(p)
    return proxies


def _fetch_dynamic() -> List[str]:
    """Fetch DSF_PROXY_LIST_URL and return parsed proxies (best-effort)."""
    url = os.getenv('DSF_PROXY_LIST_URL', '').strip()
    if not url:
        return []
    try:  # local import to avoid a cycle with providers/base
        try:
            from .providers.base import http_get
        except ImportError:  # pragma: no cover - standalone use
            from dsk.providers.base import http_get
        resp = http_get(url, timeout=30)
        if getattr(resp, 'status_code', 0) != 200:
            return []
        return _parse_list(resp.text)
    except Exception as exc:  # keep the previous list on any failure
        print(f"\033[93m[proxies] list refresh failed: {exc}\033[0m", file=__import__('sys').stderr)
        return []


def _refresh_dynamic_locked(now: float) -> None:
    ttl = float(os.getenv('DSF_PROXY_LIST_TTL', '3600') or 3600)
    if _STATE.fetched_at and now - _STATE.fetched_at < ttl:
        return
    _STATE.fetched_at = now
    fetched = _fetch_dynamic()
    if fetched:
        _STATE.dynamic = fetched
        print(f"[proxies] dynamic list refreshed: {len(fetched)} proxies", file=__import__('sys').stderr)


def _refresher_loop() -> None:
    ttl = max(60.0, float(os.getenv('DSF_PROXY_LIST_TTL', '3600') or 3600))
    while True:
        try:
            with _STATE.lock:
                _refresh_dynamic_locked(time.time())
        except Exception:
            pass
        time.sleep(ttl)


def _ensure_background_refresher() -> None:
    if os.getenv('DSF_PROXY_LIST_URL', '').strip() and not _STATE.refresher_started:
        _STATE.refresher_started = True
        threading.Thread(target=_refresher_loop, name="proxy-list-refresher",
                         daemon=True).start()


def all_proxies() -> List[str]:
    """Currently known proxies (static env ones + refreshed dynamic list)."""
    with _STATE.lock:
        _STATE.static = _static_proxies()
        if os.getenv('DSF_PROXY_LIST_URL', '').strip():
            _refresh_dynamic_locked(time.time())
        seen, merged = set(), []
        for p in _STATE.static + _STATE.dynamic:
            if p not in seen:
                seen.add(p)
                merged.append(p)
        return merged


def get_proxy(provider: Optional[str] = None) -> Optional[str]:
    """Pick a proxy for `provider` (or None to go direct)."""
    _ensure_background_refresher()
    exclude = {e.strip().lower() for e in os.getenv('DSF_PROXY_EXCLUDE', '').split(',') if e.strip()}
    if provider and provider.lower() in exclude:
        return None
    candidates = all_proxies()
    if not candidates:
        return None
    now = time.time()
    cooldown = float(os.getenv('DSF_PROXY_COOLDOWN', '120') or 120)
    with _STATE.lock:
        alive = [p for p in candidates if _STATE.cooldown.get(p, 0) <= now]
        if not alive:
            return None  # everything cooling down -> go direct
        mode = os.getenv('DSF_PROXY_MODE', 'random').strip().lower()
        if mode == 'single':
            return alive[0]
        if mode == 'round':
            proxy = alive[_STATE.rr % len(alive)]
            _STATE.rr += 1
            return proxy
        return random.choice(alive)


def mark_failure(proxy: Optional[str]) -> None:
    """Put a proxy on cooldown after a failure so it is skipped for a while."""
    if not proxy:
        return
    cooldown = float(os.getenv('DSF_PROXY_COOLDOWN', '120') or 120)
    with _STATE.lock:
        _STATE.cooldown[proxy] = time.time() + cooldown


def proxies_kwargs(provider: Optional[str] = None,
                   url: Optional[str] = None) -> Dict[str, Any]:
    """Kwargs to splat into requests/curl_cffi calls for `provider`.

    When `url` is given the provider is inferred from its host.  Returns
    ``{}`` when proxies are disabled, excluded, or all cooling down, so
    callers can do ``requests.post(..., **proxies_kwargs('deepseek'))``.
    """
    provider = provider or _provider_for_url(url or '')
    proxy = get_proxy(provider)
    if not proxy:
        return {}
    return {'proxies': {'http': proxy, 'https': proxy}}


def active_summary() -> str:
    proxies = all_proxies()
    if not proxies:
        return 'direct (no proxies configured)'
    mode = os.getenv('DSF_PROXY_MODE', 'random').strip().lower() or 'random'
    return f"{mode} rotation over {len(proxies)}: {', '.join(proxies)}"


if __name__ == '__main__':  # quick manual check: python -m dsk.proxies
    import sys
    try:
        from .providers.base import http_get
    except ImportError:
        from dsk.providers.base import http_get
    print(f"config: {active_summary()}")
    for p in all_proxies():
        try:
            resp = http_get('https://api.ipify.org?format=json',
                            **{'proxies': {'http': p, 'https': p}})
            print(f"  {p} -> {resp.text.strip()} (http {resp.status_code})")
        except Exception as exc:
            mark_failure(p)
            print(f"  {p} -> FAILED: {exc}")
    sys.exit(0)
