"""Outbound HTTP proxy support with rotation — fully dynamic.

Builds a rotating proxy pool from three layers (all optional, all merged
and deduplicated):

  1. Static env proxies
       DSF_PROXY           single proxy URL, e.g. http://user:pass@host:8080
       DSF_PROXIES         comma-separated list of proxy URLs

  2. Automatic free-proxy sources (DSF_PROXY_AUTO=true)
       Aggregates well-known public proxy lists over the web (TheSpeedX,
       monosans, proxifly, proxyscrape, roosterkid, geonode), refreshed
       every DSF_PROXY_LIST_TTL seconds. DSF_PROXY_SOURCES overrides the
       built-in list (comma-separated; "socks5=<url>" or URLs whose path
       mentions socks5/socks4 get that scheme, otherwise http).
       DSF_PROXY_MAX_POOL caps the pool (random sample, default 250) so
       health checking stays fast on small boards like a Raspberry Pi.

  3. Extra list URL(s)
       DSF_PROXY_LIST_URL / DSF_PROXY_LIST_URLS  fetched with the same TTL.

  DSF_PROXY_LIST_TTL   source refresh interval, seconds (default 1800)
  DSF_PROXY_MODE       random (default) | round | single
  DSF_PROXY_EXCLUDE    comma-separated providers that always go direct
  DSF_PROXY_COOLDOWN   seconds a proxy is skipped after a runtime failure
  DSF_PROXY_ROTATE_TTL seconds a provider keeps its assigned proxy before
                       it is re-randomized (default 300)

NO-PROXY IS A FIRST-CLASS ROUTE: the direct (no-proxy) candidate is always
part of the rotation set (DSF_PROXY_DIRECT, default true), so traffic keeps
flowing even when every pooled proxy is unhealthy or cooling down. When the
health pass finds ZERO fast proxies the draw reduces to no-proxy alone.

Provider randomization: every provider (deepseek/gemini/chatgpt/...) gets
its OWN proxy, picked randomly and preferably distinct from the proxies
already assigned to other providers — concurrent providers are spread
across different exit IPs instead of sharing one. Assignments are sticky
for DSF_PROXY_ROTATE_TTL seconds, then rotate to a new random proxy, and
are dropped immediately on runtime failure so the provider gets a fresh
random proxy on the next request.

Health checking (DSF_PROXY_CHECK=true): a background worker periodically
probes every pooled proxy (concurrent, DSF_PROXY_CHECK_CONCURRENCY workers,
DSF_PROXY_CHECK_TIMEOUT seconds each) against DSF_PROXY_CHECK_URL and only
proxies that answer keep a "healthy" lease (DSF_PROXY_CHECK_TTL seconds).
get_proxy() prefers healthy proxies and never blocks: until the first pass
completes (or if everything is cooling down) traffic simply goes direct.

All proxy URLs must be scheme-qualified (http://, https://, socks5://,
socks5h:// — DNS through the proxy); bare host:port entries are upgraded
with the source's scheme or http://. HTTP and SOCKS5 proxies are both fine
as long as they answer within the (low) latency budget; Tor is never used
anywhere (no torproxy / :9050 exit). Call sites splat
``proxies_kwargs(provider)`` into requests/curl_cffi calls; it returns {}
when no proxy applies so traffic goes direct unchanged.

SECURITY NOTE: free public proxies are untrusted. HTTPS targets are still
end-to-end TLS (the proxy only sees the hostname), but treat the pool as
opportunistic and keep DSF_PROXY_EXCLUDE for providers that misbehave.
"""

import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

# Pseudo-candidate meaning "no proxy at all" — part of the rotation set so
# every provider rotates across [no-proxy, proxy1, proxy2, ...].
DIRECT = '__direct__'

# (default_scheme_or_None, url) — None means the scheme is embedded per line
# or JSON payload. All URLs verified live (2026-09); sources may vanish, the
# aggregator tolerates that.
BUILTIN_SOURCES: List[Tuple[Optional[str], str]] = [
    ('http', 'https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/http.txt'),
    ('socks5', 'https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/socks5.txt'),
    ('http', 'https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/http.txt'),
    ('socks5', 'https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/socks5.txt'),
    (None, 'https://raw.githubusercontent.com/proxifly/free-proxy-list/main/proxies/all/data.txt'),
    ('http', 'https://api.proxyscrape.com/v2/?request=displayproxies&protocol=http&timeout=10000'),
    ('http', 'https://raw.githubusercontent.com/roosterkid/openproxylist/main/HTTPS_RAW.txt'),
    (None, 'https://proxylist.geonode.com/api/proxy-list?protocols=http%2Csocks5&limit=500&page=1&sort_by=lastChecked&sort_type=desc'),
]

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


def _normalize(entry: str, default_scheme: Optional[str] = None) -> Optional[str]:
    entry = (entry or '').strip()
    if not entry or entry.startswith('#'):
        return None
    if '://' not in entry:
        entry = (default_scheme or 'http') + '://' + entry
    return entry


def _parse_list(body: str, default_scheme: Optional[str] = None) -> List[str]:
    """Parse a fetched proxy list: plain text, scheme-tagged text or JSON."""
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
                    out.append(_normalize(item, default_scheme) or '')
                elif isinstance(item, dict):
                    out.append(_proxy_from_dict(item) or '')
            return [p for p in out if p]
        if isinstance(data, dict):
            for key in ('proxies', 'data', 'results', 'list'):
                if isinstance(data.get(key), list):
                    return _parse_list(json.dumps(data[key]), default_scheme)
            return out
    for line in body.splitlines():
        p = _normalize(line, default_scheme)
        if p:
            out.append(p)
    return out


def _proxy_from_dict(item: Dict[str, Any]) -> Optional[str]:
    """Map a JSON proxy entry ({ip,port[,protocol|protocols|proxy]}) to URL."""
    if 'proxy' in item and str(item['proxy']).strip():
        return _normalize(str(item['proxy']))
    if 'ip' not in item or 'port' not in item:
        return None
    proto: Any = None
    if isinstance(item.get('protocols'), (list, tuple)) and item['protocols']:
        proto = item['protocols'][0]
    elif item.get('protocol') or item.get('proto'):
        proto = item.get('protocol') or item.get('proto')
    proto = str(proto or 'http').split(',')[0].strip() or 'http'
    return f"{proto}://{item['ip']}:{item['port']}"


def _fetch_direct(url: str, timeout: int = 15) -> str:
    """Fetch a source list WITHOUT going through the proxy pool (no cycles)."""
    try:
        from curl_cffi import requests as cffi
    except ImportError:
        cffi = None
    if cffi is not None:
        resp = cffi.get(url, timeout=timeout, impersonate='chrome120')
    else:
        import requests as std
        resp = std.get(url, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}")
    return resp.text


def _custom_sources() -> List[Tuple[Optional[str], str]]:
    """User-overridden sources: DSF_PROXY_SOURCES=scheme=url,url2,..."""
    raw = os.getenv('DSF_PROXY_SOURCES', '').strip()
    if not raw:
        return []
    sources: List[Tuple[Optional[str], str]] = []
    for entry in raw.split(','):
        entry = entry.strip()
        if not entry:
            continue
        if '=' in entry and not entry.lower().startswith(('http://', 'https://')):
            scheme, url = entry.split('=', 1)
            sources.append((scheme.strip().lower() or None, url.strip()))
        elif 'socks5' in entry.lower():
            sources.append(('socks5', entry))
        elif 'socks4' in entry.lower():
            sources.append(('socks4', entry))
        else:
            sources.append(('http', entry))
    return sources


def _extra_list_urls() -> List[str]:
    urls = [u.strip() for u in os.getenv('DSF_PROXY_LIST_URLS', '').split(',') if u.strip()]
    single = os.getenv('DSF_PROXY_LIST_URL', '').strip()
    if single:
        urls.append(single)
    return urls


class _State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.pool: List[str] = []          # aggregated, capped, shuffled
        self.healthy: Dict[str, float] = {}  # proxy -> lease expiry (epoch)
        self.latency: Dict[str, float] = {}  # proxy -> last measured ms
        self.fetched_at: float = 0.0
        self.checked_at: float = 0.0
        self.cooldown: Dict[str, float] = {}  # proxy -> retry-after (epoch)
        self.assignments: Dict[str, Tuple[str, float]] = {}  # provider -> (proxy, expires)
        self.rr: int = 0
        self.controller_started = False
        self.last_error: str = ''


_STATE = _State()


def _list_ttl() -> float:
    return max(60.0, float(os.getenv('DSF_PROXY_LIST_TTL', '1800') or 1800))


def _check_ttl() -> float:
    return max(60.0, float(os.getenv('DSF_PROXY_CHECK_TTL', '1800') or 1800))


def _check_enabled() -> bool:
    return _env_bool('DSF_PROXY_CHECK')


def _rotate_ttl() -> float:
    """How long a provider keeps its assigned proxy before re-randomizing."""
    return max(1.0, float(os.getenv('DSF_PROXY_ROTATE_TTL', '300') or 300))


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
    return proxies


def _refresh_pool() -> None:
    """Aggregate every configured source into the (capped, shuffled) pool."""
    if _env_bool('DSF_PROXY_AUTO'):
        sources = _custom_sources() or BUILTIN_SOURCES
    else:
        sources = []
    for url in _extra_list_urls():
        low = url.lower()
        scheme = 'socks5' if 'socks5' in low else ('socks4' if 'socks4' in low else None)
        sources.append((scheme, url))

    collected: List[str] = []
    errors: List[str] = []
    for default_scheme, url in sources:
        try:
            collected.extend(_parse_list(_fetch_direct(url), default_scheme))
        except Exception as exc:
            errors.append(f"{url.split('//', 1)[-1][:60]}: {exc}")
    # static proxies (env) always survive, even if every source fails
    collected.extend(_static_proxies())

    seen: Dict[str, str] = {}
    for p in collected:
        if not p:
            continue
        key = p.split('://', 1)[-1]  # dedup by host:port across schemes
        seen.setdefault(key, p)
    pool = list(seen.values())

    max_pool = int(os.getenv('DSF_PROXY_MAX_POOL', '250') or 250)
    if len(pool) > max_pool:
        pool = random.sample(pool, max_pool)
    random.shuffle(pool)

    now = time.time()
    with _STATE.lock:
        _STATE.pool = pool
        _STATE.fetched_at = now
        healthy_keys = set(_STATE.healthy) & set(pool)
        _STATE.healthy = {p: _STATE.healthy[p] for p in healthy_keys}
        _STATE.latency = {p: ms for p, ms in _STATE.latency.items() if p in set(pool)}
        _STATE.cooldown = {p: t for p, t in _STATE.cooldown.items() if p in set(pool)}
        if errors:
            _STATE.last_error = '; '.join(errors[:3])
        else:
            _STATE.last_error = ''
    print(f"[proxies] pool refreshed: {len(pool)} proxies "
          f"(from {len(sources)} sources)", file=__import__('sys').stderr)
    if errors:
        print(f"\033[93m[proxies] source errors: {errors}\033[0m",
              file=__import__('sys').stderr)


def _max_latency_ms() -> float:
    """Fast-proxies-only budget: a proxy slower than this never gets traffic."""
    return max(50.0, float(os.getenv('DSF_PROXY_MAX_LATENCY', '800') or 800))


def _direct_rotation() -> bool:
    """Whether the no-proxy route is part of the rotation set (default on)."""
    return _env_bool('DSF_PROXY_DIRECT', 'true')


def _probe(proxy: str, url: str, timeout: float) -> Tuple[str, bool, float]:
    try:
        from curl_cffi import requests as cffi
    except ImportError:
        cffi = None
    proxies = {'http': proxy, 'https': proxy}
    t0 = time.monotonic()
    try:
        if cffi is not None:
            resp = cffi.get(url, proxies=proxies, timeout=timeout,
                            impersonate='chrome120')
        else:
            import requests as std
            resp = std.get(url, proxies=proxies, timeout=timeout)
        ms = (time.monotonic() - t0) * 1000.0
        return proxy, resp.status_code == 200, ms
    except Exception:
        return proxy, False, (time.monotonic() - t0) * 1000.0


def _run_check_pass() -> None:
    """Validate the whole pool concurrently; only proxies that answer within
    the latency budget (DSF_PROXY_MAX_LATENCY ms) get a healthy lease —
    fast proxies only, slow exits never receive traffic."""
    url = os.getenv('DSF_PROXY_CHECK_URL',
                    'https://api.ipify.org?format=json').strip()
    timeout = float(os.getenv('DSF_PROXY_CHECK_TIMEOUT', '4') or 4)
    workers = max(1, int(os.getenv('DSF_PROXY_CHECK_CONCURRENCY', '24') or 24))
    with _STATE.lock:
        pool = list(_STATE.pool)
    if not pool:
        return
    results: Dict[str, Tuple[bool, float]] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for proxy, ok, ms in ex.map(lambda p: _probe(p, url, timeout), pool):
            results[proxy] = (ok, ms)
    now = time.time()
    lease = now + _check_ttl()
    cap = _max_latency_ms()
    with _STATE.lock:
        _STATE.checked_at = now
        fast = {p: lease for p, (ok, ms) in results.items()
                if ok and ms <= cap}
        _STATE.healthy = fast
        _STATE.latency = {p: ms for p, (ok, ms) in results.items() if ok}
    if fast:
        lats = sorted(_STATE.latency[p] for p in fast)
        median = lats[len(lats) // 2]
        print(f"[proxies] health pass: {len(fast)}/{len(pool)} fast "
              f"(<= {cap:.0f} ms, median {median:.0f} ms)",
              file=__import__('sys').stderr)
    else:
        print(f"[proxies] health pass: 0/{len(pool)} fast (<= {cap:.0f} ms) "
              "— traffic rotates to direct", file=__import__('sys').stderr)


def _controller_loop() -> None:
    last_check = 0.0
    while True:
        try:
            now = time.time()
            if now - _STATE.fetched_at >= _list_ttl():
                _refresh_pool()
            if _check_enabled() and now - last_check >= max(300.0, _check_ttl() / 3):
                last_check = time.time()
                _run_check_pass()
        except Exception as exc:  # controller must never die
            print(f"\033[93m[proxies] controller error: {exc}\033[0m",
                  file=__import__('sys').stderr)
        time.sleep(15)


def _ensure_controller() -> None:
    dynamic = _env_bool('DSF_PROXY_AUTO') or _extra_list_urls()
    if dynamic and not _STATE.controller_started:
        _STATE.controller_started = True
        threading.Thread(target=_controller_loop, name="proxy-pool",
                         daemon=True).start()


def all_proxies() -> List[str]:
    """Current proxy pool (static env ones included even before refresh)."""
    _ensure_controller()
    with _STATE.lock:
        static = _static_proxies()
        seen, merged = set(), []
        for p in static + _STATE.pool:
            if p not in seen:
                seen.add(p)
                merged.append(p)
        return merged


def _ensure_timeout() -> float:
    """Max seconds ensure_pool() waits for its warm-up health pass."""
    try:
        return max(15.0, float(os.getenv('DSF_PROXY_ENSURE_TIMEOUT', '30')))
    except ValueError:
        return 30.0


def ensure_pool() -> int:
    """Blocking pool warm-up for short-lived processes (CLI one-shots).

    The background controller fills the pool asynchronously; a fresh
    process would otherwise build its first proxy ladder against an empty
    pool. Returns the current pool size.
    """
    _ensure_controller()
    with _STATE.lock:
        empty = not _STATE.pool
    if empty and (_env_bool('DSF_PROXY_AUTO') or _extra_list_urls()):
        try:
            _refresh_pool()
        except Exception:  # noqa: BLE001 — direct remains the fallback
            pass
    # Await one health pass so short-lived processes draw FAST proxies
    # instead of the unvalidated pool. Bounded: on timeout the caller
    # proceeds with the raw pool (get_proxy still samples it).
    if _check_enabled():
        with _STATE.lock:
            needs_check = (bool(_STATE.pool)
                           and not any(t > time.time()
                                       for t in _STATE.healthy.values()))
        if needs_check:
            worker = threading.Thread(target=_run_check_pass,
                                      name="proxy-ensure", daemon=True)
            worker.start()
            worker.join(_ensure_timeout())
    with _STATE.lock:
        return len(_STATE.pool)


def get_proxy(provider: Optional[str] = None, direct_ok: bool = True) -> Optional[str]:
    """Pick a proxy for `provider` (None -> go direct). Never blocks.

    With a provider key the result is a per-provider sticky assignment:
    a randomly chosen route (distinct from other providers' when possible)
    kept for DSF_PROXY_ROTATE_TTL seconds, so traffic is randomized across
    different providers/exit IPs rather than one shared route.

    ``direct_ok=False`` (callers that REQUIRE a proxy, e.g. the signup
    ladder) removes the no-proxy candidate from the draw so the sticky
    assignment can never silently be 'direct'.
    """
    _ensure_controller()
    key = provider.strip().lower() if provider and provider.strip() else None
    exclude = {e.strip().lower() for e in os.getenv('DSF_PROXY_EXCLUDE', '').split(',') if e.strip()}
    if key and key in exclude:
        return None
    pool = all_proxies()
    if not pool:
        return None
    now = time.time()
    direct = _direct_rotation() and direct_ok
    with _STATE.lock:
        if _check_enabled():
            # After a completed health pass ONLY validated fast proxies are
            # eligible; if that set is empty (every proxy failed or timed
            # out) the draw collapses to no-proxy, so traffic never rides a
            # known-dead proxy. Before the first pass, sample the raw pool
            # (each draw then relies on the low connect timeout + cooldown).
            healthy = [p for p in pool if _STATE.healthy.get(p, 0) > now]
            candidates = healthy if _STATE.checked_at else pool
        else:
            candidates = pool
        alive = [p for p in candidates if _STATE.cooldown.get(p, 0) <= now]
        # rotation candidate set: [no-proxy, proxy1, proxy2, proxy3, ...]
        if direct:
            alive = [DIRECT] + alive
        # drop assignments that expired or whose route is no longer usable
        _STATE.assignments = {prov: pair for prov, pair in _STATE.assignments.items()
                              if pair[1] > now and pair[0] in alive}
        mode = os.getenv('DSF_PROXY_MODE', 'random').strip().lower()
        if key:
            assigned = _STATE.assignments.get(key)
            if assigned:
                return None if assigned[0] == DIRECT else assigned[0]
            if not alive:
                return None
            if mode == 'single':
                # one fixed route: prefer a real proxy, direct only when the
                # pool is empty
                real = [p for p in alive if p != DIRECT]
                route = real[0] if real else DIRECT
            else:
                # randomize between providers: prefer a route not yet taken
                # by another provider; share only if the pool is too small
                taken = {pair[0] for pair in _STATE.assignments.values()}
                distinct = [p for p in alive if p not in taken]
                route = random.choice(distinct or alive)
            _STATE.assignments[key] = (route, now + _rotate_ttl())
            return None if route == DIRECT else route
        # provider=None: plain per-request selection (no stickiness)
        if not alive:
            return None
        if mode == 'single':
            real = [p for p in alive if p != DIRECT]
            return real[0] if real else None
        if mode == 'round':
            route = alive[_STATE.rr % len(alive)]
            _STATE.rr += 1
            return None if route == DIRECT else route
        route = random.choice(alive)
        return None if route == DIRECT else route


def current(provider: Optional[str]) -> Optional[str]:
    """The provider's sticky assignment, or None when direct/unassigned."""
    if not provider:
        return None
    with _STATE.lock:
        pair = _STATE.assignments.get(provider)
    if not pair or pair[0] == DIRECT:
        return None
    return pair[0]


def mark_failure(proxy: Optional[str]) -> None:
    """Put a proxy on cooldown and release any provider assigned to it."""
    if not proxy or proxy == DIRECT:
        return
    cooldown = float(os.getenv('DSF_PROXY_COOLDOWN', '120') or 120)
    now = time.time()
    with _STATE.lock:
        _STATE.cooldown[proxy] = now + cooldown
        # force a fresh random assignment for every provider that used it
        _STATE.assignments = {prov: pair for prov, pair in _STATE.assignments.items()
                              if pair[0] != proxy}


def proxies_kwargs(provider: Optional[str] = None,
                   url: Optional[str] = None,
                   no_proxy: bool = False) -> Dict[str, Any]:
    """Kwargs to splat into requests/curl_cffi calls for `provider`/`url`.

    ``no_proxy=True`` forces a DIRECT connection ({}), skipping the pool
    entirely — used by the per-request ``disable_proxy`` endpoint param.
    """
    if no_proxy:
        return {}
    # defensive: a URL accidentally passed positionally as provider
    if provider and '://' in provider:
        url = url or provider
        provider = None
    provider = provider or _provider_for_url(url or '')
    proxy = get_proxy(provider)
    if not proxy:
        return {}
    return {'proxies': {'http': proxy, 'https': proxy}}


def active_summary() -> str:
    _ensure_controller()
    with _STATE.lock:
        pool_n, healthy_n = len(_STATE.pool), len(_STATE.healthy)
        checked = _STATE.checked_at
        err = _STATE.last_error
    parts = [f"pool={pool_n}"]
    if _check_enabled():
        parts.append(f"healthy={healthy_n}"
                     + (f" (checked {time.strftime('%H:%M:%S', time.localtime(checked))})" if checked else " (not yet)"))
        parts.append(f"max-latency={_max_latency_ms():.0f}ms")
        if checked and not healthy_n:
            parts.append('all-proxies-failed->direct')
    if _direct_rotation():
        parts.append("direct-rotation=on")
    if _env_bool('DSF_PROXY_AUTO'):
        parts.append("auto-sources=on")
    mode = os.getenv('DSF_PROXY_MODE', 'random').strip().lower() or 'random'
    parts.append(f"mode={mode}")
    with _STATE.lock:
        assignments = {prov: pair[0] for prov, pair in _STATE.assignments.items()}
    if assignments:
        parts.append("assigned=" + ','.join(
            f"{prov}->{'direct' if p == DIRECT else p.split('://', 1)[-1]}"
            for prov, p in sorted(assignments.items())))
    if err:
        parts.append(f"last_error={err[:80]}")
    return 'direct' if not pool_n else ', '.join(parts)


def _force_cycle() -> None:
    """Blocking refresh + check (used by the CLI so it can report results)."""
    _refresh_pool()
    if _check_enabled():
        _run_check_pass()


if __name__ == '__main__':  # quick manual check: python -m dsk.proxies
    import sys
    if _env_bool('DSF_PROXY_AUTO') or _extra_list_urls():
        print(f"refreshing pool ({active_summary()}) ...")
        _force_cycle()
    print(f"config: {active_summary()}")
    try:
        from .providers.base import http_get
    except ImportError:
        from dsk.providers.base import http_get
    probes = get_proxy() and []  # no-op warm-up
    with _STATE.lock:
        pool_now = list(_STATE.pool)
        healthy_now = [p for p in pool_now if _STATE.healthy.get(p, 0) > time.time()]
    probes = (healthy_now or pool_now)[:8]
    for p in probes:
        try:
            resp = http_get('https://api.ipify.org?format=json',
                            proxies={'http': p, 'https': p}, timeout=10)
            print(f"  {p} -> {resp.text.strip()} (http {resp.status_code})")
        except Exception as exc:
            mark_failure(p)
            print(f"  {p} -> FAILED: {str(exc)[:90]}")
    sys.exit(0)
