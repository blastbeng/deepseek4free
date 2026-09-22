"""Hugging Face Spaces provider — free community chat endpoints.

Every Space that is (a) listed in the text-generation / code-generation
category pages, (b) a RUNNING Gradio app and (c) passes an anonymous
liveness probe is exposed as an OpenAI-style model.  Streaming goes through
the Gradio queue protocol (POST /queue/join → GET /queue/data SSE), routed
over the shared proxy pool with automatic direct fallback, exactly like the
other providers.

Discovery (dynamic, never hardcoded):
  1. https://huggingface.co/spaces?category=text-generation and
     ?category=code-generation pages are scraped — their SSR payload embeds
     a semantic-search result set (``spacesSemantcSearch``) with a
     ``semanticRelevancyScore`` per space.
  2. The global API (``/api/spaces`` with search=chat/llm/coder/... and
     sort=trendingScore/likes) broadens the pool beyond the categories.
     RANKED BY REAL USAGE — the pool is intentionally WIDE: every source
     contributes hundreds of spaces and only hard junk (image/tts/etc.)
     is dropped.  The ranking is dominated by real-world demand: likes
     (community votes, log-boosted) + total downloads of the models each
     Space actually serves (fetched per linked model repo) + trending
     momentum + semantic relevance, with a small penalty for slow
     first-token times observed during probing.
  4. Spaces whose linked models are all non-text (image/video/tts/3d
     pipelines — the giants of the global likes board) are dropped by
     pipeline_tag before anything is probed, so probing time is spent on
     actual chat/coding spaces.
  5. The top candidates are probed once with a tiny prompt; only spaces
     that actually stream a completion stay in the model list.  Probe
     results persist in ``data/hf_spaces.json`` so restarts are cheap and
     flaky spaces rotate out instead of breaking requests.

Quota reality: ZeroGPU Spaces enforce a per-identity daily GPU budget
(anonymous ≈ minutes, free token higher).  Anonymous probing/usage can hit
"exceeded your ZeroGPU quota" — the provider maps that to
ProviderRateLimitError so the router backs off, and an HF token
(``HF_TOKEN`` / jar ``token``) is attached as a Bearer header when present
to raise the quota tier.

Tool calling / agent coding needs no provider code: the OpenAI server
layer teaches every provider the DSML tool-call protocol in the prompt.
"""

import codecs
import html
import json
import logging
import math
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from .base import (
    HTTP_CONNECT_TIMEOUT,
    Provider,
    ProviderError,
    ProviderRateLimitError,
    ProviderUnavailableError,
    _resilient_request,
    http_get,
    http_post_raw,
    parse_sse_data,
    proxy_kwargs_for,
)
from .jar import env_cookies, load_jar

logger = logging.getLogger('dsk.providers.huggingface')

HF_HOST = 'https://huggingface.co'
CATEGORIES = ('text-generation', 'code-generation')
SEARCH_TERMS = ('chat', 'chatbot', 'assistant', 'llm', 'coder',
                'coding agent', 'talk')
UA = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) Chrome/120 Safari/537.36'}

# Names that are virtually never a usable text-chat endpoint.
_JUNK_RE = re.compile(
    r'image|video|flux|sdxl|diffusion|wan\d|tts|voice|speech|music|audio|'
    r'asr|whisper|lora|photo|avatar|marigold|upscale|embedding|rerank|'
    r'classif|detection|ocr|vision|leaderboard|tokenizer|dataset|'
    r'playground-?ui|humanizer|notes|annotat|translat|'
    r'try-?on|outfit|portrait|anime|animagine|midi|codeformer|3d|'
    r'parser|blip|csm-|expression|webcam|camera|makeup|coloriz|restor|'
    r'inpaint|outpaint|sketch|paint|draw|render|depth|hidream|rvc|'
    r'illustrious|pony|suno|udio|jasco|diffree', re.IGNORECASE)

DISCOVERY_TTL = 300.0          # model list freshness (matches MODELS_TTL)
PROBE_TTL = 6 * 3600.0         # re-probe working spaces twice a day
PROBE_FAIL_TTL = 30 * 60.0     # retry failed spaces every half hour
ENRICH_TOP = 900               # spaces enriched with full detail
DOWNLOAD_TOP = 1200            # spaces scored with model download counts
PROBE_TOP = 60                 # first live-probe wave per refresh
PROBE_WAVE = 20                # fill-up wave size (junk probes fail cheap:
                               # one config GET, no GPU burn)
PROBE_TARGET = 100             # stop inviting more when this many work
MAX_PROBES_PER_RUN = 400       # hard cap per discovery (quota + time); the
                               # probe cache widens coverage across runs
PROBE_TIMEOUT = 45.0           # seconds to wait for a probe completion
SSE_IDLE_TIMEOUT = 90.0        # max silence between stream events
MODEL_META_TTL = 24 * 3600.0   # downloads/pipeline-tag refetch once a day

# pipeline_tag values that make a linked model a text/chat/coding model;
# a Space whose linked models ALL carry other tags (image/video/tts/3d)
# is not a chat endpoint and never gets probed.
TEXT_PIPELINES = {'text-generation', 'text2text-generation',
                  'conversational', 'image-text-to-text'}
# spaces found on generic trending/likes boards (not via the text-gen /
# code-gen categories or chat searches) get their vote count capped in the
# cheap pre-rank so image-gen giants cannot evict real chat spaces from
# the detail-enrichment window; the exact likes still count in _score.
BOARD_LIKES_CAP = 800.0

_TEXT_IN_TYPES = ('textbox', 'multimodaltextbox')
_TEXT_OUT_TYPES = ('chatbot', 'textbox', 'markdown')

_ZEROGPU_QUOTA_RE = re.compile(r'exceeded your (?:ZeroGPU|GPU) quota', re.I)
_ILLEGAL_DURATION_RE = re.compile(
    r'(illegal duration|larger than the maximum allowed)', re.I)


def _token(auth_key: Optional[str] = None) -> str:
    """HF access token: request key → env → credential jar."""
    if auth_key and not auth_key.startswith('Bearer '):
        return auth_key.strip()
    env = (os.getenv('HF_TOKEN', '') or '').strip()
    if env:
        return env
    jar = load_jar('huggingface')
    return (jar.get('token') or '').strip()


def _auth_headers(tok: str) -> Dict[str, str]:
    h = dict(UA)
    if tok:
        h['Authorization'] = f'Bearer {tok}'
    return h


# --------------------------------------------------------------------------
# discovery sources
# --------------------------------------------------------------------------

def _hf_api(path: str, timeout: int = 12) -> Optional[Any]:
    # direct: huggingface.co is not WAF-protected, and the widened pool
    # means hundreds of API calls per discovery — the pooled proxies would
    # only add their dead-connect timeout to every one of them
    try:
        r = http_get(f'{HF_HOST}{path}', headers=UA, timeout=timeout,
                     no_proxy=True)
        if r.status_code != 200:
            return None
        return json.loads(r.text)
    except Exception:  # noqa: BLE001 — discovery is best-effort
        return None


def _semantic_category(category: str) -> List[Dict[str, Any]]:
    """Scrape a category page's embedded semantic-search result set."""
    url = (f'{HF_HOST}/spaces?category={category}&p=0'
           f'&sort=relevance&includeNonRunning=true')
    try:
        r = http_get(url, headers=UA, timeout=25, no_proxy=True)
        if r.status_code != 200:
            return []
        page = html.unescape(r.text)
    except Exception:  # noqa: BLE001
        return []
    idx = page.find('"spacesSemantcSearch":')
    if idx < 0:
        return []
    try:
        arr, _ = json.JSONDecoder().raw_decode(
            page[idx + len('"spacesSemantcSearch":'):])
    except ValueError:
        return []
    out = []
    for x in arr if isinstance(arr, list) else []:
        if not isinstance(x, dict):
            continue
        sid = x.get('id') or (
            f"{(x.get('author') or {}).get('name', '')}/{x.get('name', '')}")
        out.append({
            'id': sid,
            'likes': x.get('likes') or 0,
            'trend': x.get('trendingScore') or 0,
            'sem': x.get('semanticRelevancyScore') or 0,
            'sdk': x.get('sdk'),
            'stage': ((x.get('runtime') or {}).get('stage')
                      if x.get('runtime') else None),
        })
    return out


def _api_spaces(params: str) -> List[Dict[str, Any]]:
    data = _hf_api(f'/api/spaces?{params}')
    return data if isinstance(data, list) else []


def _collect_candidates() -> Dict[str, Dict[str, Any]]:
    """Union of category pages + global trending + search terms."""
    cands: Dict[str, Dict[str, Any]] = {}
    seen = set()

    def add(x: Dict[str, Any], sem: float = 0.0) -> None:
        sid = x.get('id') or ''
        if (not sid) or ('/' not in sid) or sid in seen:
            return
        seen.add(sid)
        cands[sid] = {
            'likes': x.get('likes') or 0,
            'trend': x.get('trendingScore') or x.get('trend') or 0,
            'sem': sem or (x.get('sem') or 0),
            'sdk': x.get('sdk'),
            'stage': x.get('stage'),
        }

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_semantic_category, c) for c in CATEGORIES]
        # global boards: trending momentum AND all-time/30d votes, so a
        # space that is past its hype but heavily used still surfaces
        futures.append(pool.submit(
            _api_spaces,
            'sort=trendingScore&limit=400&runtime=RUNNING'))
        futures.append(pool.submit(
            _api_spaces, 'sort=likes&limit=400&runtime=RUNNING'))
        futures.append(pool.submit(
            _api_spaces, 'sort=likes30d&limit=250&runtime=RUNNING'))
        for term in SEARCH_TERMS:
            futures.append(pool.submit(
                _api_spaces,
                f'search={term}&sort=trendingScore&limit=100'
                f'&runtime=RUNNING'))
        for term in ('chat', 'llm', 'coder', 'agent'):
            futures.append(pool.submit(
                _api_spaces,
                f'search={term}&sort=likes&limit=150&runtime=RUNNING'))
        for fut in as_completed(futures):
            try:
                for x in fut.result():
                    add(x)
            except Exception:  # noqa: BLE001
                pass
    # category pages + chat/llm/coder searches are chat-relevant by
    # construction; generic boards are not
    for sid, meta in cands.items():
        owner, _, name = sid.partition('/')
        low = name.lower()
        meta['chatish'] = bool(
            meta['sem']
            or re.search(r'chat|llm|assistant|agent|coder|talk|conversation',
                         low))
    return cands


def _usable_name(sid: str) -> bool:
    if _JUNK_RE.search(sid):
        return False
    owner, _, name = sid.partition('/')
    if name.lower() == 'static':
        return False
    return True


def _detail(sid: str) -> Optional[Dict[str, Any]]:
    # subdomain/stage/hardware/linked-models change rarely — cache ~30 min
    # so the widened enrichment window does not refetch hundreds of
    # /api/spaces/<id> calls every discovery cycle
    key = f'detail:{sid}'
    with _CACHE_LOCK:
        rec = _load_probe_cache().get(key)
    if rec and time.time() - float(rec.get('ts') or 0) < 1800.0:
        return rec.get('d')
    d = _hf_api(f'/api/spaces/{sid}')
    if not isinstance(d, dict):
        return None
    rt = d.get('runtime') or {}
    # gated kept RAW ('auto' | 'manual' | False): discovery admits
    # 'auto'-gated spaces when a token is present, 'manual' never
    out = {
        'sub': d.get('subdomain'),
        'sdk': d.get('sdk'),
        'stage': rt.get('stage'),
        'hw': ((rt.get('hardware') or {}).get('current')),
        'gated': d.get('gated') or False,
        'models': d.get('models') or [],
        'likes': d.get('likes') or 0,
    }
    with _CACHE_LOCK:
        cache = _load_probe_cache()
        cache[key] = {'ts': time.time(), 'd': out}
        _save_probe_cache(cache)
    return out


def _model_record(repo: str) -> Optional[Dict[str, Any]]:
    with _CACHE_LOCK:
        rec = _load_probe_cache().get(f'model:{repo}')
    if not rec:
        return None
    if time.time() - float(rec.get('ts') or 0) > MODEL_META_TTL:
        return None
    return rec


def _store_model_meta(repo: str, rec: Dict[str, Any]) -> None:
    with _CACHE_LOCK:
        cache = _load_probe_cache()
        cache[f'model:{repo}'] = rec
        _save_probe_cache(cache)


def _model_meta(repo: str) -> Optional[Dict[str, Any]]:
    """Model repo metadata: total downloads + pipeline tag (usage signals).
    Cached ~a day: download counts move slowly and the widened pool means
    hundreds of repo lookups per discovery cycle otherwise."""
    cached = _model_record(repo)
    if cached:
        return {'downloads': int(cached.get('downloads') or 0),
                'pipeline_tag': cached.get('pipeline_tag') or ''}
    d = _hf_api(f'/api/models/{repo}')
    if not isinstance(d, dict):
        return None
    try:
        dl = int(d.get('downloads') or 0)
    except (TypeError, ValueError):
        dl = 0
    rec = {'ts': time.time(), 'downloads': dl,
           'pipeline_tag': d.get('pipeline_tag') or ''}
    _store_model_meta(repo, rec)
    return {'downloads': dl, 'pipeline_tag': rec['pipeline_tag']}


# --------------------------------------------------------------------------
# gradio config → chat endpoint mapping
# --------------------------------------------------------------------------

def _gradio_config(sub: str) -> Optional[Dict[str, Any]]:
    try:
        r = http_get(f'https://{sub}.hf.space/config', headers=UA, timeout=15)
        if r.status_code != 200 or not r.text.strip().startswith('{'):
            return None
        return json.loads(r.text)
    except Exception:  # noqa: BLE001
        return None


def _pick_chat_api(cfg: Dict[str, Any]) -> Optional[Tuple[str, int, Dict[str, Any]]]:
    """Choose the dependency that drives the conversation.

    Preference: ChatInterface internals (_submit_fn / chat / respond),
    then any queued dependency with a chatbot output, then any queued
    dependency with a textual input AND a textual output — many of the
    most-used spaces are plain "text-generation playgrounds" whose result
    lands in a textbox/markdown component, not a chatbot.
    """
    deps = cfg.get('dependencies') or []
    comps = {c.get('id'): c for c in (cfg.get('components') or [])}

    def types(ids):
        return [(comps.get(i) or {}).get('type') or '' for i in ids or []]

    for api in ('_submit_fn', 'chat', 'respond'):
        for i, dep in enumerate(deps):
            if dep.get('api_name') == api:
                return api, i, dep
    for want_chatbot in (True, False):
        for i, dep in enumerate(deps):
            if not dep.get('queue'):
                continue
            outs = types(dep.get('outputs'))
            ins = types(dep.get('inputs'))
            if not (any(t in _TEXT_IN_TYPES for t in ins)
                    and any(t in _TEXT_OUT_TYPES for t in outs)):
                continue
            if want_chatbot and 'chatbot' not in outs:
                continue
            return dep.get('api_name') or f'fn{dep.get("id", i)}', i, dep
    return None


def _info_params(sub: str, api_name: str) -> List[Dict[str, Any]]:
    """Named-endpoint parameter list (names + defaults), best-effort."""
    try:
        r = http_get(f'https://{sub}.hf.space/gradio_api/info'
                     f'?api_name={api_name}', headers=UA, timeout=10)
        if r.status_code != 200:
            return []
        eps = json.loads(r.text).get('named_endpoints') or {}
        for key, ep in eps.items():
            if key.lstrip('/') == api_name:
                return ep.get('parameters') or []
    except Exception:  # noqa: BLE001
        pass
    return []


def _build_join_data(dep: Dict[str, Any], comps: Dict[int, Dict[str, Any]],
                     prompt: str) -> Tuple[List[Any], List[str]]:
    """Assemble the queue-join ``data`` array from the dependency inputs.

    Returns (data, param_names aligned with NON-state inputs,
             state_positions keyed by input id).
    """
    data: List[Any] = []
    names: List[str] = []
    for cid in dep.get('inputs') or []:
        c = comps.get(cid) or {}
        ctype = c.get('type')
        props = c.get('props') or {}
        if ctype == 'textbox':
            data.append(prompt)
            names.append(str(props.get('label') or 'message').lower())
            continue
        if ctype == 'multimodaltextbox':
            data.append({'text': prompt, 'files': []})
            names.append('message')
            continue
        v = props.get('value')
        if ctype in ('state', 'chatbot', 'browserstate'):
            if not isinstance(v, list):
                v = []
            data.append(v)
            continue
        if v is None:
            v = 0 if ctype == 'slider' else ''
        data.append(v)
        names.append(str(props.get('label') or '').lower())
    return data, names
# --------------------------------------------------------------------------
# probe cache (data/hf_spaces.json)
# --------------------------------------------------------------------------


def _cache_path() -> Path:
    base = os.getenv('COOKIES_DIR')
    if base and Path(base).is_dir():
        return Path(base) / 'hf_spaces.json'
    # repo-root data dir (same convention as refresher._data_dir) so the
    # cache survives container rebuilds on the bind-mounted volume
    return (Path(__file__).resolve().parent.parent.parent / 'data'
            / 'hf_spaces.json')


def _load_probe_cache() -> Dict[str, Any]:
    try:
        return json.loads(_cache_path().read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def _save_probe_cache(cache: Dict[str, Any]) -> None:
    path = _cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(cache, indent=1), encoding='utf-8')
        tmp.replace(path)
    except OSError:
        pass


# Short critical sections only (probe/detail/model-meta cache file
# read-modify-write in helper functions, and the final _CACHE publish).
# Reentrant so nested helper calls on one thread cannot self-deadlock.
_CACHE_LOCK = threading.RLock()


# Discovery-run serialization (one pipeline at a time; late callers
# re-check the TTL after waiting).  Kept SEPARATE from _CACHE_LOCK: the
# pipeline calls probe/detail/model-meta helpers that briefly take
# _CACHE_LOCK in worker threads — holding it across the pipeline would
# deadlock them (the original Lock vs RLock confusion, relocated).
_RUN_LOCK = threading.Lock()


def _probe_record(space_id: str) -> Optional[Dict[str, Any]]:
    with _CACHE_LOCK:
        return _load_probe_cache().get(space_id)


def _store_probe(space_id: str, rec: Dict[str, Any]) -> None:
    with _CACHE_LOCK:
        cache = _load_probe_cache()
        cache[space_id] = rec
        _save_probe_cache(cache)


def _probe_fresh(space_id: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
    rec = _probe_record(space_id)
    if not rec:
        return False, None
    age = time.time() - float(rec.get('ts') or 0)
    if rec.get('ok'):
        return age < PROBE_TTL, rec
    return age < PROBE_FAIL_TTL, rec


# --------------------------------------------------------------------------
# live probe
# --------------------------------------------------------------------------

def _extract_text(value: Any) -> str:
    """Pull assistant text out of any gradio chatbot/value shape."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ('text', 'value'):
            if isinstance(value.get(key), str):
                return value[key]
        return ''
    if isinstance(value, list):
        # message list (gradio 4+): [{"role": "assistant", "content": ...}]
        for item in reversed(value):
            if not isinstance(item, (list, dict)):
                continue
            if isinstance(item, list):
                # legacy tuples format: [user, assistant]
                if len(item) >= 2 and isinstance(item[-1], str):
                    return item[-1]
                continue
            role = str(item.get('role') or '').lower()
            if role not in ('assistant', ''):
                continue
            content = item.get('content')
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts = [p.get('text') for p in content
                         if isinstance(p, dict) and p.get('type') == 'text']
                return ''.join(t or '' for t in texts)
            if isinstance(content, dict):
                t = content.get('text') or ''
                return t if isinstance(t, str) else ''
    return ''


def _chat_text_from_output(out: Any, out_types: List[str]) -> str:
    """Pull assistant text from outputs, strictly at known TEXT positions.

    No promiscuous fallback: media spaces (image/audio/3D) also emit
    string pieces (seed numbers, status lines, preview URLs) that would
    otherwise pass the probe as 'text'.  Unknown layouts simply fail the
    probe and are excluded — a much better tradeoff than media junk.
    """
    data = (out or {}).get('data')
    if not isinstance(data, list):
        return ''
    for pos, otype in enumerate(out_types):
        if pos >= len(data):
            break
        if otype in ('chatbot', 'textbox', 'markdown', 'json'):
            txt = _extract_text(data[pos])
            if txt:
                return txt
    return ''


def _join_prefix(sub: str, fn_index: int, data: List[Any], tok: str,
                 session_hash: Optional[str] = None
                 ) -> Tuple[Optional[str], str]:
    """POST /queue/join, trying /gradio_api first then the legacy path.

    ``session_hash`` must be the SAME one the SSE stream will listen on —
    a mismatch makes the queue answer 'Session not found'.
    """
    sess = session_hash or _session_hash()
    body = json.dumps({'fn_index': fn_index, 'data': data,
                       'event_data': None, 'trigger_id': None,
                       'session_hash': sess})
    for prefix in ('/gradio_api', ''):
        try:
            r = http_post_raw(
                f'https://{sub}.hf.space{prefix}/queue/join',
                data=body.encode('utf-8'),
                headers={**_auth_headers(tok),
                         'Content-Type': 'application/json'},
                timeout=15)
            if r.status_code == 200:
                return prefix, r.text[:200]
            if r.status_code not in (404, 405, 501):
                return None, f'join-{r.status_code}: {r.text[:120]}'
        except Exception as exc:  # noqa: BLE001
            return None, f'join-failed: {type(exc).__name__}: {str(exc)[:120]}'
    return None, 'join-404-both-paths'


def _session_hash() -> str:
    return uuid.uuid4().hex[:16]


def _sse_stream(sub: str, sess: str, prefix: str, tok: str, timeout: float):
    """GET the queue SSE stream, resilient over the proxy pool."""
    url = f'https://{sub}.hf.space{prefix}/queue/data?session_hash={sess}'
    extra = proxy_kwargs_for(url)

    def _do(kwargs: Dict[str, Any]):
        px = bool(kwargs.get('proxies'))
        eff = (HTTP_CONNECT_TIMEOUT, timeout) if px else (timeout, timeout)
        try:
            import curl_cffi.requests as cffi
            return cffi.get(url, headers={**_auth_headers(tok),
                                          'Accept': 'text/event-stream'},
                            stream=True, impersonate='chrome120',
                            timeout=eff, **kwargs)
        except ImportError:
            import requests as std
            return std.get(url, headers={**_auth_headers(tok),
                                         'Accept': 'text/event-stream'},
                           stream=True, timeout=eff, **kwargs)

    return _resilient_request(url, extra, _do, pooled=True)


def _probe_space(space_id: str, sub: str, tok: str) -> Dict[str, Any]:
    """Tiny end-to-end completion. Returns a persistent probe record."""
    rec: Dict[str, Any] = {'ts': time.time(), 'ok': False}
    try:
        cfg = _gradio_config(sub)
        if not cfg:
            rec['err'] = 'no-config'
            return rec
        pick = _pick_chat_api(cfg)
        if not pick:
            rec['err'] = 'no-chat-api'
            return rec
        api_name, fn_index, dep = pick
        comps = {c.get('id'): c for c in (cfg.get('components') or [])}
        params = _info_params(sub, api_name)
        msg_mm = any((comps.get(cid) or {}).get('type') == 'multimodaltextbox'
                     for cid in dep.get('inputs') or [])
        data: List[Any] = []
        pos = 0
        out_types = [(comps.get(oid) or {}).get('type') or ''
                     for oid in dep.get('outputs') or []]
        pdefaults: Dict[str, Any] = {}
        for cid in dep.get('inputs') or []:
            c = comps.get(cid) or {}
            ctype = c.get('type')
            if ctype == 'textbox':
                data.append('Say OK and nothing else.')
            elif ctype == 'multimodaltextbox':
                data.append({'text': 'Say OK and nothing else.', 'files': []})
            elif ctype in ('state', 'chatbot', 'browserstate'):
                data.append([] if not isinstance(
                    (c.get('props') or {}).get('value'), list) else
                    (c.get('props') or {}).get('value'))
            else:
                v = (c.get('props') or {}).get('value')
                if v is None:
                    v = 0 if ctype == 'slider' else ''
                data.append(v)
                if pos < len(params):
                    pdefaults[str(params[pos].get('parameter_name') or '')
                              ] = v
            if ctype not in ('state', 'browserstate'):
                pos += 1
        sess = _session_hash()
        prefix, join_err = _join_prefix(sub, fn_index, data, tok, sess)
        if prefix is None:
            rec['err'] = join_err
            return rec
        resp = _sse_stream(sub, sess, prefix, tok, PROBE_TIMEOUT)
        decoder = codecs.getincrementaldecoder('utf-8')('replace')
        buf = ''
        gen_txt = 0
        t0 = time.time()
        final = ''
        for chunk in resp.iter_content(chunk_size=None):
            buf += decoder.decode(chunk or b'')
            while '\n' in buf:
                line, buf = buf.split('\n', 1)
                ev = parse_sse_data(line.encode('utf-8'))
                if not ev:
                    continue
                m = ev.get('msg')
                if m == 'process_generating':
                    # count extracted TEXT only: image/video previews also
                    # stream large JSON that would otherwise look like
                    # generation and pass the probe (false positive)
                    gen_txt += len(_chat_text_from_output(
                        ev.get('output') or {}, out_types))
                elif m == 'process_completed':
                    out = ev.get('output') or {}
                    final = _chat_text_from_output(out, out_types)
                    err = out.get('error')
                    if err:
                        rec['err'] = str(err)[:200]
                        return rec
                    if ev.get('success') is False:
                        rec['err'] = 'completed-unsuccessful'
                        return rec
                    break
                elif m == 'unexpected_error':
                    rec['err'] = 'unexpected: ' + str(ev.get('message'))[:160]
                    return rec
                if time.time() - t0 > PROBE_TIMEOUT + 5:
                    break
            if final or gen_txt > 40:
                break
            if time.time() - t0 > PROBE_TIMEOUT + 5:
                break
        ok = bool(final.strip()) or gen_txt > 40
        rec.update(ok=ok, prefix=prefix, api=api_name, fn_index=fn_index,
                   out_types=out_types, msg_mm=msg_mm,
                   pdefaults=pdefaults, ver=str(cfg.get('version') or ''),
                   gen=gen_txt, final=final[:80],
                   ttft=round(time.time() - t0, 1))
        if not ok:
            rec.setdefault('err', 'no-output')
        return rec
    except ProviderError as exc:
        rec['err'] = str(exc)[:200]
        return rec
    except Exception as exc:  # noqa: BLE001
        rec['err'] = f'{type(exc).__name__}: {str(exc)[:160]}'
        return rec


# --------------------------------------------------------------------------
# discovery pipeline
# --------------------------------------------------------------------------

_CACHE: Dict[str, Any] = {'ts': 0.0, 'models': [], 'entries': {}}


def _pre_score(entry: Dict[str, Any]) -> float:
    """Cheap rank used BEFORE per-space details/downloads are known.

    Spaces found on generic trending/likes boards (``chatish`` False — not
    from the text-gen/code-gen categories or chat/llm/coder searches) get
    their vote count capped: likes alone do not mean a space is a chat
    endpoint, and the global likes board is dominated by image-generation
    giants that would otherwise evict real chat spaces from the
    detail-enrichment window. Exact likes still count in ``_score``.
    """
    likes = float(entry.get('likes') or 0)
    if not entry.get('chatish'):
        likes = min(likes, BOARD_LIKES_CAP)
    return (likes + math.log10(likes + 1.0) * 40.0
            + float(entry.get('trend') or 0) * 10.0
            + float(entry.get('sem') or 0) * 5.0)


def _score(entry: Dict[str, Any]) -> float:
    """Usage score — 'most used' wins, per demand signals:
    votes (likes) and downloads of the models each Space serves dominate,
    both log-scaled so a handful of mega-liked entries cannot drown solid
    mid-tier spaces; downloads are weighted highest because they measure
    real-world usage of the model behind the Space. Trend momentum and
    semantic relevance only break ties; slow spaces are demoted."""
    likes = float(entry.get('likes') or 0)
    trend = float(entry.get('trend') or 0)
    sem = float(entry.get('sem') or 0)
    dl = float(entry.get('downloads') or 0)
    ttft = float((entry.get('probe') or {}).get('ttft')
                 or entry.get('ttft') or 0)
    return (math.log10(likes + 1.0) * 150.0
            + math.log10(dl + 1.0) * 170.0
            + trend * 10.0 + sem * 5.0
            - min(ttft, 30.0) * 2.0)


def _discover(force: bool = False,
              auth_key: Optional[str] = None) -> Dict[str, Any]:
    """Refresh the model list (TTL cached). Returns _CACHE."""
    now = time.time()
    if not force and (now - float(_CACHE.get('ts') or 0)) < DISCOVERY_TTL \
            and _CACHE.get('models'):
        return _CACHE
    with _RUN_LOCK:
        if not force and (time.time() - float(_CACHE.get('ts') or 0)) \
                < DISCOVERY_TTL and _CACHE.get('models'):
            return _CACHE
        tok = _token(auth_key)
        cands = _collect_candidates()
        usable = {sid: meta for sid, meta in cands.items()
                  if _usable_name(sid)}
        # cheap first rank to bound the detail-fetch fan-out
        for sid, meta in usable.items():
            meta['pre'] = _pre_score(meta)
        ranked = sorted(usable.items(),
                        key=lambda kv: kv[1]['pre'], reverse=True)
        top = ranked[:ENRICH_TOP]

        entries: Dict[str, Dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=12) as pool:
            futs = {pool.submit(_detail, sid): sid for sid, _ in top}
            for fut in as_completed(futs):
                sid = futs[fut]
                d = None
                try:
                    d = fut.result()
                except Exception:  # noqa: BLE001
                    pass
                if not d:
                    continue
                gated = d.get('gated')
                # a logged-in token admits 'auto'-gated spaces (an unaccepted
                # gate only costs one cheap failed probe); 'manual' stays
                # out — there is no autonomous approval path for it
                if (d['sdk'] != 'gradio' or d['stage'] != 'RUNNING'
                        or gated == 'manual' or (gated and not tok)
                        or not d['sub']):
                    continue
                meta = dict(usable[sid])
                meta.update(d)
                meta['space_id'] = sid  # probe cache key (slug ≠ space id)
                entries[sid] = meta

        # usage scoring via the models each space serves: downloads are a
        # 'most used' signal, and the pipeline_tag separates chat/coding
        # models from image/video/tts giants so they never reach probing
        dl_targets = sorted(entries.items(),
                            key=lambda kv: _pre_score(kv[1]), reverse=True
                            )[:DOWNLOAD_TOP]
        meta_cache: Dict[str, Optional[Dict[str, Any]]] = {}

        def _meta_for(sid: str) -> Tuple[float, set]:
            total = 0
            tags = set()
            for repo in entries[sid].get('models') or []:
                if repo not in meta_cache:
                    meta_cache[repo] = _model_meta(repo)
                mm = meta_cache[repo] or {}
                total += mm.get('downloads') or 0
                if mm.get('pipeline_tag'):
                    tags.add(mm['pipeline_tag'])
            return total, tags

        with ThreadPoolExecutor(max_workers=8) as pool:
            futs = {pool.submit(_meta_for, sid): sid
                    for sid, _ in dl_targets}
            for fut in as_completed(futs):
                sid = futs[fut]
                try:
                    dl, tags = fut.result()
                except Exception:  # noqa: BLE001
                    dl, tags = 0, set()
                entries[sid]['downloads'] = dl
                entries[sid]['pipeline_tags'] = sorted(tags)
        # drop spaces whose linked models are ALL non-text (image gen UIs,
        # 3d viewers, tts demos…); spaces without linked models stay in —
        # the name filter and the live probe decide for them
        for sid in [s for s, e in entries.items()
                    if e.get('pipeline_tags')
                    and not (set(e['pipeline_tags']) & TEXT_PIPELINES)]:
            del entries[sid]

        def _probe_batch(sids: List[str]) -> None:
            """Probe a wave, reusing any cached record (ok or fresh fail)."""
            todo = []
            for sid in sids:
                fresh, rec = _probe_fresh(sid)
                if fresh and rec and rec.get('ok'):
                    entries[sid]['probe'] = rec
                    entries[sid]['ok'] = True
                elif fresh:
                    entries[sid]['probe'] = rec or {}
                else:
                    todo.append(sid)
            if not todo:
                return
            with ThreadPoolExecutor(max_workers=8) as pool:
                futs = {pool.submit(_probe_space, sid,
                                    entries[sid]['sub'], tok): sid
                        for sid in todo}
                for fut in as_completed(futs):
                    sid = futs[fut]
                    try:
                        rec = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        rec = {'ts': time.time(), 'ok': False,
                               'err': str(exc)[:120]}
                    _store_probe(sid, rec)
                    entries[sid]['probe'] = rec
                    entries[sid]['ok'] = bool(rec.get('ok'))

        # live probe of the best candidates, reusing cached results; then
        # fill-up waves: usage-ranked media UIs fail the probe cheaply
        # (no-chat-api = one config GET, no GPU burn) but must not crowd
        # usable chat spaces out of the probe window
        order = [sid for sid, _ in sorted(entries.items(),
                                          key=lambda kv: _score(kv[1]),
                                          reverse=True)]
        _probe_batch(order[:PROBE_TOP])
        probed = PROBE_TOP
        while (sum(1 for e in entries.values() if e.get('ok')) < PROBE_TARGET
               and probed < min(MAX_PROBES_PER_RUN, len(order))):
            _probe_batch(order[probed:probed + PROBE_WAVE])
            probed += PROBE_WAVE
        # one more pass: any previously-ok cached probe counts
        for sid, entry in entries.items():
            if not entry.get('ok'):
                fresh, rec = _probe_fresh(sid)
                if fresh and rec and rec.get('ok'):
                    entry['probe'] = rec
                    entry['ok'] = True
        working = {sid: e for sid, e in entries.items() if e.get('ok')}
        models: List[Dict[str, Any]] = []
        entries_out: Dict[str, Dict[str, Any]] = {}
        for sid, e in sorted(working.items(),
                             key=lambda kv: _score(kv[1]), reverse=True):
            slug = re.sub(r'[^a-z0-9]+', '-', sid.lower()).strip('-')
            entries_out[slug] = e
            models.append({
                'id': f'hf-{slug}',
                'upstream_model': sid,
                'thinking_enabled': False,
                'search_enabled': False,
                'vision': bool(e.get('probe', {}).get('msg_mm')),
                'context_length': 16384,
                'max_output_tokens': 2048,
                'extra': {
                    'hf_space': sid,
                    'score': round(_score(e)),
                    'hw': e.get('hw') or '',
                    'likes': e.get('likes') or 0,
                    'downloads': e.get('downloads') or 0,
                },
            })
        # virtual router model: best working space first, rotates on quota
        if working:
            models.append({
                'id': 'hf-auto',
                'upstream_model': '@auto',
                'thinking_enabled': False,
                'search_enabled': False,
                'vision': False,
                'context_length': 16384,
                'max_output_tokens': 2048,
                'extra': {'hf_space': '(best available space by usage)',
                          'score': round(max(_score(e) for e
                                             in working.values())),
                },
            })
        if not models and _CACHE.get('models'):
            # transient total failure (upstream throttling): keep the last
            # good model list and retry in a minute instead of serving an
            # empty provider until the next TTL window
            _CACHE_LOCK.acquire()
            try:
                _CACHE['ts'] = time.time() - (DISCOVERY_TTL - 60.0)
            finally:
                _CACHE_LOCK.release()
            return _CACHE
        _CACHE_LOCK.acquire()
        try:
            _CACHE.update(ts=time.time(), models=models, entries=entries_out)
        finally:
            _CACHE_LOCK.release()
        return _CACHE


# --------------------------------------------------------------------------
# provider
# --------------------------------------------------------------------------


class HuggingFaceProvider(Provider):
    name = 'huggingface'

    def available(self, auth_key: Optional[str] = None) -> bool:
        return True  # anonymous access is first-class

    def list_models(self, auth_key: Optional[str] = None
                    ) -> List[Dict[str, Any]]:
        return list(_discover(auth_key=auth_key).get('models') or [])

    # -- streaming -------------------------------------------------------

    def _resolve(self, model: str) -> Tuple[str, Dict[str, Any]]:
        cache = _discover()
        entries = cache.get('entries') or {}
        if model == 'hf-auto' or model == 'auto':
            ranked = sorted(entries.items(),
                            key=lambda kv: _score(kv[1]), reverse=True)
            for slug, entry in ranked:
                if entry.get('ok'):
                    return slug, entry
            raise ProviderUnavailableError(
                'no working huggingface space right now')
        slug = re.sub(r'^hf-', '', model)
        entry = entries.get(slug)
        if not entry:
            raise ProviderError(f'unknown huggingface space: {model}')
        return slug, entry

    def stream(self, prompt: str, *, model: str,
               thinking_enabled: bool = False, search_enabled: bool = False,
               temperature: Optional[float] = None,
               max_tokens: Optional[int] = None,
               images: Optional[List[Dict[str, Any]]] = None,
               image_generation: bool = False,
               auth_key: Optional[str] = None,
               no_proxy: bool = False
               ) -> Generator[Dict[str, Any], None, None]:
        tok = _token(auth_key)
        tried: List[str] = []
        # auto rotation: walk working spaces by score until one streams
        for slug, entry in self._rotation(model):
            tried.append(slug)
            try:
                yield from self._stream_space(
                    slug, entry, prompt, thinking_enabled=thinking_enabled,
                    temperature=temperature, max_tokens=max_tokens,
                    auth_key=auth_key, no_proxy=no_proxy, tok=tok)
                return
            except (ProviderRateLimitError, ProviderUnavailableError) as exc:
                if model not in ('hf-auto', 'auto') or len(tried) >= 3:
                    raise
                logger.info('hf space %s failed (%s); rotating',
                            slug, str(exc)[:80])
                continue
        raise ProviderUnavailableError(
            f'all huggingface spaces failed: {", ".join(tried)}')

    def _rotation(self, model: str) -> List[Tuple[str, Dict[str, Any]]]:
        cache = _discover()
        entries = cache.get('entries') or {}
        if model in ('hf-auto', 'auto'):
            return sorted(entries.items(),
                          key=lambda kv: _score(kv[1]), reverse=True)
        slug = re.sub(r'^hf-', '', model)
        entry = entries.get(slug)
        if not entry:
            raise ProviderError(f'unknown huggingface space: {model}')
        return [(slug, entry)]

    # -- single space execution ------------------------------------------

    def _stream_space(self, slug: str, entry: Dict[str, Any], prompt: str, *,
                      thinking_enabled: bool, temperature: Optional[float],
                      max_tokens: Optional[int], auth_key: Optional[str],
                      no_proxy: bool, tok: str
                      ) -> Generator[Dict[str, Any], None, None]:
        rec = entry.get('probe') or {}
        if not rec.get('ok'):
            # stale probe: try a quick refresh before giving up
            fresh, rec2 = _probe_fresh(entry.get('space_id', slug))
            if not (fresh and rec2 and rec2.get('ok')):
                fresh_probe = _probe_space(entry.get('space_id', slug),
                                           entry.get('sub', ''), tok)
                _store_probe(entry.get('space_id', slug), fresh_probe)
                if not fresh_probe.get('ok'):
                    raise ProviderUnavailableError(
                        f"space {entry.get('space_id')} probe failed: "
                        f"{fresh_probe.get('err')}")
                rec = fresh_probe
                entry['probe'] = rec
                entry['ok'] = True
        sub = entry.get('sub') or ''
        prefix = rec.get('prefix') or '/gradio_api'
        fn_index = int(rec.get('fn_index') or 0)
        out_types = rec.get('out_types') or []
        cfg = _gradio_config(sub)
        if not cfg:
            raise ProviderUnavailableError(f'space {sub} config unavailable')
        pick = _pick_chat_api(cfg)
        if not pick:
            raise ProviderUnavailableError(f'space {sub} lost its chat api')
        api_name, idx_now, dep = pick
        comps = {c.get('id'): c for c in (cfg.get('components') or [])}
        params = _info_params(sub, api_name)
        data, _names = _build_join_data(dep, comps, prompt)
        if fn_index != idx_now:
            fn_index = idx_now  # config shifted since the probe
        self._apply_params(data, dep, comps, params, thinking_enabled,
                           temperature, max_tokens)
        sess = _session_hash()
        join_prefix, join_err = _join_prefix(sub, fn_index, data, tok, sess)
        if join_prefix is None:
            raise ProviderUnavailableError(f'space {sub} join failed: '
                                           f'{join_err}')
        prefix = join_prefix or prefix
        resp = _sse_stream(sub, sess, prefix, tok, SSE_IDLE_TIMEOUT)
        decoder = codecs.getincrementaldecoder('utf-8')('replace')
        buf = ''
        last = ''
        t_last_event = time.time()
        for chunk in resp.iter_content(chunk_size=None):
            buf += decoder.decode(chunk or b'')
            if '\n' not in buf:
                if time.time() - t_last_event > SSE_IDLE_TIMEOUT:
                    raise ProviderUnavailableError(
                        f'space {sub} stream stalled (no SSE data)')
                continue
            t_last_event = time.time()
            while '\n' in buf:
                line, buf = buf.split('\n', 1)
                ev = parse_sse_data(line.encode('utf-8'))
                if not ev:
                    continue
                m = ev.get('msg')
                if m == 'process_generating':
                    text = _chat_text_from_output(ev.get('output') or {},
                                                  out_types)
                    delta = ''
                    if text and text.startswith(last):
                        delta = text[len(last):]
                    elif text and not last:
                        delta = text
                    elif text:
                        # restructure: emit the new snapshot fully
                        delta = text
                        last = ''
                    if delta:
                        last += delta
                        yield {'content': delta, 'type': 'text',
                               'finish_reason': None}
                elif m == 'process_completed':
                    out = ev.get('output') or {}
                    err = out.get('error')
                    if err:
                        msg = str(err)
                        if _ZEROGPU_QUOTA_RE.search(msg):
                            raise ProviderRateLimitError(
                                f'{sub}: {msg[:160]}', retry_after=1800)
                        if _ILLEGAL_DURATION_RE.search(msg):
                            self._mark_bad(slug, entry, 'illegal-duration')
                            raise ProviderUnavailableError(
                                f'{sub}: {msg[:160]}')
                        raise ProviderUnavailableError(
                            f'{sub}: {msg[:160]}')
                    if ev.get('success') is False:
                        self._mark_bad(slug, entry, 'completed-false')
                        raise ProviderUnavailableError(
                            f'{sub}: run failed without error message')
                    text = _chat_text_from_output(out, out_types)
                    if text and text.startswith(last) and text != last:
                        yield {'content': text[len(last):], 'type': 'text',
                               'finish_reason': None}
                    elif text and not text.startswith(last) and text:
                        yield {'content': text, 'type': 'text',
                               'finish_reason': None}
                    yield {'content': '', 'type': 'text',
                           'finish_reason': 'stop'}
                    return
                elif m == 'unexpected_error':
                    raise ProviderUnavailableError(
                        f"{sub}: unexpected error: "
                        f"{str(ev.get('message'))[:140]}")
                elif m in ('close_stream',):
                    yield {'content': '', 'type': 'text',
                           'finish_reason': 'stop'}
                    return
            if time.time() - t_last_event > SSE_IDLE_TIMEOUT:
                raise ProviderUnavailableError(
                    f'space {sub} stream stalled')
        # stream ended without completion
        if last:
            yield {'content': '', 'type': 'text', 'finish_reason': 'stop'}
            return
        raise ProviderUnavailableError(f'space {sub} ended without output')

    def _apply_params(self, data: List[Any], dep: Dict[str, Any],
                      comps: Dict[int, Dict[str, Any]],
                      params: List[Dict[str, Any]],
                      thinking_enabled: bool,
                      temperature: Optional[float],
                      max_tokens: Optional[int]) -> None:
        """Fill temperature / top_p / max_tokens / reasoning slots."""
        inputs = dep.get('inputs') or []
        pos = 0
        for cid in inputs:
            c = comps.get(cid) or {}
            ctype = c.get('type')
            if ctype in ('state', 'browserstate'):
                continue
            if pos >= len(data):
                break
            name = ''
            if pos < len(params):
                name = str(params[pos].get('parameter_name') or '').lower()
            label = str((c.get('props') or {}).get('label') or '').lower()
            nm = name or label
            if 'temp' in nm and isinstance(data[pos], (int, float)):
                data[pos] = 1.0 if temperature is None else max(
                    0.0, min(2.0, float(temperature)))
            elif 'top_p' in nm and isinstance(data[pos], (int, float)):
                data[pos] = 0.95
            elif 'max' in nm and isinstance(data[pos], (int, float)):
                hi = (c.get('props') or {}).get('maximum')
                want = int(max_tokens or 2048)
                data[pos] = min(want, int(hi)) if hi else want
            elif 'reason' in nm and isinstance(data[pos], str):
                data[pos] = 'medium' if thinking_enabled else 'off'
            pos += 1

    def _mark_bad(self, slug: str, entry: Dict[str, Any], err: str) -> None:
        space_id = entry.get('space_id') or f'{slug}'
        _store_probe(space_id, {'ts': time.time(), 'ok': False, 'err': err})
        entry['ok'] = False
