"""Self-maintenance: detect upstream web-app changes and auto-patch provider code.

The three providers are reverse-engineered clients of live web apps
(chat.deepseek.com, gemini.google.com, chatgpt.com). When one of those sites
changes its endpoints/payload shapes, the matching provider module breaks.
This module closes the loop autonomously:

  1. PROBE   a background daemon health-checks every configured provider on a
             TTL (cheap calls: PoW-challenge fetch / model discovery). Each
             failure is classified:

               ok          all good
               network     transient (proxy/reset/timeout) — ignored
               rate        rate limited — ignored
               auth        credentials rejected — handed to dsk.refresher
               structural  404 / unexpected payload shape / parse errors —
                           the upstream probably changed -> heal candidate

  2. HEAL    after DSF_SELFHEAL_TRIGGER consecutive structural failures:
               a. collect evidence: the provider module source, recent error
                  messages and LIVE excerpts fetched from the upstream site
                  (its JS bundles are grepped for API paths, header names,
                  etc.)
               b. ask a free LLM to rewrite the module. Fixer chain (first
                  configured wins):
                    DSF_SELFHEAL_FIXER_*   any OpenAI-compatible endpoint
                    DSF_OPENROUTER_API_KEY OpenRouter free models
                    local-self             this very server via 127.0.0.1,
                                           routed to a *working* provider
               c. validate the proposed file in a throwaway subprocess (real
                  import + real probe against the live site)
               d. only then replace the module on disk (timestamped backup in
                  data/selfheal/backups/), hot-reload it and re-probe. On any
                  failure the backup is restored automatically.

Guardrails: whitelisted files only, attempt/incident caps and per-provider
cooldowns, JSONL audit trail in data/selfheal/history.jsonl, everything can
turned off with DSF_SELFHEAL=false. The LLM never sees or touches credentials.

Configuration (env):
    DSF_SELFHEAL                  false disables everything (default on)
    DSF_SELFHEAL_PROBE_TTL        seconds between probe cycles (default 600)
    DSF_SELFHEAL_TRIGGER          consecutive structural failures before a
                                  heal is attempted (default 3)
    DSF_SELFHEAL_COOLDOWN         seconds between heal incidents per provider
                                  (default 3600)
    DSF_SELFHEAL_MAX_ATTEMPTS     LLM patch attempts per incident (default 3)
    DSF_SELFHEAL_MAX_INCIDENTS    heal incidents per provider per day
                                  (default 4)
    DSF_SELFHEAL_EXCLUDE          providers never probed/healed
    DSF_SELFHEAL_FIXER_BASE_URL / _API_KEY / _MODELS
                                  generic OpenAI-compatible fixer endpoint
    DSF_OPENROUTER_API_KEY        enables the OpenRouter free-model fallback
    DSF_SELFHEAL_OPENROUTER_MODELS  comma list (see default in code)
    DSF_SELFHEAL_LOCAL            allow using this server's own routes as the
                                  fixer (default true)

CLI:
    python -m dsk.selfheal probe          run one probe cycle now
    python -m dsk.selfheal status         current state
    python -m dsk.selfheal heal gemini    force a heal incident
    python -m dsk.selfheal _validate gemini /tmp/candidate.py   (internal)
"""

import importlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .providers.base import provider_enabled

_BASE = Path(__file__).resolve().parent

# Files the fixer LLM is allowed to rewrite (everything else is off-limits).
HEALABLE: Dict[str, Path] = {
    'deepseek': _BASE / 'api.py',
    'gemini': _BASE / 'providers' / 'gemini_provider.py',
    'chatgpt': _BASE / 'providers' / 'chatgpt_provider.py',
    'claude': _BASE / 'providers' / 'claude_provider.py',
    'grok': _BASE / 'providers' / 'grok_provider.py',
    'mistral': _BASE / 'providers' / 'mistral_provider.py',
    'qwen': _BASE / 'providers' / 'qwen_provider.py',
    'kimi': _BASE / 'providers' / 'kimi_provider.py',
    'copilot': _BASE / 'providers' / 'copilot_provider.py',
    'perplexity': _BASE / 'providers' / 'perplexity_provider.py',
    'glm': _BASE / 'providers' / 'glm_provider.py',
}

_MODULE_NAMES = {
    'deepseek': 'dsk.api',
    'gemini': 'dsk.providers.gemini_provider',
    'chatgpt': 'dsk.providers.chatgpt_provider',
    'claude': 'dsk.providers.claude_provider',
    'grok': 'dsk.providers.grok_provider',
    'mistral': 'dsk.providers.mistral_provider',
    'qwen': 'dsk.providers.qwen_provider',
    'kimi': 'dsk.providers.kimi_provider',
    'copilot': 'dsk.providers.copilot_provider',
    'perplexity': 'dsk.providers.perplexity_provider',
    'glm': 'dsk.providers.glm_provider',
}

_PROVIDER_MODULES = {
    'deepseek': 'dsk.providers.deepseek_provider',
    'gemini': 'dsk.providers.gemini_provider',
    'chatgpt': 'dsk.providers.chatgpt_provider',
    'claude': 'dsk.providers.claude_provider',
    'grok': 'dsk.providers.grok_provider',
    'mistral': 'dsk.providers.mistral_provider',
    'qwen': 'dsk.providers.qwen_provider',
    'kimi': 'dsk.providers.kimi_provider',
    'copilot': 'dsk.providers.copilot_provider',
    'perplexity': 'dsk.providers.perplexity_provider',
    'glm': 'dsk.providers.glm_provider',
}

# Provider class name inside each module (used by probe/configured).
_PROVIDER_CLASSES = {
    'deepseek': 'DeepSeekProvider',
    'gemini': 'GeminiWebProvider',
    'chatgpt': 'ChatGPTProvider',
    'claude': 'ClaudeWebProvider',
    'grok': 'GrokProvider',
    'mistral': 'MistralProvider',
    'qwen': 'QwenProvider',
    'kimi': 'KimiProvider',
    'copilot': 'CopilotProvider',
    'perplexity': 'PerplexityProvider',
    'glm': 'GlmProvider',
}

# Markers grepped out of the upstream's JS bundles as fixer evidence.
_EVIDENCE_PATTERNS = {
    'deepseek': [r'api/v0/[A-Za-z0-9_/{}$.\-]{0,60}', r'x-ds-[A-Za-z0-9_-]+',
                 r'pow[A-Za-z_]{0,20}'],
    'chatgpt': [r'backend-api/[A-Za-z0-9_/{}$.\-]{0,60}', r'accessToken',
                r'require-account', r'/conversation'],
    'gemini': [r'BardChatUi[A-Za-z0-9_/.$\-]{0,80}', r'StreamGenerate',
               r'batchexecute', r'SNlM0e', r'assistant\.lamda\.[A-Za-z]+'],
    'claude': [r'chat_conversations[A-Za-z0-9_/{}$.\-]{0,60}', r'sessionKey',
               r'completion[A-Za-z]{0,20}', r'anthropic\-[a-z\-]+',
               r'content_block_[a-z]+'],
    'grok': [r'rest/app-chat[A-Za-z0-9_/{}$.\-]{0,60}', r'conversations/new',
             r'sso\-rw?', r'x\-xai\-[a-z\-]+', r'modeId'],
    'mistral': [r'trpc[A-Za-z0-9_/{}$.\-]{0,60}', r'message\.newChat',
                r'/api/chat', r'stable_anon_id'],
    'qwen': [r'api/v2/chats[A-Za-z0-9_/{}$.\-]{0,60}', r'chat/completions',
             r'bx\-[a-z]+', r'feature_config'],
    'kimi': [r'kimi\.gateway\.chat[A-Za-z0-9_.]{0,80}', r'ChatService',
             r'application/connect\+json', r'multiStage'],
    'copilot': [r'c/api/chat[A-Za-z0-9_/{}$.\-]{0,60}', r'participantId',
                r'chainOfThought', r'websocket'],
    'perplexity': [r'rest/sse/perplexity_ask', r'model_preference',
                   r'ask_text', r'markdown_block'],
    'glm': [r'api/chat/completions', r'assistant/stream', r'refresh_token',
            r'chatglm', r'delta_content'],
}


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, '').strip().lower()
    if not raw:
        return default
    return raw in ('1', 'true', 'yes', 'on')


def _probe_ttl() -> float:
    return max(60.0, float(os.getenv('DSF_SELFHEAL_PROBE_TTL', '600') or 600))


def _trigger() -> int:
    return max(1, int(os.getenv('DSF_SELFHEAL_TRIGGER', '3') or 3))


def _cooldown() -> float:
    return max(60.0, float(os.getenv('DSF_SELFHEAL_COOLDOWN', '3600') or 3600))


def _max_attempts() -> int:
    return max(1, int(os.getenv('DSF_SELFHEAL_MAX_ATTEMPTS', '3') or 3))


def _max_incidents() -> int:
    return max(1, int(os.getenv('DSF_SELFHEAL_MAX_INCIDENTS', '4') or 4))


def _heal_dir() -> Path:
    base = (os.getenv('DSF_SELFHEAL_DIR') or os.getenv('COOKIES_DIR')
            or str(_BASE.parent / 'data'))
    path = Path(base) / 'selfheal'
    (path / 'backups').mkdir(parents=True, exist_ok=True)
    return path


def _history_path() -> Path:
    return _heal_dir() / 'history.jsonl'


class _State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.started = False
        self.healing: Dict[str, bool] = {}
        self.probes: Dict[str, Dict[str, Any]] = {}
        self.last_heal: Dict[str, Any] = {}
        self.incidents: Dict[str, Tuple[str, int]] = {}  # provider -> (day, n)


_STATE = _State()


def _rotate_history(path) -> None:
    """Keep the JSONL log bounded: over ~1 MB keep only the newest 2000 lines."""
    try:
        if path.exists() and path.stat().st_size > 1_000_000:
            lines = path.read_text(encoding='utf-8').splitlines()
            path.write_text('\n'.join(lines[-2000:]) + '\n', encoding='utf-8')
    except OSError:
        pass


def _log_history(provider: str, event: str, detail: Any = '') -> None:
    entry = {'ts': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
             'provider': provider, 'event': event,
             'detail': str(detail)[:2000]}
    with _STATE.lock:
        _STATE.last_heal[provider] = entry
    try:
        _rotate_history(_history_path())
        with _history_path().open('a', encoding='utf-8') as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + '\n')
    except OSError:
        pass


# --------------------------------------------------------------------- probes
def _probe_once(name: str) -> Tuple[str, str]:
    """Cheap live check of one provider using freshly built client objects.

    Returns (status, detail) with status in ok|auth|network|rate|structural.
    """
    try:
        if name == 'deepseek':
            module = importlib.import_module(_PROVIDER_MODULES[name])
            provider = module.DeepSeekProvider()
            api = provider._get_api()
            if api is None:
                return 'auth', 'no DeepSeek credentials configured'
            api._get_pow_challenge()
            return 'ok', 'pow challenge ok'
        if name == 'qwen':
            # The model picker is anonymous, so listing models proves
            # nothing — validate the session token against /api/v1/auths.
            from .providers import qwen_provider as qp
            verdict = qp.validate_token()
            if verdict == 'ok':
                return 'ok', 'session token valid'
            if verdict == 'unauth':
                if qp._relay_enabled():
                    return 'ok', 'no token; guest browser relay serves qwen'
                return 'auth', 'no working chat.qwen.ai session token'
            if verdict.startswith('unreachable'):
                return 'network', verdict
            return 'structural', verdict
        module = importlib.import_module(_PROVIDER_MODULES[name])
        provider = getattr(module, _PROVIDER_CLASSES[name])()
        if not provider.available():
            return 'auth', 'no credentials configured'
        models = provider.list_models()
        if not models:
            return 'structural', f'{name} discovered 0 models (unexpected payload?)'
        return 'ok', f'{len(models)} models'
    except Exception as e:  # noqa: BLE001 - classification is the point
        return _classify(e), f'{type(e).__name__}: {e}'[:500]


def _classify(exc: BaseException) -> str:
    text = f'{type(exc).__name__}: {exc}'
    low = text.lower()
    try:
        from .providers.base import (ProviderAuthError, ProviderRateLimitError,
                                     ProviderUnavailableError)
        if isinstance(exc, ProviderAuthError):
            return 'auth'
        if isinstance(exc, ProviderRateLimitError):
            return 'rate'
        if isinstance(exc, ProviderUnavailableError):
            return 'network'
    except Exception:  # pragma: no cover
        pass
    if any(m in low for m in ('401', '403', 'unauthorized', 'forbidden',
                              'no credentials', 'expired', 'sign in',
                              'session', 'authorization', 'authentication',
                              'invalid token')):
        return 'auth'
    if any(m in low for m in ('429', 'rate limit', 'too many requests')):
        return 'rate'
    if any(m in low for m in ('connection', 'timed out', 'timeout', 'proxy',
                              'reset by peer', 'curl', 'network', 'resolve',
                              'unreachable', 'ssl')):
        return 'network'
    return 'structural'


def _provider_configured(name: str) -> bool:
    try:
        module = importlib.import_module(_PROVIDER_MODULES[name])
        return bool(getattr(module, _PROVIDER_CLASSES[name])().available())
    except Exception:
        return False


# ------------------------------------------------------------------- evidence
def _fetch(url: str, timeout: int = 25) -> str:
    try:
        from curl_cffi import requests as cffi
        resp = cffi.get(url, timeout=timeout, impersonate='chrome120')
    except ImportError:
        import requests
        headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
                                 '(KHTML, like Gecko) Chrome/126.0 Safari/537.36'}
        resp = requests.get(url, timeout=timeout, headers=headers)
    if resp.status_code != 200:
        raise RuntimeError(f'HTTP {resp.status_code} for {url[:120]}')
    return resp.text


def _script_urls(html: str, base: str) -> List[str]:
    urls = re.findall(r'src=["\'](/{1,2}[^"\']+\.js)["\']', html)
    out: List[str] = []
    for u in urls[:8]:
        if u.startswith('//'):
            out.append('https:' + u)
        elif u.startswith('/'):
            out.append(base.rstrip('/') + u)
        else:
            out.append(u)
    return out


def _upstream_evidence(name: str, cap: int = 7000) -> str:
    """Fetch the upstream web app and grep its JS for API-path style markers."""
    base = {'deepseek': 'https://chat.deepseek.com',
            'gemini': 'https://gemini.google.com',
            'chatgpt': 'https://chatgpt.com',
            'claude': 'https://claude.ai',
            'grok': 'https://grok.com',
            'mistral': 'https://chat.mistral.ai',
            'qwen': 'https://chat.qwen.ai',
            'kimi': 'https://www.kimi.com',
            'copilot': 'https://copilot.microsoft.com',
            'perplexity': 'https://www.perplexity.ai',
            'glm': 'https://chat.z.ai'}[name]
    patterns = [re.compile(p) for p in _EVIDENCE_PATTERNS[name]]
    chunks: List[str] = []
    seen: set = set()
    total = 0
    try:
        pages = [base]
        html = _fetch(base)
        pages += _script_urls(html, base)[:3]
        for url in pages:
            try:
                text = html if url == base else _fetch(url)
            except Exception as e:  # noqa: BLE001
                chunks.append(f'-- {url[:120]}: fetch failed ({e})')
                continue
            for pat in patterns:
                for match in pat.finditer(text):
                    frag = text[max(0, match.start() - 80):match.end() + 120]
                    frag = re.sub(r'\s+', ' ', frag).strip()
                    key = frag[:100]
                    if key in seen or total > cap:
                        continue
                    seen.add(key)
                    chunks.append(f'-- {url[:100]}: ...{frag}...')
                    total += len(frag)
    except Exception as e:  # noqa: BLE001
        chunks.append(f'-- evidence collection failed: {e}')
    return '\n'.join(chunks)[:cap + 500] or '(no evidence collected)'


# --------------------------------------------------------------- fixer chain
def _fixer_candidates() -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    base = os.getenv('DSF_SELFHEAL_FIXER_BASE_URL', '').strip().rstrip('/')
    if base:
        key = os.getenv('DSF_SELFHEAL_FIXER_API_KEY', '').strip()
        for m in [x.strip() for x in os.getenv('DSF_SELFHEAL_FIXER_MODELS', '').split(',')
                  if x.strip()]:
            out.append({'base': base, 'key': key, 'model': m, 'label': f'fixer:{m}'})
    orkey = (os.getenv('DSF_OPENROUTER_API_KEY', '')
             or os.getenv('DSF_SELFHEAL_OPENROUTER_KEY', '')).strip()
    if orkey:
        defaults = ('deepseek/deepseek-chat-v3.1:free,'
                    'meta-llama/llama-3.3-70b-instruct:free,'
                    'qwen/qwen3-235b-a22b:free')
        for m in [x.strip() for x in
                  os.getenv('DSF_SELFHEAL_OPENROUTER_MODELS', defaults).split(',')
                  if x.strip()]:
            out.append({'base': 'https://openrouter.ai/api/v1', 'key': orkey,
                        'model': m, 'label': f'openrouter:{m}'})
    if _env_bool('DSF_SELFHEAL_LOCAL', True):
        out.append({'base': f"http://127.0.0.1:{os.getenv('DSF_PORT', '8000')}",
                    'key': os.getenv('DSF_API_KEY', '').strip(),
                    'model': '__auto__', 'label': 'local-self'})
    return out


def _pick_local_model(broken: str) -> Optional[str]:
    """A route served by a provider OTHER than the broken one (else any)."""
    try:
        from dsk.openai_server import ROUTER  # lazy: only exists in-server
        routes = list(ROUTER.routes.values())
        for r in routes:
            if r.provider_name != broken:
                return r.model_id
        return routes[0].model_id if routes else None
    except Exception:
        return None


def _fixer_chat(messages: List[Dict[str, str]], broken: str) -> Tuple[str, str]:
    """Try every configured fixer endpoint; return (content, label)."""
    import requests
    errors: List[str] = []
    for cand in _fixer_candidates():
        model = cand['model']
        if model == '__auto__':
            model = _pick_local_model(broken)
            if not model:
                errors.append(f"{cand['label']}: no local routes")
                continue
        try:
            headers = {'Content-Type': 'application/json'}
            if cand['key']:
                headers['Authorization'] = f"Bearer {cand['key']}"
            resp = requests.post(f"{cand['base']}/chat/completions",
                                 json={'model': model, 'messages': messages,
                                       'temperature': 0.1, 'max_tokens': 16000,
                                       'stream': False},
                                 headers=headers, timeout=420)
            if resp.status_code != 200:
                errors.append(f"{cand['label']}: HTTP {resp.status_code} "
                              f"{resp.text[:200]}")
                continue
            data = resp.json()
            content = str(data['choices'][0]['message']['content'] or '')
            if content.strip():
                return content, f"{cand['label']} ({model})"
            errors.append(f"{cand['label']}: empty completion")
        except Exception as e:  # noqa: BLE001
            errors.append(f"{cand['label']}: {type(e).__name__}: {e}"[:300])
    raise RuntimeError('all fixers failed: ' + ' | '.join(errors)[:1000])


def _build_prompt(name: str, source: str, errors: List[str],
                  evidence: str, feedback: List[str]) -> List[Dict[str, str]]:
    path = HEALABLE[name]
    feedback_text = '\n\n'.join(f'- {f}' for f in feedback) if feedback \
        else '- (none yet)'
    system = (
        'You are an expert Python engineer maintaining a reverse-engineered '
        'client for a live web app. The web app changed and the client broke. '
        'Rewrite the module so it works against the CURRENT upstream, using '
        'the supplied live evidence (JS bundle excerpts, error messages). '
        'Rules: output ONLY the complete patched Python file inside one '
        '```python fenced block; keep every public class/function name and '
        'signature; change as little as possible; never invent credentials; '
        'keep the module importable with its existing dependencies.'
    )
    user = (
        f'File to fix: {path.name} (module {_MODULE_NAMES[name]})\n\n'
        f'=== CURRENT SOURCE ===\n{source}\n\n'
        f'=== RECENT ERRORS ===\n' + '\n'.join(errors[-5:]) + '\n\n'
        f'=== LIVE UPSTREAM EVIDENCE ===\n{evidence}\n\n'
        f'=== VALIDATION FEEDBACK FROM PREVIOUS ATTEMPTS ===\n{feedback_text}\n\n'
        'Return the complete fixed file now.'
    )
    return [{'role': 'system', 'content': system},
            {'role': 'user', 'content': user}]


def _extract_code(text: str) -> Optional[str]:
    blocks = re.findall(r'```(?:python|py)?\s*\n(.*?)```', text, re.DOTALL)
    for block in blocks:
        code = block.strip()
        try:
            compile(code, '<candidate>', 'exec')
            return code
        except SyntaxError:
            continue
    text = text.strip()
    if text.startswith(('"""', '#', 'import ', 'from ', 'class ')):
        try:
            compile(text, '<candidate>', 'exec')
            return text
        except SyntaxError:
            pass
    return None


# ------------------------------------------------------------- validation
def _validate_subprocess(name: str, candidate: Path) -> Tuple[bool, str]:
    """Import + live-probe the candidate file in a throwaway interpreter."""
    cmd = [sys.executable, '-m', 'dsk.selfheal', '_validate', name, str(candidate)]
    try:
        proc = subprocess.run(cmd, cwd=str(_BASE.parent), timeout=300,
                              capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        return False, 'validation subprocess timed out after 300s'
    if proc.returncode == 0:
        return True, (proc.stdout or '').strip()[-500:]
    detail = (proc.stderr or proc.stdout or '').strip()
    return False, detail[-2000:] if detail else f'exit code {proc.returncode}'


def _cmd_validate(name: str, candidate: str) -> int:
    """Subprocess entry: load the candidate AS the real module, then probe."""
    try:
        import dsk  # noqa: F401  (parent package first)
        if name == 'deepseek':
            import dsk.providers  # noqa: F401
        target = _MODULE_NAMES[name]
        spec = importlib.util.spec_from_file_location(target, candidate)
        module = importlib.util.module_from_spec(spec)
        sys.modules[target] = module
        spec.loader.exec_module(module)
        status, detail = _probe_once(name)
        print(f'validate[{name}]: {status}: {detail}')
        return 0 if status == 'ok' else 3
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 3


# ------------------------------------------------------------------ reload
def _reload(name: str) -> None:
    if name == 'deepseek':
        api = importlib.import_module('dsk.api')
        importlib.reload(api)
    provider_mod = importlib.import_module(_PROVIDER_MODULES[name])
    importlib.reload(provider_mod)
    try:  # rebuild live provider instances + force re-discovery (server mode)
        from dsk.openai_server import ROUTER
        ROUTER.reload_providers()
    except Exception:  # CLI mode: no running server
        pass


# -------------------------------------------------------------------- heal
def _incident_budget(name: str) -> bool:
    today = time.strftime('%Y-%m-%d')
    day, n = _STATE.incidents.get(name, (today, 0))
    if day != today:
        day, n = today, 0
    if n >= _max_incidents():
        return False
    _STATE.incidents[name] = (day, n + 1)
    return True


def heal(name: str, reason: str = '',
         fixer: Optional[Callable[[str], str]] = None,
         validate: Optional[Callable[[str], Tuple[bool, str]]] = None) -> Dict[str, Any]:
    """Attempt to auto-patch one provider module. Returns a result dict.

    `fixer`/`validate` are injectable for tests.
    """
    if not _env_bool('DSF_SELFHEAL', True):
        return {'healed': False, 'skipped': 'selfheal disabled'}
    if name not in HEALABLE:
        return {'healed': False, 'skipped': f'{name} is not healable'}
    excl = {e.strip().lower() for e in
            os.getenv('DSF_SELFHEAL_EXCLUDE', '').split(',') if e.strip()}
    if name in excl:
        return {'healed': False, 'skipped': f'{name} excluded'}
    with _STATE.lock:
        if _STATE.healing.get(name):
            return {'healed': False, 'skipped': 'heal already in progress'}
        last = _STATE.last_heal.get(name, {})
        if last.get('event') == 'incident-exhausted' and \
                time.time() - _last_ts(last) < _cooldown():
            return {'healed': False, 'skipped': 'cooldown'}
        _STATE.healing[name] = True
    try:
        return _heal_locked(name, reason, fixer, validate)
    finally:
        with _STATE.lock:
            _STATE.healing[name] = False


def _last_ts(entry: Dict[str, Any]) -> float:
    try:
        return time.mktime(time.strptime(entry['ts'][:19], '%Y-%m-%dT%H:%M:%S'))
    except Exception:
        return 0.0


def _heal_locked(name: str, reason: str,
                 fixer: Optional[Callable[[str], str]],
                 validate: Optional[Callable[[str, Path], Tuple[bool, str]]]
                 ) -> Dict[str, Any]:
    if not _incident_budget(name):
        _log_history(name, 'incident-budget-exhausted', reason)
        return {'healed': False, 'skipped': 'daily incident budget exhausted'}
    target = HEALABLE[name]
    source = target.read_text(encoding='utf-8')
    with _STATE.lock:
        errors = list((_STATE.probes.get(name, {}).get('errors') or [])[-5:])
    if not errors:
        errors = [reason or 'structural failures observed']
    _log_history(name, 'incident-start', reason)
    evidence = _upstream_evidence(name)
    feedback: List[str] = []
    attempts: List[Dict[str, Any]] = []
    max_attempts = _max_attempts()
    for i in range(1, max_attempts + 1):
        messages = _build_prompt(name, source, errors, evidence, feedback)
        try:
            if fixer is not None:
                raw, label = fixer(messages[-1]['content']), 'test-fixer'
            else:
                raw, label = _fixer_chat(messages, name)
        except Exception as e:  # noqa: BLE001
            attempts.append({'attempt': i, 'error': str(e)[:500]})
            _log_history(name, 'fixer-unavailable', e)
            break
        code = _extract_code(raw)
        if code is None:
            feedback.append(f'attempt {i}: no parsable ```python block with '
                            f'compilable code in the fixer reply')
            attempts.append({'attempt': i, 'fixer': label, 'error': 'no code'})
            continue
        candidate = _heal_dir() / f'{name}.candidate.py'
        candidate.write_text(code, encoding='utf-8')
        ok, detail = (validate(name, candidate) if validate
                      else _validate_subprocess(name, candidate))
        if not ok:
            feedback.append(f'attempt {i} failed live validation: {detail}')
            attempts.append({'attempt': i, 'fixer': label, 'error': detail[:500]})
            _log_history(name, 'validation-failed', detail)
            continue
        backup = _heal_dir() / 'backups' / \
            f"{target.name}.{time.strftime('%Y%m%d-%H%M%S')}.py"
        try:
            backup.write_text(source, encoding='utf-8')
            tmp = target.with_suffix(target.suffix + '.new')
            tmp.write_text(code, encoding='utf-8')
            os.replace(tmp, target)
            try:
                _reload(name)
                status, detail = _probe_once(name)
                applied_ok = status == 'ok'
            except Exception as e:  # noqa: BLE001
                applied_ok, detail = False, f'{type(e).__name__}: {e}'
            if applied_ok:
                with _STATE.lock:
                    probe_state = _STATE.probes.get(name, {})
                    probe_state.update({'status': 'ok', 'consecutive': 0,
                                        'last_error': '',
                                        'last_probe_at': time.time()})
                    _STATE.probes[name] = probe_state
                _log_history(name, 'healed',
                             f'{label}; attempt {i}; backup {backup.name}')
                return {'healed': True, 'attempt': i, 'fixer': label,
                        'backup': str(backup)}
            # in-process verification failed -> roll back
            target.write_text(source, encoding='utf-8')
            try:
                _reload(name)
            except Exception:  # pragma: no cover
                pass
            feedback.append(f'attempt {i} applied but post-reload probe '
                            f'failed ({detail}); file restored from backup')
            attempts.append({'attempt': i, 'fixer': label,
                             'error': f'post-reload: {detail[:400]}'})
            _log_history(name, 'rollback', detail)
        except Exception as e:  # noqa: BLE001
            target.write_text(source, encoding='utf-8')
            feedback.append(f'attempt {i} crashed: {e}')
            attempts.append({'attempt': i, 'error': str(e)[:500]})
            _log_history(name, 'apply-crash', e)
    _log_history(name, 'incident-exhausted',
                 f'{len(attempts)} attempts; feedback: {feedback[-1] if feedback else ""}')
    return {'healed': False, 'attempts': attempts}


# ------------------------------------------------------------- daemon cycle
def probe_cycle() -> Dict[str, Dict[str, Any]]:
    """Probe every configured provider once; maybe trigger heal / refresh."""
    results: Dict[str, Dict[str, Any]] = {}
    excl = {e.strip().lower() for e in
            os.getenv('DSF_SELFHEAL_EXCLUDE', '').split(',') if e.strip()}
    for name in HEALABLE:
        if name in excl or not provider_enabled(name) \
                or not _provider_configured(name):
            results[name] = {'status': 'skipped'}
            continue
        status, detail = _probe_once(name)
        with _STATE.lock:
            st = _STATE.probes.setdefault(name, {})
            st['status'] = status
            st['last_detail'] = detail
            st['last_probe_at'] = time.time()
            st['errors'] = (st.get('errors') or [])[-4:]
            if status == 'ok':
                st['consecutive'] = 0
            elif status == 'structural':
                st['consecutive'] = int(st.get('consecutive') or 0) + 1
                st['errors'].append(f'{time.strftime("%H:%M:%S")} {detail}')
        results[name] = {'status': status, 'detail': detail,
                         'consecutive': int(st.get('consecutive') or 0)}
        if status == 'structural' and results[name]['consecutive'] >= _trigger():
            results[name]['heal'] = heal(name, reason=detail)
        elif status == 'auth':
            try:  # credentials rejected -> let the refresher bot try to renew
                from dsk import refresher
                results[name]['renew'] = refresher.renew(name, reason='auth')
            except Exception as e:  # noqa: BLE001
                results[name]['renew'] = {'error': str(e)[:300]}
    return results


def start_daemon() -> bool:
    """Idempotently start the background probe/heal loop."""
    with _STATE.lock:
        if _STATE.started:
            return False
        _STATE.started = True

    def _loop() -> None:
        while True:
            try:
                probe_cycle()
            except Exception:  # pragma: no cover - daemon must never die
                pass
            time.sleep(_probe_ttl())

    threading.Thread(target=_loop, name='selfheal', daemon=True).start()
    return True


def status() -> Dict[str, Any]:
    with _STATE.lock:
        probes = {k: {kk: vv for kk, vv in v.items() if kk != 'errors'}
                  for k, v in _STATE.probes.items()}
        last_heal = dict(_STATE.last_heal)
        started = _STATE.started
    incidents = {k: v[1] for k, v in _STATE.incidents.items()}
    return {'enabled': _env_bool('DSF_SELFHEAL', True),
            'daemon': started,
            'probe_ttl': _probe_ttl(),
            'trigger': _trigger(),
            'fixers': [c['label'] for c in _fixer_candidates()],
            'probes': probes,
            'incidents_today': incidents,
            'last_heal': last_heal}


def main(argv: List[str]) -> int:  # pragma: no cover - CLI
    cmd = argv[1] if len(argv) > 1 else 'status'
    if cmd == 'probe':
        print(json.dumps(probe_cycle(), indent=2, default=str))
        return 0
    if cmd == 'status':
        print(json.dumps(status(), indent=2, default=str))
        return 0
    if cmd == 'heal' and len(argv) > 2:
        print(json.dumps(heal(argv[2], reason='manual CLI heal'),
                         indent=2, default=str))
        return 0
    if cmd == '_validate' and len(argv) > 3:
        return _cmd_validate(argv[2], argv[3])
    print(__doc__)
    return 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
