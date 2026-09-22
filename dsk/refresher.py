"""Credential renewal bot: keeps the three web sessions alive autonomously.

Nothing here ever asks a human to re-copy a cookie — the ladder of renewal
strategies is, from cheapest to most invasive:

  1. HTTP cookie refresh (safe, automatic)
       gemini    load the cookie jar, GET gemini.google.com/app, capture the
                 rotated ``__Secure-1PSIDTS`` from the Set-Cookie header and
                 write it back to ``data/gemini_cookies.json``
       chatgpt   GET chatgpt.com/api/auth/session with the jar, persist the
                 fresh ``accessToken``-issuing session cookies
       deepseek  nothing to rotate over HTTP (the userToken only changes on
                 login) — verified with a live probe instead

  2. Headless-browser re-login (default ON: DSF_REFRESHER_LOGIN=true, set
     false to disable). Uses the same DrissionPage/Chromium stack as
     the Cloudflare bypass; exports the fresh cookies/token automatically.
       DEEPSEEK_LOGIN_EMAIL / DEEPSEEK_LOGIN_PASSWORD
       CHATGPT_LOGIN_EMAIL  / CHATGPT_LOGIN_PASSWORD
       GEMINI_LOGIN_EMAIL   / GEMINI_LOGIN_PASSWORD   (Google anti-bot: best
                                                        effort only)
     Email verification codes (OTP) during login are fetched from an IMAP
     mailbox (see DSF_MAIL_* below), so the loop stays unmanned.

  3. Account auto-signup (default ON for ALL providers:
     DSF_REFRESHER_AUTOSIGNUP). Creates a fresh free account when even the
     login session is dead — and BOOTSTRAPS providers that have no
     credentials at all (the refresher daemon signs every missing provider
     up on its first cycle, so a fresh install comes up unattended). The
     e-mail address is AUTO-GENERATED (dsk/mailgen.py): a catch-all IMAP
     domain (DSF_MAIL_DOMAIN) when available, else a mail.tm throwaway
     account — the verification code is read from that mailbox
     automatically. Disable the auto-generation with DSF_MAIL_AUTOGEN=false.
     Created accounts are persisted to data/accounts.json so later renewal
     cycles can re-login with them. Google/OpenAI may still throw captcha
     or phone walls at automation — those rungs are best effort and their
     failures surface in history.jsonl like any other ladder miss.

Renewals are triggered two ways: the self-healing daemon calls ``renew``
whenever a provider probe classifies as ``auth``, and the refresher daemon
proactively rotates cookies every DSF_REFRESHER_TTL seconds. Every action is
logged to data/refresher/history.jsonl; all ladders respect per-provider
cooldowns and daily attempt caps, and DSF_REFRESHER=false disables everything.

Bot-managed credential files take precedence over env vars (documented in
README): data/deepseek_token, data/gemini_cookies.json, data/chatgpt_cookies.json.
Delete the file to hand control back to the environment.

Mail config (for OTP during browser flows):
    DSF_MAIL_AUTOGEN       auto-create throwaway mailboxes (default true)
    DSF_MAIL_DOMAIN        catch-all domain for autogen (optional; without
                           it mail.tm public temp-mail is used)
    DSF_MAIL_IMAP_HOST / _PORT (993) / _USER / _PASS
    DSF_MAIL_OTP_SENDER    substring matched against the sender (default deepseek)
    DSF_MAIL_OTP_REGEX     code regex (default \\\\b(\\\\d{6})\\\\b)
    DSF_MAIL_OTP_MAX_AGE   ignore older mail, minutes (default 30)

CLI:
    python -m dsk.refresher status
    python -m dsk.refresher refresh gemini
    python -m dsk.refresher login deepseek
    python -m dsk.refresher signup [deepseek|chatgpt|gemini]
    python -m dsk.refresher bootstrap   (create credentials for every provider that has none)
    python -m dsk.refresher mailgen   (create a throwaway mailbox as a test)
"""

import base64
import imaplib
import json
import os
import random
import re
import threading
import time
from datetime import date, datetime, timezone
from email import message_from_bytes
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import mailgen
from .providers.base import provider_enabled

_BASE = Path(__file__).resolve().parent


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, '').strip().lower()
    if not raw:
        return default
    return raw in ('1', 'true', 'yes', 'on')


def _data_dir() -> Path:
    base = (os.getenv('COOKIES_DIR') or os.getenv('DSF_SELFHEAL_DIR')
            or str(_BASE.parent / 'data'))
    path = Path(base)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _ttl() -> float:
    return max(300.0, float(os.getenv('DSF_REFRESHER_TTL', '21600') or 21600))


def _cooldown() -> float:
    return max(60.0, float(os.getenv('DSF_REFRESHER_COOLDOWN', '1800') or 1800))


def _max_renews() -> int:
    return max(1, int(os.getenv('DSF_REFRESHER_MAX_RENEWS', '6') or 6))


def _jar_path(name: str) -> Path:
    files = {'gemini': 'gemini_cookies.json', 'chatgpt': 'chatgpt_cookies.json',
             'deepseek': 'cookies.json', 'claude': 'claude_cookies.json',
             'grok': 'grok_cookies.json', 'mistral': 'mistral_cookies.json',
             'qwen': 'qwen_cookies.json', 'kimi': 'kimi_cookies.json',
             'huggingface': 'huggingface_jar.json',
             'openrouter': 'openrouter_jar.json', 'groq': 'groq_jar.json',
             'copilot': 'copilot_cookies.json',
             'perplexity': 'perplexity_cookies.json', 'glm': 'glm_cookies.json'}
    return _data_dir() / files[name]


class _State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.started = False
        self.renewing: Dict[str, bool] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        self.counts: Dict[str, Tuple[str, int]] = {}  # provider -> (day, n)
        self.counts_seeded = False
        # inline request-path remediation bookkeeping
        self.inline_counts: Dict[str, Tuple[str, int]] = {}  # provider -> (hour, n)
        self.inline_last: Dict[str, Tuple[str, float]] = {}  # provider -> (ok|failed, ts)
        self.inline_threads: Dict[str, threading.Thread] = {}


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
             'provider': provider, 'event': event, 'detail': str(detail)[:1000]}
    with _STATE.lock:
        _STATE.results[provider] = entry
    try:
        path = _data_dir() / 'refresher' / 'history.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        _rotate_history(path)
        with path.open('a', encoding='utf-8') as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + '\n')
    except OSError:
        pass


# --------------------------------------------------------------- cookie jars
def _load_jar(name: str) -> Dict[str, str]:
    """Cookies as a flat {name: value} dict (jar file first, env fallback)."""
    jar: Dict[str, str] = {}
    path = _jar_path(name)
    try:
        if path.is_file():
            data = json.loads(path.read_text(encoding='utf-8'))
            if isinstance(data, dict) and isinstance(data.get('cookies'), dict):
                data = data['cookies']  # deepseek bypass format
            if isinstance(data, dict):
                jar = {str(k): str(v) for k, v in data.items()}
            elif isinstance(data, list):
                jar = {str(e.get('name')): str(e.get('value'))
                       for e in data if isinstance(e, dict) and e.get('name')}
    except (OSError, ValueError):
        jar = {}
    if name == 'gemini':
        jar.setdefault('__Secure-1PSID',
                       (os.getenv('GEMINI_1PSID', '') or
                        os.getenv('GEMINI_COOKIES_1PSID', '')).strip())
        jar.setdefault('__Secure-1PSIDTS',
                       (os.getenv('GEMINI_1PSIDTS', '') or
                        os.getenv('GEMINI_COOKIES_1PSIDTS', '')).strip())
    else:
        # Token providers keep their primary credential in one env var; merge
        # it so _has_creds and the live verifiers below agree on one source.
        primary = {'claude': ('sessionKey', 'CLAUDE_SESSION_KEY'),
                   'grok': ('sso', 'GROK_SSO'),
                   'groq': ('api_key', 'GROQ_API_KEY'),
                   'huggingface': ('token', 'HF_TOKEN'),
                   'kimi': ('token', 'KIMI_TOKEN'),
                   'mistral': ('session_token', 'MISTRAL_SESSION_TOKEN'),
                   'openrouter': ('api_key', 'OPENROUTER_API_KEY'),
                   'qwen': ('token', 'QWEN_TOKEN')}.get(name)
        if primary:
            jar.setdefault(primary[0], (os.getenv(primary[1], '') or '').strip())
    return {k: v for k, v in jar.items() if k and v and k != 'cookies'}


def _save_jar(name: str, updates: Dict[str, str]) -> None:
    """Merge cookie updates into the jar file (atomic, bot-managed)."""
    path = _jar_path(name)
    merged: Any
    try:
        existing = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None
    except (OSError, ValueError):
        existing = None
    if isinstance(existing, dict) and isinstance(existing.get('cookies'), dict):
        base, fmt = dict(existing['cookies']), 'bypass'
    elif isinstance(existing, list):
        base = {str(e.get('name')): str(e.get('value'))
                for e in existing if isinstance(e, dict) and e.get('name')}
        fmt = 'list'
    else:
        base, fmt = dict(existing or {}), 'dict'
    base.update({k: v for k, v in updates.items() if k and v})
    if fmt == 'bypass':
        merged = {'cookies': base,
                  'user_agent': (existing or {}).get('user_agent', '')}
    elif fmt == 'list':
        merged = [{'name': k, 'value': v} for k, v in base.items()]
    else:
        merged = base
    tmp = path.with_suffix(path.suffix + '.new')
    tmp.write_text(json.dumps(merged, indent=2, ensure_ascii=False),
                   encoding='utf-8')
    os.replace(tmp, path)
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _save_deepseek_token(token: str) -> Path:
    path = _data_dir() / 'deepseek_token'
    tmp = path.with_suffix('.new')
    tmp.write_text(token.strip(), encoding='utf-8')
    os.replace(tmp, path)
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return path


# -------------------------------------------------- credential bootstrap
def _has_creds(name: str) -> bool:
    """True when the provider has any usable credential (file or env).

    Used to decide whether the daemon must bootstrap (auto-signup) a
    provider. For deepseek only the userToken counts — bypass cookies
    alone cannot serve requests."""
    if name == 'deepseek':
        if os.getenv('DEEPSEEK_AUTH_TOKEN', '').strip():
            return True
        try:
            f = _data_dir() / 'deepseek_token'
            return bool(f.is_file() and f.read_text(encoding='utf-8').strip())
        except OSError:
            return False
    if name == 'chatgpt':
        if (os.getenv('CHATGPT_ACCESS_TOKEN', '')
                or os.getenv('CHATGPT_SESSION_TOKEN', '')).strip():
            return True
        if os.getenv('CHATGPT_SESSION_COOKIES', '').strip():
            return True
        return bool(_load_jar('chatgpt'))
    if name in ('claude', 'grok', 'groq', 'openrouter', 'qwen', 'kimi',
                'huggingface'):
        env_key = {'claude': 'CLAUDE_SESSION_KEY', 'grok': 'GROK_SSO',
                   'groq': 'GROQ_API_KEY', 'openrouter': 'OPENROUTER_API_KEY',
                   'qwen': 'QWEN_TOKEN', 'kimi': 'KIMI_TOKEN',
                   'huggingface': 'HF_TOKEN'}[name]
        if os.getenv(env_key, '').strip():
            return True
        jar = _load_jar(name)
        if name == 'claude':
            return bool(jar.get('sessionKey'))
        if name == 'grok':
            return bool(jar.get('sso') or jar.get('sso-rw'))
        if name == 'groq':
            return bool(jar.get('api_key') or jar.get('key')
                        or jar.get('gsk_key'))
        if name == 'openrouter':
            return bool(jar.get('api_key') or jar.get('key')
                        or jar.get('token'))
        if name == 'kimi':
            return bool(jar.get('token') or jar.get('jwt'))
        return bool(jar.get('token'))
    if name in ('copilot', 'perplexity', 'glm'):
        return True  # anonymous reverse-engineered modes always available
    if name == 'mistral':
        if os.getenv('MISTRAL_SESSION_TOKEN', '').strip():
            return True
        return bool((_load_jar('mistral') or {}).get('session_token'))
    return bool(_load_jar('gemini'))          # jar already merges env 1PSID


_ACCOUNTS_FILE = 'accounts.json'


def _load_accounts() -> Dict[str, Dict[str, str]]:
    """Accounts the bot created itself (provider -> {email, password,...})."""
    try:
        data = json.loads((_data_dir() / _ACCOUNTS_FILE)
                          .read_text(encoding='utf-8'))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_account(name: str, email: str, password: str,
                  backend: str = '', extra: Optional[Dict[str, Any]] = None
                  ) -> None:
    """Persist a bot-created account so later renewals can re-login.

    When the caller doesn't supply a new ``mail_session`` (e.g. a signup
    rung that reuses a pre-existing email/password but has no fresh
    session from ``mailgen.create_email``), the previously stored
    ``mail_session`` / ``backend`` are PRESERVED — dropping it would
    orphan the signup mailbox and break the activation-link polling on
    the next cycle.
    """
    accs = _load_accounts()
    prev = dict(accs.get(name) or {})
    entry = {'email': email, 'password': password, 'backend': backend,
             'ts': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    if extra:
        entry.update(extra)
    same_email = (prev.get('email') or '') == (email or '')
    if same_email and not entry.get('mail_session') and prev.get('mail_session'):
        entry['mail_session'] = prev['mail_session']
    if same_email and not entry.get('backend') and prev.get('backend'):
        entry['backend'] = prev['backend']
    accs[name] = entry
    path = _data_dir() / _ACCOUNTS_FILE
    tmp = path.with_suffix('.new')
    tmp.write_text(json.dumps(accs, indent=2, ensure_ascii=False),
                   encoding='utf-8')
    os.replace(tmp, path)
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _proxies_kwargs(url: str) -> Dict[str, Any]:
    try:
        from . import proxies as _px
        return _px.proxies_kwargs(url=url)
    except Exception:
        return {}


_UA = ('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
       '(KHTML, like Gecko) Chrome/126.0 Safari/537.36')


# ------------------------------------------------------- HTTP refresh layer
def _http_get(url: str, cookies: Dict[str, str], timeout: int = 30):
    import requests
    return requests.get(url, cookies=cookies,
                        headers={'User-Agent': _UA}, timeout=timeout,
                        allow_redirects=True, **_proxies_kwargs(url))


def refresh_gemini() -> Tuple[bool, str]:
    jar = _load_jar('gemini')
    if not jar.get('__Secure-1PSID'):
        return False, 'no gemini credentials (jar/env)'
    resp = _http_get('https://gemini.google.com/app', jar)
    if resp.status_code in (401, 403) or 'accounts.google.com' in str(resp.url):
        _save_jar('gemini', dict(resp.cookies))
        return False, 'session rejected (cookies expired) - re-login needed'
    rotated = {k: v for k, v in dict(resp.cookies).items()
               if k in ('__Secure-1PSID', '__Secure-1PSIDTS') and v}
    if rotated:
        _save_jar('gemini', rotated)
        return True, f"rotated: {','.join(rotated)}"
    return True, 'session valid (no rotation offered)'


def refresh_chatgpt() -> Tuple[bool, str]:
    jar = _load_jar('chatgpt')
    if not jar:
        return False, 'no chatgpt credentials (jar/env)'
    resp = _http_get('https://chatgpt.com/api/auth/session', jar)
    new_cookies = {k: v for k, v in dict(resp.cookies).items() if v}
    if new_cookies:
        _save_jar('chatgpt', new_cookies)
    if resp.status_code in (401, 403):
        return False, 'session cookies rejected - re-login needed'
    if resp.status_code == 200:
        try:
            has_token = bool((resp.json() or {}).get('accessToken'))
        except ValueError:
            has_token = False
        return True, ('session valid, token issued' if has_token
                      else 'session reachable but no accessToken (expired?)')
    return False, f'HTTP {resp.status_code}'


def refresh_deepseek() -> Tuple[bool, str]:
    """userToken cannot be rotated over HTTP; just verify it live."""
    try:
        from . import selfheal
        status, detail = selfheal._probe_once('deepseek')
        return (status == 'ok'), f'{status}: {detail}'
    except Exception as e:  # noqa: BLE001
        return False, f'probe failed: {e}'


def _manual_only(provider: str, hint: str):
    """Refresher/sign-up stub for token-based RE providers that have no
    automated account creation: their credential is a manually exported
    cookie/token, so the bot only reports how to set it."""
    def _f() -> Tuple[bool, str]:
        return False, f'{provider}: no automated signup — {hint}'
    return _f


def _anonymous(provider: str):
    """Refresher stub for providers that need no credential at all."""
    def _f() -> Tuple[bool, str]:
        return True, f'{provider}: anonymous access — nothing to refresh'
    return _f


def refresh_claude() -> Tuple[bool, str]:
    """claude.ai sessionKey cannot be rotated over HTTP; verify it live."""
    if not _has_creds('claude'):
        return False, 'no claude credentials — set CLAUDE_SESSION_KEY or claude_cookies.json'
    try:
        resp = _http_get('https://claude.ai/api/organizations',
                         {'sessionKey': _load_jar('claude').get('sessionKey', '')})
    except Exception as e:  # noqa: BLE001
        return False, f'verify failed: {type(e).__name__}: {e}'
    if resp.status_code == 200:
        return True, 'session valid'
    if resp.status_code in (401, 403):
        return False, 'sessionKey rejected — re-export from claude.ai'
    return False, f'HTTP {resp.status_code}'


def refresh_grok() -> Tuple[bool, str]:
    if not _has_creds('grok'):
        return False, 'no grok credentials — set GROK_SSO or grok_cookies.json'
    jar = _load_jar('grok')
    cookies = {k: v for k, v in jar.items() if k in ('sso', 'sso-rw')}
    try:
        resp = _http_get('https://grok.com/rest/rate-limits', cookies)
    except Exception as e:  # noqa: BLE001
        return False, f'verify failed: {type(e).__name__}: {e}'
    if resp.status_code == 200:
        return True, 'session valid'
    if resp.status_code in (401, 403):
        return False, 'sso cookie rejected — re-export from grok.com'
    if resp.status_code == 429:
        return True, 'session valid (rate limited)'
    return False, f'HTTP {resp.status_code}'


def _qwen_headers() -> Dict[str, str]:
    """WAF-safe request headers for chat.qwen.ai (static bx-ua fingerprint
    from the provider module; falls back to a plain UA on import failure)."""
    try:
        from .providers.qwen_provider import _headers
        return _headers()
    except Exception:  # noqa: BLE001
        return {'User-Agent': _UA}


def _qwen_request(method: str, url: str, **kw) -> Tuple[Any, Optional[str]]:
    """``requests`` with the pooled egress, retried once DIRECT on failure.

    Pooled exits intermittently break TLS to chat.qwen.ai (self-signed
    cert / SOCKS refused). The qwen auth endpoints are WAF-free, so a
    direct connection is a safe second rung. Returns (response, note)
    where note explains the fallback (or None); the direct attempt's
    exception propagates if it also fails.
    """
    import requests
    timeout = kw.pop('timeout', 30)
    try:
        return (requests.request(method, url, timeout=timeout,
                                 **kw, **_proxies_kwargs(url)), None)
    except Exception as pooled_exc:  # noqa: BLE001
        note = f'direct-retry after {type(pooled_exc).__name__}'
    return requests.request(method, url, timeout=timeout, **kw), note


def _qwen_signin(email: str, password: str) -> Tuple[bool, str]:
    """HTTP re-login on chat.qwen.ai (OpenWebUI-style /api/v1/auths/signin).

    The signin path is NOT WAF-challenged (unlike /signup, which sits behind
    an Aliyun slide-captcha), so token renewal needs no browser: exchange the
    stored email/password for a fresh JWT and persist it in the jar.
    """
    import requests
    try:
        resp, note = _qwen_request(
            'POST', 'https://chat.qwen.ai/api/v1/auths/signin',
            json={'email': email, 'password': password},
            headers=_qwen_headers())
    except Exception as e:  # noqa: BLE001
        return False, f'signin failed: {type(e).__name__}: {e}'
    if resp.status_code != 200:
        detail = ''
        try:
            detail = str((resp.json() or {}).get('detail') or '')[:80]
        except Exception:  # noqa: BLE001
            pass
        return False, (f'signin rejected (HTTP {resp.status_code})'
                       + (f': {detail}' if detail else ''))
    try:
        body = resp.json() or {}
    except ValueError:
        return False, 'signin returned a non-JSON body'
    token = str(body.get('token') or '').strip()
    if not token:
        return False, 'signin ok but no token in response'
    _save_jar('qwen', {'token': token, 'email': email})
    return True, 're-signed in; fresh qwen token saved'


def refresh_qwen() -> Tuple[bool, str]:
    """Verify the stored token; re-login, or converge a pending activation."""
    jar = _load_jar('qwen')
    token = (jar.get('token') or '').strip()
    if token:
        try:
            resp, note = _qwen_request(
                'GET', 'https://chat.qwen.ai/api/v1/auths',
                headers={**_qwen_headers(),
                         'Authorization': f'Bearer {token}'})
            if note:
                _log_history('qwen', 'stage', f'verify fell back to direct ({note})')
        except Exception as e:  # noqa: BLE001
            return False, f'verify failed: {type(e).__name__}: {e}'
        if resp.status_code == 200:
            return True, 'token valid'
        if resp.status_code not in (401, 403):
            return False, f'HTTP {resp.status_code}'
    # token missing/rejected -> HTTP re-login from stored credentials
    # (chat.qwen.ai /signin is not WAF-gated, so no browser is needed)
    email, password = _creds('qwen')
    if not email or not password:
        return False, ('no qwen credentials — set QWEN_TOKEN, or '
                       'QWEN_LOGIN_EMAIL/QWEN_LOGIN_PASSWORD (or let the '
                       'qwen signup rung create an account) to enable '
                       'automatic re-login')
    ok, detail = _qwen_signin(email, password)
    if not ok:
        # 'pending activation' means the account exists but was never
        # confirmed — poll the signup mailbox and open the activation link
        # so renewal converges autonomously across cycles.
        if _qwen_pending(detail):
            _log_history('qwen', 'stage',
                         f'signin pending activation; activating ({detail})')
            return _qwen_activate(email, password)
        return False, f'token rejected; {detail}'
    return True, detail


def _qwen_pending(detail: str) -> bool:
    """True when a signin failure means 'account exists but unactivated'."""
    d = (detail or '').lower()
    return any(n in d for n in ('pending', 'unverified', 'not verified',
                                'verify', 'activation', 'activat'))


def _qwen_mail_session(email: str) -> Optional[Dict[str, Any]]:
    """Rebuild a mailgen session for the qwen account's signup mailbox."""
    stored = _load_accounts().get('qwen') or {}
    if stored.get('email') and stored.get('email') != email:
        return None
    ms = stored.get('mail_session')
    if isinstance(ms, dict) and ms.get('backend') and ms.get('address'):
        return dict(ms)
    if (stored.get('backend') or '').strip() and email:
        return {'backend': stored['backend'], 'address': email}
    return None


def _qwen_activate(email: str, password: str) -> Tuple[bool, str]:
    """Finish a pending chat.qwen.ai activation autonomously.

    New accounts submit to 'pending activation': the confirmation mail lands
    in the bot's signup mailbox. Poll it for the activation URL, open the
    link (plain GET first, then a real headed browser as fallback), and HTTP
    re-login. Called from the login and signup rungs, so the daemon converges
    on activation across renewal cycles with no operator action.
    """
    session = _qwen_mail_session(email)
    if not session:
        return False, 'no mailbox backend recorded for this qwen account'
    link = mailgen.fetch_otp(session, max_wait_s=120, sender_needle='qwen',
                             code_re=re.compile(
                                 r'(https://chat\.qwen\.ai/[^\s"\'<>]*'
                                 r'activate[^\s"\'<>]*)'))
    if not link:
        return False, 'activation link not in mailbox yet'
    _log_history('qwen', 'stage', 'activation link found; opening')
    try:
        _http_get(link, cookies={}, timeout=30)
    except Exception:  # noqa: BLE001 — the browser visit is authoritative
        pass
    page = None
    try:
        page = _browser(proxy=_pool_proxy(), headed=True)
        page.get(link)
        time.sleep(6)
    except Exception:  # noqa: BLE001 — the plain GET above may have sufficed
        pass
    finally:
        if page is not None:
            try:
                page.quit()
            except Exception:  # noqa: BLE001
                pass
    ok, detail = _qwen_signin(email, password)
    if ok:
        _log_history('qwen', 'stage', 'activation completed; re-login ok')
    return ok, detail


def refresh_kimi() -> Tuple[bool, str]:
    """Lightweight liveness check; the authoritative probe runs at request time."""
    if not _has_creds('kimi'):
        return False, 'no kimi credentials — set KIMI_TOKEN or kimi_cookies.json'
    jar = _load_jar('kimi')
    token = jar.get('token') or jar.get('jwt') or ''
    try:
        resp = _http_get('https://www.kimi.com/', {'token': token})
    except Exception as e:  # noqa: BLE001
        return False, f'verify failed: {type(e).__name__}: {e}'
    if resp.status_code in (401, 403):
        return False, 'token rejected — re-export from kimi.com'
    return True, f'session reachable (HTTP {resp.status_code})'


def refresh_mistral() -> Tuple[bool, str]:
    """Best-effort token check: Ory Kratos whoami, lenient when unverifiable.

    The authoritative validation runs at request time (the provider raises
    on a rejected token), so an unavailable whoami endpoint is not an error.
    """
    if not _has_creds('mistral'):
        return False, ('no mistral credentials — set MISTRAL_SESSION_TOKEN '
                       'or mistral_cookies.json')
    token = (os.getenv('MISTRAL_SESSION_TOKEN', '') or '').strip()
    if not token:
        token = (_load_jar('mistral') or {}).get('session_token') or ''
    try:
        resp = _http_get('https://auth.mistral.ai/sessions/whoami',
                         {'ory_kratos_session': token})
    except Exception as e:  # noqa: BLE001
        return False, f'verify failed: {type(e).__name__}: {e}'
    if resp.status_code == 200:
        return True, 'session valid'
    if resp.status_code in (401, 403):
        return False, 'session token rejected — re-export from chat.mistral.ai'
    return True, (f'whoami unavailable (HTTP {resp.status_code}) — '
                  'token unverified (validated at request time)')


# --------------------------------------------------------- huggingface
_HF_VERIFY_URL = 'https://huggingface.co/api/whoami-v2'
_HF_LOGIN_URL = 'https://huggingface.co/login'
_HF_JOIN_URL = 'https://huggingface.co/join'


def _hf_verify_token(token: str) -> Tuple[bool, str]:
    """GET /api/whoami-v2 with the Bearer token (200 = valid)."""
    import requests
    try:
        resp = requests.get(
            _HF_VERIFY_URL,
            headers={'Authorization': f'Bearer {token}', 'User-Agent': _UA},
            timeout=20, **_proxies_kwargs(_HF_VERIFY_URL))
    except Exception as e:  # noqa: BLE001
        return False, f'verify failed: {type(e).__name__}: {e}'
    if resp.status_code == 200:
        return True, 'token valid'
    if resp.status_code in (401, 403):
        return False, 'token rejected'
    return False, f'HTTP {resp.status_code}'


def _hf_page_with_egress(url: str, want_css: str = 'css:input',
                         want_wait_s: float = 60.0):
    """Open a huggingface.co page in a real browser, egress-aware.

    huggingface.co sits behind AWS WAF (goku) behind CloudFront: plain
    HTTP POSTs get 202 challenge pages, and the container's direct egress
    IP is often flat-out refused ('The request could not be satisfied').
    Same medicine as the qwen signup: try direct first, then fresh pooled
    exits; return the page that shows the expected element (or None).
    """
    from DrissionPage import ChromiumPage, ChromiumOptions

    def _open(cand: Optional[str]):
        opts = ChromiumOptions().auto_port()
        opts.set_argument('--no-sandbox')
        opts.set_argument('--disable-gpu')
        opts.set_argument('--disable-blink-features=AutomationControlled')
        opts.headless(True)  # WAF JS challenge passes headless; cf needs no X
        if cand:
            opts.set_argument(f'--proxy-server={cand}')
        page = ChromiumPage(addr_or_opts=opts)
        try:
            page.get(url, timeout=40)
        except Exception:  # noqa: BLE001 — navigation races are fine
            pass
        deadline = time.time() + want_wait_s
        while time.time() < deadline:
            try:
                if page.ele(want_css, timeout=2):
                    return page
            except Exception:  # noqa: BLE001
                pass
            time.sleep(2)
        page.quit()
        return None

    candidates: List[Optional[str]] = [None]
    try:
        from . import proxies as _proxies
        _proxies.ensure_pool()
        raw = list(_proxies.all_proxies())
        random.shuffle(raw)
        candidates += raw[:6]
    except Exception:  # noqa: BLE001 — direct remains the fallback
        pass
    last = ''
    for cand in candidates:
        try:
            page = _open(cand)
        except Exception as exc:  # noqa: BLE001
            last = f'{type(exc).__name__}: {exc}'
            continue
        if page is not None:
            _log_history('huggingface', 'stage',
                         f'egress ok ({cand or "direct"})')
            return page
        last = f'no form with egress {cand or "direct"}'
    _log_history('huggingface', 'stage', f'all egresses failed: {last}')
    return None


def _hf_session_token(page) -> str:
    """Extract the hf_ web-session cookie after a login.

    The cookie is literally named ``token`` and its value doubles as the
    API Bearer credential (lifts anonymous ZeroGPU quotas, admits
    'auto'-gated spaces).
    """
    for _ in range(20):
        try:
            for c in (page.cookies(all_domains=True) or []):
                name = (c.get('name') or c.get('Name') or '') if isinstance(c, dict) else getattr(c, 'name', '')
                value = (c.get('value') or c.get('Value') or '') if isinstance(c, dict) else getattr(c, 'value', '')
                if name == 'token' and str(value).startswith('hf_'):
                    return str(value)
        except Exception:  # noqa: BLE001 — page may be mid-navigation
            pass
        time.sleep(2)
    return ''


def _hf_browser_login(email: str, password: str) -> Tuple[bool, str]:
    """Browser re-login on huggingface.co, then persist the hf_ session.

    The WAF challenge executes silently in the real browser; the form is
    the classic username/password POST to /login. On success the session
    cookie lands in the huggingface jar (verified via whoami-v2 first).
    """
    page = _hf_page_with_egress(_HF_LOGIN_URL, 'css:input[name=username]')
    if page is None:
        return False, 'hf login form never appeared (WAF/egress blocked)'
    try:
        if not _fill_first(page, ['css:input[name=username]'], email):
            return False, 'hf login: username field unusable'
        if not _fill_first(page, ['css:input[name=password]'], password):
            return False, 'hf login: password field unusable'
        try:
            page('css:form[action="/login"] button[type=submit]').click(
                by_js=False)
        except Exception:  # noqa: BLE001 — Enter in the password field works too
            try:
                page.ele('css:input[name=password]').input('\n')
            except Exception:  # noqa: BLE001
                pass
        token = _hf_session_token(page)
        if not token:
            return False, ('login submitted but no session token — '
                           f'captcha or bad credentials? {_body_head(page)}')
        ok, detail = _hf_verify_token(token)
        if not ok:
            return False, f'cookie captured but rejected: {detail}'
        _save_jar('huggingface', {'token': token, 'email': email})
        return True, 'browser re-login ok; fresh hf token saved'
    finally:
        try:
            page.quit()
        except Exception:  # noqa: BLE001
            pass


def refresh_huggingface() -> Tuple[bool, str]:
    """Verify the stored hf_ token; browser re-login when missing/rejected.

    Anonymous access stays first-class for the provider itself; the
    credential exists to lift ZeroGPU quotas and admit 'auto'-gated
    spaces (the discovery probe list widens with a valid token).
    """
    token = (_load_jar('huggingface').get('token') or '').strip()
    if token:
        ok, detail = _hf_verify_token(token)
        if ok:
            return True, detail
        if detail not in ('token rejected',) and not detail.startswith('HTTP'):
            # transient verifier failure: keep the token (validated at
            # request time anyway) instead of burning a browser login
            return False, detail
        _log_history('huggingface', 'stage', f'token rejected; re-login ({detail})')
    email, password = _creds('huggingface')
    if not email or not password:
        return False, ('no huggingface credentials — set HF_TOKEN, or '
                       'HUGGINGFACE_LOGIN_EMAIL/PASSWORD (or let the '
                       'huggingface signup rung create an account) to '
                       'lift anonymous quotas')
    return _hf_browser_login(email, password)


def _hf_mail_session(email: str) -> Optional[Dict[str, Any]]:
    """Rebuild a mailgen session for the huggingface account's mailbox."""
    stored = _load_accounts().get('huggingface') or {}
    if stored.get('email') and stored.get('email') != email:
        return None
    ms = stored.get('mail_session')
    if isinstance(ms, dict) and ms.get('backend') and ms.get('address'):
        return dict(ms)
    if (stored.get('backend') or '').strip() and email:
        return {'backend': stored['backend'], 'address': email}
    return None


def signup_huggingface() -> Tuple[bool, str]:
    """Create a huggingface.co account autonomously (email-verified).

    /join runs in a real browser (AWS WAF + CloudFront egress blocks).
    Flow: enter the mailgen address -> poll the mailbox for huggingface's
    verification link -> open it and complete the username/password form.
    Credentials persist in data/accounts.json; the login rung then
    captures the hf_ session cookie into the jar. Accounts stuck before
    verification still persist — the mailbox is rebuilt from the stored
    session on later cycles, so activation converges autonomously.
    """
    email, password = _creds('huggingface')
    session = None
    if not email or not password:
        if not mailgen.autogen_enabled():
            return False, ('no HUGGINGFACE_LOGIN_EMAIL/PASSWORD and mail '
                           'autogen off')
        session, err = mailgen.create_email()
        if not session:
            return False, f'autogen mailbox unavailable: {err}'
        email = session['address']
        password = mailgen.gen_password()
    username = re.sub(r'[^a-z0-9-]', '',
                      f'dsf-{email.split("@")[0][:16]}'.lower()) or 'dsf-bot'
    _log_history('huggingface', 'stage', f'signup as {email}')
    page = _hf_page_with_egress(
        _HF_JOIN_URL, 'css:input[type=email],css:input[name=email]')
    if page is None:
        _save_account('huggingface', email, password,
                      backend=(session or {}).get('backend', ''),
                      extra={'mail_session': session} if session else None,
                      username=username)
        return False, 'hf join form never appeared (WAF/egress blocked)'
    try:
        if not _fill_first(page, ['css:input[type=email]',
                                  'css:input[name=email]'], email):
            return False, 'hf join: email field unusable'
        try:
            page('css:button[type=submit]').click(by_js=False)
        except Exception:  # noqa: BLE001
            try:
                page.ele('css:input[type=email]').input('\n')
            except Exception:  # noqa: BLE001
                pass
        time.sleep(6)
        body = _body_head(page)
        if 'already' in body:
            _save_account('huggingface', email, password,
                          backend=(session or {}).get('backend', ''),
                          extra={'mail_session': session} if session else None,
                          username=username)
            return False, 'email already registered — login rung should take over'
        _save_account('huggingface', email, password,
                      backend=(session or {}).get('backend', ''),
                      extra={'mail_session': session} if session else None,
                      username=username)
        # wait out the verification mail, then follow the link
        link = mailgen.fetch_otp(
            session, max_wait_s=300, sender_needle='huggingface',
            code_re=re.compile(r'(https://huggingface\.co/[^\s"\'<>]+)'))
        if not link:
            ms = _hf_mail_session(email)
            if ms and ms is not session:
                link = mailgen.fetch_otp(
                    ms, max_wait_s=120, sender_needle='huggingface',
                    code_re=re.compile(
                        r'(https://huggingface\.co/[^\s"\'<>]+)'))
        if not link:
            return False, ('verification link not in mailbox yet — '
                           'renews on the next cycle')
        _log_history('huggingface', 'stage', 'verification link found; opening')
        try:
            page.get(link, timeout=40)
        except Exception:  # noqa: BLE001 — the form loads on retry
            pass
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                if page.ele('css:input[type=password]', timeout=2):
                    break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(2)
        if not _fill_first(page, ['css:input[name=username]',
                                  'css:#username'], username):
            _log_history('huggingface', 'stage',
                         f'no username field: {_body_head(page)}')
        _fill_first(page, ['css:input[name=password]',
                           'css:input[type=password]'], password)
        try:
            page('css:button[type=submit]').click(by_js=False)
        except Exception:  # noqa: BLE001
            pass
        time.sleep(10)
        token = _hf_session_token(page)
        if token:
            ok, detail = _hf_verify_token(token)
            if ok:
                _save_jar('huggingface', {'token': token, 'email': email})
                return True, 'hf account created; session token saved'
        # account may exist but need a plain login — the next renewal
        # cycle's browser login rung converges on it
        return False, ('signup submitted but no session yet — login rung '
                       'converges next cycle')
    finally:
        try:
            page.quit()
        except Exception:  # noqa: BLE001
            pass


# ------------------------------------- openrouter / groq (API-key rungs)
def _or_verify_key(key: str) -> Tuple[bool, str]:
    """GET /api/v1/auth/key with the Bearer key (200 = valid)."""
    import requests
    try:
        resp = requests.get(
            'https://openrouter.ai/api/v1/auth/key',
            headers={'Authorization': f'Bearer {key}', 'User-Agent': _UA},
            timeout=20, **_proxies_kwargs('https://openrouter.ai'))
    except Exception as e:  # noqa: BLE001
        return False, f'verify failed: {type(e).__name__}: {e}'
    if resp.status_code == 200:
        return True, 'api key valid'
    if resp.status_code in (401, 403):
        return False, 'api key rejected'
    return False, f'HTTP {resp.status_code}'


def _groq_verify_key(key: str) -> Tuple[bool, str]:
    """GET /openai/v1/models with the Bearer key (200 = valid)."""
    import requests
    try:
        resp = requests.get(
            'https://api.groq.com/openai/v1/models',
            headers={'Authorization': f'Bearer {key}', 'User-Agent': _UA},
            timeout=20, **_proxies_kwargs('https://api.groq.com'))
    except Exception as e:  # noqa: BLE001
        return False, f'verify failed: {type(e).__name__}: {e}'
    if resp.status_code == 200:
        return True, 'api key valid'
    if resp.status_code in (401, 403):
        return False, 'api key rejected'
    return False, f'HTTP {resp.status_code}'


def _stored_api_key(name: str) -> str:
    """First jar key-ish field, then the provider's primary env var."""
    jar = _load_jar(name) or {}
    for field in ('api_key', 'key', 'token'):
        val = (jar.get(field) or '').strip()
        if val:
            return val
    env = {'openrouter': 'OPENROUTER_API_KEY', 'groq': 'GROQ_API_KEY'}[name]
    return (os.getenv(env, '') or '').strip()


def refresh_openrouter() -> Tuple[bool, str]:
    """Verify the stored OpenRouter key (keys don't expire like cookies)."""
    key = _stored_api_key('openrouter')
    if key:
        ok, detail = _or_verify_key(key)
        if ok:
            return True, detail
        if detail != 'api key rejected' and not detail.startswith('HTTP'):
            return False, detail  # transient verifier failure
        _log_history('openrouter', 'stage', f'key rejected ({detail})')
    return False, ('no working OpenRouter key — set OPENROUTER_API_KEY '
                   '(create at openrouter.ai/keys; :free models need no '
                   'credit) or let the openrouter signup rung mint one')


def refresh_groq() -> Tuple[bool, str]:
    """Verify the stored Groq key."""
    key = _stored_api_key('groq')
    if key:
        ok, detail = _groq_verify_key(key)
        if ok:
            return True, detail
        if detail != 'api key rejected' and not detail.startswith('HTTP'):
            return False, detail
        _log_history('groq', 'stage', f'key rejected ({detail})')
    return False, ('no working Groq key — set GROQ_API_KEY (create at '
                   'console.groq.com/keys, free tier) or let the groq '
                   'signup rung mint one')


def _mail_session_for(name: str, email: str) -> Optional[Dict[str, Any]]:
    """Rebuild a mailgen session for a stored account's mailbox."""
    stored = _load_accounts().get(name) or {}
    if stored.get('email') and stored.get('email') != email:
        return None
    ms = stored.get('mail_session')
    if isinstance(ms, dict) and ms.get('backend') and ms.get('address'):
        return dict(ms)
    if (stored.get('backend') or '').strip() and email:
        return {'backend': stored['backend'], 'address': email}
    return None


def _mint_key_via_page(page: Any, path: str, body: str) -> str:
    """Same-origin synchronous XHR in the logged-in page; raw JSON or ''.

    DrissionPage's run_js cannot await fetch(), so a sync XMLHttpRequest
    is used instead — fine for a one-shot same-origin POST.
    """
    js = ("var xhr = new XMLHttpRequest();"
          f"xhr.open('POST', {json.dumps(path)}, false);"
          "xhr.setRequestHeader('Content-Type', 'application/json');"
          f"xhr.send({json.dumps(body)});"
          "return xhr.responseText;")
    try:
        return (page.run_js(js) or '').strip()
    except Exception:  # noqa: BLE001
        return ''


def signup_openrouter() -> Tuple[bool, str]:
    """Create an openrouter.ai account and mint a free-tier API key.

    Signup runs in a real browser (WAF/egress blocks plain HTTP); after
    the email-verified session exists, a same-origin XHR against
    /api/v1/keys (cookie auth) mints the key saved into the jar.
    """
    email, password = _creds('openrouter')
    session = None
    if not email or not password:
        if not mailgen.autogen_enabled():
            return False, ('no OPENROUTER_LOGIN_EMAIL/PASSWORD and mail '
                           'autogen off')
        session, err = mailgen.create_email()
        if not session:
            return False, f'autogen mailbox unavailable: {err}'
        email = session['address']
        password = mailgen.gen_password()
    _log_history('openrouter', 'stage', f'signup as {email}')
    page = _hf_page_with_egress(
        'https://openrouter.ai/sign-up',
        'css:input[name=email],css:input[type=email]')
    if page is None:
        _save_account('openrouter', email, password,
                      backend=(session or {}).get('backend', ''),
                      extra={'mail_session': session} if session else None)
        return False, 'openrouter signup form never appeared (WAF/egress)'
    try:
        if not _fill_first(page, ['css:input[name=email]',
                                  'css:input[type=email]'], email):
            return False, 'openrouter signup: email field unusable'
        _fill_first(page, ['css:input[name=password]',
                           'css:input[type=password]'], password)
        try:
            page('css:button[type=submit]').click(by_js=False)
        except Exception:  # noqa: BLE001
            try:
                page.ele('css:input[type=password]').input('\n')
            except Exception:  # noqa: BLE001
                pass
        time.sleep(8)
        _save_account('openrouter', email, password,
                      backend=(session or {}).get('backend', ''),
                      extra={'mail_session': session} if session else None)
        if session is None:
            session = _mail_session_for('openrouter', email)
        if session:
            link = mailgen.fetch_otp(
                session, max_wait_s=300, sender_needle='openrouter',
                code_re=re.compile(r'(https://openrouter\.ai/[^\s"\'<>]+)'))
            if link:
                _log_history('openrouter', 'stage', 'verification link found')
                try:
                    page.get(link, timeout=40)
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(8)
        raw = _mint_key_via_page(
            page, '/api/v1/keys', json.dumps({'name': 'dsf-bot'}))
        key = ''
        if raw:
            try:
                key = ((json.loads(raw) or {}).get('data')
                       or {}).get('key', '')
            except Exception:  # noqa: BLE001
                key = ''
        if not key:
            m = re.search(r'sk-or-v1-[A-Za-z0-9]{16,}', _body_head(page))
            key = m.group(0) if m else ''
        if key:
            ok, detail = _or_verify_key(key)
            if ok:
                _save_jar('openrouter', {'api_key': key, 'email': email})
                return True, 'openrouter account created; api key saved'
            _log_history('openrouter', 'stage',
                         f'minted key rejected: {detail}')
        return False, ('signup submitted but no key yet — next cycle logs '
                       'in and mints it')
    finally:
        try:
            page.quit()
        except Exception:  # noqa: BLE001
            pass


def signup_groq() -> Tuple[bool, str]:
    """Create a console.groq.com account and mint an API key.

    The console is client-rendered behind Cloudflare; flow is best-effort
    with defensive selectors: email/password signup, verification link,
    then the /keys page — click Create API Key and scrape the gsk_ token.
    Accounts persist in data/accounts.json so later cycles converge.
    """
    email, password = _creds('groq')
    session = None
    if not email or not password:
        if not mailgen.autogen_enabled():
            return False, ('no GROQ_LOGIN_EMAIL/PASSWORD and mail autogen '
                           'off — Groq needs an account before a key can be '
                           'minted')
        session, err = mailgen.create_email()
        if not session:
            return False, f'autogen mailbox unavailable: {err}'
        email = session['address']
        password = mailgen.gen_password()
    _log_history('groq', 'stage', f'signup as {email}')
    page = _hf_page_with_egress(
        'https://console.groq.com/sign-up',
        'css:input[type=email],css:input[name=email]')
    if page is None:
        _save_account('groq', email, password,
                      backend=(session or {}).get('backend', ''),
                      extra={'mail_session': session} if session else None)
        return False, 'groq signup form never appeared (WAF/egress)'
    try:
        if not _fill_first(page, ['css:input[name=email]',
                                  'css:input[type=email]'], email):
            return False, 'groq signup: email field unusable'
        _fill_first(page, ['css:input[name=password]',
                           'css:input[type=password]'], password)
        try:
            page('css:button[type=submit]').click(by_js=False)
        except Exception:  # noqa: BLE001
            try:
                page.ele('css:input[type=password]').input('\n')
            except Exception:  # noqa: BLE001
                pass
        time.sleep(8)
        _save_account('groq', email, password,
                      backend=(session or {}).get('backend', ''),
                      extra={'mail_session': session} if session else None)
        if session is None:
            session = _mail_session_for('groq', email)
        if session:
            link = mailgen.fetch_otp(
                session, max_wait_s=300, sender_needle='groq',
                code_re=re.compile(r'(https://[^\s"\'<>]*groq[^\s"\'<>]+)'))
            if link:
                _log_history('groq', 'stage', 'verification link found')
                try:
                    page.get(link, timeout=40)
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(8)
        # keys page: click Create, then scrape the one-time gsk_ token
        try:
            page.get('https://console.groq.com/keys', timeout=40)
        except Exception:  # noqa: BLE001
            pass
        time.sleep(5)
        _click_any(page, ['Create API Key', 'New API Key', 'Create key'])
        time.sleep(6)
        m = re.search(r'gsk_[A-Za-z0-9]{20,}', _body_head(page))
        if m:
            key = m.group(0)
            ok, detail = _groq_verify_key(key)
            if ok:
                _save_jar('groq', {'api_key': key, 'email': email})
                return True, 'groq account created; api key saved'
            _log_history('groq', 'stage', f'minted key rejected: {detail}')
        return False, ('signup submitted but no key yet — next cycle logs '
                       'in and mints it')
    finally:
        try:
            page.quit()
        except Exception:  # noqa: BLE001
            pass


REFRESH = {'gemini': refresh_gemini, 'chatgpt': refresh_chatgpt,
           'deepseek': refresh_deepseek, 'claude': refresh_claude,
           'grok': refresh_grok, 'qwen': refresh_qwen, 'kimi': refresh_kimi,
           'mistral': refresh_mistral, 'copilot': _anonymous('copilot'),
           'perplexity': _anonymous('perplexity'), 'glm': _anonymous('glm'),
           'huggingface': refresh_huggingface,
           'openrouter': refresh_openrouter, 'groq': refresh_groq}


# ------------------------------------------------------------------ IMAP OTP
def imap_otp(max_wait_s: int = 120, to_needle: Optional[str] = None) -> Optional[str]:
    """Poll the configured IMAP mailbox for a fresh verification code.

    ``to_needle`` restricts matches to mails addressed to that recipient —
    used by the catch-all autogen backend so unrelated codes are ignored.
    """
    host = os.getenv('DSF_MAIL_IMAP_HOST', '').strip()
    if not host:
        return None
    user = os.getenv('DSF_MAIL_IMAP_USER', '').strip()
    password = os.getenv('DSF_MAIL_IMAP_PASS', '').strip()
    port = int(os.getenv('DSF_MAIL_IMAP_PORT', '993') or 993)
    sender_needle = os.getenv('DSF_MAIL_OTP_SENDER', 'deepseek').strip().lower()
    code_re = re.compile(os.getenv('DSF_MAIL_OTP_REGEX', r'\b(\d{6})\b'))
    max_age_min = float(os.getenv('DSF_MAIL_OTP_MAX_AGE', '30') or 30)
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        try:
            box = imaplib.IMAP4_SSL(host, port, timeout=15)
            box.login(user, password)
            box.select('INBOX')
            since = date.today().strftime('%d-%b-%Y')
            typ, data = box.search(None, f'(SINCE "{since}")')
            for num in reversed((data[0] or b'').split()):
                typ, msg_data = box.fetch(num, '(RFC822)')
                if not msg_data or not msg_data[0]:
                    continue
                msg = message_from_bytes(msg_data[0][1])
                if to_needle:
                    recipients = ' '.join(str(msg.get(h, ''))
                                          for h in ('To', 'Delivered-To',
                                                    'X-Original-To')).lower()
                    if to_needle.lower() not in recipients:
                        continue
                sender = str(msg.get('From', '')).lower()
                if sender_needle and sender_needle not in sender:
                    continue
                age = _mail_age_min(msg)
                if age is not None and age > max_age_min:
                    continue
                body = _mail_body(msg)
                match = code_re.search(body)
                if match:
                    try:
                        box.logout()
                    except Exception:  # noqa: BLE001
                        pass
                    return match.group(1) or match.group(0)
            try:
                box.logout()
            except Exception:  # noqa: BLE001
                pass
        except Exception:  # noqa: BLE001
            pass
        time.sleep(6)
    return None


def _mail_age_min(msg) -> Optional[float]:
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(str(msg.get('Date', '')))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds() / 60.0
    except Exception:
        return None


def _mail_body(msg) -> str:
    parts: List[str] = [str(msg.get('Subject', ''))]
    for part in msg.walk():
        ctype = part.get_content_type()
        if ctype not in ('text/plain', 'text/html'):
            continue
        payload = part.get_payload(decode=True)
        if not payload:
            continue
        charset = part.get_content_charset() or 'utf-8'
        try:
            parts.append(payload.decode(charset, errors='ignore'))
        except (LookupError, UnicodeDecodeError):
            parts.append(payload.decode('utf-8', errors='ignore'))
    return '\n'.join(parts)


# ------------------------------------------------------------ browser layer
def _signup_proxy() -> Optional[str]:
    """Egress for signup browsers.

    DeepSeek (CloudFront) blocks some datacenter/host IPs outright, so
    signups prefer an explicit ``DSF_SIGNUP_PROXY``; otherwise the ladder
    falls through to the dynamic pool and finally direct. Tor is never
    used. Returns None = direct connection.
    """
    explicit = os.getenv('DSF_SIGNUP_PROXY', '').strip()
    if explicit:
        return explicit
    return None


_DISPLAY = None  # pyvirtualdisplay handle kept alive for non-headless runs


def _ensure_display() -> bool:
    """Best-effort X server for non-headless runs. Returns True when a
    DISPLAY is available (existing socket, pyvirtualdisplay, or env)."""
    global _DISPLAY
    if os.environ.get('DISPLAY'):
        return True
    import glob as _glob
    sockets = sorted(_glob.glob('/tmp/.X11-unix/X[0-9]*'))
    if sockets:
        os.environ['DISPLAY'] = f':{sockets[0].rsplit("X", 1)[1]}'
        return True
    try:
        from pyvirtualdisplay import Display
        _DISPLAY = Display(visible=False, size=(1440, 900))
        _DISPLAY.start()
        os.environ['DISPLAY'] = _DISPLAY.new_display_var
        return True
    except Exception:  # noqa: BLE001
        return False


def _browser(proxy: Optional[str] = None, headed: bool = False):
    from DrissionPage import ChromiumPage, ChromiumOptions
    options = ChromiumOptions().auto_port()
    options.set_argument('--no-sandbox')
    options.set_argument('--disable-gpu')
    # Aliyun's slider scores the client: hide automation and run windowed
    # (real Chrome under Xvfb) whenever the rung asks for non-headless.
    options.set_argument('--disable-blink-features=AutomationControlled')
    options.set_argument('--window-size=1440,900')
    if proxy:
        if proxy.startswith('socks'):
            # DrissionPage's set_proxy only speaks HTTP; chromium itself
            # handles SOCKS via the command line. Chromium accepts the
            # plain "socks5://" scheme only ("socks5h://" is a curl-ism
            # and yields ERR_NO_SUPPORTED_PROXIES); DNS is forced through
            # the proxy with a resolver rule so the exit stays consistent.
            scheme, _, hostport = proxy.partition('://')
            host = hostport.split('/')[0].split(':')[0]
            options.set_argument(f'--proxy-server=socks5://{hostport}')
            options.set_argument(
                f'--host-resolver-rules=MAP * ~NOTFOUND , EXCLUDE {host}')
        else:
            options.set_proxy(proxy)
    # headed=True forces a windowed real Chrome (anti-bot services score
    # headless clients far lower — Aliyun slider, Cloudflare, Google) and
    # degrades to headless only when no X server can be obtained.
    if headed:
        if not _ensure_display():
            options.headless(True)  # no X server obtainable -> degrade quietly
    elif _env_bool('DSF_REFRESHER_HEADLESS', True):
        options.headless(True)
    elif not _ensure_display():
        options.headless(True)  # no X server obtainable -> degrade quietly
    return ChromiumPage(addr_or_opts=options)


def _fill_first(page, selectors: List[str], value: str) -> bool:
    for sel in selectors:
        try:
            ele = page.ele(sel, timeout=5)
            if ele:
                ele.clear()
                ele.input(value)
                return True
        except Exception:  # noqa: BLE001
            continue
    return False


def _body_head(page) -> str:
    """First 300 chars of the rendered body (lowercased), '' on failure."""
    try:
        return page.ele('tag:body').text[:300].lower()
    except Exception:  # noqa: BLE001 — detached/blank page
        return ''


_NET_LOG_JS = r"""
window.__dsf_log = window.__dsf_log || [];
if (!window.__dsf_hooked) {
  window.__dsf_hooked = true;
  const of = window.fetch;
  window.fetch = function(...a){
    return of.apply(this, a).then(r => {
      try { const c = r.clone();
        c.text().then(t => window.__dsf_log.push(
          [String((a[0]&&a[0].url)||a[0]), r.status, String(t).slice(0,500)])); }
      catch(e){}
      return r;
    });
  };
  const oo = XMLHttpRequest.prototype.open;
  const os = XMLHttpRequest.prototype.send;
  XMLHttpRequest.prototype.open = function(m,u){ this.__u=u; return oo.apply(this,arguments); };
  XMLHttpRequest.prototype.send = function(b){
    this.addEventListener('load', () => {
      try { window.__dsf_log.push(
        [String(this.__u), this.status, String(this.responseText).slice(0,500)]); }
      catch(e){}
    });
    return os.apply(this, arguments);
  };
}
"""


def _net_log_install(page) -> None:
    """Record fetch/XHR responses on the page (signup API verdicts)."""
    try:
        page.run_js(_NET_LOG_JS)
    except Exception:  # noqa: BLE001 — diagnostics only
        pass


def _net_log_read(page, needle: str = '', limit: int = 6) -> str:
    """Last few recorded request/response pairs, filtered by URL ``needle``."""
    try:
        log = page.run_js('return window.__dsf_log || [];') or []
    except Exception:  # noqa: BLE001 — diagnostics only
        return ''
    out = []
    for entry in log:
        try:
            url, status, body = entry[0], entry[1], entry[2]
        except Exception:  # noqa: BLE001
            continue
        if needle and needle.lower() not in str(url).lower():
            continue
        out.append(f'{status} {url} -> {str(body)[:200]}')
    return ' || '.join(out[-limit:])


def _click_any(page, targets: List[str]) -> bool:
    for t in targets:
        try:
            ele = page.ele(f'text:{t}', timeout=4)
            if ele:
                ele.click()
                return True
        except Exception:  # noqa: BLE001
            continue
    return False


def _select_first(page, selectors: List[str], value: str) -> bool:
    for sel in selectors:
        try:
            ele = page.ele(sel, timeout=3)
            if ele:
                ele.select.by_text(value)
                return True
        except Exception:  # noqa: BLE001
            continue
    return False


def _wait_token(page, timeout_s: int = 150) -> Optional[str]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            raw = page.run_js('return window.localStorage.getItem("userToken");')
            if raw:
                try:
                    return str(json.loads(raw).get('value') or '').strip() or str(raw)
                except (ValueError, AttributeError):
                    return str(raw)
        except Exception:  # noqa: BLE001
            pass
        time.sleep(3)
    return None


def _export_cookies(page, name: str, domains: Tuple[str, ...]) -> int:
    try:
        cookies = page.cookies(all_domains=True) or []
    except TypeError:
        cookies = page.cookies() or []
    updates = {str(c.get('name')): str(c.get('value')) for c in cookies
               if isinstance(c, dict) and c.get('name') and c.get('value')
               and any(d in str(c.get('domain', '')) for d in domains)}
    if updates:
        _save_jar(name, updates)
    return len(updates)


def _creds(name: str) -> Tuple[str, str]:
    """Login credentials: env first, then accounts the bot created itself
    (data/accounts.json, written by the signup rungs)."""
    prefix = {'deepseek': 'DEEPSEEK', 'chatgpt': 'CHATGPT', 'gemini': 'GEMINI',
              'claude': 'CLAUDE', 'grok': 'GROK', 'mistral': 'MISTRAL',
              'qwen': 'QWEN', 'kimi': 'KIMI', 'copilot': 'COPILOT',
              'perplexity': 'PERPLEXITY', 'glm': 'GLM',
              'openrouter': 'OPENROUTER', 'groq': 'GROQ',
              'huggingface': 'HUGGINGFACE'}.get(name, name.upper())
    email = os.getenv(f'{prefix}_LOGIN_EMAIL', '').strip()
    password = os.getenv(f'{prefix}_LOGIN_PASSWORD', '').strip()
    if email and password:
        return email, password
    stored = _load_accounts().get(name) or {}
    return (stored.get('email') or email, stored.get('password') or password)


_DEEPSEEK_EMAIL_SELECTORS = ['@placeholder:email', '@placeholder:Email',
                             'css:input[type=text]', 'css:input[name=email]']
_CHATGPT_EMAIL_SELECTORS = ['css:input[name=email]', 'css:input[type=email]',
                            '@placeholder:Email address']
_GEMINI_EMAIL_SELECTORS = ['css:input[type=email]', '@placeholder:Email or phone']
_PASSWORD_SELECTORS = ['css:input[type=password]']


def browser_login(name: str) -> Tuple[bool, str]:
    """Headless re-login; exports fresh cookies/token into the data dir."""
    email, password = _creds(name)
    if not email or not password:
        return False, f'{name}: no login credentials configured'
    if name == 'qwen':
        # chat.qwen.ai /signin is not WAF-gated -> plain HTTP re-login,
        # no browser needed (and /signup sits behind an Aliyun slider,
        # so the API path is strictly more reliable here)
        ok, detail = _qwen_signin(email, password)
        if ok:
            return ok, detail
        if _qwen_pending(detail):
            # account exists but is unactivated: poll the signup mailbox for
            # the activation link, open it, then re-login — no operator needed
            ok_a, detail_a = _qwen_activate(email, password)
            if ok_a:
                return _qwen_signin(email, password)
            return False, f'{detail}; {detail_a}'
        return ok, detail
    proxy = _deepseek_egress() if name == 'deepseek' else None
    try:
        # headed: CloudFront's WAF hard-403s headless clients but serves the
        # SPA (JS challenge -> aws-waf-token) to a windowed real Chrome
        page = _browser(proxy=proxy, headed=True)
    except Exception as e:  # noqa: BLE001
        return False, f'browser unavailable: {e}'
    try:
        if name == 'deepseek':
            # Root SPA first: /sign_in document GETs are CloudFront-403'd,
            # client-side routing is not.
            page.get('https://chat.deepseek.com/')
            time.sleep(6)
            _net_log_install(page)
            root_head = ((page.title or '') + ' ' + _body_head(page)).lower()
            if ('could not be satisfied' in root_head
                    or '403 error' in root_head):
                return False, (
                    f'CloudFront 403 via {proxy or "direct"} — DeepSeek '
                    f'blocks datacenter/host IPs; set DSF_SIGNUP_PROXY to '
                    f'a RESIDENTIAL proxy to unblock signup')
            if not _click_any(page, ['Log in', 'Login', '登录']):
                page.get('https://chat.deepseek.com/sign_in')
            time.sleep(4)
            if not _fill_first(page, _DEEPSEEK_EMAIL_SELECTORS, email):
                return False, 'email field not found'
            _fill_first(page, _PASSWORD_SELECTORS, password)
            _click_any(page, ['Log In', 'Log in', 'Sign In', '登录'])
            # some flows demand an emailed code
            if _fill_first(page, ['@placeholder:code', '@placeholder:Code',
                                  'css:input[name=code]'], ' ', timeout=2):
                code = imap_otp()
                if not code:
                    return False, 'login needs an email code but OTP not found'
                _fill_first(page, ['@placeholder:code', '@placeholder:Code',
                                   'css:input[name=code]'], code)
                _click_any(page, ['Log In', 'Verify', '确认', '验证'])
            token = _wait_token(page)
            _export_cookies(page, 'deepseek', ('deepseek.com',))
            if token:
                _save_deepseek_token(token)
                return True, 'userToken captured from browser session'
            return False, 'login finished but no userToken in localStorage'
        if name == 'chatgpt':
            page.get('https://chatgpt.com/auth/login')
            time.sleep(5)
            _click_any(page, ['Log in', 'Log In', 'Sign up'])
            time.sleep(4)
            if not _fill_first(page, _CHATGPT_EMAIL_SELECTORS, email):
                return False, 'email field not found (bot wall?)'
            _click_any(page, ['Continue', 'Next'])
            time.sleep(3)
            _fill_first(page, _PASSWORD_SELECTORS, password)
            _click_any(page, ['Continue', 'Log in'])
            time.sleep(12)
            n = _export_cookies(page, 'chatgpt', ('chatgpt.com', 'openai.com'))
            return (n > 0), f'{n} session cookies exported'
        if name == 'gemini':
            # gemini (Google) — best effort, heavy anti-bot
            page.get('https://accounts.google.com/ServiceLogin')
            time.sleep(4)
            if not _fill_first(page, _GEMINI_EMAIL_SELECTORS, email):
                return False, 'google email field not found'
            _click_any(page, ['Next', 'Weiter'])
            time.sleep(4)
            _fill_first(page, _PASSWORD_SELECTORS, password)
            _click_any(page, ['Next', 'Weiter'])
            time.sleep(12)
            page.get('https://gemini.google.com/app')
            time.sleep(6)
            n = _export_cookies(page, 'gemini', ('google.com',))
            return (n > 0), f'{n} google cookies exported (2FA/anti-bot may block)'
        # claude / grok / kimi / mistral: their credentials are HTTP-only
        # tokens (sessionKey / sso / JWT / Ory session) that no login form
        # re-issues — nothing to rotate in a browser here.
        return False, f'{name}: browser re-login not applicable'
    except Exception as e:  # noqa: BLE001
        return False, f'browser flow failed: {type(e).__name__}: {e}'
    finally:
        try:
            page.quit()
        except Exception:  # noqa: BLE001
            pass


def _cool(proxy: Optional[str]) -> None:
    """Cooldown a blocked egress (mark_failure); safe on None/direct."""
    if not proxy:
        return
    try:
        from . import proxies as _proxies
        _proxies.mark_failure(proxy)
    except Exception:  # noqa: BLE001 — rotation is best-effort
        pass


def _ds_egress_ok(proxy: Optional[str]) -> bool:
    """Cheap CloudFront reachability probe for a signup/login egress.

    chat.deepseek.com hard-403s document GETs by IP reputation; a *browser*
    session (headed Chrome) solves the AWS WAF JS challenge when the IP is
    merely challenged rather than blocked. ``proxy=None`` probes the direct
    egress, so the ladder can prefer local traffic when the host IP passes.
    """
    try:
        from .providers.base import http_get
        r = http_get('https://chat.deepseek.com/',
                     proxies=({'http': proxy, 'https': proxy} if proxy
                              else None),
                     timeout=12)
    except Exception:  # noqa: BLE001 — treat as unusable
        return False
    if r.status_code == 200:
        return True
    # 202 + goku = AWS WAF JS challenge: the BROWSER solves it, so the exit
    # is usable. Only the hard CloudFront 403 ("could not be satisfied")
    # means the IP is blocked and the browser would fail too.
    body = (r.text[:500] or '').lower()
    return not ('could not be satisfied' in body or '403 error' in body)


def _pool_egresses(limit: int = 3, samples: int = 6) -> List[str]:
    """Up to ``limit`` DISTINCT pool exits that pass the reachability probe.

    Samples the sticky pool, cools down blocked exits, and returns only
    egresses CloudFront currently lets through."""
    try:
        from . import proxies as _proxies
        _proxies.ensure_pool()
    except Exception:  # noqa: BLE001 — direct remains the fallback
        pass
    out: List[str] = []
    seen: set = set()
    for _ in range(samples):
        if len(out) >= limit:
            break
        p = _pool_proxy()
        if not p:
            break
        if p in seen:
            _cool(p)  # sticky assignment: rotate the exit away, re-sample
            p = _pool_proxy()
            if not p or p in seen:
                break
        seen.add(p)
        if _ds_egress_ok(p):
            out.append(p)
        else:
            _cool(p)  # blocked exit: cooldown + force a fresh one
    return out


def _deepseek_egress() -> Optional[str]:
    """Best single egress for a DeepSeek browser session (login/renewal).

    Direct traffic wins whenever the host IP passes the reachability probe
    (a headed browser solves the AWS WAF challenge); free-proxy pool exits
    are only consulted when direct is hard-blocked.
    """
    explicit = _signup_proxy()
    if explicit and _ds_egress_ok(explicit):
        return explicit
    if _ds_egress_ok(None):
        return None
    egresses = _pool_egresses(limit=1, samples=4)
    return egresses[0] if egresses else None


def _pool_proxy() -> Optional[str]:
    """A random egress from the dynamic free-proxy pool (when enabled).

    ``direct_ok=False`` — the signup ladder already has its own direct
    rung, so the pool must never hand back the no-proxy sentinel.
    """
    try:
        from . import proxies as _proxies
        return _proxies.get_proxy('deepseek-signup', direct_ok=False)
    except Exception:  # pragma: no cover
        return None


def signup_deepseek() -> Tuple[bool, str]:
    """Create a fresh DeepSeek account — fully autonomous when possible.

    Credentials ladder:
      1. DEEPSEEK_LOGIN_EMAIL / DEEPSEEK_LOGIN_PASSWORD if configured;
      2. otherwise an auto-generated throwaway mailbox (dsk/mailgen.py):
         catch-all IMAP domain when DSF_MAIL_DOMAIN is set, else a mail.tm
         temp account. The verification code is read from that mailbox, so
         no human and no pre-existing account are needed.
    """
    email, password = _creds('deepseek')
    session = None
    generated = False
    if not email or not password:
        if not mailgen.autogen_enabled():
            return False, 'no DEEPSEEK_LOGIN_EMAIL/PASSWORD and mail autogen off'
        session, err = mailgen.create_email()
        if not session:
            return False, f'autogen mailbox unavailable: {err}'
        email = session['address']
        # mailbox password: the signup form needs one; tempmail.lol sessions
        # don't carry one (the inbox is token-addressed), so mint a form
        # password independent of the mailbox credentials.
        password = session.get('password') or mailgen.gen_password()
        generated = True
    # egress ladder: explicit DSF_SIGNUP_PROXY first, then up to 3 distinct
    # dynamic-pool exits that PASS the root-page reachability probe, then
    # direct (duplicates dropped). Tor is never used. Blocked exits are
    # cooled down so the next rung samples a fresh, hopefully-working IP
    # instead of the same blocked one.
    ladder: List[Optional[str]] = []
    seen: set = set()
    # direct first when the host IP passes the (cheap) reachability probe —
    # the free-proxy pool is unreliable and each dead rung costs a browser
    # startup; explicit DSF_SIGNUP_PROXY and pool exits follow, direct last
    # as the always-present fallback.
    ordered: List[Optional[str]] = []
    if _ds_egress_ok(None):
        ordered.append(None)
    if _signup_proxy():
        ordered.append(_signup_proxy())
    ordered += _pool_egresses()
    ordered.append(None)
    for p in ordered:
        if p is None or p not in seen:
            ladder.append(p)
            if p is not None:
                seen.add(p)
    for proxy in ladder or [None]:
        page = None
        try:
            # headed: the CloudFront WAF blocks headless clients outright
            # (403 "request blocked") while a windowed Chrome passes the JS
            # challenge and reaches the SPA — the direct rung is the reliable
            # one, so run it like a real browser.
            page = _browser(proxy=proxy, headed=True)
            # CloudFront 403s document GETs of /sign_up by IP reputation,
            # but the root SPA loads and routes to /sign_up CLIENT-SIDE
            # (no document request -> no WAF block). Root first, click
            # through; fall back to the direct document GET only if the
            # SPA entry point is missing.
            page.get('https://chat.deepseek.com/')
            time.sleep(6)
            root_head = ((page.title or '') + ' ' + _body_head(page)).lower()
            if ('could not be satisfied' in root_head
                    or '403 error' in root_head):
                last_error = (
                    f'CloudFront 403 via {proxy or "direct"} — DeepSeek '
                    f'blocks datacenter/host IPs; set DSF_SIGNUP_PROXY to '
                    f'a RESIDENTIAL proxy to unblock signup')
                _log_history('deepseek', 'signup-blocked', last_error)
                _cool(proxy)  # blocked exit: cooldown + force a fresh one
                continue
            if not _click_any(page, ['Sign up', 'Sign Up', '注册']):
                page.get('https://chat.deepseek.com/sign_up')
            time.sleep(4)
            body_head = _body_head(page)
            if ('could not be satisfied' in body_head
                    or '403 error' in body_head):
                last_error = (
                    f'CloudFront 403 via {proxy or "direct"} — DeepSeek '
                    f'blocks datacenter/host IPs; set DSF_SIGNUP_PROXY to '
                    f'a RESIDENTIAL proxy to unblock signup')
                _log_history('deepseek', 'signup-blocked', last_error)
                _cool(proxy)  # blocked exit: cooldown + force a fresh one
                continue
            if not _fill_first(page, _DEEPSEEK_EMAIL_SELECTORS, email):
                last_error = 'email field not found'
                continue
            _fill_first(page, _PASSWORD_SELECTORS, password)
            clicked_send = _click_any(page, ['Send Code', 'Send code',
                                            '获取验证码'])
            # Capture the form's reaction to the send: a visible error means
            # the request was refused (rate limit, captcha, domain rejected
            # with a UI message); a silent accept followed by no OTP means
            # the mail was delivered nowhere (domain dropped server-side).
            time.sleep(4)
            send_feedback = _body_head(page).lower()
            send_note = ''
            for needle in ('too many', 'rate limit', 'captcha', 'verify',
                           'invalid', 'exist', 'error', 'failed'):
                if needle in send_feedback:
                    send_note = f' (page feedback: {needle})'
                    break
            send_api = _net_log_read(page, needle='code')
            _log_history('deepseek', 'stage',
                         f'send-code seen ({email}) clicked={clicked_send} '
                         + (send_api or 'no code API call recorded')
                         + ' || ALL: ' + (_net_log_read(page) or 'none'))
            if 'exist' in send_feedback and generated and session:
                # shared gmail dot/plus variants may already be registered —
                # mint a fresh mailbox and resend on this rung
                fresh, _ = mailgen.create_email()
                if fresh:
                    session = fresh
                    email = fresh['address']
                    password = (fresh.get('password')
                                or mailgen.gen_password())
                    _fill_first(page, _DEEPSEEK_EMAIL_SELECTORS, email)
                    _click_any(page, ['Send Code', 'Send code', '获取验证码'])
                    time.sleep(4)
                    send_note += ' (mailbox regenerated)'
            if generated:
                code = mailgen.fetch_otp(session, max_wait_s=180)
                if not code:
                    # fall back to the plain IMAP poller (no recipient filter)
                    code = imap_otp(max_wait_s=30)
            else:
                code = imap_otp(max_wait_s=180)
            if not code:
                return False, ('signup code email not found in mailbox — '
                               'no OTP arrived (gmail + disposable backends '
                               'tried); configure DSF_MAIL_DOMAIN + '
                               'DSF_MAIL_IMAP_HOST with a catch-all inbox '
                               'for guaranteed delivery'
                               + send_note)
            if not _fill_first(page, ['@placeholder:code', '@placeholder:Code',
                                      'css:input[name=code]'], code):
                return False, 'code field not found'
            _click_any(page, ['Sign Up', 'Sign up', '注册'])
            # Harvest: the fresh session's userToken lands in localStorage
            # once the SPA logs in — quit() without capturing it would throw
            # the whole signup away. Token + credentials both persisted so
            # every later renewal re-logs in instead of re-signing-up.
            time.sleep(8)
            token = _wait_token(page, timeout_s=90)
            _export_cookies(page, 'deepseek', ('deepseek.com',))
            if token:
                _save_deepseek_token(token)
                if generated:
                    _save_account('deepseek', email, password,
                                  (session or {}).get('backend', ''))
                _log_history('deepseek', 'signup-token',
                             f'captured via {proxy or "direct"}')
                return True, (f'account created and userToken captured '
                              f'({(session or {}).get("backend", "manual")}: '
                              f'{email})')
            return False, ('signup submitted but no userToken in localStorage '
                           '(verification may still be pending)')
        except Exception as e:  # noqa: BLE001
            last_error = f'signup flow failed: {type(e).__name__}: {e}'
            _log_history('deepseek', 'egress-failed',
                         f'{proxy or "direct"}: {last_error}')
        finally:
            if page is not None:
                try:
                    page.quit()
                except Exception:  # noqa: BLE001
                    pass
    return False, last_error or 'all signup egresses failed'


def signup_chatgpt() -> Tuple[bool, str]:
    """Create a fresh ChatGPT account — fully autonomous (best effort).

    Uses an auto-generated throwaway mailbox (dsk/mailgen.py) for the
    verification code. OpenAI may still show an Arkose captcha or demand
    phone verification for some IPs; those cases end the attempt with a
    clear detail string and the ladder records a normal miss. Created
    account credentials are persisted so later renewals can re-login."""
    if not mailgen.autogen_enabled():
        return False, 'mail autogen disabled (DSF_MAIL_AUTOGEN=false)'
    session, err = mailgen.create_email()
    if not session:
        return False, f'autogen mailbox unavailable: {err}'
    email = session['address']
    # token-addressed backends (emailnator, tempmail.lol) carry no mailbox
    # password: mint a form password independent of the mailbox creds.
    password = session.get('password') or mailgen.gen_password()
    try:
        page = _browser(headed=True)
    except Exception as e:  # noqa: BLE001
        return False, f'browser unavailable: {e}'
    try:
        page.get('https://chatgpt.com/auth/login')
        time.sleep(6)
        if not _click_any(page, ['Sign up', 'Sign Up', 'Create account']):
            return False, 'sign-up entry not found (bot wall?)'
        time.sleep(4)
        _click_any(page, ['Continue with email', 'Email'])
        time.sleep(2)
        if not _fill_first(page, _CHATGPT_EMAIL_SELECTORS, email):
            return False, 'email field not found (bot wall?)'
        _click_any(page, ['Continue', 'Next'])
        time.sleep(3)
        if not _fill_first(page, _PASSWORD_SELECTORS, password):
            return False, 'password field not found'
        _click_any(page, ['Continue', 'Next'])
        time.sleep(8)
        code = mailgen.fetch_otp(session, max_wait_s=240,
                                 sender_needle='openai')
        if not code:
            return False, 'verification email not found (captcha/phone wall may have blocked signup)'
        if not _fill_first(page, ['css:input[name=code]',
                                  'css:input[inputmode=numeric]',
                                  'css:input[autocomplete=one-time-code]',
                                  '@placeholder:code', '@placeholder:Code',
                                  'css:input[type=text]'], code):
            return False, 'verification code field not found'
        _click_any(page, ['Continue', 'Verify'])
        time.sleep(12)
        _save_account('chatgpt', email, password,
                      session.get('backend', ''))
        n = _export_cookies(page, 'chatgpt', ('chatgpt.com', 'openai.com'))
        via = f'account created ({session["backend"]}: {email})'
        if n > 0:
            return True, f'{via}, {n} session cookies exported'
        return False, f'{via} but no session cookies captured'
    except Exception as e:  # noqa: BLE001
        return False, f'chatgpt signup failed: {type(e).__name__}: {e}'
    finally:
        try:
            page.quit()
        except Exception:  # noqa: BLE001
            pass


def signup_gemini() -> Tuple[bool, str]:
    """Create a fresh Google account for Gemini — best effort.

    Google's anti-bot (captcha, phone verification, unusual-traffic
    checks) blocks most automated attempts; every failure surfaces as a
    ladder miss. Uses an auto-generated mailbox for the verification
    code; the account is persisted for later re-login attempts."""
    if not mailgen.autogen_enabled():
        return False, 'mail autogen disabled (DSF_MAIL_AUTOGEN=false)'
    session, err = mailgen.create_email()
    if not session:
        return False, f'autogen mailbox unavailable: {err}'
    email = session['address']
    # token-addressed backends (emailnator, tempmail.lol) carry no mailbox
    # password: mint a form password independent of the mailbox creds.
    password = session.get('password') or mailgen.gen_password()
    try:
        page = _browser(headed=True)
    except Exception as e:  # noqa: BLE001
        return False, f'browser unavailable: {e}'
    try:
        page.get('https://accounts.google.com/signup/v2/createaccount'
                 '?flowName=GlifWebSignIn&flowEntry=AccountSignUp')
        time.sleep(6)
        if not _fill_first(page, ['css:input#firstName',
                                  'css:input[name=firstName]'], 'Alex'):
            return False, 'google first-name field not found (bot wall?)'
        _fill_first(page, ['css:input#lastName',
                           'css:input[name=lastName]'], 'Free')
        _click_any(page, ['Next', 'Weiter'])
        time.sleep(4)
        _fill_first(page, ['css:input#day', 'css:input[name=day]'], '12')
        _select_first(page, ['css:select#month'], 'June')
        _fill_first(page, ['css:input#year', 'css:input[name=year]'], '1994')
        _select_first(page, ['css:select#gender'], 'Rather not say')
        _click_any(page, ['Next', 'Weiter'])
        time.sleep(4)
        # prefer the "use existing email" branch so no Gmail is required
        _click_any(page, ['Use your existing email', 'current email address'])
        time.sleep(2)
        if not _fill_first(page, ['css:input#userName',
                                  'css:input[name=userName]',
                                  'css:input[type=email]'], email):
            return False, 'existing-email field not found'
        _click_any(page, ['Next', 'Weiter'])
        time.sleep(3)
        if not _fill_first(page, ['css:input[name=Passwd]',
                                  'css:input[type=password]'], password):
            return False, 'google password field not found'
        _fill_first(page, ['css:input[name=PasswdAgain]'], password)
        _click_any(page, ['Next', 'Weiter'])
        time.sleep(6)
        code = mailgen.fetch_otp(session, max_wait_s=240,
                                 sender_needle='accounts.google')
        if not code:
            return False, 'google verification email not found (captcha/phone wall likely)'
        if not _fill_first(page, ['css:input#code',
                                  'css:input[name=code]'], code):
            return False, 'google code field not found'
        _click_any(page, ['Next', 'Weiter'])
        time.sleep(6)
        _click_any(page, ["Yes, I'm in", 'Skip', 'Not now', 'Confirm'])
        time.sleep(3)
        _click_any(page, ['I agree'])
        time.sleep(8)
        _save_account('gemini', email, password, session.get('backend', ''))
        page.get('https://gemini.google.com/app')
        time.sleep(6)
        n = _export_cookies(page, 'gemini', ('google.com',))
        via = f'account created ({session["backend"]}: {email})'
        if n > 0:
            return True, f'{via}, {n} google cookies exported'
        return False, f'{via} but no cookies captured (2FA/anti-bot?)'
    except Exception as e:  # noqa: BLE001
        return False, f'gemini signup failed: {type(e).__name__}: {e}'
    finally:
        try:
            page.quit()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------- token signups
# claude.ai / grok.com / kimi.com / chat.mistral.ai all expose free email+password
# signup forms; the resulting session lands as an HTTP-only cookie or local JWT
# that the refresher exports automatically — no human, no cookie export.


def signup_claude() -> Tuple[bool, str]:
    """Create a fresh claude.ai account and harvest the sessionKey cookie.

    claude.ai signup is email + password + emailed OTP; the sessionKey
    cookie appears in the jar once the SPA lands in the app. Created
    credentials are persisted for later re-login.
    """
    if not mailgen.autogen_enabled():
        return False, 'mail autogen disabled (DSF_MAIL_AUTOGEN=false)'
    session, err = mailgen.create_email()
    if not session:
        return False, f'autogen mailbox unavailable: {err}'
    email = session['address']
    password = session.get('password') or mailgen.gen_password()
    page = None
    try:
        page = _browser(headed=True)
        page.get('https://claude.ai/login')
        time.sleep(6)
        if not _click_any(page, ['Sign up', 'Create account']):
            page.get('https://claude.ai/signup')
        time.sleep(4)
        if not _fill_first(page, _CHATGPT_EMAIL_SELECTORS, email):
            return False, 'email field not found (bot wall?)'
        _fill_first(page, _PASSWORD_SELECTORS, password)
        _click_any(page, ['Continue', 'Sign up', 'Create account'])
        time.sleep(6)
        code = mailgen.fetch_otp(session, max_wait_s=240,
                                 sender_needle='claude')
        if not code:
            return False, 'claude verification email not found (bot wall/OTP)'
        if not _fill_first(page, ['css:input[name=code]',
                                  '@placeholder:code', '@placeholder:Code',
                                  'css:input[inputmode=numeric]'], code):
            return False, 'code field not found'
        _click_any(page, ['Verify', 'Continue', 'Sign up'])
        time.sleep(12)
        jar_cookies = {}
        try:
            for c in (page.cookies(all_domains=True) or []):
                if 'claude.ai' in str(c.get('domain', '')):
                    jar_cookies[c['name']] = c['value']
        except TypeError:
            for c in (page.cookies() or []):
                if 'claude.ai' in str(c.get('domain', '')):
                    jar_cookies[c['name']] = c['value']
        session_key = jar_cookies.get('sessionKey', '')
        if session_key:
            _save_jar('claude', {'sessionKey': session_key})
            _save_account('claude', email, password, session.get('backend', ''))
            return True, (f'account created, sessionKey harvested '
                          f'({session.get("backend")}: {email})')
        return False, 'signup finished but no sessionKey cookie captured'
    except Exception as e:  # noqa: BLE001
        return False, f'claude signup failed: {type(e).__name__}: {e}'
    finally:
        if page is not None:
            try:
                page.quit()
            except Exception:  # noqa: BLE001
                pass


def signup_grok() -> Tuple[bool, str]:
    """Create a grok.com account (X SSO-less email signup) and export sso."""
    if not mailgen.autogen_enabled():
        return False, 'mail autogen disabled (DSF_MAIL_AUTOGEN=false)'
    session, err = mailgen.create_email()
    if not session:
        return False, f'autogen mailbox unavailable: {err}'
    email = session['address']
    password = session.get('password') or mailgen.gen_password()
    page = None
    try:
        page = _browser(headed=True)
        page.get('https://accounts.x.ai/sign-up')
        time.sleep(6)
        if not _fill_first(page, _CHATGPT_EMAIL_SELECTORS, email):
            if not _click_any(page, ['Sign up', 'Create account', 'Sign in']):
                return False, 'grok signup entry not found (bot wall?)'
            time.sleep(4)
            if not _fill_first(page, _CHATGPT_EMAIL_SELECTORS, email):
                return False, 'email field not found'
        _click_any(page, ['Continue', 'Next'])
        time.sleep(3)
        _fill_first(page, _PASSWORD_SELECTORS, password)
        _click_any(page, ['Continue', 'Sign up'])
        time.sleep(10)
        code = mailgen.fetch_otp(session, max_wait_s=240, sender_needle='x.ai')
        if not code:
            code = mailgen.fetch_otp(session, max_wait_s=60, sender_needle='')
        if not code:
            return False, 'grok verification email not found'
        if not _fill_first(page, ['css:input[name=code]', '@placeholder:code',
                                  'css:input[inputmode=numeric]',
                                  'css:input[type=text]'], code):
            return False, 'code field not found'
        _click_any(page, ['Verify', 'Continue'])
        time.sleep(12)
        page.get('https://grok.com/')
        time.sleep(6)
        jar_cookies = {}
        try:
            for c in (page.cookies(all_domains=True) or []):
                if 'grok.com' in str(c.get('domain', '')):
                    jar_cookies[c['name']] = c['value']
        except TypeError:
            for c in (page.cookies() or []):
                if 'grok.com' in str(c.get('domain', '')):
                    jar_cookies[c['name']] = c['value']
        sso = jar_cookies.get('sso') or jar_cookies.get('sso-rw') or ''
        if sso:
            _save_jar('grok', {'sso': sso, 'sso-rw': sso})
            _save_account('grok', email, password, session.get('backend', ''))
            return True, (f'account created, sso exported '
                          f'({session.get("backend")}: {email})')
        return False, 'signup finished but no sso cookie captured'
    except Exception as e:  # noqa: BLE001
        return False, f'grok signup failed: {type(e).__name__}: {e}'
    finally:
        if page is not None:
            try:
                page.quit()
            except Exception:  # noqa: BLE001
                pass


def signup_kimi() -> Tuple[bool, str]:
    """Create a kimi.com account (phone-free email signup) and save the JWT."""
    if not mailgen.autogen_enabled():
        return False, 'mail autogen disabled (DSF_MAIL_AUTOGEN=false)'
    session, err = mailgen.create_email()
    if not session:
        return False, f'autogen mailbox unavailable: {err}'
    email = session['address']
    password = session.get('password') or mailgen.gen_password()
    page = None
    try:
        page = _browser(headed=True)
        page.get('https://www.kimi.com/')
        time.sleep(6)
        if not _click_any(page, ['Sign up', 'Sign Up', '注册', 'Log in', '登录']):
            return False, 'kimi auth entry not found'
        time.sleep(4)
        # prefer email/password over phone (no phone wall for email)
        _click_any(page, ['Email', '邮箱', 'Password login', '密码登录'])
        time.sleep(2)
        if not _fill_first(page, _CHATGPT_EMAIL_SELECTORS, email):
            return False, 'email field not found'
        _fill_first(page, _PASSWORD_SELECTORS, password)
        _click_any(page, ['Sign up', 'Sign Up', '注册', 'Continue'])
        time.sleep(10)
        code = mailgen.fetch_otp(session, max_wait_s=240, sender_needle='kimi')
        if not code:
            code = mailgen.fetch_otp(session, max_wait_s=60, sender_needle='')
        if not code:
            return False, 'kimi verification email not found'
        if not _fill_first(page, ['css:input[name=code]', '@placeholder:code',
                                  'css:input[inputmode=numeric]',
                                  'css:input[type=text]'], code):
            return False, 'code field not found'
        _click_any(page, ['Verify', 'Continue', '确认'])
        time.sleep(12)
        token = ''
        try:
            token = str(page.run_js(
                'let hit="";'
                'for(let i=0;i<localStorage.length;i++){'
                'const k=localStorage.key(i);const v=localStorage.getItem(k);'
                'if(v&&v.length>40&&/eyJ[A-Za-z0-9_-]/.test(v)){hit=v;break;}}'
                'return hit;') or '')
        except Exception:  # noqa: BLE001
            pass
        if token:
            try:
                token = str(json.loads(token).get('value') or token)
            except (ValueError, AttributeError):
                pass
            _save_jar('kimi', {'token': token, 'email': email})
            _save_account('kimi', email, password, session.get('backend', ''))
            return True, (f'account created, JWT saved '
                          f'({session.get("backend")}: {email})')
        return False, 'signup finished but no JWT found in localStorage'
    except Exception as e:  # noqa: BLE001
        return False, f'kimi signup failed: {type(e).__name__}: {e}'
    finally:
        if page is not None:
            try:
                page.quit()
            except Exception:  # noqa: BLE001
                pass


def signup_mistral() -> Tuple[bool, str]:
    """Create a chat.mistral.ai account and export the Ory session cookie."""
    if not mailgen.autogen_enabled():
        return False, 'mail autogen disabled (DSF_MAIL_AUTOGEN=false)'
    session, err = mailgen.create_email()
    if not session:
        return False, f'autogen mailbox unavailable: {err}'
    email = session['address']
    password = session.get('password') or mailgen.gen_password()
    page = None
    try:
        page = _browser(headed=True)
        page.get('https://auth.mistral.ai/ui/registration')
        time.sleep(6)
        if not _fill_first(page, _CHATGPT_EMAIL_SELECTORS, email):
            return False, 'email field not found (bot wall?)'
        _fill_first(page, _PASSWORD_SELECTORS, password)
        _fill_first(page, ['css:input[name=reveal_password]',
                           'css:input[name=confirm_password]'], password)
        _click_any(page, ['Create an account', 'Sign up', 'Continue'])
        time.sleep(10)
        code = mailgen.fetch_otp(session, max_wait_s=240,
                                 sender_needle='mistral')
        if not code:
            code = mailgen.fetch_otp(session, max_wait_s=60, sender_needle='')
        if not code:
            return False, 'mistral verification email not found'
        if not _fill_first(page, ['css:input[name=code]', '@placeholder:code',
                                  'css:input[name=code*]',
                                  'css:input[inputmode=numeric]',
                                  'css:input[type=text]'], code):
            return False, 'code field not found'
        _click_any(page, ['Verify', 'Continue', 'Submit'])
        time.sleep(12)
        page.get('https://chat.mistral.ai/chat')
        time.sleep(6)
        jar_cookies = {}
        try:
            for c in (page.cookies(all_domains=True) or []):
                if 'mistral' in str(c.get('domain', '')):
                    jar_cookies[c['name']] = c['value']
        except TypeError:
            for c in (page.cookies() or []):
                if 'mistral' in str(c.get('domain', '')):
                    jar_cookies[c['name']] = c['value']
        if jar_cookies.get('ory_kratos_session'):
            _save_jar('mistral', jar_cookies)
            _save_account('mistral', email, password, session.get('backend', ''))
            return True, (f'account created, session cookie exported '
                          f'({session.get("backend")}: {email})')
        return False, 'signup finished but no Ory session cookie captured'
    except Exception as e:  # noqa: BLE001
        return False, f'mistral signup failed: {type(e).__name__}: {e}'
    finally:
        if page is not None:
            try:
                page.quit()
            except Exception:  # noqa: BLE001
                pass


_QWEN_SIGNUP_URL = 'https://chat.qwen.ai/auth?action=signup'
_QWEN_NAME_SELECTORS = ['@placeholder:Full Name', 'css:input[type=text]']
_QWEN_EMAIL_SELECTORS = ['@placeholder:Email', 'css:input[type=email]']


def _largest_component(mask):
    """Bounding box + size of the largest 4-connected True region."""
    import numpy as np
    from collections import deque
    H, W = mask.shape
    lbl = np.zeros(mask.shape, dtype=int)
    best, bestn = None, 0
    cur = 0
    for y in range(H):
        for x in range(W):
            if mask[y, x] and lbl[y, x] == 0:
                cur += 1
                q = deque([(y, x)])
                lbl[y, x] = cur
                n = 0
                x0 = x1 = x
                y0 = y1 = y
                while q:
                    cy, cx = q.popleft()
                    n += 1
                    x0, x1 = min(x0, cx), max(x1, cx)
                    y0, y1 = min(y0, cy), max(y1, cy)
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = cy + dy, cx + dx
                        if (0 <= ny < H and 0 <= nx < W and mask[ny, nx]
                                and lbl[ny, nx] == 0):
                            lbl[ny, nx] = cur
                            q.append((ny, nx))
                if n > bestn:
                    bestn, best = n, (x0, x1, y0, y1, n)
    return best


def _masked_ncc(gray_b, gray_p, mask):
    """Best normalized cross-correlation of the masked piece patch over the
    background. Returns [(score, bx, by), ...] best-first."""
    import numpy as np
    ph, pw = gray_p.shape
    bh, bw = gray_b.shape
    m = mask.astype(float)
    n = m.sum()
    sp = (gray_p * m).sum()
    sp2 = (gray_p * gray_p * m).sum()
    var_p = max(sp2 - sp * sp / n, 1e-9)
    out = []
    for by in range(0, bh - ph + 1):
        for bx in range(0, bw - pw + 1):
            win = gray_b[by:by + ph, bx:bx + pw]
            sw = (win * m).sum()
            sw2 = (win * win * m).sum()
            var_w = max(sw2 - sw * sw / n, 1e-9)
            num = (win * gray_p * m).sum() - sp * sw / n
            out.append((num / np.sqrt(var_w * var_p), bx, by))
    out.sort(reverse=True)
    return out


# piece element-left as a function of handle travel D on the 300px embed
# track: left(D) = A*D^2 + B*D  (accelerating slider easing, measured)
_QWEN_TRACK = (0.0035503, 0.0769223)


def _qwen_widget_state(page) -> Optional[dict]:
    """Locate the Aliyun slider widget and dump what the solver needs.

    Returns None when no widget is on the page (not armed, or already
    solved and gone). The piece/bg <img> elements are found by displayed
    size; their src is either an inline data: URI or a static-captcha CDN
    URL — both are handled by _qwen_fetch_challenge_images. The knob has
    no DOM node (CSS-drawn); it sits at the left edge of the text-box.
    """
    js = (
        'const pick=(lo,hi,hlo,hhi)=>{'
        'for(const im of document.querySelectorAll("img")){'
        'const r=im.getBoundingClientRect();'
        'if(r.width>=lo&&r.width<=hi&&r.height>=hlo&&r.height<=hhi)'
        'return [im.src,r.x,r.y,r.width,r.height];}return null;};'
        'const bg=pick(285,315,185,215), piece=pick(45,60,100,220);'
        'const box=document.querySelector(".aliyunCaptcha-sliding-text-box");'
        'const br=box?box.getBoundingClientRect():null;'
        'const tx=document.querySelector(".aliyunCaptcha-sliding-text");'
        'return JSON.stringify({bg:bg,piece:piece,'
        'box:br?[br.x,br.y,br.width,br.height]:null,'
        'state:tx?tx.textContent:null});')
    try:
        raw = page.run_js(js)
    except Exception:  # noqa: BLE001
        return None
    if not raw:
        return None
    try:
        d = json.loads(raw)
    except Exception:  # noqa: BLE001
        return None
    if not d.get('bg') or not d.get('piece') or not d.get('box'):
        return None
    return d


def _qwen_fetch_challenge_images(state: dict) -> Tuple[bytes, bytes]:
    """Raw PNG bytes for (background, piece sprite) from widget state."""
    out = []
    for key in ('bg', 'piece'):
        src = state[key][0]
        if src.startswith('data:'):
            out.append(base64.b64decode(src.split(',', 1)[1]))
        else:
            import requests as _rq
            # public static CDN — no WAF, no cookies, egress irrelevant
            r = _rq.get(src, timeout=20, headers={'User-Agent': _UA})
            r.raise_for_status()
            out.append(r.content)
    return out[0], out[1]


def _qwen_solve_challenge(bg_bytes: bytes,
                          piece_bytes: bytes,
                          disp_w: float = 300.0) -> Optional[Tuple[float, str]]:
    """Closed-form slide target for one challenge.

    Handles both observed styles — a white ghost piece baked into the
    background (whiteness-blob detection) and a dark hole cut out of it
    (masked NCC of the piece sprite) — and arbitrates between them.
    Returns (L_star, method): L* is the piece element-left target on the
    DISPLAYED background in px, or None when the challenge is unsolvable.
    """
    try:
        import io as _io
        import numpy as np
        from PIL import Image
        bg = np.asarray(
            Image.open(_io.BytesIO(bg_bytes)).convert('RGB'), float)
        pc = np.asarray(
            Image.open(_io.BytesIO(piece_bytes)).convert('RGBA'), float)
    except Exception:  # noqa: BLE001
        return None
    A, B = _QWEN_TRACK
    alpha = pc[:, :, 3]
    pys, pxs = np.where(alpha > 128)
    if not len(pxs):
        return None
    px0, py0 = int(pxs.min()), int(pys.min())
    pw = int(pxs.max() - pxs.min() + 1)
    ph = int(pys.max() - pys.min() + 1)
    patch = pc[py0:py0 + ph, px0:px0 + pw, :3]
    pmask = alpha[py0:py0 + ph, px0:px0 + pw] > 128
    gray_p = patch.mean(axis=2)
    gray_b = bg.mean(axis=2)
    sx = disp_w / bg.shape[1]  # displayed/natural scale (300/296)
    # style 1: white ghost blob (largest bright low-saturation component)
    R, G, Bl = bg[:, :, 0], bg[:, :, 1], bg[:, :, 2]
    mx = np.maximum(np.maximum(R, G), Bl)
    mn = np.minimum(np.minimum(R, G), Bl)
    white = (mx > 150) & ((mx - mn) < 40)
    white[py0 + 3:py0 + ph - 3, px0 + 3:px0 + pw - 3] = False
    ghost = None
    try:
        comp = _largest_component(white)
        if comp and comp[4] > 200:
            x0, x1, y0, y1, _n = comp
            ghost = (x0, y0, x1 - x0 + 1, y1 - y0 + 1)
    except Exception:  # noqa: BLE001
        ghost = None
    # style 2: masked NCC of the piece sprite
    try:
        top = _masked_ncc(gray_b, gray_p, pmask)
        score, ndx, ndy = top[0]
    except Exception:  # noqa: BLE001
        return None
    if score >= 0.6:
        dx, dy, method = ndx, ndy, 'ncc'
    elif ghost and abs(ghost[2] - pw) < 14 and abs(ghost[3] - ph) < 14:
        dx, dy, method = ghost[0], ghost[1], 'ghost'
    elif score >= 0.35:
        dx, dy, method = ndx, ndy, 'ncc-weak'
    elif ghost:
        dx, dy, method = ghost[0], ghost[1], 'ghost-weak'
    else:
        return None
    return (dx * sx - px0, method)


def _qwen_drag_closed_loop(page, state: dict,
                           l_star: float) -> Optional[float]:
    """Humanized glide + vision-corrected micro-creep to the target.

    The piece <img> is read live from the DOM; each correction inverts the
    calibrated piece-left(D) easing, so the landing error converges below
    a pixel regardless of where the glide phase stopped. Returns the final
    piece element-left error in px (None if the piece could not be read).
    """
    A, B = _QWEN_TRACK
    bx = state['bg'][1]
    box = state['box']
    gx, gy = box[0] + 20, box[1] + box[3] / 2  # knob at the box's left edge
    d_est = (-B + (B * B + 4 * A * l_star) ** 0.5) / (2 * A)
    js = ('for(const im of document.querySelectorAll("img")){'
          'const r=im.getBoundingClientRect();'
          'if(r.width>=45&&r.width<=60&&r.height>=100)return r.x;}'
          'return null;')
    ac = page.actions
    # human approach: stray hovers before grabbing the knob
    ac.move(gx - random.randint(15, 35), gy + random.randint(15, 40),
            duration=.3)
    ac.move(gx - random.randint(3, 8), gy + random.randint(1, 4),
            duration=.2)
    time.sleep(random.uniform(.2, .5))
    ac.move_to((gx, gy)).hold()
    time.sleep(random.uniform(.15, .4))
    x = gx
    # glide to ~92% on a smoothstep velocity profile with jitter
    d1 = d_est * .92
    segs = random.randint(7, 10)
    for i in range(1, segs + 1):
        u = i / segs
        e = (3 * u * u - 2 * u * u * u) * d1
        nx = gx + e + random.uniform(-1, 1)
        ac.move(nx - x, random.uniform(-1.5, 1.5),
                duration=random.uniform(.04, .12))
        x = nx
        if random.random() < .25:
            time.sleep(random.uniform(.02, .09))
    time.sleep(random.uniform(.05, .2))
    # vision-corrected creep: read the piece, invert the easing, land
    err = None
    for _ in range(22):
        try:
            p = page.run_js(js)
        except Exception:  # noqa: BLE001
            p = None
        if p is None:
            break
        err = (bx + l_star) - p
        if abs(err) <= 0.8:
            break
        gain = 2 * A * (x - gx) + B
        dh = max(-12.0, min(12.0, err / gain))
        if abs(dh) < .25:
            dh = -0.25 if dh < 0 else 0.25
        x += dh
        ac.move(dh, 0, duration=random.uniform(.03, .08))
        time.sleep(random.uniform(.12, .25))
    time.sleep(random.uniform(.2, .45))
    ac.release()
    time.sleep(random.uniform(.8, 1.4))
    return err


def _qwen_rearm(page) -> None:
    """Click the slider's text area to fetch a fresh challenge (this also
    restarts the 20-minute init clock that expires with VerifyCode F014)."""
    try:
        el = page.ele('.aliyunCaptcha-sliding-text-box', timeout=3)
        if el:
            el.click(by_js=False)
            return
    except Exception:  # noqa: BLE001
        pass
    try:
        page.run_js('const e=document.querySelector('
                    '".aliyunCaptcha-sliding-text-box");if(e)e.click();')
    except Exception:  # noqa: BLE001
        pass


def _qwen_last_verify_code(page) -> str:
    """VerifyCode from the newest -verify.captcha response in the net log
    (T001 = pass; F001 = behavioral reject; F015 = wrong position; F014 =
    init expired)."""
    try:
        body = _net_log_read(page, needle='-verify.captcha')
        if body:
            last = body.split(' || ')[-1]
            raw = last.split(' -> ', 1)[1] if ' -> ' in last else ''
            return str((json.loads(raw).get('Result') or {})
                       .get('VerifyCode') or '')
    except Exception:  # noqa: BLE001
        pass
    return ''


def _qwen_slider_pass(page, attempts: int = 6) -> bool:
    """Solve Aliyun's noCaptcha slider with a closed-loop vision drag.

    chat.qwen.ai's signup POST is replayed by the WAF once the slider is
    solved. The widget's piece/background images are read straight from
    the DOM, the slide target is computed offline (_qwen_solve_challenge:
    ghost-blob / masked-NCC arbitration), and a humanized glide lands the
    piece within a pixel of the target (_qwen_drag_closed_loop). The whole
    arm-solve-drag cycle runs in seconds — well inside the challenge's
    20-minute init validity (an expired init fails with VerifyCode F014
    no matter how accurate the landing was).

    Position is never the bottleneck once landed accurately; the residual
    F001 rejects are behavioral/device-trust scoring, so each failed drag
    is followed by a re-arm (fresh challenge + fresh init clock) instead
    of grinding on one widget.
    Returns True when no widget is present (already passed / not armed) or
    a drag was accepted; False when every attempt was rejected.
    """
    for _ in range(attempts):
        state = _qwen_widget_state(page)
        if not state:
            return True  # no widget — the signup POST went through
        time.sleep(random.uniform(.4, 1.0))
        sol = None
        try:
            bg_b, piece_b = _qwen_fetch_challenge_images(state)
            sol = _qwen_solve_challenge(bg_b, piece_b)
        except Exception:  # noqa: BLE001
            sol = None
        if not sol:
            _qwen_rearm(page)
            time.sleep(1.5)
            continue
        try:
            err = _qwen_drag_closed_loop(page, state, sol[0])
        except Exception:  # noqa: BLE001
            err = None
        time.sleep(2.0)  # verify POST + verdict
        code = _qwen_last_verify_code(page)
        _log_history('qwen', 'stage',
                     f'slider method={sol[1]} err={err} verify={code or "?"}')
        if code == 'T001' or not _qwen_widget_state(page):
            time.sleep(3)  # let the WAF replay the original POST
            return True
        _qwen_rearm(page)
        time.sleep(random.uniform(1.0, 2.0))
    return False


def signup_qwen() -> Tuple[bool, str]:
    """Create a chat.qwen.ai account autonomously.

    chat.qwen.ai is an OpenWebUI-style deployment; the signup form asks for
    name/email/password (no phone). The signup API sits behind an Aliyun
    slide-captcha, so this runs in a real (windowed, humanized-warmup)
    browser and solves the slider with a closed-loop vision drag
    (_qwen_slider_pass: DOM image dump -> offline target solve ->
    humanized glide + sub-pixel creep). The mailbox comes
    from dsk/mailgen (emailnator's real gmail.com inboxes first). On success
    the JWT lands in the qwen jar and the credentials in data/accounts.json
    — every later renewal is then a plain HTTP re-login (see
    _qwen_signin/refresh_qwen). Accounts that only reach 'pending
    activation' still persist their credentials (never orphaned): the
    login rung polls the signup mailbox for the activation link on every
    later cycle, so activation converges with no operator action.
    """
    email, password = _creds('qwen')
    session = None
    preexisting = bool(_load_accounts().get('qwen'))
    if not email or not password:
        if not mailgen.autogen_enabled():
            return False, 'no QWEN_LOGIN_EMAIL/PASSWORD and mail autogen off'
        session, err = mailgen.create_email()
        if not session:
            return False, f'autogen mailbox unavailable: {err}'
        email = session['address']
        password = mailgen.gen_password()
    name = f'DSF {email.split("@")[0][:8]}'.strip()
    last_error = ''
    typed_pw: Dict[str, str] = {}  # password value actually in the form
    ladder: List[Optional[str]] = [None]
    try:
        # slider verdicts are per-IP-reputation: a burned direct IP never
        # passes, so sample FRESH pool exits directly (latency is irrelevant
        # for a one-off signup; reachability is pre-probed cheaply)
        from . import proxies as _proxies
        _proxies.ensure_pool()
        raw = list(_proxies.all_proxies())
        random.shuffle(raw)
        import requests as _rq
        for cand in raw[:24]:
            if len(ladder) >= 4:
                break
            if cand in ladder:
                continue
            try:
                _rq.get('https://chat.qwen.ai/', timeout=12,
                        proxies={'http': cand, 'https': cand},
                        headers={'User-Agent': _UA})
                ladder.append(cand)
            except Exception:  # noqa: BLE001 — dead exit, skip
                continue
    except Exception:  # noqa: BLE001 — direct remains the fallback
        pass
    def _fill_signup_form(page) -> bool:
        """Fill name/email/passwords, tick the custom agree widget, click
        Create Account. Returns False when the form or button is unusable."""
        if not _fill_first(page, _QWEN_NAME_SELECTORS, name):
            return False
        if not _fill_first(page, _QWEN_EMAIL_SELECTORS, email):
            return False
        pw_fields = page.eles('css:input[type=password]')
        if len(pw_fields) < 2:
            return False
        pw_fields[0].clear()
        pw_fields[0].input(password)
        pw_fields[1].clear()
        pw_fields[1].input(password)
        try:
            # keep what the form actually holds: page JS may normalize the
            # typed value, and re-login later must use the POSTed password
            v0 = str(pw_fields[0].attr('value') or '')
            v1 = str(pw_fields[1].attr('value') or '')
            typed_pw['v'] = v0 or v1
        except Exception:  # noqa: BLE001 — best-effort observation
            pass
        try:
            # the agree control is a custom widget: ARIA role=checkbox
            # (element-plus style) — no real input[type=checkbox] exists
            agreed = False
            for sel in ('css:[role=checkbox]', 'css:input[type=checkbox]'):
                try:
                    cb = page.ele(sel, timeout=2)
                except Exception:  # noqa: BLE001
                    cb = None
                if cb:
                    try:
                        cb.click(by_js=False)
                    except Exception:  # noqa: BLE001
                        cb.click(by_js=True)
                    agreed = True
                    break
            if not agreed:
                page.run_js(
                    'const el=document.querySelector('
                    '"[role=checkbox],[class*=checkbox],[class*=agree]");'
                    'if(el) el.click();')
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1)
        btn_ready = page.run_js(
            '[...document.querySelectorAll("button")]'
            '.filter(b=>/create account/i.test(b.textContent))'
            '.map(b=>b.disabled)[0]')
        if btn_ready:
            return False  # agree toggle failed — button still disabled
        return bool(_click_any(page,
                               ['Create Account', 'Create account', '注册']))

    for proxy in ladder:
        page = None
        try:
            page = _browser(proxy=proxy, headed=True)
            # WAF reputation warmup: browse like a human before touching the
            # challenge (headless + cold single-page sessions score lowest)
            try:
                page.get('https://chat.qwen.ai/')
                time.sleep(random.uniform(6, 10))
                ac = page.actions
                for _ in range(random.randint(2, 4)):
                    ac.move(random.randint(150, 900),
                            random.randint(120, 500),
                            duration=random.uniform(.2, .5))
                page.run_js('window.scrollBy(0, 400);')
                time.sleep(1)
            except Exception:  # noqa: BLE001 — best-effort warmup
                pass
            # slider scoring accumulates per WAF session: a few failed drags
            # poison the page — so limit drags per load and RELOAD for a
            # fresh verdict instead of grinding on one widget
            submitted = False
            for round_no in range(3):
                page.get(_QWEN_SIGNUP_URL)
                time.sleep(5)
                if 'qwen' not in (page.url or ''):
                    page.get(_QWEN_SIGNUP_URL)
                    time.sleep(4)
                _net_log_install(page)
                if not _fill_signup_form(page):
                    last_error = 'qwen signup form unusable (fields/agree)'
                    break
                submitted = True
                time.sleep(3)
                # the signup POST may be intercepted by Aliyun's slider
                slider_ok = _qwen_slider_pass(page, attempts=4)
                signup_resp = _net_log_read(page, needle='/signup')
                _log_history('qwen', 'stage',
                             f'round={round_no} slider_pass={slider_ok}'
                             + (f' api={signup_resp}' if signup_resp else ''))
            # never orphan a submitted account: persist the credentials now
            # so later cycles re-login (and poll the activation mail) instead
            # of minting yet another mailbox
            if submitted:
                stored_pw = password
                obs = (typed_pw.get('v') or '').strip()
                if obs and obs != password:
                    # The form holds a different password than generated —
                    # keep whichever one actually re-logins (INVALID_CRED on
                    # the generated value was the historical root cause of
                    # un-renewable qwen accounts).
                    ok_g, _d = _qwen_signin(email, password)
                    if not ok_g:
                        ok_o, det_o = _qwen_signin(email, obs)
                        if ok_o:
                            _log_history('qwen', 'stage',
                                         're-login works with the '
                                         'form-observed password; saving it')
                            stored_pw = obs
                        else:
                            _log_history('qwen', 'stage',
                                         f'observed-password signin failed: '
                                         f'{det_o}')
                _save_account('qwen', email, stored_pw,
                              (session or {}).get('backend', ''),
                              {'mail_session': session} if session else None)
            # the session JWT: scan localStorage for any JWT-shaped value
            # (the storage key name is fork-specific) and fall back to the
            # conventional 'token' key / cookies
            token = ''
            deadline = time.time() + 90
            while time.time() < deadline and not token:
                try:
                    token = str(page.run_js(
                        'let hit="";'
                        'const re=/^[A-Za-z0-9_-]{20,}\\.[A-Za-z0-9_-]{20,}'
                        '\\.[A-Za-z0-9_-]{20,}$/;'
                        'for(let i=0;i<localStorage.length;i++){'
                        'const k=localStorage.key(i);'
                        'const v=localStorage.getItem(k)||"";'
                        'if(k==="token"&&v){hit=v;break;}'
                        'if(v.length>80&&re.test(v)){hit=v;break;}}'
                        'return hit;') or '').strip()
                except Exception:  # noqa: BLE001
                    token = ''
                if not token:
                    time.sleep(3)
            # guard against analytics junk that merely LOOKS token-ish:
            # the session JWT must authenticate against /api/v1/auths
            import requests as _rq
            verified = False
            if token:
                try:
                    vcheck = _rq.get(
                        'https://chat.qwen.ai/api/v1/auths',
                        headers={**_qwen_headers(),
                                 'Authorization': f'Bearer {token}'},
                        timeout=30,
                        **_proxies_kwargs('https://chat.qwen.ai'))
                    verified = vcheck.status_code == 200
                except Exception:  # noqa: BLE001
                    verified = False
            _log_history('qwen', 'stage',
                         f'token={"yes" if token else "no"} verified={verified}'
                         + (f' body={_body_head(page)[:80]}' if not verified else ''))
            if not verified:
                # new accounts are "pending activation": the activation mail
                # lands in our mailbox — open its link, then re-login via the
                # WAF-free HTTP signin for a fresh, activated session token
                pending = 'pending activation' in _body_head(page).lower()
                if pending and session:
                    _log_history('qwen', 'stage', 'pending-activation detected')
                    link = mailgen.fetch_otp(
                        session, max_wait_s=180, sender_needle='qwen',
                        code_re=re.compile(
                            r'(https://chat\.qwen\.ai/[^\s"\'<>]*'
                            r'activate[^\s"\'<>]*)'))
                    _log_history('qwen', 'stage',
                                 'activation link '
                                 + ('found' if link else 'MISSING'))
                    if link:
                        try:
                            page.get(link)
                            time.sleep(6)
                        except Exception:  # noqa: BLE001
                            pass
                        ok2, detail2 = _qwen_signin(email, password)
                        if ok2:
                            try:
                                token = (_load_jar('qwen').get('token') or '')
                                vcheck = _rq.get(
                                    'https://chat.qwen.ai/api/v1/auths',
                                    headers={**_qwen_headers(),
                                             'Authorization':
                                                 f'Bearer {token}'},
                                    timeout=30,
                                    **_proxies_kwargs('https://chat.qwen.ai'))
                                verified = vcheck.status_code == 200
                            except Exception:  # noqa: BLE001
                                verified = False
            if not verified and preexisting:
                # the account likely already exists (this signup POST was a
                # duplicate): its activation mail is the missing piece — poll
                # the recorded mailbox, open the link, re-login
                _log_history('qwen', 'stage',
                             'preexisting creds; activation retry')
                if _qwen_activate(email, password)[0]:
                    token = (_load_jar('qwen').get('token') or '')
                    verified = bool(token)
            if not verified:
                last_error = ('qwen signup submitted but no working session '
                              f'captured via {proxy or "direct"} '
                              '(slider/WAF rejected the POST, or the '
                              'activation mail never arrived)')
                continue
            _save_jar('qwen', {'token': token, 'email': email})
            _save_account('qwen', email, password,
                          (session or {}).get('backend', ''),
                          {'mail_session': session} if session else None)
            return True, (f'qwen account ready for {email} (mailbox: '
                          f'{(session or {}).get("backend", "stored creds")})')
        except Exception as e:  # noqa: BLE001
            last_error = f'qwen signup failed: {type(e).__name__}: {e}'
        finally:
            if page is not None:
                try:
                    page.quit()
                except Exception:  # noqa: BLE001
                    pass
    return False, last_error or 'all qwen signup egresses failed'


SIGNUP = {'deepseek': signup_deepseek, 'chatgpt': signup_chatgpt,
          'gemini': signup_gemini,
          'claude': signup_claude,
          'grok': signup_grok,
          'qwen': signup_qwen,
          'kimi': signup_kimi,
          'mistral': signup_mistral,
          'huggingface': signup_huggingface,
          'openrouter': signup_openrouter,
          'groq': signup_groq,
          'copilot': _anonymous('copilot'),
          'perplexity': _anonymous('perplexity'), 'glm': _anonymous('glm')}


# ------------------------------------------------------------------ renew
def _seed_counts() -> None:
    """Seed today's per-provider attempt counts from history.jsonl.

    The counts live in memory, so a container restart would otherwise
    bypass the daily attempt cap; today's ``renew-start`` events make the
    budget continuous across restarts. Runs once per process.

    Only strict proactive attempts count against the budget: today's
    ``renew-start`` events whose reason is exactly ``proactive``
    (the TTL-cadence bootstrap sweep). Reactive attempts ('auth' from
    the self-heal probe, 'inline-auth' from the request path) and
    operator-initiated runs ('manual-*', CLI) are excluded — so a heavy
    debug or reactive day can never starve the daemon's own retries.
    """
    if _STATE.counts_seeded:
        return
    _STATE.counts_seeded = True
    try:
        path = _data_dir() / 'refresher' / 'history.jsonl'
        today = time.strftime('%Y-%m-%d')
        counts: Dict[str, int] = {}
        with path.open('r', encoding='utf-8') as fh:
            for line in fh:
                try:
                    entry = json.loads(line)
                except (ValueError, TypeError):
                    continue
                reason = str(entry.get('detail', ''))
                if (entry.get('event') == 'renew-start'
                        and str(entry.get('ts', '')).startswith(today)
                        and reason == 'proactive'):
                    counts[entry.get('provider', '')] = \
                        counts.get(entry.get('provider', ''), 0) + 1
        for name, n in counts.items():
            _STATE.counts[name] = (today, n)
        if counts:
            print(f'[refresher] daily budget seeded from history: {counts}')
    except OSError:
        pass  # no history yet


def renew(name: str, reason: str = '') -> Dict[str, Any]:
    """Run the full renewal ladder for one provider. Returns a status dict."""
    if not _env_bool('DSF_REFRESHER', True):
        return {'renewed': False, 'skipped': 'refresher disabled'}
    if not provider_enabled(name):
        return {'renewed': False, 'skipped': 'provider disabled (DSF_PROVIDERS)'}
    excl = {e.strip().lower() for e in
            os.getenv('DSF_REFRESHER_EXCLUDE', '').split(',') if e.strip()}
    if name in excl:
        return {'renewed': False, 'skipped': f'{name} excluded'}
    with _STATE.lock:
        if _STATE.renewing.get(name):
            return {'renewed': False, 'skipped': 'renewal already running'}
        last = _STATE.results.get(name, {})
        if last.get('ts') and time.time() - _entry_ts(last) < _cooldown() \
                and not reason.startswith('manual'):
            return {'renewed': False, 'skipped': 'cooldown'}
        today = time.strftime('%Y-%m-%d')
        _seed_counts()
        day, n = _STATE.counts.get(name, (today, 0))
        n = n + 1 if day == today else 1
        manual = reason.startswith('manual')
        if n > _max_renews() and not manual:
            return {'renewed': False, 'skipped': 'daily attempt budget exhausted'}
        if not manual:  # manual/CLI attempts never consume the daemon budget
            _STATE.counts[name] = (day, n)
        _STATE.renewing[name] = True
    try:
        return _renew_locked(name, reason)
    finally:
        with _STATE.lock:
            _STATE.renewing[name] = False


def _entry_ts(entry: Dict[str, Any]) -> float:
    try:
        return time.mktime(time.strptime(entry['ts'][:19], '%Y-%m-%dT%H:%M:%S'))
    except Exception:
        return 0.0


def _renew_locked(name: str, reason: str) -> Dict[str, Any]:
    steps: List[str] = []
    _log_history(name, 'renew-start', reason or 'proactive')

    ok, detail = REFRESH[name]()
    steps.append(f'refresh: {detail}')
    _log_history(name, 'refresh', detail)
    status = _verify(name)
    if status == 'ok':
        _log_history(name, 'renewed', '; '.join(steps))
        return {'renewed': True, 'via': 'http-refresh', 'steps': steps}

    if _env_bool('DSF_REFRESHER_LOGIN', True):
        ok, detail = browser_login(name)
        steps.append(f'login: {detail}')
        _log_history(name, 'browser-login', detail)
        status = _verify(name)
        if status == 'ok':
            _log_history(name, 'renewed', '; '.join(steps))
            return {'renewed': True, 'via': 'browser-login', 'steps': steps}

    if _env_bool('DSF_REFRESHER_AUTOSIGNUP', True):
        ok, detail = SIGNUP[name]()   # all providers: create what is missing
        steps.append(f'signup: {detail}')
        _log_history(name, 'autosignup', detail)
        status = _verify(name)
        if status == 'ok':
            _log_history(name, 'renewed', '; '.join(steps))
            return {'renewed': True, 'via': 'autosignup', 'steps': steps}

    _log_history(name, 'renew-failed', '; '.join(steps))
    return {'renewed': False, 'steps': steps,
            'hint': 'needs manual credential update' if status == 'auth' else status}


def _verify(name: str) -> str:
    try:
        from . import selfheal
        status, _ = selfheal._probe_once(name)
        return status
    except Exception as e:  # noqa: BLE001
        return f'probe-error: {e}'


def renew_inline(name: str, detail: str = '') -> Dict[str, Any]:
    """Request-path remediation: fire a background renewal on an auth error.

    Called from the router the moment a request classified as
    ``ProviderAuthError`` — the ladder (refresh -> browser re-login ->
    auto-signup) runs in a background thread so the failing request is
    not blocked; the NEXT request picks up the fresh credential.
    Guarded by an hourly per-provider attempt cap (DSF_REFRESHER_INLINE_HOURLY,
    default 2) and a 60s silence window after a completed attempt so a
    burst of failing requests cannot spin the ladder.
    """
    if not _env_bool('DSF_REFRESHER', True):
        return {'triggered': False, 'skipped': 'refresher disabled'}
    if not provider_enabled(name):
        return {'triggered': False, 'skipped': 'provider disabled'}
    hourly = max(1, int(os.getenv('DSF_REFRESHER_INLINE_HOURLY', '2') or 2))
    now = time.time()
    hour = time.strftime('%Y%m%d%H')
    with _STATE.lock:
        if _STATE.renewing.get(name):
            return {'triggered': False, 'skipped': 'renewal already running'}
        last = _STATE.inline_last.get(name)
        if last and now - last[1] < 60:
            return {'triggered': False, 'skipped': 'inline silence window'}
        (h, cnt) = _STATE.inline_counts.get(name, (hour, 0))
        if h == hour and cnt >= hourly:
            return {'triggered': False, 'skipped': 'inline hourly cap'}
        _STATE.inline_counts[name] = (hour, cnt + 1 if h == hour else 1)

    def _run() -> None:
        try:
            res = renew(name, reason='inline-auth')
            ok = bool(res.get('renewed'))
        except Exception:  # noqa: BLE001 — remediation must never raise
            ok = False
        with _STATE.lock:
            _STATE.inline_last[name] = ('ok' if ok else 'failed', time.time())

    t = threading.Thread(target=_run, name=f'inline-renew-{name}', daemon=True)
    with _STATE.lock:
        _STATE.inline_threads[name] = t
    t.start()
    _log_history(name, 'inline-renew-triggered', detail[:200])
    return {'triggered': True, 'reason': detail[:120]}


# ------------------------------------------------------------------ daemon
def refresh_cycle() -> Dict[str, Any]:
    """Proactive daemon cycle: rotate refreshable cookies (gemini/chatgpt)
    and BOOTSTRAP any provider that has no credentials at all — the signup
    rung creates fresh ones unattended."""
    if not _env_bool('DSF_REFRESHER', True):
        return {'refresher': 'disabled'}
    out: Dict[str, Any] = {}
    for name in tuple(REFRESH):
        if not provider_enabled(name):
            continue  # disabled via DSF_PROVIDERS: no routes, no probes, no bot
        if not _has_creds(name):
            if not _env_bool('DSF_REFRESHER_AUTOSIGNUP', True):
                out[name] = 'skipped (no credentials, autosignup off)'
                continue
            try:
                out[name] = renew(name, reason='bootstrap')
            except Exception as e:  # noqa: BLE001
                out[name] = f'bootstrap error: {type(e).__name__}: {e}'
            continue
        if name == 'deepseek':
            continue  # token is verified live by the self-heal probe
        try:
            ok, detail = REFRESH[name]()
            out[name] = detail
            _log_history(name, 'proactive-refresh' if ok else 'refresh-issue', detail)
        except Exception as e:  # noqa: BLE001
            out[name] = f'error: {e}'
    return out


def bootstrap_all() -> Dict[str, Any]:
    """Force credential creation for every provider that currently has
    none (CLI / manual trigger; bypasses the per-provider cooldown)."""
    out: Dict[str, Any] = {}
    for name in REFRESH:
        if not provider_enabled(name):
            out[name] = {'renewed': False, 'skipped': 'provider disabled (DSF_PROVIDERS)'}
            continue
        if _has_creds(name):
            out[name] = {'renewed': False, 'skipped': 'credentials present'}
            continue
        try:
            out[name] = renew(name, reason='manual-bootstrap')
        except Exception as e:  # noqa: BLE001
            out[name] = {'renewed': False, 'error': str(e)[:300]}
    return out


def start_daemon() -> bool:
    with _STATE.lock:
        if _STATE.started:
            return False
        _STATE.started = True

    def _loop() -> None:
        while True:
            try:
                refresh_cycle()
            except Exception:  # pragma: no cover
                pass
            time.sleep(_ttl())

    threading.Thread(target=_loop, name='refresher', daemon=True).start()
    return True


def status() -> Dict[str, Any]:
    with _STATE.lock:
        results = dict(_STATE.results)
        started = _STATE.started
    enabled = [p for p in REFRESH if provider_enabled(p)]
    return {'enabled': _env_bool('DSF_REFRESHER', True),
            'daemon': started,
            'ttl': _ttl(),
            'browser_login': _env_bool('DSF_REFRESHER_LOGIN', True),
            'autosignup': _env_bool('DSF_REFRESHER_AUTOSIGNUP', True),
            'autosignup_providers': sorted(SIGNUP),
            'mail_autogen': mailgen.autogen_enabled(),
            'mail_configured': bool(os.getenv('DSF_MAIL_IMAP_HOST', '').strip()),
            'providers': {'enabled': enabled,
                          'disabled': [p for p in REFRESH if p not in enabled]},
            'credentials': {p: bool(all(_creds(p))) for p in REFRESH},
            'has_credentials': {p: _has_creds(p) for p in REFRESH},
            'bootstrap': {'enabled': _env_bool('DSF_REFRESHER_AUTOSIGNUP', True),
                          'missing': [p for p in enabled if not _has_creds(p)]},
            'last_results': results}


def main(argv: List[str]) -> int:  # pragma: no cover - CLI
    cmd = argv[1] if len(argv) > 1 else 'status'
    if cmd == 'status':
        print(json.dumps(status(), indent=2, default=str))
        return 0
    if cmd == 'refresh' and len(argv) > 2:
        print(json.dumps({argv[2]: REFRESH[argv[2]]()[1]}, indent=2))
        return 0
    if cmd == 'login' and len(argv) > 2:
        print(json.dumps(dict(zip(('ok', 'detail'), browser_login(argv[2]))), indent=2))
        return 0
    if cmd == 'signup':
        prov = argv[2] if len(argv) > 2 else 'deepseek'
        print(json.dumps(dict(zip(('ok', 'detail'), SIGNUP[prov]())), indent=2))
        return 0
    if cmd == 'bootstrap':
        print(json.dumps(bootstrap_all(), indent=2, default=str))
        return 0
    if cmd == 'mailgen':
        session, err = mailgen.create_email()
        print(json.dumps({'ok': bool(session),
                          'detail': session or err,
                          'address': (session or {}).get('address')}, indent=2))
        return 0
    print(__doc__)
    return 1


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
