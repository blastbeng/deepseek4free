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

import imaplib
import json
import os
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
                   'kimi': ('token', 'KIMI_TOKEN'),
                   'mistral': ('session_token', 'MISTRAL_SESSION_TOKEN'),
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
    if name in ('claude', 'grok', 'qwen', 'kimi'):
        env_key = {'claude': 'CLAUDE_SESSION_KEY', 'grok': 'GROK_SSO',
                   'qwen': 'QWEN_TOKEN', 'kimi': 'KIMI_TOKEN'}[name]
        if os.getenv(env_key, '').strip():
            return True
        jar = _load_jar(name)
        if name == 'claude':
            return bool(jar.get('sessionKey'))
        if name == 'grok':
            return bool(jar.get('sso') or jar.get('sso-rw'))
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
                  backend: str = '') -> None:
    """Persist a bot-created account so later renewals can re-login."""
    accs = _load_accounts()
    accs[name] = {'email': email, 'password': password, 'backend': backend,
                  'ts': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
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


def refresh_qwen() -> Tuple[bool, str]:
    if not _has_creds('qwen'):
        return False, 'no qwen credentials — set QWEN_TOKEN or qwen_cookies.json'
    import requests
    try:
        resp = requests.get(
            'https://chat.qwen.ai/api/v1/auths',
            headers={'Authorization': f"Bearer {_load_jar('qwen').get('token', '')}",
                     'User-Agent': _UA},
            timeout=30, **_proxies_kwargs('https://chat.qwen.ai'))
    except Exception as e:  # noqa: BLE001
        return False, f'verify failed: {type(e).__name__}: {e}'
    if resp.status_code == 200:
        return True, 'token valid'
    if resp.status_code == 401:
        return False, 'token rejected — re-export from chat.qwen.ai'
    return False, f'HTTP {resp.status_code}'


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


REFRESH = {'gemini': refresh_gemini, 'chatgpt': refresh_chatgpt,
           'deepseek': refresh_deepseek, 'claude': refresh_claude,
           'grok': refresh_grok, 'qwen': refresh_qwen, 'kimi': refresh_kimi,
           'mistral': refresh_mistral, 'copilot': _anonymous('copilot'),
           'perplexity': _anonymous('perplexity'), 'glm': _anonymous('glm')}


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
    falls through to the dynamic pool and finally direct. Tor is NOT used
    anywhere (slow). Returns None = direct connection.
    """
    explicit = os.getenv('DSF_SIGNUP_PROXY', '').strip()
    if explicit:
        return explicit
    return None


def _browser(proxy: Optional[str] = None):
    from DrissionPage import ChromiumPage, ChromiumOptions
    options = ChromiumOptions().auto_port()
    options.set_argument('--no-sandbox')
    options.set_argument('--disable-gpu')
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
    if _env_bool('DSF_REFRESHER_HEADLESS', True):
        options.headless(True)
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
              'perplexity': 'PERPLEXITY', 'glm': 'GLM'}[name]
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
    proxy = _deepseek_egress() if name == 'deepseek' else None
    try:
        page = _browser(proxy=proxy)
    except Exception as e:  # noqa: BLE001
        return False, f'browser unavailable: {e}'
    try:
        if name == 'deepseek':
            # Root SPA first: /sign_in document GETs are CloudFront-403'd,
            # client-side routing is not.
            page.get('https://chat.deepseek.com/')
            time.sleep(6)
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

    chat.deepseek.com hard-403s document GETs by IP reputation. A 200 on
    the root page predicts the browser SPA entry will work; 403/challenge
    responses mark the exit unusable before we pay browser startup cost.
    """
    if not proxy:
        return False
    try:
        from .providers.base import http_get
        r = http_get('https://chat.deepseek.com/',
                     proxies={'http': proxy, 'https': proxy}, timeout=12)
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
    """Best single egress for a DeepSeek browser session (login/renewal)."""
    explicit = _signup_proxy()
    if explicit and _ds_egress_ok(explicit):
        return explicit
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
        email, password = session['address'], session['password']
        generated = True
    # egress ladder: explicit DSF_SIGNUP_PROXY first, then SEVERAL distinct
    # dynamic-pool exits, then direct (duplicates dropped). DeepSeek's
    # CloudFront blocks many datacenter/host IPs, and a blocked exit stays
    # "healthy" for generic checks — so every CloudFront block puts that
    # proxy on cooldown (mark_failure) and the next rung samples a fresh
    # exit instead of retrying the same blocked IP.
    # egress ladder: explicit DSF_SIGNUP_PROXY first, then up to 3 distinct
    # dynamic-pool exits that PASS the root-page reachability probe, then
    # direct (duplicates dropped). CloudFront blocks many datacenter/host
    # IPs; blocked exits are cooled down so the next rung samples a fresh,
    # hopefully-working IP instead of retrying the same blocked one.
    ladder: List[Optional[str]] = []
    seen: set = set()
    for p in (([_signup_proxy()] if _signup_proxy() else [])
              + _pool_egresses() + [None]):
        if p is None or p not in seen:
            ladder.append(p)
            if p is not None:
                seen.add(p)
    for proxy in ladder or [None]:
        page = None
        try:
            page = _browser(proxy=proxy)
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
            _click_any(page, ['Send Code', 'Send code', '获取验证码'])
            if generated:
                code = mailgen.fetch_otp(session, max_wait_s=180)
                if not code:
                    # fall back to the plain IMAP poller (no recipient filter)
                    code = imap_otp(max_wait_s=30)
            else:
                code = imap_otp(max_wait_s=180)
            if not code:
                return False, ('signup code email not found in mailbox — '
                               'DeepSeek silently drops disposable domains; '
                               'configure DSF_MAIL_DOMAIN + DSF_MAIL_IMAP_HOST '
                               'with a catch-all inbox for reliable delivery')
            if not _fill_first(page, ['@placeholder:code', '@placeholder:Code',
                                      'css:input[name=code]'], code):
                return False, 'code field not found'
            _click_any(page, ['Sign Up', 'Sign up', '注册'])
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
    email, password = session['address'], session['password']
    try:
        page = _browser()
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
    email, password = session['address'], session['password']
    try:
        page = _browser()
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


SIGNUP = {'deepseek': signup_deepseek, 'chatgpt': signup_chatgpt,
          'gemini': signup_gemini,
          'claude': _manual_only('claude', 'export the sessionKey cookie into CLAUDE_SESSION_KEY'),
          'grok': _manual_only('grok', 'export the sso cookie into GROK_SSO'),
          'qwen': _manual_only('qwen', 'export the Bearer token into QWEN_TOKEN'),
          'kimi': _manual_only('kimi', 'export the token cookie into KIMI_TOKEN'),
          'mistral': _manual_only('mistral',
                                  'export the session token into MISTRAL_SESSION_TOKEN'),
          'copilot': _anonymous('copilot'),
          'perplexity': _anonymous('perplexity'), 'glm': _anonymous('glm')}


# ------------------------------------------------------------------ renew
def _seed_counts() -> None:
    """Seed today's per-provider attempt counts from history.jsonl.

    The counts live in memory, so a container restart would otherwise
    bypass the daily attempt cap; today's ``renew-start`` events make the
    budget continuous across restarts. Runs once per process."""
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
                if (entry.get('event') == 'renew-start'
                        and str(entry.get('ts', '')).startswith(today)):
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
