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

  3. Account auto-signup (default ON, deepseek only: DSF_REFRESHER_AUTOSIGNUP).
     Creates a fresh free account when even the login session is dead. With
     no DEEPSEEK_LOGIN_EMAIL/PASSWORD configured, the e-mail address is
     AUTO-GENERATED (dsk/mailgen.py): a catch-all IMAP domain
     (DSF_MAIL_DOMAIN) when available, else a mail.tm throwaway account —
     the verification code is read from that mailbox automatically. Disable
     the auto-generation with DSF_MAIL_AUTOGEN=false.

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
    python -m dsk.refresher signup
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
from email.header import decode_header
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import mailgen

_BASE = Path(__file__).resolve().parent


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, '').strip().lower()
    if not raw:
        return default
    return raw in ('1', 'true', 'yes', 'on')


def _data_dir() -> Path:
    base = os.getenv('COOKIES_DIR') or os.getenv('DSF_SELFHEAL_DIR') or str(_BASE)
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
             'deepseek': 'cookies.json'}
    return _data_dir() / files[name]


class _State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.started = False
        self.renewing: Dict[str, bool] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        self.counts: Dict[str, Tuple[str, int]] = {}  # provider -> (day, n)


_STATE = _State()


def _log_history(provider: str, event: str, detail: Any = '') -> None:
    entry = {'ts': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
             'provider': provider, 'event': event, 'detail': str(detail)[:1000]}
    with _STATE.lock:
        _STATE.results[provider] = entry
    try:
        path = _data_dir() / 'refresher' / 'history.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
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


REFRESH = {'gemini': refresh_gemini, 'chatgpt': refresh_chatgpt,
           'deepseek': refresh_deepseek}


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
            box = imaplib.IMAP4_SSL(host, port)
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
def _browser():
    from DrissionPage import ChromiumPage, ChromiumOptions
    options = ChromiumOptions().auto_port()
    options.set_argument('--no-sandbox')
    options.set_argument('--disable-gpu')
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
    prefix = {'deepseek': 'DEEPSEEK', 'chatgpt': 'CHATGPT', 'gemini': 'GEMINI'}[name]
    email = os.getenv(f'{prefix}_LOGIN_EMAIL', '').strip()
    password = os.getenv(f'{prefix}_LOGIN_PASSWORD', '').strip()
    return email, password


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
    try:
        page = _browser()
    except Exception as e:  # noqa: BLE001
        return False, f'browser unavailable: {e}'
    try:
        if name == 'deepseek':
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
    try:
        page = _browser()
    except Exception as e:  # noqa: BLE001
        return False, f'browser unavailable: {e}'
    try:
        page.get('https://chat.deepseek.com/sign_up')
        time.sleep(5)
        if not _fill_first(page, _DEEPSEEK_EMAIL_SELECTORS, email):
            return False, 'email field not found'
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
            return False, 'signup code email not found in mailbox'
        if not _fill_first(page, ['@placeholder:code', '@placeholder:Code',
                                  'css:input[name=code]'], code):
            return False, 'code field not found'
        _click_any(page, ['Sign Up', 'Sign up', '注册'])
        token = _wait_token(page)
        _export_cookies(page, 'deepseek', ('deepseek.com',))
        if token:
            _save_deepseek_token(token)
            via = f'account created (autogen {session["backend"]}: {email})' \
                if generated else 'account created'
            return True, f'{via}, userToken captured'
        return False, 'signup finished but no userToken appeared'
    except Exception as e:  # noqa: BLE001
        return False, f'signup flow failed: {type(e).__name__}: {e}'
    finally:
        try:
            page.quit()
        except Exception:  # noqa: BLE001
            pass


# ------------------------------------------------------------------ renew
def renew(name: str, reason: str = '') -> Dict[str, Any]:
    """Run the full renewal ladder for one provider. Returns a status dict."""
    if not _env_bool('DSF_REFRESHER', True):
        return {'renewed': False, 'skipped': 'refresher disabled'}
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
        day, n = _STATE.counts.get(name, (today, 0))
        n = n + 1 if day == today else 1
        if n > _max_renews():
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

    if name == 'deepseek' and _env_bool('DSF_REFRESHER_AUTOSIGNUP', True):
        ok, detail = signup_deepseek()
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
    """Proactively rotate refreshable cookies (gemini/chatgpt) once."""
    out: Dict[str, Any] = {}
    for name in ('gemini', 'chatgpt'):
        if not _load_jar(name):
            out[name] = 'skipped (no credentials)'
            continue
        try:
            ok, detail = REFRESH[name]()
            out[name] = detail
            _log_history(name, 'proactive-refresh' if ok else 'refresh-issue', detail)
        except Exception as e:  # noqa: BLE001
            out[name] = f'error: {e}'
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
    return {'enabled': _env_bool('DSF_REFRESHER', True),
            'daemon': started,
            'ttl': _ttl(),
            'browser_login': _env_bool('DSF_REFRESHER_LOGIN', True),
            'autosignup': _env_bool('DSF_REFRESHER_AUTOSIGNUP', True),
            'mail_autogen': mailgen.autogen_enabled(),
            'mail_configured': bool(os.getenv('DSF_MAIL_IMAP_HOST', '').strip()),
            'credentials': {p: bool(all(_creds(p))) for p in REFRESH},
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
        print(json.dumps(dict(zip(('ok', 'detail'), signup_deepseek())), indent=2))
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
