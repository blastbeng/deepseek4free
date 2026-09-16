"""Automatic email-address generation for unattended credential flows.

Used by the refresher (browser re-login / auto-signup) when no login e-mail
is configured: a throwaway mailbox is created on the fly, the verification
e-mail is fetched from it and the OTP is extracted — so the renewal ladder
stays fully unmanned with zero mail configuration.

Four backends, tried in order (first that is *configured* wins, then the
first that *works*):

1. IMAP catch-all (``DSF_MAIL_IMAP_HOST`` + ``DSF_MAIL_DOMAIN``):
   a random local part is invented under your own domain
   (``dsf-<hex8>@<domain>``); the OTP is read through the existing IMAP
   poller. Most reliable — use this when you own a catch-all mailbox.

2. emailnator.com (https://www.emailnator.com, free, no key): the only
   free inbox service handing out REAL ``@gmail.com`` addresses (dot/plus
   variants of pooled master accounts). gmail.com is allowlisted
   essentially everywhere — this is the autonomous path that finally
   delivers DeepSeek OTP mail, which silently drops every disposable
   domain.

3. tempmail.lol (https://tempmail.lol, free public API, no key): a mailbox
   on a rotating pool of obscure domains — the best chance among the
   disposable pools against domain blocklists. No signup; the inbox is
   addressed by an opaque token stored in the session.

4. mail.tm / mail.gw (https://mail.tm, free public API, no key): a real
   throwaway account is created on a public temp-mail domain and its
   inbox is polled over HTTPS. Works out of the box, but public domains
   are often rejected by signup forms (DeepSeek silently drops them)
   — the caller treats failures as a normal renewal-ladder miss.

Env switches:
    DSF_MAIL_AUTOGEN   master switch for this module (default: true)
    DSF_MAIL_DOMAIN    domain for the IMAP catch-all backend (optional)

CLI smoke test:
    python -m dsk.mailgen
"""

import http.cookiejar
import json
import os
import re
import secrets
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

MAILTM_API = 'https://api.mail.tm'
MAILGW_API = 'https://api.mail.gw'
TEMPMAIL_API = 'https://api.tempmail.lol'
_EMAILNATOR_BASE = 'https://www.emailnator.com'
_EMAILNATOR_UA = ('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/131.0 Safari/537.36')
_UA = 'deepseek4free-refresher/1.0 (+autonomous credential maintenance)'


def autogen_enabled() -> bool:
    raw = os.getenv('DSF_MAIL_AUTOGEN', '').strip().lower()
    if not raw:
        return True  # default ON
    return raw in ('1', 'true', 'yes', 'on')


def _gen_local_part() -> str:
    return f"dsf-{secrets.token_hex(4)}"


def _gen_password() -> str:
    # letter + digit prefix keeps even the pickiest signup forms happy
    return f"Aa1{secrets.token_urlsafe(12)}"


def gen_password() -> str:
    """Public alias — signup forms need a password even when the mailbox
    backend (tempmail.lol) is token-addressed and carries none."""
    return _gen_password()


def _http(method: str, url: str, body: Optional[Dict[str, Any]] = None,
          token: Optional[str] = None, timeout: int = 25) -> Tuple[int, Any]:
    """Minimal JSON HTTP client (stdlib only — no proxy deps)."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header('User-Agent', _UA)
    req.add_header('Accept', 'application/json')
    if data is not None:
        req.add_header('Content-Type', 'application/json')
    if token:
        req.add_header('Authorization', f'Bearer {token}')
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode('utf-8', 'replace')
            return resp.status, (json.loads(raw) if raw.strip() else {})
    except urllib.error.HTTPError as e:
        try:
            raw = e.read().decode('utf-8', 'replace')
            return e.code, json.loads(raw)
        except Exception:  # noqa: BLE001
            return e.code, {}
    except Exception as e:  # noqa: BLE001
        return 0, {'error': f'{type(e).__name__}: {e}'}


# ------------------------------------------------------------------ mail.tm
def _items(body: Any) -> List[Any]:
    """mail.tm returns either a JSON-LD object (hydra:member) or a plain
    list depending on the requested content type — accept both."""
    if isinstance(body, list):
        return body
    if isinstance(body, dict):
        return body.get('hydra:member') or body.get('member') or []
    return []


def _mailtm_domains(api: str) -> List[str]:
    code, body = _http('GET', f'{api}/domains?page=1')
    if code != 200:
        return []
    return [d['domain'] for d in _items(body)
            if isinstance(d, dict) and d.get('domain') and d.get('isActive', True)]


def _mailtm_extract(body: Any, code_re: re.Pattern) -> Optional[str]:
    """Extract the OTP from a mail.tm message payload (text or html)."""
    if not isinstance(body, dict):
        return None
    text = ''
    for key in ('text', 'intro'):
        if isinstance(body.get(key), str):
            text += body[key] + '\n'
    html = body.get('html')
    if isinstance(html, list):
        text += '\n'.join(str(h) for h in html)
    elif isinstance(html, str):
        text += html
    match = code_re.search(text)
    return match.group(1) or match.group(0) if match else None


def _mailtm_create() -> Optional[Dict[str, Any]]:
    """Create a throwaway account on the first working temp-mail service.

    mail.tm and mail.gw expose the same API shape but run different domain
    pools; ESP blocklists often cover one pool and not the other, so both
    are tried. The chosen service's API base is stored in the session so
    fetch_otp polls the right inbox.
    """
    for backend, api in (('mail.tm', MAILTM_API), ('mail.gw', MAILGW_API)):
        for domain in _mailtm_domains(api)[:3]:
            address = f"{_gen_local_part()}@{domain}"
            password = _gen_password()
            code, body = _http('POST', f'{api}/accounts',
                               {'address': address, 'password': password})
            if code in (200, 201) and isinstance(body, dict) and body.get('id'):
                code2, tok = _http('POST', f'{api}/token',
                                   {'address': address, 'password': password})
                if code2 == 200 and isinstance(tok, dict) and tok.get('token'):
                    return {'backend': backend, 'api': api, 'address': address,
                            'password': password, 'token': tok['token'],
                            'account_id': body.get('id')}
            # rate-limited / domain rejected → try the next domain
            time.sleep(1.5)
    return None


def _mailtm_fetch_otp(session: Dict[str, Any], sender_needle: str,
                      code_re: re.Pattern, max_age_min: float,
                      deadline: float, seen_ids: set) -> Optional[str]:
    """Poll the temp-mail inbox (mail.tm or mail.gw) until a fresh OTP shows."""
    api = str(session.get('api') or MAILTM_API)
    while time.time() < deadline:
        code, body = _http('GET', f'{api}/messages?page=1',
                           token=session['token'])
        if code == 200:
            for msg in _items(body):
                if not isinstance(msg, dict) or msg.get('id') in seen_ids:
                    continue
                seen_ids.add(msg.get('id'))
                sender = str((msg.get('from') or {}).get('address', '')).lower()
                if sender_needle and sender_needle not in sender:
                    continue
                # age filter: mail.tm timestamps are ISO-8601
                age_min = _iso_age_min(msg.get('createdAt'))
                if age_min is not None and age_min > max_age_min:
                    continue
                # need the full message for the body
                mcode, full = _http('GET',
                                    f"{api}/messages/{msg.get('id')}",
                                    token=session['token'])
                if mcode == 200:
                    otp = _mailtm_extract(full, code_re)
                    if otp:
                        return otp
        time.sleep(6)
    return None


def _iso_age_min(ts: Any) -> Optional[float]:
    if not ts:
        return None
    try:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(str(ts).replace('Z', '+00:00'))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds() / 60.0
    except Exception:
        return None


# --------------------------------------------------------------- tempmail.lol
def _tempmail_create() -> Optional[Dict[str, Any]]:
    """Mint a mailbox on tempmail.lol.

    The service rotates a large pool of obscure domains per mailbox, which
    dodges the disposable-domain blocklists that DeepSeek (and others) apply
    to the well-known mail.tm/mail.gw pools. No account signup needed — the
    inbox is addressed by an opaque token.
    """
    code, body = _http('GET', f'{TEMPMAIL_API}/generate')
    if code != 200 or not isinstance(body, dict):
        return None
    address = str(body.get('address') or '').strip()
    token = str(body.get('token') or '').strip()
    if not address or not token:
        return None
    return {'backend': 'tempmail.lol', 'address': address, 'token': token}


def _tempmail_fetch_otp(session: Dict[str, Any], sender_needle: str,
                        code_re: re.Pattern, max_age_min: float,
                        deadline: float, seen_ids: set) -> Optional[str]:
    """Poll the tempmail.lol inbox for the verification code."""
    token = session.get('token') or ''
    while time.time() < deadline:
        code, body = _http('GET', f'{TEMPMAIL_API}/auth/{token}')
        if code == 200 and isinstance(body, dict):
            for msg in body.get('email') or []:
                if not isinstance(msg, dict):
                    continue
                mid = str(msg.get('date') or '') + str(msg.get('from') or '') \
                    + str(msg.get('subject') or '')
                if mid in seen_ids:
                    continue
                seen_ids.add(mid)
                sender = str(msg.get('from') or '').lower()
                if sender_needle and sender_needle not in sender:
                    continue
                age = _iso_age_min(msg.get('date'))
                if age is not None and age > max_age_min:
                    continue
                text = '\n'.join(str(msg.get(k) or '')
                                 for k in ('body', 'html', 'subject'))
                match = code_re.search(text)
                if match:
                    return match.group(1) or match.group(0)
        time.sleep(6)
    return None


# ------------------------------------------------- emailnator (real gmail.com)
class _Emailnator:
    """emailnator.com client — the only free inbox service that hands out
    REAL ``@gmail.com`` addresses (dot/plus variants of pooled master
    accounts). DeepSeek's mail pipeline silently drops every accessible
    disposable domain, but gmail.com is allowlisted essentially everywhere,
    so this is the first public backend for OTP delivery.

    Endpoints (reverse-engineered from the site bundle):
      POST /api/generate-email  {"ids": [2,3,8]}  -> {"email": "..."}
      POST /api/message-list    {"email": addr, "limit": 20} -> {"messages": [...]}
      GET  /api/message/{id}    -> message body
    CSRF: XSRF-TOKEN cookie echoed (URL-decoded) as X-XSRF-TOKEN; the pair
    is refreshed on 403/419.
    """

    def __init__(self) -> None:
        # emailnator fingerprints TLS: python urllib/requests get the page
        # but never receive session cookies — curl's fingerprint does, so
        # this client shells out to curl with a persistent cookie jar.
        self._jar = ''
        self._xsrf = ''

    def _jar_path(self) -> str:
        if not self._jar:
            import tempfile
            self._jar = os.path.join(tempfile.gettempdir(),
                                     f'dsf-emailnator-{os.getpid()}.jar')
        return self._jar

    def _curl(self, args: List[str], timeout: int = 40) -> Tuple[int, str]:
        cmd = (['curl', '-sS', '-m', '30',
                '-b', self._jar_path(), '-c', self._jar_path(),
                '-A', _EMAILNATOR_UA,
                '-H', 'Accept-Language: en-US,en;q=0.9'] + args)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=timeout)
        except (OSError, subprocess.TimeoutExpired):
            return 0, ''
        out = proc.stdout or ''
        status = 0
        if '\n' in out[-12:]:
            raw, _, code = out.rpartition('\n')
            try:
                status = int(code.strip())
                out = raw
            except ValueError:
                status = 0
        return (status or (200 if proc.returncode == 0 else 0)), out

    def _ensure(self) -> None:
        code, _ = self._curl(['-H', 'Accept: text/html,application/xhtml+xml',
                              _EMAILNATOR_BASE + '/'])
        xsrf = ''
        try:
            with open(self._jar_path(), 'r', encoding='utf-8') as fh:
                for line in fh:
                    parts = line.rstrip('\n').split('\t')
                    if len(parts) >= 7 and parts[5] == 'XSRF-TOKEN':
                        xsrf = urllib.parse.unquote(parts[6])
        except OSError:
            pass
        # the API currently works without any session/CSRF; XSRF is acquired
        # only when a 403/419 says it is needed — never fail hard here
        self._xsrf = xsrf

    def _call(self, method: str, path: str,
              body: Optional[Dict[str, Any]] = None,
              timeout: int = 25) -> Tuple[int, Any]:
        args: List[str] = []
        if method == 'POST':
            args += ['-X', 'POST', '-H', 'Content-Type: application/json',
                     '-d', json.dumps(body or {})]
        args += ['-H', 'Accept: application/json',
                 '-H', 'X-Requested-With: XMLHttpRequest',
                 '-H', 'Referer: https://www.emailnator.com/',
                 _EMAILNATOR_BASE + path]
        if self._xsrf:
            args += ['-H', f'X-XSRF-TOKEN: {self._xsrf}']
        code, out = self._curl(args, timeout)
        if code in (403, 419):  # CSRF challenge -> acquire pair, retry once
            try:
                self._ensure()
            except Exception:  # noqa: BLE001
                return code, {'error': 'session refresh failed'}
            args = [a for a in args if not a.startswith('X-XSRF-TOKEN:')]
            if self._xsrf:
                args += ['-H', f'X-XSRF-TOKEN: {self._xsrf}']
            code, out = self._curl(args, timeout)
        try:
            parsed = json.loads(out) if out.strip() else {}
        except ValueError:
            parsed = {'error': 'non-json response'}
        return code, parsed

    def generate(self) -> Optional[str]:
        # 3 = dotGmail, 8 = googleMail — real gmail.com inbox variants.
        # id 2 (plusGmail, user+tag@gmail.com) is deliberately EXCLUDED:
        # DeepSeek's mail pipeline silently drops plus-addressed recipients,
        # so the OTP never arrives (verified live 2026-09-16). Dot-variants
        # and googlemail.com aliases are delivered.
        code, body = self._call('POST', '/api/generate-email', {'ids': [3, 8]})
        if code == 200 and isinstance(body, dict) \
                and str(body.get('status') or '') == 'success':
            address = str(body.get('email') or '').strip()
            return address or None
        return None

    def messages(self, address: str) -> List[Dict[str, Any]]:
        code, body = self._call('POST', '/api/message-list',
                                {'email': address, 'limit': 20})
        if code == 200 and isinstance(body, dict) \
                and isinstance(body.get('messages'), list):
            return [m for m in body['messages'] if isinstance(m, dict)]
        return []

    def message_body(self, mid: str) -> str:
        code, body = self._call(
            'GET', f"/api/message/{urllib.parse.quote(mid, safe='')}")
        if code == 200:
            if isinstance(body, dict):
                return '\n'.join(str(body.get(k) or '')
                                 for k in ('html', 'text', 'body', 'content'))
            return body if isinstance(body, str) else ''
        return ''


_EMAILNATOR = _Emailnator()


def _emailnator_create() -> Optional[Dict[str, Any]]:
    """Mint a real @gmail.com inbox via emailnator (no account needed)."""
    try:
        address = _EMAILNATOR.generate()
    except Exception:  # noqa: BLE001
        return None
    if not address:
        return None
    return {'backend': 'emailnator-gmail', 'address': address}


def _emailnator_fetch_otp(session: Dict[str, Any], sender_needle: str,
                          code_re: re.Pattern, max_age_min: float,
                          deadline: float, seen_ids: set) -> Optional[str]:
    """Poll the emailnator gmail inbox for the verification code."""
    address = session.get('address') or ''
    while time.time() < deadline:
        try:
            msgs = _EMAILNATOR.messages(address)
        except Exception:  # noqa: BLE001
            msgs = []
        for msg in msgs:
            mid = str(msg.get('id') or '')
            if not mid or mid in seen_ids or msg.get('locked'):
                continue
            seen_ids.add(mid)
            sender = str(msg.get('from') or '').lower()
            subject = str(msg.get('subject') or '').lower()
            if sender_needle and sender_needle not in sender \
                    and sender_needle not in subject:
                continue
            try:
                age = (time.time() - float(msg.get('timestamp'))) / 60.0
            except (TypeError, ValueError):
                age = None
            if age is not None and age > max_age_min:
                continue
            match = code_re.search(_EMAILNATOR.message_body(mid))
            if match:
                return match.group(1) or match.group(0)
        time.sleep(6)
    return None


# ------------------------------------------------------------- IMAP catch-all
def _imap_catchall_create() -> Optional[Dict[str, Any]]:
    domain = os.getenv('DSF_MAIL_DOMAIN', '').strip()
    host = os.getenv('DSF_MAIL_IMAP_HOST', '').strip()
    if not domain or not host:
        return None
    address = f"{_gen_local_part()}@{domain}"
    return {'backend': 'imap-catchall', 'address': address}


def _imap_fetch_otp(session: Dict[str, Any], sender_needle: str,
                    code_re: re.Pattern, max_age_min: float,
                    deadline: float, seen_ids: set) -> Optional[str]:
    from . import refresher
    # late import avoids a circular dependency (refresher imports mailgen)
    return refresher.imap_otp(max_wait_s=max(1, int(deadline - time.time())),
                              to_needle=session['address'])


# -------------------------------------------------------------------- public
def available() -> bool:
    """True when at least one backend is plausibly configured."""
    if not autogen_enabled():
        return False
    if (os.getenv('DSF_MAIL_DOMAIN', '').strip()
            and os.getenv('DSF_MAIL_IMAP_HOST', '').strip()):
        return True
    return True  # mail.tm needs no configuration


def create_email() -> Tuple[Optional[Dict[str, Any]], str]:
    """Create a throwaway mailbox. Returns (session, error).

    Session is a dict with backend/address and (for mail.tm) credentials.
    The caller passes ``session`` to :func:`fetch_otp` once the signup form
    has asked for the verification code.

    Backend order: the operator's catch-all IMAP domain (most reliable,
    needs configuration), then emailnator's REAL gmail.com inboxes (gmail
    is allowlisted where disposable domains are dropped), then tempmail.lol
    (rotating obscure domains), then the well-known mail.tm/mail.gw pools
    (often blocklisted by big providers).
    """
    if not autogen_enabled():
        return None, 'DSF_MAIL_AUTOGEN disabled'
    backends = (_imap_catchall_create, _emailnator_create,
                _tempmail_create, _mailtm_create)
    errors: List[str] = []
    for make in backends:
        try:
            session = make()
        except Exception as e:  # noqa: BLE001
            session = None
            errors.append(f'{make.__name__}: {type(e).__name__}: {e}')
        if session:
            return session, ''
    # public temp-mail backends rate-limit in bursts: one retry pass after a
    # short pause usually gets a mailbox without failing the whole signup
    time.sleep(3.0)
    for make in backends[1:]:  # retry the non-IMAP backends once
        try:
            session = make()
        except Exception as e:  # noqa: BLE001
            session = None
            errors.append(f'{make.__name__} retry: {type(e).__name__}: {e}')
        if session:
            return session, ''
    return None, '; '.join(errors) or 'no backend produced a mailbox'


def fetch_otp(session: Dict[str, Any], max_wait_s: int = 180,
              sender_needle: Optional[str] = None,
              code_re: Optional[re.Pattern] = None,
              max_age_min: float = 30.0) -> Optional[str]:
    """Block until the OTP lands in the generated mailbox (or timeout)."""
    if not session:
        return None
    sender_needle = (sender_needle
                     or os.getenv('DSF_MAIL_OTP_SENDER', 'deepseek')).strip().lower()
    code_re = code_re or re.compile(os.getenv('DSF_MAIL_OTP_REGEX', r'\b(\d{6})\b'))
    max_age_min = float(os.getenv('DSF_MAIL_OTP_MAX_AGE', str(max_age_min)) or max_age_min)
    fetcher = {'imap-catchall': _imap_fetch_otp,
               'emailnator-gmail': _emailnator_fetch_otp,
               'tempmail.lol': _tempmail_fetch_otp}.get(session['backend']) \
        or _mailtm_fetch_otp
    try:
        return fetcher(session, sender_needle, code_re, max_age_min,
                       time.time() + max_wait_s, set())
    except Exception as e:  # noqa: BLE001
        print(f'[mailgen] fetch_otp failed: {type(e).__name__}: {e}',
              file=__import__('sys').stderr)
        return None


def main(argv: List[str]) -> int:  # pragma: no cover - CLI smoke test
    session, err = create_email()
    if not session:
        print(json.dumps({'ok': False, 'error': err}, indent=2))
        return 1
    print(json.dumps({'ok': True, 'backend': session['backend'],
                      'address': session['address'],
                      'hint': 'send a code to this address, then rerun with fetch'
                      if len(argv) < 2 else ''}, indent=2))
    if len(argv) > 1 and argv[1] == 'wait':
        print('waiting up to 120s for any code…')
        otp = fetch_otp(session, max_wait_s=120)
        print(json.dumps({'otp': otp}, indent=2))
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
