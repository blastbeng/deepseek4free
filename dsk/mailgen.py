"""Automatic email-address generation for unattended credential flows.

Used by the refresher (browser re-login / auto-signup) when no login e-mail
is configured: a throwaway mailbox is created on the fly, the verification
e-mail is fetched from it and the OTP is extracted — so the renewal ladder
stays fully unmanned with zero mail configuration.

Two backends, tried in order (first that is *configured* wins, then the
first that *works*):

1. IMAP catch-all (``DSF_MAIL_IMAP_HOST`` + ``DSF_MAIL_DOMAIN``):
   a random local part is invented under your own domain
   (``dsf-<hex8>@<domain>``); the OTP is read through the existing IMAP
   poller. Most reliable — use this when you own a catch-all mailbox.

2. mail.tm (https://mail.tm, free public API, no key): a real throwaway
   account is created on a public temp-mail domain and its inbox is polled
   over HTTPS. Works out of the box, but public domains are sometimes
   rejected by signup forms — the caller treats failures as a normal
   renewal-ladder miss.

Env switches:
    DSF_MAIL_AUTOGEN   master switch for this module (default: true)
    DSF_MAIL_DOMAIN    domain for the IMAP catch-all backend (optional)

CLI smoke test:
    python -m dsk.mailgen
"""

import json
import os
import re
import secrets
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

MAILTM_API = 'https://api.mail.tm'
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


def _mailtm_domains() -> List[str]:
    code, body = _http('GET', f'{MAILTM_API}/domains?page=1')
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
    """Create a throwaway mail.tm account. Returns session dict or None."""
    for domain in _mailtm_domains()[:3]:
        address = f"{_gen_local_part()}@{domain}"
        password = _gen_password()
        code, body = _http('POST', f'{MAILTM_API}/accounts',
                           {'address': address, 'password': password})
        if code in (200, 201) and isinstance(body, dict) and body.get('id'):
            code2, tok = _http('POST', f'{MAILTM_API}/token',
                               {'address': address, 'password': password})
            if code2 == 200 and isinstance(tok, dict) and tok.get('token'):
                return {'backend': 'mail.tm', 'address': address,
                        'password': password, 'token': tok['token'],
                        'account_id': body.get('id')}
        # rate-limited / domain rejected → try the next domain
        time.sleep(1.5)
    return None


def _mailtm_fetch_otp(session: Dict[str, Any], sender_needle: str,
                      code_re: re.Pattern, max_age_min: float,
                      deadline: float, seen_ids: set) -> Optional[str]:
    """Poll the mail.tm inbox until a fresh OTP shows up."""
    while time.time() < deadline:
        code, body = _http('GET', f'{MAILTM_API}/messages?page=1',
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
                                    f"{MAILTM_API}/messages/{msg.get('id')}",
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
    """
    if not autogen_enabled():
        return None, 'DSF_MAIL_AUTOGEN disabled'
    backends = (_imap_catchall_create, _mailtm_create)
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
    fetcher = _imap_fetch_otp if session['backend'] == 'imap-catchall' \
        else _mailtm_fetch_otp
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
