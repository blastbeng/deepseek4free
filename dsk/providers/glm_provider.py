"""GLM provider — Z.ai (chat.z.ai) anonymous + chatglm.cn (reverse-engineered).

Covers both free GLM chat surfaces with web-session credentials — no official
BigModel/Zhipu API keys:

- ``glm-*`` z.ai routes: chat.z.ai anonymous mode via the browser-transport
  engine (``_ZaiBrowser``): Aliyun Captcha 2.0 gates ``/api/v2/chat/
  completions`` and only passes inside a real browser session, so requests
  run in-page through a fetch hook while the UI's own captcha flow completes
  automatically (traceless verification). Model selection and the thinking
  toggle are enforced by rewriting the request body inside the hook, so no
  fragile UI toggles are needed.
- ``glm-4.6`` / ``glm-4.6-thinking``: chatglm.cn assistant stream with a
  refresh-token session (X-Sign md5 headers).

Model discovery is dynamic: ``GET /api/models`` is merged over the static
route list (TTL-cached), so newly released z.ai models show up automatically.
"""

import atexit
import hashlib
import json
import logging
import os
import re
import threading
import time
import uuid
from typing import Any, Dict, Generator, List, Optional

from .base import (
    Provider,
    ProviderAuthError,
    ProviderError,
    ProviderUnavailableError,
    classify_http_error,
    http_get,
    http_post_raw,
    http_post_stream,
    parse_sse_data,
)
from .jar import env_cookies, load_jar, save_jar

logger = logging.getLogger('dsk.providers.glm')

ZAI_BASE_URL = 'https://chat.z.ai'
ZAI_AUTH_URL = f'{ZAI_BASE_URL}/api/v1/auths/'
ZAI_MODELS_URL = f'{ZAI_BASE_URL}/api/models'
# Browser-only endpoint (Aliyun Captcha 2.0 rejects plain HTTP clients).
ZAI_COMPLETIONS_PATH = '/api/v2/chat/completions'
ZAI_FE_VERSION = 'prod-fe-1.1.95'

GLM_BASE_URL = 'https://chatglm.cn/chatglm'
GLM_REFRESH_URL = f'{GLM_BASE_URL}/user-api/user/refresh'
GLM_STREAM_URL = f'{GLM_BASE_URL}/backend-api/assistant/stream'
GLM_ASSISTANT_ID = '65940acff94777010aa6b796'
GLM_SIGN_SECRET = '8a1317a7468aa3ad86e997d08f3f31cb'

GLM_CONTEXT_LENGTH = int(os.getenv('DSF_GLM_CONTEXT_LENGTH', '128000'))
# The z.ai WEB transport (chat.z.ai in a browser) silently stops answering
# somewhere between 40k and 100k prompt characters — declare a realistic
# context so llmtrim trims the conversation BEFORE it reaches the page.
ZAI_WEB_CONTEXT = int(os.getenv('DSF_ZAI_CONTEXT_LENGTH', '10000'))
GLM_MAX_OUTPUT = int(os.getenv('DSF_GLM_MAX_OUTPUT', '8192'))

# Browser transport tuning (see _ZaiBrowser).
ZAI_HEADLESS = os.getenv('DSF_ZAI_HEADLESS', '').strip().lower() in ('1', 'true', 'yes')
ZAI_START_TIMEOUT = int(os.getenv('DSF_ZAI_START_TIMEOUT', '90'))
# Idle/total bounds also cap how long a wedged browser request holds the
# singleton session lock after a client has already timed out (defaults
# tuned so the lock frees before typical client timeouts cascade).
ZAI_IDLE_TIMEOUT = int(os.getenv('DSF_ZAI_IDLE_TIMEOUT', '60'))
ZAI_TOTAL_TIMEOUT = int(os.getenv('DSF_ZAI_TOTAL_TIMEOUT', '300'))
# Max time a queued request waits for the browser session slot (FIFO). With
# concurrency, parallel glm requests queue here instead of failing fast.
ZAI_BUSY_TIMEOUT = int(os.getenv('DSF_ZAI_BUSY_TIMEOUT', '180'))

# Dynamic z.ai model discovery cache.
_ZAI_MODELS_TTL = int(os.getenv('DSF_ZAI_MODELS_TTL', '900'))
_ZAI_DYNAMIC: List[Dict[str, Any]] = []
_ZAI_DYNAMIC_AT = 0.0
_ZAI_DYNAMIC_LOCK = threading.Lock()

_USER_AGENT = (
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
    '(KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36 Edg/139.0.0.0'
)

# Static routes: chatglm.cn signed-in surface (needs GLM_REFRESH_TOKEN).
# z.ai routes come from dynamic discovery (_zai_dynamic_models); the minimal
# fallback below is used only while discovery has never succeeded.
GLM_MODELS: List[Dict[str, Any]] = [
    {'id': 'glm-4.6', 'backend': 'chatglm', 'thinking': False},
    {'id': 'glm-4.6-thinking', 'backend': 'chatglm', 'thinking': True},
]

# Verified-working anonymous z.ai models (kept as discovery fallback).
GLM_ZAI_FALLBACK: List[Dict[str, Any]] = [
    {'id': 'glm-4.7', 'backend': 'zai', 'thinking': True,
     'upstream': 'glm-4.7', 'name': 'GLM-4.7'},
    {'id': 'glm-5.3-flash', 'backend': 'zai', 'thinking': True,
     'upstream': 'x-preview-l', 'name': 'GLM-5.3-Flash'},
]

_DETAILS_RE = re.compile(r'<details[^>]*>|</details>|</Full>')


def _refresh_token() -> str:
    raw = (os.getenv('GLM_REFRESH_TOKEN', '') or '').strip()
    if raw:
        return raw
    jar = load_jar('glm') or env_cookies('GLM')
    return (jar.get('refresh_token') or '').strip()


def _zai_token(no_proxy: bool = False) -> str:
    jar = load_jar('glm') or env_cookies('GLM')
    token = (jar.get('zai_token') or '').strip()
    if token:
        return token
    response = http_get(ZAI_AUTH_URL, headers={
        'User-Agent': _USER_AGENT,
        'Accept': 'application/json',
        'X-FE-Version': ZAI_FE_VERSION,
        'Origin': ZAI_BASE_URL,
        'Referer': f'{ZAI_BASE_URL}/',
    }, no_proxy=no_proxy)
    if response.status_code != 200:
        raise classify_http_error(response.status_code,
                                  response.text or '', response.headers)
    try:
        token = str((response.json() or {}).get('token') or '')
    except ValueError as e:
        raise ProviderError(f'z.ai auth returned non-JSON: {e}') from e
    if not token:
        raise ProviderAuthError('z.ai anonymous auth returned no token')
    save_jar('glm', {'zai_token': token})
    return token


def _zai_headers(token: str) -> Dict[str, str]:
    return {
        'Authorization': f'Bearer {token}',
        'User-Agent': _USER_AGENT,
        'Accept': 'application/json',
        'X-FE-Version': ZAI_FE_VERSION,
        'Origin': ZAI_BASE_URL,
        'Referer': f'{ZAI_BASE_URL}/',
    }


def _zai_dynamic_models(no_proxy: bool = False) -> List[Dict[str, Any]]:
    """Discover z.ai models from /api/models (TTL-cached, best effort).

    Returns normalized entries ``{'id', 'backend', 'thinking', 'upstream',
    'name'}``. Vision-only and deep-research routes are skipped (image input
    and agentic web UIs are not supported by this provider).
    """
    global _ZAI_DYNAMIC, _ZAI_DYNAMIC_AT
    now = time.time()
    with _ZAI_DYNAMIC_LOCK:
        if _ZAI_DYNAMIC and now - _ZAI_DYNAMIC_AT < _ZAI_MODELS_TTL:
            return _ZAI_DYNAMIC
    models: List[Dict[str, Any]] = []
    try:
        token = _zai_token(no_proxy=no_proxy)
        response = http_get(ZAI_MODELS_URL, headers=_zai_headers(token),
                            no_proxy=no_proxy)
        if response.status_code == 200:
            payload = response.json() or {}
            for entry in payload.get('data') or []:
                if not isinstance(entry, dict):
                    continue
                raw_id = str(entry.get('id') or '').strip()
                if not raw_id:
                    continue
                low = raw_id.lower()
                if 'research' in low or low.endswith('-dr') \
                        or low.startswith('x-preview'):
                    continue
                info = entry.get('info') if isinstance(entry.get('info'), dict) else {}
                meta = info.get('meta') if isinstance(info.get('meta'), dict) else {}
                caps = meta.get('capabilities')
                if isinstance(caps, dict):
                    # Boolean flags: keep only the enabled capabilities.
                    caps = {k for k, v in caps.items()
                            if v and str(v).lower() not in ('false', '0', 'no')}
                elif isinstance(caps, (list, tuple, set)):
                    caps = set(caps)
                else:
                    caps = set()
                if 'vision' in caps:
                    pass  # vision routes still accept text prompts
                think = 'think' in caps
                # Prefer the human-readable display name as the exposed id
                # (e.g. 'GLM-5.3-Flash'); fall back to the raw upstream id.
                name = str((info.get('name') if isinstance(info, dict) else None)
                           or raw_id)
                exposed = re.sub(r'[^a-z0-9._-]+', '-', name.lower()).strip('-') \
                    or re.sub(r'[^a-z0-9._-]+', '-', low).strip('-')
                if not exposed:
                    continue
                models.append({'id': exposed, 'backend': 'zai',
                               'thinking': think, 'upstream': raw_id,
                               'vision': 'vision' in caps,
                               'name': name})
    except Exception as exc:  # noqa: BLE001 — discovery is best effort
        logger.debug('z.ai dynamic model discovery failed: %s', exc)
    with _ZAI_DYNAMIC_LOCK:
        # Keep the previous list on transient failures.
        if models or not _ZAI_DYNAMIC:
            _ZAI_DYNAMIC = models
            _ZAI_DYNAMIC_AT = now
        return _ZAI_DYNAMIC


def _browser_enabled() -> bool:
    raw = os.getenv('DSF_ZAI_BROWSER', 'true').strip().lower()
    return raw not in ('0', 'false', 'no', 'off')


def _all_models() -> List[Dict[str, Any]]:
    """Discovery output with the verified fallback routes merged in."""
    entries = list(_zai_dynamic_models())
    present = {e['id'] for e in entries}
    for fallback in GLM_ZAI_FALLBACK:
        if fallback['id'] not in present:
            entries.append(fallback)
    return entries


def _glm_sign() -> Dict[str, str]:
    """chatglm.cn request signature (observed in the web client)."""
    digits = str(int(time.time() * 1000))
    t = len(digits)
    checksum = sum(int(c) for c in digits) - int(digits[t - 2])
    timestamp = digits[0:t - 2] + str(checksum % 10) + digits[t - 1]
    nonce = str(uuid.uuid4())
    sign = hashlib.md5(
        f'{timestamp}-{nonce}-{GLM_SIGN_SECRET}'.encode()).hexdigest()
    return {'X-Sign': sign, 'X-Timestamp': timestamp, 'X-Nonce': nonce,
            'X-Request-Id': str(uuid.uuid4()),
            'X-Device-Id': str(uuid.uuid4())}


def _glm_access_token(no_proxy: bool = False) -> str:
    """Exchange the refresh token for a fresh access token."""
    refresh = _refresh_token()
    if not refresh:
        raise ProviderAuthError('no chatglm.cn refresh token configured')
    headers = {
        'Authorization': f'Bearer {refresh}',
        'User-Agent': _USER_AGENT,
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Origin': 'https://chatglm.cn',
        'Referer': 'https://chatglm.cn/',
        **_glm_sign(),
    }
    response = http_post_raw(GLM_REFRESH_URL, b'{}', headers=headers,
                             no_proxy=no_proxy)
    if response.status_code != 200:
        raise classify_http_error(response.status_code,
                                  response.text or '', response.headers)
    try:
        result = (response.json() or {}).get('result') or {}
    except ValueError as e:
        raise ProviderError(f'glm refresh returned non-JSON: {e}') from e
    access = str(result.get('access_token') or '')
    if not access:
        raise ProviderAuthError('glm refresh returned no access token '
                                '(refresh token expired?)')
    if result.get('refresh_token'):
        save_jar('glm', {'refresh_token': str(result['refresh_token'])})
    return access


def _clean_thinking(text: str) -> str:
    return _DETAILS_RE.sub('', text).replace('&gt; ', '> ').strip()


# The fetch hook installed in the page. It (re)arms the per-request capture
# buffers, rewrites the completion request body with the requested model and
# thinking toggle (so no UI toggles are needed), and tees the SSE response
# stream into ``window.__zaiResp`` for incremental draining from Python.
_HOOK_JS = """
window.__zaiResp = '';
window.__zaiOffset = 0;
window.__zaiDone = false;
window.__zaiStatus = 0;
window.__zaiErr = '';
window.__zaiReqBody = '';
if (!window.__zaiHooked) {
  window.__zaiHooked = true;
  const __origFetch = window.fetch;
  window.fetch = async function (...args) {
    let url = '';
    try {
      url = String((args[0] && (args[0].url !== undefined ? args[0].url
                : args[0])) || '');
      const init = args[1] || {};
      if (url.includes('%(completions)s') && init &&
          typeof init.body === 'string') {
        window.__zaiReqBody = init.body;
        try {
          const obj = JSON.parse(init.body);
          if (window.__zaiModel) {
            obj.model = window.__zaiModel;
            if (obj.model_item) obj.model_item.id = window.__zaiModel;
          }
          if (obj.features) {
            obj.features.enable_thinking = !!window.__zaiThinking;
          }
          args[1] = Object.assign({}, init, {body: JSON.stringify(obj)});
        } catch (e) {}
      }
    } catch (e) {}
    const resp = await __origFetch.apply(this, args);
    try {
      if (url.includes('%(completions)s')) {
        window.__zaiStatus = resp.status;
        if (resp.status !== 200) {
          resp.clone().text().then(function (t) {
            window.__zaiErr = String(t || '').slice(0, 4000);
            window.__zaiDone = true;
          }).catch(function () { window.__zaiDone = true; });
        } else if (resp.body) {
          const reader = resp.clone().body.getReader();
          const dec = new TextDecoder();
          (function pump() {
            reader.read().then(function (chunk) {
              if (chunk.done) { window.__zaiDone = true; return; }
              window.__zaiResp += dec.decode(chunk.value, {stream: true});
              pump();
            }).catch(function () { window.__zaiDone = true; });
          })();
        } else {
          resp.clone().text().then(function (t) {
            window.__zaiResp += String(t || '');
            window.__zaiDone = true;
          }).catch(function () { window.__zaiDone = true; });
        }
      }
    } catch (e) { window.__zaiDone = true; }
    return resp;
  };
}
return true;
"""


class _FifoTicket:
    """Fair single-slot scheduler: concurrent z.ai requests queue in arrival
    order instead of a ``threading.Lock``'s unspecified wake-up order (or an
    immediate 'busy' error). ``acquire(timeout)`` waits in line; expiry just
    gives the slot back to the next waiter.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._busy = False
        self._waiters: List[int] = []
        self._next = 0

    def acquire(self, timeout: float) -> bool:
        with self._cond:
            ticket = self._next
            self._next += 1
            self._waiters.append(ticket)
            deadline = time.monotonic() + timeout
            try:
                while self._busy or self._waiters[0] != ticket:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    self._cond.wait(remaining)
                self._busy = True
                return True
            finally:
                if ticket in self._waiters:
                    self._waiters.remove(ticket)

    def release(self) -> None:
        with self._cond:
            self._busy = False
            self._cond.notify_all()


class _ZaiBrowser:
    """Single lazy Chromium session driving the chat.z.ai web UI.

    Aliyun Captcha 2.0 only issues ``captcha_verify_param`` inside a real
    browser session, so z.ai completions run in-page: the hook above tees the
    SSE stream while the UI's own traceless captcha flow completes
    automatically. One request at a time (anonymous session limits), thread
    safe, self-healing across crashes (one restart+retry per request).
    """

    _instance: Optional['_ZaiBrowser'] = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._page = None
        self._display = None
        self._busy = _FifoTicket()

    @classmethod
    def instance(cls) -> '_ZaiBrowser':
        with cls._instance_lock:
            if cls._instance is None:
                browser = cls()
                atexit.register(browser.close)
                cls._instance = browser
            return cls._instance

    # ------------------------------------------------------------- lifecycle
    def _alive(self) -> bool:
        if self._page is None:
            return False
        try:
            return bool(self._page.run_js('return 1;'))
        except Exception:  # noqa: BLE001
            return False

    def close(self) -> None:
        try:
            if self._page is not None:
                self._page.quit()
        except Exception:  # noqa: BLE001
            pass
        self._page = None
        try:
            if self._display is not None:
                self._display.stop()
        except Exception:  # noqa: BLE001
            pass
        self._display = None

    def _ensure(self) -> None:
        if self._alive():
            return
        self.close()
        from DrissionPage import ChromiumPage, ChromiumOptions  # heavy import
        if not os.environ.get('DISPLAY'):
            try:
                from pyvirtualdisplay import Display
                self._display = Display(visible=False, size=(1440, 900))
                self._display.start()
            except Exception as exc:  # noqa: BLE001 — fall back to headless
                logger.debug('z.ai Xvfb unavailable (%s); using headless', exc)
                self._display = None
        options = ChromiumOptions().auto_port()
        options.set_argument('--no-sandbox')
        options.set_argument('--disable-gpu')
        options.set_argument('--disable-blink-features=AutomationControlled')
        options.set_argument('--window-size=1440,900')
        if ZAI_HEADLESS or not os.environ.get('DISPLAY'):
            options.headless(True)
        self._page = ChromiumPage(addr_or_opts=options)
        self._page.get(ZAI_BASE_URL)
        self._wait_ready()

    def _wait_ready(self) -> None:
        """Wait for the anonymous guest token the UI stores in localStorage."""
        deadline = time.time() + ZAI_START_TIMEOUT
        while time.time() < deadline:
            try:
                token = self._page.run_js(
                    'return window.localStorage.getItem("token");')
                if token:
                    save_jar('glm', {'zai_token': str(token)})
                    return
            except Exception:  # noqa: BLE001 — page still loading
                pass
            time.sleep(1)
        self.close()
        raise ProviderUnavailableError(
            'z.ai browser session did not initialize (no guest token)')

    # ----------------------------------------------------------- utilities
    def _js(self, script: str, default: Any = None) -> Any:
        try:
            return self._page.run_js(script)
        except Exception:  # noqa: BLE001
            return default

    def _click(self, selector: str, timeout: int = 6) -> bool:
        try:
            ele = self._page.ele(selector, timeout=timeout)
            if ele:
                ele.click()
                return True
        except Exception:  # noqa: BLE001
            pass
        return False

    def _dismiss_alerts(self) -> None:
        """Close any native JS dialog blocking page interactions."""
        try:
            self._page.handle_alert(accept=True, timeout=2)
        except Exception:  # noqa: BLE001 — no dialog present
            pass

    def _new_chat(self) -> None:
        self._dismiss_alerts()
        if not self._click('#new-chat-button'):
            self._page.get(ZAI_BASE_URL)
            time.sleep(2)

    def _drain(self) -> Generator[str, None, None]:
        """Yield incremental SSE text until the hooked reader is done."""
        deadline = time.time() + ZAI_TOTAL_TIMEOUT
        idle_at = time.time() + ZAI_IDLE_TIMEOUT
        while True:
            total = self._js('return window.__zaiResp.length;', 0) or 0
            offset = self._js('return window.__zaiOffset || 0;', 0) or 0
            if total > offset:
                piece = self._js(
                    'const p = window.__zaiResp.slice(window.__zaiOffset || 0);'
                    'window.__zaiOffset = (window.__zaiOffset || 0) + p.length;'
                    'return p;', '') or ''
                if piece:
                    idle_at = time.time() + ZAI_IDLE_TIMEOUT
                    yield piece
                    continue
            if self._js('return window.__zaiDone;', False):
                return
            if time.time() > deadline:
                raise ProviderError('z.ai browser stream timed out')
            if time.time() > idle_at:
                return  # deliver whatever arrived
            time.sleep(0.3)

    # ------------------------------------------------------------------ ask
    def ask(self, prompt: str, upstream: str, thinking: bool,
            no_proxy: bool = False,
            image_paths: Optional[List[str]] = None
            ) -> Generator[Dict[str, Any], None, None]:
        if not self._busy.acquire(timeout=ZAI_BUSY_TIMEOUT):
            raise ProviderUnavailableError(
                'z.ai browser session is busy with another request')
        try:
            yield from self._ask_inner(prompt, upstream, thinking, no_proxy,
                                       retry=True, image_paths=image_paths)
        finally:
            self._busy.release()
            for p in (image_paths or []):
                try:
                    os.unlink(p)
                except OSError:
                    pass

    def _ask_inner(self, prompt: str, upstream: str, thinking: bool,
                   no_proxy: bool, retry: bool = True,
                   image_paths: Optional[List[str]] = None
                   ) -> Generator[Dict[str, Any], None, None]:
        self._ensure()
        try:
            self._new_chat()
            try:
                box = self._page.ele('#chat-input', timeout=20)
            except Exception as exc:  # noqa: BLE001 — dead session?
                box = None
                logger.debug('z.ai chat input lookup failed: %s', exc)
            if box is None:
                if retry:
                    logger.warning('z.ai session stale; restarting browser')
                    self.close()
                    yield from self._ask_inner(prompt, upstream, thinking,
                                               no_proxy, retry=False,
                                               image_paths=image_paths)
                    return
                raise ProviderUnavailableError(
                    'z.ai chat input not found (session dead)')
            # Attach images through the UI file input BEFORE the prompt is
            # sent — vision-capable upstreams then receive them with the
            # message. The hidden input accepts png/jpg/jpeg/bmp/gif.
            if image_paths:
                try:
                    fi = self._page.ele('css:input[type=file]', timeout=8)
                    fi.input(list(image_paths))
                    time.sleep(3 + 2 * len(image_paths))  # upload settle
                except Exception as exc:  # noqa: BLE001
                    raise ProviderError(
                        f'z.ai image upload failed: {exc}') from exc
            # Model selection is enforced by rewriting the request body in
            # the fetch hook — clicking selector buttons corrupts UI state.
            self._js(_HOOK_JS % {'completions': ZAI_COMPLETIONS_PATH}, False)
            self._js('window.__zaiModel = ' + json.dumps(upstream) + ';\n'
                     'window.__zaiThinking = '
                     + ('true' if thinking else 'false') + ';\n'
                     'return true;', False)
            try:
                box.clear()
                box.input(prompt)
            except Exception as exc:
                raise ProviderError(f'z.ai prompt input failed: {exc}') from exc
            if not self._click('#send-message-button', timeout=10):
                self._dismiss_alerts()  # a login/consent dialog may block it
                if not self._click('#send-message-button', timeout=6):
                    raise ProviderError('z.ai send button not found')

            status = 0
            buf = ''
            emitted = False
            for piece in self._drain():
                buf += piece
                if '\n' not in buf:
                    continue
                *lines, buf = buf.split('\n')
                for line in lines:
                    data = parse_sse_data(line)
                    if not data:
                        continue
                    inner = data.get('data') or {}
                    error = data.get('error')
                    if not error and isinstance(inner, dict):
                        error = inner.get('error') or (inner.get('data') or {}) \
                            .get('error')
                    if error:
                        raise ProviderError(f'z.ai stream error: {error}')
                    if not isinstance(inner, dict):
                        continue
                    phase = inner.get('phase') or ''
                    delta = inner.get('delta_content') or ''
                    if delta:
                        emitted = True
                        if phase == 'thinking':
                            yield {'content': _clean_thinking(delta),
                                   'type': 'thinking', 'finish_reason': None}
                        else:
                            yield {'content': delta, 'type': 'text',
                                   'finish_reason': None}
                    if inner.get('done') or phase == 'done':
                        return
            # Stream closed (or idle timeout): classify non-200 responses.
            status = self._js('return window.__zaiStatus;', 0) or 0
            if status and status != 200:
                body = self._js('return window.__zaiErr;', '') or ''
                raise classify_http_error(status, body or buf, {})
            if not emitted and buf and not self._js(
                    'return window.__zaiDone;', False):
                pass  # partial SSE tail; parsed below for stragglers
            for line in buf.split('\n'):
                data = parse_sse_data(line)
                if not data:
                    continue
                inner = data.get('data') or {}
                if isinstance(inner, dict):
                    delta = inner.get('delta_content') or ''
                    if delta:
                        phase = inner.get('phase') or ''
                        if phase == 'thinking':
                            yield {'content': _clean_thinking(delta),
                                   'type': 'thinking', 'finish_reason': None}
                        else:
                            yield {'content': delta, 'type': 'text',
                                   'finish_reason': None}
            if not emitted:
                raise ProviderError(
                    f'z.ai browser returned no content (status={status or "?"})')
            yield {'content': '', 'type': 'text', 'finish_reason': 'stop'}
        except (ProviderError, ProviderUnavailableError,
                ProviderAuthError):
            raise
        except Exception as exc:  # noqa: BLE001 — browser transport failure
            if retry:
                logger.warning('z.ai browser transport failed (%s); retrying',
                               exc)
                self.close()
                yield from self._ask_inner(prompt, upstream, thinking,
                                           no_proxy, retry=False)
                return
            raise ProviderError(f'z.ai browser transport failed: {exc}') from exc


_ZAI_BROWSER: Optional[_ZaiBrowser] = None
_ZAI_BROWSER_LOCK = threading.Lock()


def _zai_browser() -> _ZaiBrowser:
    global _ZAI_BROWSER
    with _ZAI_BROWSER_LOCK:
        if _ZAI_BROWSER is None:
            _ZAI_BROWSER = _ZaiBrowser.instance()
        return _ZAI_BROWSER


class GlmProvider(Provider):
    name = 'glm'

    # ---------------------------------------------------------------- provider
    def available(self, auth_key: Optional[str] = None) -> bool:
        return _browser_enabled() or bool(_refresh_token())

    def list_models(self, auth_key: Optional[str] = None) -> List[Dict[str, Any]]:
        """Dynamically discovered z.ai routes + chatglm.cn signed-in routes.

        z.ai is validated anonymously (guest token, cached in the jar);
        chatglm routes require a refresh token. Discovery is TTL-cached and
        keeps the last good list on transient failures.
        """
        has_chatglm = bool(_refresh_token())
        dynamic = _all_models() if _browser_enabled() else []
        out: List[Dict[str, Any]] = []
        seen = set()
        if dynamic:
            _zai_token()  # cheap anonymous validation, cached in the jar

        def add(entry_id: str, upstream: str, thinking: bool,
                backend: str, name: Optional[str] = None,
                vision: bool = False, context_length: int = GLM_CONTEXT_LENGTH
                ) -> None:
            if entry_id in seen:
                return
            seen.add(entry_id)
            extra: Dict[str, Any] = {'backend': backend}
            if name:
                extra['name'] = name
            out.append({
                'id': entry_id,
                'upstream_model': upstream,
                'thinking_enabled': thinking,
                'search_enabled': False,
                'vision': bool(vision),
                'image_gen': False,
                'context_length': context_length,
                'max_output_tokens': GLM_MAX_OUTPUT,
                'extra': extra,
            })

        # Verified anonymous-friendly models first, then the rest.
        preferred = ('glm-4.7', 'glm-5.3-flash')
        for entry in dynamic:
            if entry['id'] in preferred:
                add(entry['id'], entry['upstream'], entry['thinking'], 'zai',
                    entry.get('name'), context_length=ZAI_WEB_CONTEXT)
        for entry in dynamic:
            add(entry['id'], entry['upstream'], entry['thinking'], 'zai',
                entry.get('name'), vision=bool(entry.get('vision')),
                context_length=ZAI_WEB_CONTEXT)
        for entry in GLM_MODELS:
            if entry['backend'] == 'chatglm' and not has_chatglm:
                continue
            add(entry['id'], entry.get('upstream', entry['id']),
                entry['thinking'], entry['backend'])
        return out

    def stream(self, prompt: str, *, model: str, thinking_enabled: bool = False,
               search_enabled: bool = False, temperature: Optional[float] = None,
               max_tokens: Optional[int] = None,
               images: Optional[List[Dict[str, Any]]] = None,
               image_generation: bool = False,
               no_proxy: bool = False,
               auth_key: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
        backend, thinking, upstream = 'zai', thinking_enabled, model
        entry = next((e for e in GLM_MODELS if e['id'] == model), None)
        if entry is None and _browser_enabled():
            entry = next((e for e in _all_models()
                          if e['id'] == model), None)
        if entry is not None:
            backend, thinking = entry['backend'], entry['thinking']
            upstream = entry.get('upstream', entry['id'])
        if images:
            # z.ai's headless UI upload is not reliably automatable (the
            # file input ignores programmatic DataTransfer/input events and
            # native dialogs can't be driven), so image input stays
            # unsupported here; chatgpt/gemini providers handle images.
            raise ProviderError(
                'glm image input is not supported yet — z.ai web upload '
                'cannot be automated headlessly; use a vision-capable '
                'provider with credentials (chatgpt/gemini)')
        if backend == 'chatglm':
            return self._stream_chatglm(prompt, thinking, no_proxy=no_proxy)
        if not _browser_enabled():
            raise ProviderUnavailableError(
                'z.ai browser transport is disabled (DSF_ZAI_BROWSER=false)')
        return _zai_browser().ask(prompt, upstream, thinking, no_proxy=no_proxy)

    # ------------------------------------------------------------- chatglm.cn
    def _stream_chatglm(self, prompt: str, thinking: bool,
                        no_proxy: bool = False) -> Generator[Dict[str, Any], None, None]:
        access = _glm_access_token(no_proxy=no_proxy)
        body: Dict[str, Any] = {
            'assistant_id': GLM_ASSISTANT_ID,
            'conversation_id': '',
            'project_id': '',
            'chat_type': 'user_chat',
            'messages': [{'role': 'user',
                          'content': [{'type': 'text', 'text': prompt}]}],
            'meta_data': {
                'channel': '',
                'draft_id': '',
                'if_plus_model': True,
                'input_question_type': 'xxxx',
                'is_networking': False,
                'is_test': False,
                'platform': 'pc',
                'quote_log_id': '',
                'cogview': {'rm_label_watermark': False},
            },
        }
        if thinking:
            body['meta_data']['chat_mode'] = 'zero'
        headers = {
            'Authorization': f'Bearer {access}',
            'User-Agent': _USER_AGENT,
            'Content-Type': 'application/json',
            'Accept': 'text/event-stream',
            'Origin': 'https://chatglm.cn',
            'Referer': 'https://chatglm.cn/',
            **_glm_sign(),
        }
        response = http_post_stream(GLM_STREAM_URL, headers=headers,
                                    json_body=body, no_proxy=no_proxy)
        if response.status_code != 200:
            try:
                error_text = response.text or ''
            except Exception:  # pragma: no cover
                error_text = f'HTTP {response.status_code}'
            raise classify_http_error(response.status_code, error_text,
                                      response.headers)
        return self._iter_chatglm(response)

    def _iter_chatglm(self, response) -> Generator[Dict[str, Any], None, None]:
        for line in response.iter_lines():
            data = parse_sse_data(line)
            if not data:
                continue
            result = data.get('result') or {}
            status = result.get('status') or ''
            for part in result.get('parts') or []:
                if not isinstance(part, dict):
                    continue
                for item in part.get('content') or []:
                    if not isinstance(item, dict):
                        continue
                    kind = item.get('type')
                    if kind == 'text' and item.get('text'):
                        yield {'content': item['text'], 'type': 'text',
                               'finish_reason': None}
                    elif kind == 'think' and item.get('think'):
                        yield {'content': item['think'], 'type': 'thinking',
                               'finish_reason': None}
                    elif kind == 'image' and isinstance(item.get('image'), list) \
                            and part.get('status') == 'finish':
                        for image in item['image']:
                            url = (image or {}).get('image_url') or ''
                            if url.startswith('http'):
                                yield {'content': f'![image]({url})',
                                       'type': 'image', 'url': url,
                                       'finish_reason': None}
            if status in ('finish', 'intervene'):
                break
        yield {'content': '', 'type': 'text', 'finish_reason': 'stop'}
