"""Browser relay for chat.qwen.ai completions.

The Aliyun WAF punishes EVERY non-browser POST to /api/v2/chat/completions
(browser cookies, SPA-exact headers/body, five TLS impersonations, HTTP/1.1
— all replayed and still punished) and even stalls signed-in SPA sessions
indefinitely. The one transport that streams is the qwen SPA in GUEST mode:
the landing-page composer, type + Enter, the SPA fires the completions POST
itself. So the relay runs one persistent guest Chrome session (DrissionPage)
and drives the real UI:

1. A hook records the SPA's completions response incrementally (fetch
   ``clone().body.getReader()`` tee + XHR ``onprogress`` deltas).
2. Each request: (re)load the landing composer, pick the requested model in
   the guest model dropdown, type the prompt, press Enter.
3. Hooked chunks are drained into Python with short run_js polls and parsed
   as SSE events. Guest chats are IP-rate-limited by qwen and the guest
   picker only offers a subset of the catalog — the relay refuses models
   the picker does not offer instead of silently serving a different one.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, Generator, List, Optional, Tuple

logger = logging.getLogger(__name__)


_CAPTURE_HOOK = r"""
window.__dsfCap = window.__dsfCap || {reqs: [], seq: 0};
window.__dsfEnc = window.__dsfEnc || function(text) {
  var bytes = new TextEncoder().encode(text);
  var bin = '';
  for (var j = 0; j < bytes.length; j += 0x8000)
    bin += String.fromCharCode.apply(null, bytes.subarray(j, j + 0x8000));
  return btoa(bin);
};
if (!window.__dsfCapHooked) {
  window.__dsfCapHooked = true;
  var of = window.fetch;
  window.fetch = function() {
    var a = arguments;
    var url = String((a[0] && a[0].url) || a[0]);
    var p = of.apply(this, a);
    if (url.indexOf('completions') === -1) return p;
    var init = a[1] || {};
    var entry = {id: ++window.__dsfCap.seq, url: url,
                 body: (typeof init.body === 'string' ? init.body : ''),
                 status: 0, ctype: '', chunks: [], done: false, err: ''};
    window.__dsfCap.reqs.push(entry);
    return p.then(function(r) {
      entry.status = r.status;
      try { entry.ctype = r.headers.get('content-type') || ''; } catch (e) {}
      try {
        var dec = new TextDecoder('utf-8', {stream: true});
        var reader = r.clone().body.getReader();
        (function pump() {
          return reader.read().then(function(step) {
            if (step.done) { entry.done = true; return; }
            entry.chunks.push(window.__dsfEnc(
              dec.decode(step.value, {stream: true})));
            return pump();
          });
        })().catch(function(e) { entry.done = true; entry.err = String(e); });
      } catch (e) { entry.done = true; entry.err = String(e); }
      return r;
    }, function(e) { entry.done = true; entry.err = String(e); throw e; });
  };
  var oo = XMLHttpRequest.prototype.open;
  var os = XMLHttpRequest.prototype.send;
  XMLHttpRequest.prototype.open = function(m, u) {
    this.__dsfUrl = String(u);
    return oo.apply(this, arguments);
  };
  XMLHttpRequest.prototype.send = function(b) {
    var xhr = this;
    if (xhr.__dsfUrl && xhr.__dsfUrl.indexOf('completions') !== -1 &&
        (!xhr.responseType || xhr.responseType === 'text')) {
      var entry = {id: ++window.__dsfCap.seq, url: xhr.__dsfUrl,
                   body: (typeof b === 'string' ? b : ''),
                   status: 0, ctype: '', chunks: [], done: false, err: ''};
      window.__dsfCap.reqs.push(entry);
      xhr.addEventListener('progress', function() {
        try {
          entry.status = xhr.status;
          entry.ctype = xhr.getResponseHeader('content-type') || '';
          var text = xhr.responseText || '';
          var prev = entry.xlen || 0;
          if (text.length > prev) {
            entry.chunks.push(window.__dsfEnc(text.slice(prev)));
            entry.xlen = text.length;
          }
        } catch (e) {}
      });
      xhr.addEventListener('loadend', function() {
        entry.done = true;
        try { entry.status = xhr.status; } catch (e) {}
      });
    }
    return os.apply(this, arguments);
  };
}
window.__dsfCapReset = function() { window.__dsfCap.reqs.length = 0;
                                    return 'ok'; };
window.__dsfCapList = function() {
  return JSON.stringify(window.__dsfCap.reqs.map(function(r) {
    return {id: r.id, status: r.status, ctype: r.ctype, done: !!r.done,
            n: r.chunks.length, url: r.url};
  }));
};
window.__dsfCapTake = function(id, from) {
  var r = window.__dsfCap.reqs.find(function(x) { return x.id === id; });
  if (!r) return null;
  return JSON.stringify({id: r.id, status: r.status, ctype: r.ctype,
                         done: !!r.done, err: r.err || '', body: r.body || '',
                         total: r.chunks.length, chunks: r.chunks.slice(from)});
};
return 'hooked';
"""


class RelayPunish(RuntimeError):
    """WAF interstitial / stall swallowed the stream — rebuild and retry."""


def norm_model(model: str) -> str:
    """'qwen3.8-max' and display title 'Qwen3.8-Max' both -> 'qwen38max'."""
    return re.sub(r'[^a-z0-9]+', '', str(model or '').lower())


def _sse_events(buffer: bytes) -> Tuple[List[Dict[str, Any]], bytes]:
    """Split complete ``data:`` SSE events out of `buffer` -> (events, rest)."""
    events: List[Dict[str, Any]] = []
    while b'\n' in buffer:
        line, buffer = buffer.split(b'\n', 1)
        text = line.decode('utf-8', errors='replace').strip()
        if not text.startswith('data:'):
            continue
        payload = text[5:].strip()
        if not payload or payload == '[DONE]':
            continue
        try:
            events.append(json.loads(payload))
        except ValueError:
            continue
    return events, buffer


class QwenRelay:
    """One persistent guest browser session driving the qwen UI."""

    RELOAD_TTL = 1800      # re-mint WAF clearance twice an hour
    SEND_TIMEOUT = 45      # wait for the SPA to fire the completions request
    FIRST_BYTE_TIMEOUT = 60   # headers+body must start after Enter
    CHUNK_STALL_TIMEOUT = 120  # max idle gap between stream chunks
    STREAM_TIMEOUT = 600   # cap on one generation
    POLL_S = 0.5

    def __init__(self) -> None:
        self._page = None
        self._lock = threading.RLock()
        self._loaded_at: float = 0.0
        self._offered: List[str] = []  # display titles seen in the picker

    # ------------------------------------------------------------- lifecycle
    def enabled(self) -> bool:
        return os.getenv('DSF_QWEN_RELAY', '1').strip().lower() not in \
            ('0', 'false', 'no', 'off')

    def offered(self) -> List[str]:
        """Model display titles seen in the guest picker (empty until the
        first dropdown open). Used by the provider to keep /v1/models honest
        about what the relay can actually serve."""
        return list(self._offered)

    def _alive(self) -> bool:
        try:
            return self._page is not None and bool(
                self._page.run_js('return 1;') == 1)
        except Exception:  # noqa: BLE001 — crashed/closed browser
            return False

    def _slider_pass(self) -> None:
        from dsk import refresher
        if 'Captcha Interception' in (self._page.title or ''):
            refresher._qwen_slider_pass(self._page)
            time.sleep(4)

    def _accept_dialogs(self) -> None:
        """Dismiss cookie-consent overlays that block the composer."""
        for text in ('Accept all cookies', 'Accept all strict', 'I agree'):
            try:
                ele = self._page.ele(f'text:{text}', timeout=2)
                if ele:
                    ele.click()
                    time.sleep(1)
                    return
            except Exception:  # noqa: BLE001
                continue

    def _install_hook(self) -> None:
        out = str(self._page.run_js(_CAPTURE_HOOK) or '')
        if out != 'hooked':
            raise RuntimeError(f'qwen relay hook failed: {out!r}')

    def _open_home(self) -> None:
        """(Re)load the landing composer and reinstall the hook."""
        page = self._page
        page.get('https://chat.qwen.ai/')
        time.sleep(6)
        self._slider_pass()
        self._accept_dialogs()
        self._install_hook()
        self._loaded_at = time.time()
        if self._composer() is None:
            raise RuntimeError('qwen relay: composer not found on landing')
        logger.info('qwen relay: guest composer ready')

    def _build(self) -> None:
        from dsk import refresher
        if self._page is not None:
            try:
                self._page.quit()
            except Exception:  # noqa: BLE001
                pass
            self._page = None
        refresher._ensure_display()
        self._page = refresher._browser(headed=True)
        self._open_home()

    def _ensure(self) -> None:
        if self._alive() and (time.time() - self._loaded_at) <= self.RELOAD_TTL:
            return
        if self._alive():
            try:
                self._open_home()
                return
            except Exception as e:  # noqa: BLE001 — fall through to rebuild
                logger.warning('qwen relay refresh failed (%s); rebuilding', e)
        self._build()

    def shutdown(self) -> None:
        with self._lock:
            if self._page is not None:
                try:
                    self._page.quit()
                except Exception:  # noqa: BLE001
                    pass
                self._page = None

    # ------------------------------------------------------------- streaming
    def _composer(self):
        for sel in ('css:#chat-input', 'css:div[contenteditable=true]',
                    'css:textarea'):
            try:
                ele = self._page.ele(sel, timeout=4)
                if ele:
                    return ele
            except Exception:  # noqa: BLE001
                continue
        return None

    def _select_model(self, model: str) -> bool:
        """Pick `model` in the guest model dropdown; False if not offered.

        Also records every displayed title in ``_offered`` so the provider
        can keep its model list honest."""
        page = self._page
        trig = None
        for sel in ('css:.ant-dropdown-trigger', 'css:.wms-trigger'):
            try:
                trig = page.ele(sel, timeout=4)
                if trig:
                    break
            except Exception:  # noqa: BLE001
                continue
        if trig is None:
            return False
        try:
            trig.click()
        except Exception:  # noqa: BLE001
            return False
        time.sleep(1.2)
        want = norm_model(model)
        try:
            picked = str(page.run_js('''
              var want = %s;
              var items = document.querySelectorAll(
                '.wms-list__item, .ant-dropdown-menu-item');
              var seen = [];
              var out = 'not-found';
              for (var i = 0; i < items.length; i++) {
                var t = (items[i].innerText||'').trim().split('\\n')[0];
                if (!t) continue;
                seen.push(t);
                var norm = t.toLowerCase().replace(/[^a-z0-9]+/g, '');
                if (norm === want) {
                  items[i].click();
                  out = 'picked:' + t;
                }
              }
              window.__dsfOffered = seen;
              return out;
            ''' % json.dumps(want)) or '')
        except Exception:  # noqa: BLE001
            picked = 'err'
        titles = []
        try:
            titles = json.loads(str(page.run_js(
                'return JSON.stringify(window.__dsfOffered || []);') or '[]'))
        except Exception:  # noqa: BLE001
            titles = []
        if titles:
            self._offered = [str(t) for t in titles]
        time.sleep(0.8)
        try:
            page.run_js('if (document.body) document.body.click();')
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.4)
        if picked.startswith('picked'):
            logger.info('qwen relay: model selected -> %s', picked[7:])
            return True
        logger.warning('qwen relay: model %r not offered (picker shows %s)',
                       model, ', '.join(self._offered) or '?')
        return False

    def _type_and_send(self, prompt: str) -> None:
        page = self._page
        typed = str(page.run_js(
            'var ed = document.querySelector("#chat-input, '
            'div[contenteditable=true], textarea");'
            'if (!ed) return "no-editor";'
            'ed.focus();'
            'if (!document.execCommand("insertText", false, %s)) '
            'return "exec-failed";'
            'return "typed";' % json.dumps(prompt)) or '')
        if typed != 'typed':
            ele = self._composer()
            if ele is None:
                raise RuntimeError(f'qwen relay: cannot type ({typed})')
            ele.input(prompt)
        time.sleep(0.8)
        from DrissionPage.common import Actions
        Actions(page).key_down('Enter').key_up('Enter')

    def _wait_entry(self) -> Dict[str, Any]:
        deadline = time.time() + self.SEND_TIMEOUT
        while time.time() < deadline:
            time.sleep(self.POLL_S)
            raw = self._page.run_js('return window.__dsfCapList();') or '[]'
            try:
                listing = json.loads(raw) or []
            except ValueError:
                listing = []
            if listing:
                return listing[0]
        raise RuntimeError('qwen relay: SPA never fired the completions '
                           'request')

    def _take(self, eid: int, consumed: int) -> Optional[Dict[str, Any]]:
        raw = self._page.run_js(
            f'return window.__dsfCapTake({int(eid)}, {int(consumed)});')
        if not raw:
            return None
        try:
            return json.loads(raw)
        except ValueError:
            return None

    def _check_model(self, entry_body: str, model: str) -> None:
        try:
            body = json.loads(entry_body) if entry_body else {}
            sent = body.get('model') or ''
            if not sent:
                msgs = body.get('messages') or [{}]
                models = (msgs[0] or {}).get('models') or []
                sent = models[0] if models else ''
            if sent and norm_model(sent) != norm_model(model):
                raise RuntimeError(
                    f'qwen relay: SPA is serving model {sent!r} instead of '
                    f'{model!r}')
        except ValueError:
            pass

    def _drain(self, eid: int, model: str) -> Generator[Dict[str, Any],
                                                        None, None]:
        consumed = 0
        buf = b''
        saw_any = False
        checked_model = False
        first_byte_at = time.time() + self.FIRST_BYTE_TIMEOUT
        chunk_deadline = time.time() + self.CHUNK_STALL_TIMEOUT
        deadline = time.time() + self.STREAM_TIMEOUT
        while time.time() < deadline:
            time.sleep(self.POLL_S)
            take = self._take(eid, consumed)
            if take is None:
                break
            if not checked_model:
                self._check_model(str(take.get('body') or ''), model)
                checked_model = True
            ctype = str(take.get('ctype') or '').lower()
            status = int(take.get('status') or 0)
            if 'text/html' in ctype:
                raise RelayPunish(
                    f'WAF interstitial on completions (http {status})')
            new = take.get('chunks') or []
            if new:
                for chunk in new:
                    buf += base64.b64decode(chunk)
                chunk_deadline = time.time() + self.CHUNK_STALL_TIMEOUT
            consumed = int(take.get('total') or consumed)
            total = consumed
            events, buf = _sse_events(buf)
            for ev in events:
                saw_any = True
                yield ev
            done = bool(take.get('done'))
            if done and status >= 400 and not saw_any:
                snippet = buf[:200].decode('utf-8', errors='replace')
                raise RelayPunish(f'http {status}: {snippet}')
            if not total and time.time() > first_byte_at:
                raise RelayPunish('completions stalled: no response bytes')
            if total and not new and time.time() > chunk_deadline:
                raise RelayPunish('completions stalled: stream idle')
            if done and consumed >= total:
                break
        if buf.strip():
            events, _ = _sse_events(buf + b'\n')
            for ev in events:
                saw_any = True
                yield ev
        if not saw_any:
            snippet = buf[:200].decode('utf-8', errors='replace')
            raise RuntimeError('qwen relay: no SSE events captured; '
                               f'buffer head: {snippet!r}')

    def stream(self, token: str, model: str, prompt: str,
               chat_id: str = '', thinking_enabled: bool = False,
               search_enabled: bool = False) -> Generator[Dict[str, Any],
                                                          None, None]:
        """Yield upstream qwen SSE dicts for one prompt via the guest UI.

        ``token``/``chat_id`` are unused (guest sessions) and kept for API
        compatibility with the provider call site.
        """
        if not self.enabled():
            raise RuntimeError('qwen relay disabled')
        with self._lock:
            self._ensure()
            try:
                self._open_home()
                if not self._select_model(model):
                    raise RuntimeError(
                        f'model {model!r} is not offered in the qwen guest '
                        f'picker (offered: {", ".join(self._offered) or "?"})')
                self._type_and_send(prompt)
                entry = self._wait_entry()
                yield from self._drain(int(entry['id']), model)
            except RelayPunish:
                logger.warning('qwen relay: stream failed; rebuilding session')
                try:
                    self._build()
                except Exception as e:  # noqa: BLE001
                    logger.warning('qwen relay rebuild failed: %s', e)
                raise


_RELAY: Optional[QwenRelay] = None
_RELAY_LOCK = threading.Lock()


def get_relay() -> QwenRelay:
    """Process-wide relay singleton."""
    global _RELAY
    with _RELAY_LOCK:
        if _RELAY is None:
            _RELAY = QwenRelay()
        return _RELAY
