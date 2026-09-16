"""Microsoft Copilot provider via copilot.microsoft.com (reverse-engineered).

The free Copilot web app accepts anonymous identities (synthetic cookies) and
streams over a websocket — no official Azure OpenAI API, no paid keys.

How it works
------------
1. Anonymous identity: synthetic cookies (``__Host-copilot-anon``, MUID,
   ``_EDGE_S``…) persisted in ``copilot_cookies.json`` so the identity is
   stable across requests; ``COPILOT_COOKIES`` env JSON can override.
2. Conversation: POST ``/c/api/start`` returns ``currentConversationId``.
3. Stream: ``wss://copilot.microsoft.com/c/api/chat?api-version=2``; sending
   ``{"event": "send", ...}`` yields ``appendText`` / ``chainOfThought`` /
   ``imageGenerated`` / ``done`` events.

Tool calling is emulated by the shared TOOL_CALL protocol in the
OpenAI-compatible server.
"""

import json
import logging
import os
import random
import secrets
import string
import urllib.parse
from typing import Any, Dict, Generator, List, Optional

from .base import (
    Provider,
    ProviderAuthError,
    ProviderError,
    classify_http_error,
    http_post_raw,
)
from .jar import env_cookies, load_jar, save_jar

logger = logging.getLogger('dsk.providers.copilot')

COPILOT_BASE_URL = 'https://copilot.microsoft.com'
COPILOT_START_URL = f'{COPILOT_BASE_URL}/c/api/start'
COPILOT_WS_URL = 'wss://copilot.microsoft.com/c/api/chat?api-version=2'

COPILOT_CONTEXT_LENGTH = int(os.getenv('DSF_COPILOT_CONTEXT_LENGTH', '32000'))
COPILOT_MAX_OUTPUT = int(os.getenv('DSF_COPILOT_MAX_OUTPUT', '4096'))

_USER_AGENT = (
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
    '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0'
)

# Mode picker (mapped from exposed ids, like the web UI toggle).
COPILOT_MODELS: List[Dict[str, Any]] = [
    {'id': 'copilot-chat', 'mode': 'chat', 'thinking': False},
    {'id': 'copilot-think', 'mode': 'Think', 'thinking': True},
    {'id': 'copilot-smart', 'mode': 'Smart', 'thinking': True},
]


def _random_hex(length: int) -> str:
    return ''.join(random.choices('0123456789abcdef', k=length))


def _random_base64(length: int) -> str:
    alphabet = string.ascii_letters + string.digits + '/='
    return ''.join(random.choices(alphabet, k=length))


def _harvest_browser_cookies() -> Dict[str, str]:
    """Mint a real anonymous identity with headless Chromium (DrissionPage).

    Synthetic cookies are rejected by ``/c/api/start`` (401); only cookies a
    real browser mints are accepted. Runs once per missing identity, guarded
    by ``DSF_COPILOT_HARVEST`` (default on). Never raises.
    """
    if os.getenv('DSF_COPILOT_HARVEST', 'true').strip().lower() in \
            ('0', 'false', 'no', 'off'):
        return {}
    try:
        import time as _time
        from DrissionPage import ChromiumOptions, ChromiumPage
        try:
            from pyvirtualdisplay import Display
        except ImportError:
            Display = None
        import logging as _logging

        _log = _logging.getLogger('dsk.providers.copilot')
        _log.info('copilot: harvesting anonymous identity with headless '
                  'chromium')
        display = None
        if Display is not None and os.getenv('DISPLAY') is None:
            display = Display(visible=False, size=(1366, 900))
            display.start()
        page = None
        try:
            opts = ChromiumOptions()
            opts.set_browser_path('/usr/bin/chromium')
            opts.set_argument('--no-sandbox')
            opts.set_argument('--disable-gpu')
            opts.set_argument('--disable-blink-features=AutomationControlled')
            opts.set_argument('--window-size=1366,900')
            page = ChromiumPage(addr_or_opts=opts)
            page.get(f'{COPILOT_BASE_URL}/', retry=2, interval=3, timeout=90)
            deadline = _time.time() + 30
            anon = ''
            while _time.time() < deadline and not anon:
                for c in page.cookies(all_domains=True):
                    if c.get('name') == '__Host-copilot-anon' and c.get('value'):
                        anon = c['value']
                        break
                if not anon:
                    _time.sleep(1.5)
            out: Dict[str, str] = {}
            domains = ('.copilot.microsoft.com', '.microsoft.com',
                       '.bing.com', '.live.com')
            for c in page.cookies(all_domains=True):
                name, value = c.get('name'), c.get('value')
                dom = (c.get('domain') or '').lower()
                if name and value and any(dom.endswith(d) or d in dom
                                          for d in domains):
                    out[name] = value
            if anon:
                out['__Host-copilot-anon'] = anon
            if out.get('MUID') or anon:
                save_jar('copilot', out)
                _log.info('copilot: harvested %d browser cookies '
                          '(anon token present: %s)', len(out), bool(anon))
                return out
            _log.warning('copilot: browser harvest produced no anon token')
            return {}
        except Exception as e:  # pragma: no cover - browser flakiness
            _log.warning('copilot: browser harvest failed: %s', e)
            return {}
        finally:
            if page is not None:
                try:
                    page.quit()
                except Exception:  # pragma: no cover
                    pass
            if display is not None:
                try:
                    display.stop()
                except Exception:  # pragma: no cover
                    pass
    except Exception as e:  # pragma: no cover - import failures
        logger.warning('copilot: harvest unavailable: %s', e)
        return {}


def _rotate_identity() -> None:
    """Discard the stale anonymous identity and mint a fresh browser one."""
    try:
        save_jar('copilot', {})
    except Exception:  # noqa: BLE001 — rotation is best-effort
        pass
    if not _harvest_browser_cookies():
        raise ProviderAuthError(
            'copilot identity rotation failed — browser harvest produced '
            'no cookies')


def _anon_cookies() -> Dict[str, str]:
    """Stable identity: env JSON > jar > real browser harvest > synthetic."""
    cookies = env_cookies('COPILOT')
    if cookies:
        return cookies
    jar = load_jar('copilot')
    if jar.get('MUID') and jar.get('__Host-copilot-anon'):
        return jar
    harvested = _harvest_browser_cookies()
    if harvested.get('__Host-copilot-anon'):
        return harvested
    if jar.get('MUID'):
        return jar
    cookies = {
        '_C_ETH': '1',
        '_C_Auth': '',
        'MUID': _random_hex(32),
        'MUIDB': _random_hex(32),
        '_EDGE_S': f'F=1&SID={_random_hex(32)}',
        '_EDGE_V': '1',
        'ak_bmsc': (f'{_random_hex(32)}~{"0" * 48}~'
                    f'{urllib.parse.quote(_random_base64(300))}'),
        '__Host-copilot-anon': secrets.token_hex(16),
    }
    save_jar('copilot', cookies)
    return cookies


def _cookie_header(cookies: Dict[str, str]) -> str:
    return '; '.join(f'{k}={v}' for k, v in cookies.items() if v)


def _egress_url(no_proxy: bool) -> Optional[str]:
    """Current pool egress URL for Copilot WS traffic (None = direct)."""
    if no_proxy:
        return None
    try:
        from .. import proxies as _proxies
        return _proxies.get_proxy('copilot', direct_ok=True)
    except Exception:  # noqa: BLE001 — direct remains the fallback
        return None


def _cool_egress(proxy_url: Optional[str]) -> None:
    """Cooldown a WS egress that failed to connect or was edge-blocked."""
    if not proxy_url:
        return
    try:
        from .. import proxies as _proxies
        _proxies.mark_failure(proxy_url)
    except Exception:  # noqa: BLE001 — rotation is best-effort
        pass


class CopilotProvider(Provider):
    name = 'copilot'

    # ---------------------------------------------------------------- provider
    def available(self, auth_key: Optional[str] = None) -> bool:
        return True  # anonymous identities are generated on demand

    def list_models(self, auth_key: Optional[str] = None) -> List[Dict[str, Any]]:
        """Web-app mode picker (no public discovery endpoint)."""
        return [{
            'id': entry['id'],
            'upstream_model': entry['mode'],
            'thinking_enabled': entry['thinking'],
            'search_enabled': True,   # Copilot is web-grounded by default
            'vision': False,
            'image_gen': True,        # DALL-E-backed image generation
            'context_length': COPILOT_CONTEXT_LENGTH,
            'max_output_tokens': COPILOT_MAX_OUTPUT,
            'extra': {'mode': entry['mode']},
        } for entry in COPILOT_MODELS]

    def stream(self, prompt: str, *, model: str, thinking_enabled: bool = False,
               search_enabled: bool = False, temperature: Optional[float] = None,
               max_tokens: Optional[int] = None,
               images: Optional[List[Dict[str, Any]]] = None,
               image_generation: bool = False,
               no_proxy: bool = False,
               auth_key: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
        if images:
            raise ProviderError('copilot image input is not supported yet')
        mode = 'chat'
        for entry in COPILOT_MODELS:
            if entry['id'] == model:
                mode = entry['mode']
                break
        conversation_id: Optional[str] = None
        # A rejected anonymous identity triggers ONE automatic rotation:
        # discard the stale jar, mint a fresh browser identity, retry.
        last_auth: Optional[ProviderAuthError] = None
        for attempt in range(2):
            try:
                conversation_id = self._start_conversation(no_proxy=no_proxy)
                return self._ws_stream(conversation_id, prompt, mode,
                                       image_generation, no_proxy=no_proxy)
            except ProviderAuthError as e:
                last_auth = e
                if attempt or os.getenv('COPILOT_COOKIES'):
                    raise
                logger.warning('copilot: identity rejected (%s) — rotating '
                               'anonymous identity', e)
                _rotate_identity()
        raise last_auth or ProviderError('copilot stream failed')

    def _headers(self) -> Dict[str, str]:
        return {
            'User-Agent': _USER_AGENT,
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Cookie': _cookie_header(_anon_cookies()),
            'Origin': COPILOT_BASE_URL,
            'Referer': f'{COPILOT_BASE_URL}/',
        }

    def _start_conversation(self, no_proxy: bool = False) -> str:
        body = json.dumps({
            'timeZone': 'America/Los_Angeles',
            'startNewConversation': True,
            'teenSupportEnabled': True,
            'correctPersonalizationSetting': True,
            'performUserMerge': True,
            'deferredDataUseCapable': True,
        }).encode()
        response = http_post_raw(COPILOT_START_URL, body,
                                 headers=self._headers(), no_proxy=no_proxy)
        if response.status_code == 401:
            raise ProviderAuthError('copilot anonymous identity rejected — '
                                    'rotate copilot_cookies.json')
        if response.status_code != 200:
            try:
                error_text = response.text or ''
            except Exception:  # pragma: no cover
                error_text = f'HTTP {response.status_code}'
            raise classify_http_error(response.status_code, error_text,
                                      response.headers)
        try:
            conversation = response.json().get('currentConversationId')
        except ValueError as e:
            raise ProviderError(f'copilot /c/api/start non-JSON: {e}') from e
        if not conversation:
            raise ProviderError('copilot /c/api/start returned no conversation')
        return str(conversation)

    def _connect_ws(self, proxy_url: Optional[str]):
        """Open the Copilot chat websocket over `proxy_url` (None = direct)."""
        import websocket  # websocket-client
        kwargs: Dict[str, Any] = {}
        if proxy_url:
            parsed = urllib.parse.urlparse(proxy_url)
            scheme = parsed.scheme or 'http'
            kwargs = {'http_proxy_host': parsed.hostname,
                      'http_proxy_port': parsed.port}
            if parsed.username:
                kwargs['http_proxy_auth'] = (parsed.username,
                                             parsed.password or '')
            kwargs['proxy_type'] = (scheme if scheme.startswith('socks')
                                    else 'http')
        return websocket.create_connection(
            COPILOT_WS_URL,
            header=[
                f'Cookie: {_cookie_header(_anon_cookies())}',
                f'User-Agent: {_USER_AGENT}',
                f'Origin: {COPILOT_BASE_URL}',
                'Referer: https://copilot.microsoft.com/',
                'Accept-Language: en-US,en;q=0.9',
                'Sec-CH-UA: "Chromium";v="120", "Not_A Brand";v="24", '
                '"Microsoft Edge";v="120"',
                'Sec-CH-UA-Mobile: ?0',
                'Sec-CH-UA-Platform: "Windows"',
                'Sec-Fetch-Dest: websocket',
                'Sec-Fetch-Mode: websocket',
                'Sec-Fetch-Site: same-origin',
            ],
            enable_multithread=True,
            timeout=120,
            suppress_origin=True,
            **kwargs,
        )

    def _ws_stream(self, conversation_id: str, prompt: str, mode: str,
                   image_generation: bool,
                   no_proxy: bool = False) -> Generator[Dict[str, Any], None, None]:
        try:
            import websocket  # websocket-client
        except ImportError as e:  # pragma: no cover
            raise ProviderError('websocket-client is required for the '
                                'copilot provider (pip install '
                                'websocket-client)') from e

        # Egress ladder for the WS handshake: requested egress, then a FRESH
        # pool draw (a dead or edge-blocked exit is cooled down first), then
        # direct. Copilot's edge geo-blocks host/datacenter IPs (WS 460), so
        # a working proxy exit is often required.
        ws = None
        last_auth: Optional[Exception] = None
        last_connect: Optional[Exception] = None
        first = _egress_url(no_proxy)
        ladder: List[Optional[str]] = []
        if first:
            ladder.append(first)
            ladder.append(_egress_url(no_proxy))  # fresh draw after a miss
        ladder.append(None)  # direct last
        seen_eg: set = set()
        for proxy_url in ladder:
            if proxy_url and proxy_url in seen_eg:
                continue
            if proxy_url:
                seen_eg.add(proxy_url)
            try:
                ws = self._connect_ws(proxy_url)
                break
            except websocket.WebSocketBadStatusException as e:
                status = getattr(e, 'status_code', 0) or 0
                if status in (401, 403, 440, 460):
                    last_auth = e
                    if proxy_url:
                        _cool_egress(proxy_url)
                    continue
                raise ProviderError(f'copilot WS handshake failed: {e}') from e
            except (OSError, websocket.WebSocketException) as e:
                last_connect = e
                if proxy_url:
                    _cool_egress(proxy_url)
                continue
        if ws is None:
            if last_auth is not None and last_connect is None:
                raise ProviderAuthError(
                    'copilot anonymous chat refused by the edge — anonymous '
                    'Copilot is geo-blocked in some regions (notably the '
                    'EU). Provide cookies from a non-EU browser session via '
                    'COPILOT_COOKIES env JSON or copilot_cookies.json'
                ) from last_auth
            raise ProviderError(
                'copilot WS connection failed: '
                f'{last_connect or last_auth or "all egresses failed"}')
        try:
            ws.send(json.dumps({
                'event': 'send',
                'conversationId': conversation_id,
                'content': [{'type': 'text', 'text': prompt}],
                'mode': mode,
                'participantId': _anon_cookies().get('__Host-copilot-anon', ''),
            }))
            prev_reasoning = ''
            while True:
                try:
                    raw = ws.recv()
                except (websocket.WebSocketTimeoutException, ConnectionError):
                    break
                if not raw:
                    break
                if isinstance(raw, bytes):
                    raw = raw.decode('utf-8', errors='replace')
                try:
                    event = json.loads(raw)
                except ValueError:
                    continue
                etype = event.get('event')
                if etype == 'appendText':
                    text = event.get('text') or ''
                    if text:
                        yield {'content': text, 'type': 'text',
                               'finish_reason': None}
                elif etype == 'replaceText':
                    text = event.get('text') or ''
                    if text.startswith(prev_reasoning):
                        text = text[len(prev_reasoning):]
                    if text:
                        prev_reasoning += text
                        yield {'content': text, 'type': 'text',
                               'finish_reason': None}
                elif etype == 'chainOfThought':
                    thought = event.get('text') or event.get('content') or ''
                    if isinstance(thought, list):
                        thought = ''.join(str(t) for t in thought)
                    if thought:
                        yield {'content': str(thought), 'type': 'thinking',
                               'finish_reason': None}
                elif etype == 'imageGenerated':
                    url = (event.get('url') or event.get('imageUrl') or '')
                    if url:
                        yield {'content': f'![image]({url})', 'type': 'image',
                               'url': url, 'finish_reason': None}
                elif etype == 'error':
                    message = (event.get('message') or event.get('error')
                               or 'unknown copilot error')
                    raise ProviderError(f'copilot stream error: {message}')
                elif etype == 'done':
                    break
                # generatingImage / citation / titleUpdate / suggestedFollowups
                # / received / startMessage / partCompleted / connected: ignore
        finally:
            try:
                ws.close()
            except Exception:  # pragma: no cover - defensive
                pass
        yield {'content': '', 'type': 'text', 'finish_reason': 'stop'}
