"""Perplexity provider via www.perplexity.ai (reverse-engineered).

Talks to the web app's own SSE answer endpoint with browser cookies — no
official Sonar API subscription, no paid keys.

How it works
------------
1. Credentials: browser cookies of a (free) perplexity.ai session
   (bot-managed ``perplexity_cookies.json`` or ``PERPLEXITY_COOKIES`` env
   JSON). Anonymous requests also work but are heavily rate limited.
2. Stream: POST ``/rest/sse/perplexity_ask`` with the web app's full
   ``params`` payload; the SSE body is a sequence of frames whose payload
   lives in a ``blocks`` array (legacy frames carried the fields at the
   top level — both shapes are parsed):
   - ``intended_usage == 'ask_text'`` → ``markdown_block.chunks`` (cumulative)
     and ``markdown_block.answer`` (authoritative full text)
   - ``diff_block.patches`` with ``/goals`` paths → reasoning text
   - ``diff_block.field == 'markdown_block'`` → answer deltas
3. Anonymous sessions that hit the fraud wall receive an
   ``upsell_information`` block (``fraud_authwall_upsell`` / ``LOGIN``);
   that raises :class:`ProviderAuthError` so callers can rotate identity.

Model ids are the upstream ``model_preference`` values the web app offers
(turbo/pplx_pro/pplx_reasoning plus partner models like gpt5, claude45sonnet,
o3 — routed through Perplexity's own search stack).
"""

import logging
import os
import uuid
from typing import Any, Dict, Generator, List, Optional

from .base import (
    Provider,
    ProviderAuthError,
    ProviderError,
    classify_http_error,
    http_get,
    http_post_stream,
    parse_sse_data,
)
from .jar import env_cookies, load_jar

logger = logging.getLogger('dsk.providers.perplexity')

PERPLEXITY_BASE_URL = 'https://www.perplexity.ai'
PERPLEXITY_ASK_URL = f'{PERPLEXITY_BASE_URL}/rest/sse/perplexity_ask'
PERPLEXITY_AUTH_URL = f'{PERPLEXITY_BASE_URL}/api/auth/session'

PERPLEXITY_CONTEXT_LENGTH = int(os.getenv('DSF_PERPLEXITY_CONTEXT_LENGTH',
                                          '128000'))
PERPLEXITY_MAX_OUTPUT = int(os.getenv('DSF_PERPLEXITY_MAX_OUTPUT', '4096'))

_USER_AGENT = (
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) '
    'Chrome/120.0.0.0 Safari/537.36'
)

# upstream model_preference values (exposed as <provider>-<name> routes)
PERPLEXITY_MODELS: List[Dict[str, Any]] = [
    {'id': 'perplexity-turbo', 'upstream': 'turbo', 'thinking': False},
    {'id': 'perplexity-pro', 'upstream': 'pplx_pro', 'thinking': False},
    {'id': 'perplexity-reasoning', 'upstream': 'pplx_reasoning',
     'thinking': True},
    {'id': 'perplexity-gpt5', 'upstream': 'gpt5', 'thinking': False},
    {'id': 'perplexity-claude-4.5-sonnet', 'upstream': 'claude45sonnet',
     'thinking': False},
    {'id': 'perplexity-o3', 'upstream': 'o3', 'thinking': False},
]

_SUPPORTED_BLOCKS = [
    'answer_modes', 'media_items', 'knowledge_cards', 'inline_entity_cards',
    'place_widgets', 'finance_widgets', 'prediction_market_widgets',
    'sports_widgets', 'flight_status_widgets', 'news_widgets',
    'shopping_widgets', 'jobs_widgets', 'search_result_widgets',
    'inline_images', 'inline_assets', 'placeholder_cards', 'diff_blocks',
    'inline_knowledge_cards', 'entity_group_v2', 'refinement_filters',
    'canvas_mode', 'maps_preview', 'answer_tabs',
    'price_comparison_widgets', 'preserve_latex', 'generic_onboarding_widgets',
    'in_context_suggestions', 'inline_claims',
]


def _cookies() -> Dict[str, str]:
    jar = load_jar('perplexity') or env_cookies('PERPLEXITY')
    return jar


def _cookie_header() -> str:
    return '; '.join(f'{k}={v}' for k, v in _cookies().items() if v)


def _headers(accept: str = 'application/json') -> Dict[str, str]:
    return {
        'Cookie': _cookie_header(),
        'User-Agent': _USER_AGENT,
        'Content-Type': 'application/json',
        'Accept': accept,
        'Origin': PERPLEXITY_BASE_URL,
        'Referer': f'{PERPLEXITY_BASE_URL}/',
        'x-perplexity-request-reason': 'perplexity-query-state-provider',
        'x-request-id': str(uuid.uuid4()),
    }


def _user_id(no_proxy: bool = False) -> str:
    """Best-effort account id from the auth session (empty when anonymous)."""
    try:
        response = http_get(PERPLEXITY_AUTH_URL, headers=_headers(),
                            no_proxy=no_proxy)
        if response.status_code == 200:
            user = (response.json() or {}).get('user') or {}
            return str(user.get('id') or '')
    except Exception:  # noqa: BLE001 - anonymous mode has no session
        pass
    return ''


class PerplexityProvider(Provider):
    name = 'perplexity'

    # ---------------------------------------------------------------- provider
    def available(self, auth_key: Optional[str] = None) -> bool:
        return True  # anonymous mode works (with lower quotas)

    def list_models(self, auth_key: Optional[str] = None) -> List[Dict[str, Any]]:
        """Upstream model_preference values (validated live via ask).

        ``pplx_pro`` / partner models require a free signed-in session;
        anonymous visitors effectively get ``turbo``.
        """
        return [{
            'id': entry['id'],
            'upstream_model': entry['upstream'],
            'thinking_enabled': entry['thinking'],
            'search_enabled': True,   # every answer is web-grounded
            'vision': False,
            'image_gen': False,
            'context_length': PERPLEXITY_CONTEXT_LENGTH,
            'max_output_tokens': PERPLEXITY_MAX_OUTPUT,
            'extra': {'model_preference': entry['upstream']},
        } for entry in PERPLEXITY_MODELS]

    def stream(self, prompt: str, *, model: str, thinking_enabled: bool = False,
               search_enabled: bool = False, temperature: Optional[float] = None,
               max_tokens: Optional[int] = None,
               images: Optional[List[Dict[str, Any]]] = None,
               image_generation: bool = False,
               no_proxy: bool = False,
               auth_key: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
        if images:
            raise ProviderError('perplexity attachments are not supported yet')
        preference = model
        for entry in PERPLEXITY_MODELS:
            if entry['id'] == model:
                preference = entry['upstream']
                break
        frontend_uuid = str(uuid.uuid4())
        payload: Dict[str, Any] = {
            'query_str': prompt,
            'params': {
                'attachments': [],
                'language': 'en-US',
                'timezone': 'America/Los_Angeles',
                'search_focus': 'internet',
                'sources': ['web'],
                'search_recency_filter': None,
                'frontend_uuid': frontend_uuid,
                'mode': 'copilot',
                'model_preference': preference,
                'is_related_query': False,
                'is_sponsored': False,
                'frontend_context_uuid': str(uuid.uuid4()),
                'prompt_source': 'user',
                'query_source': 'home',
                'is_incognito': False,
                'time_from_first_type': 18361,
                'local_search_enabled': False,
                'use_schematized_api': True,
                'send_back_text_in_streaming_api': False,
                'supported_block_use_cases': _SUPPORTED_BLOCKS,
                'client_coordinates': None,
                'mentions': [],
                'dsl_query': prompt,
                'skip_search_enabled': False,
                'is_nav_suggestions_disabled': False,
                'source': 'default',
                'always_search_override': False,
                'override_no_search': False,
                'should_ask_for_mcp_tool_confirmation': True,
                'browser_agent_allow_once_from_toggle': False,
                'force_enable_browser_agent': False,
                'supported_features': ['browser_agent_permission_banner_v1.1'],
                'version': '2.18',
            },
        }
        response = http_post_stream(PERPLEXITY_ASK_URL,
                                    headers=_headers('text/event-stream'),
                                    json_body=payload, no_proxy=no_proxy)
        if response.status_code != 200:
            try:
                error_text = response.text or ''
            except Exception:  # pragma: no cover
                error_text = f'HTTP {response.status_code}'
            raise classify_http_error(response.status_code, error_text,
                                      response.headers)
        return self._iter_chunks(response)

    def _iter_chunks(self, response) -> Generator[Dict[str, Any], None, None]:
        full_response = ''
        full_reasoning = ''
        for line in response.iter_lines():
            data = parse_sse_data(line)
            if not data:
                continue
            if data.get('error'):
                raise ProviderError(f"perplexity stream error: {data['error']}")
            # Modern frames nest the payload in a ``blocks`` array; legacy
            # frames carried the block fields at the top level. Normalize
            # both into a list of block events.
            blocks = data.get('blocks')
            if isinstance(blocks, list) and blocks:
                events = [b for b in blocks if isinstance(b, dict)]
            else:
                events = [data]
            upsell = data.get('upsell_information') or {}
            if not upsell:
                for b in events:
                    upsell = b.get('upsell_information') or {}
                    if upsell:
                        break
            if (upsell.get('name') == 'fraud_authwall_upsell'
                    or upsell.get('upsell_type') == 'LOGIN'):
                raise ProviderAuthError(
                    'perplexity anonymous session is auth-walled '
                    '(fraud_authwall_upsell) — provide a signed-in session '
                    'via PERPLEXITY_COOKIES or perplexity_cookies.json')
            for block in events:
                if block.get('intended_usage') == 'ask_text':
                    chunk = ''.join((block.get('markdown_block') or {})
                                    .get('chunks') or [])
                    answer = (block.get('markdown_block') or {}).get('answer')
                    if answer and isinstance(answer, str):
                        # ``answer`` is authoritative and cumulative
                        if answer.startswith(full_response):
                            chunk = answer[len(full_response):]
                        else:
                            chunk = answer
                    if (chunk
                            and (not full_response
                                 or not full_response.endswith(chunk))
                            and not full_reasoning.startswith(chunk)
                            and (not full_response
                                 or not chunk.startswith(full_response))):
                        full_response += chunk
                        yield {'content': chunk, 'type': 'text',
                               'finish_reason': None}
                diff = block.get('diff_block') or {}
                for patch in diff.get('patches') or []:
                    if patch.get('path') == '/progress':
                        continue
                    value = patch.get('value', '')
                    if isinstance(value, dict) and 'chunks' in value:
                        value = ''.join(value.get('chunks') or [])
                    if str(patch.get('path', '')).startswith('/goals'):
                        if isinstance(value, str) and value:
                            if value.startswith(full_reasoning):
                                value = value[len(full_reasoning):]
                            if value:
                                full_reasoning += value
                                yield {'content': value, 'type': 'thinking',
                                       'finish_reason': None}
                        continue
                    if diff.get('field') != 'markdown_block':
                        continue
                    value = value.get('answer', '') if isinstance(value, dict) \
                        else value
                    if value and isinstance(value, str):
                        if full_response and full_response.startswith(value):
                            continue
                        piece = value
                        if piece.startswith(full_response):
                            piece = piece[len(full_response):]
                        full_response += piece
                        yield {'content': piece, 'type': 'text',
                               'finish_reason': None}
        yield {'content': '', 'type': 'text', 'finish_reason': 'stop'}
