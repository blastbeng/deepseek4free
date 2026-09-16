"""Model router: dynamic discovery, retries and cross-provider fallback chains.

The router owns every model id exposed on ``/v1/models``. Models are
DYNAMICALLY discovered from each provider's web session — no model lists are
hardcoded anywhere:

    deepseek  the web app has three fixed chat modes (plain/think/search); the
              exposed ids follow the operator's DSF_MODEL_* configuration
    gemini    discovered live from the gemini.google.com web app (batchexecute
              user-status RPC) using browser cookies
    chatgpt   discovered live from chatgpt.com/backend-api/models using the
              web session access token

``Router.refresh_models`` re-runs discovery (TTL-cached, thread-safe). A
provider whose discovery fails keeps its previously known routes, so a
transient outage never empties the registry. ``Router.stream`` transparently
retries rate limits / network failures with backoff (honoring ``Retry-After``)
and then falls back down the chain, so a single OpenAI request is always served
by the best available backend.

Configuration (env):
    DSF_MODEL_THINKER      exposed id of the DeepSeek thinking mode
                           (default deepseek-reasoner)
    DSF_MODEL_FAST         exposed id of the DeepSeek fast mode (default deepseek-chat)
    DSF_MODEL_SEARCH       exposed id of the DeepSeek search mode (default deepseek-search)
    DSF_MODELS_TTL         seconds between model re-discoveries (default 300)
    DSF_MAX_RETRIES        retries per provider before falling back (default 2)
    DSF_RETRY_BACKOFF      base backoff seconds, doubled each retry (default 2.0)
    DSF_FALLBACKS          JSON object {model_id: [fallback_id, ...]}
    DSF_DEFAULT_FALLBACKS  comma list applied to routes without explicit fallbacks

The synthetic ``auto`` model is the smart router: every request is classified
(coding / general / translation / summarize / vision / image generation) and
served by the best available provider, falling back through all other healthy
models on rate limits, auth failures (which also trigger an inline credential
renewal), outages or blocks. ``auto`` is also the default when a client omits
the model field.
"""

import importlib
import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, Generator, List, Optional

from .base import (
    Route,
    Provider,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderUnavailableError,
    provider_enabled,
)

# Provider classes are imported LAZILY via PROVIDER_MODULES below — a broken
# module then only disables its own provider instead of crashing startup.

logger = logging.getLogger('dsk.router')

# Registry master list: name -> (module, provider class). Both __init__ and
# the self-healing reload walk this single map so no provider can be wired
# in one place and forgotten in the other.
PROVIDER_MODULES = (
    ('deepseek', '.deepseek_provider', 'DeepSeekProvider'),
    ('gemini', '.gemini_provider', 'GeminiWebProvider'),
    ('chatgpt', '.chatgpt_provider', 'ChatGPTProvider'),
    ('claude', '.claude_provider', 'ClaudeWebProvider'),
    ('grok', '.grok_provider', 'GrokProvider'),
    ('mistral', '.mistral_provider', 'MistralProvider'),
    ('qwen', '.qwen_provider', 'QwenProvider'),
    ('kimi', '.kimi_provider', 'KimiProvider'),
    ('copilot', '.copilot_provider', 'CopilotProvider'),
    ('perplexity', '.perplexity_provider', 'PerplexityProvider'),
    ('glm', '.glm_provider', 'GlmProvider'),
)

OWNED_BY = {
    'deepseek': 'deepseek4free',
    'gemini': 'google',
    'chatgpt': 'openai',
    'claude': 'anthropic',
    'grok': 'xai',
    'mistral': 'mistral',
    'qwen': 'alibaba',
    'kimi': 'moonshot',
    'copilot': 'microsoft',
    'perplexity': 'perplexity',
    'glm': 'zai',
}

MAX_RETRIES = int(os.getenv('DSF_MAX_RETRIES', '2'))
RETRY_BACKOFF = float(os.getenv('DSF_RETRY_BACKOFF', '2.0'))
RETRY_CAP = 30.0
MODELS_TTL = float(os.getenv('DSF_MODELS_TTL', '300'))


def _csv_env(name: str, default: str) -> List[str]:
    raw = os.getenv(name, '') or default
    return [item.strip() for item in raw.split(',') if item.strip()]


def _parse_fallbacks() -> Dict[str, List[str]]:
    """Parse DSF_FALLBACKS JSON ({model_id: [fallback, ...]})."""
    raw = (os.getenv('DSF_FALLBACKS', '') or '').strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        logger.warning('DSF_FALLBACKS is not valid JSON, ignoring: %.100s', raw)
        return {}
    out: Dict[str, List[str]] = {}
    if isinstance(data, dict):
        for model_id, chain in data.items():
            if isinstance(chain, list):
                out[str(model_id)] = [str(f) for f in chain]
    return out


# ---------------------------------------------------------------------------
# 'auto' smart router: request classification + category preference tables.
# The tables list PREFERRED PROVIDERS per request category (never concrete
# model ids): whatever each web provider discovers dynamically is picked up
# automatically. At serve time the chain is filtered by live credential state
# (available()) and every other discovered model is appended as a safety net.
# ---------------------------------------------------------------------------
AUTO_MODEL_ID = 'auto'

AUTO_CATEGORIES: Dict[str, List[str]] = {
    'image_gen':   ['chatgpt', 'gemini', 'glm'],
    'vision':      ['chatgpt', 'gemini', 'glm'],
    'translation': ['gemini', 'chatgpt', 'deepseek', 'glm', 'qwen', 'mistral'],
    'summarize':   ['chatgpt', 'gemini', 'glm', 'qwen', 'mistral', 'deepseek'],
    'coding':      ['deepseek', 'glm', 'qwen', 'kimi', 'mistral', 'chatgpt'],
    'general':     ['chatgpt', 'gemini', 'glm', 'deepseek', 'qwen', 'mistral', 'kimi'],
}

_RE_CODE_FENCE = re.compile(
    r'```|\bdef\s+\w+\s*\(|\bclass\s+\w+\s*[(:]|\bfunction\s+\w+\s*\('
    r'|\bconsole\.log\s*\(|^\s*(?:import|from)\s+\w+', re.MULTILINE)
_RE_CODE_HINTS = re.compile(
    r'\b(?:python|javascript|typescript|golang|rust|sql|regex|json|yaml|html|css|'
    r'bug|debug|traceback|exception|compile|refactor|npm|pytest|docker|bash|'
    r'script|snippet|function|algorithm|program|code|api|endpoint|database|'
    r'java|c\+\+|c#|php|swift|kotlin)\b', re.IGNORECASE)
_RE_CODE_ASK = re.compile(
    r'\b(?:write|create|fix|debug|refactor|optimize|implement|generate|convert|'
    r'explain|review)\b[^.?!]{0,80}\b(?:function|script|class|code|program|'
    r'query|regex|component|endpoint|algorithm)\b', re.IGNORECASE)
_RE_EXPLAIN_LANG = re.compile(
    r'\b(?:explain|how)\b[^.?!]{0,60}\b(?:in|with|using)\s+'
    r'(?:python|javascript|typescript|java|golang|rust|c\+\+|php|sql|bash)\b',
    re.IGNORECASE)
_RE_TRANSLATE = re.compile(
    r'\btranslat(?:e|ion|ing)\b|tradu[cz]|\u00fcbersetz|\u7ffb\u8bd1',
    re.IGNORECASE)
_RE_SUMMARIZE = re.compile(
    r'\bsummari[sz]e\b|\bsummary\b|\btldr\b|\btl;dr\b|\bkey points\b'
    r'|\bin brief\b|\bcondense\b', re.IGNORECASE)


def classify_request(prompt: str, has_images: bool = False,
                     image_generation: bool = False) -> str:
    """Cheap heuristic classification of a request into an 'auto' category.

    Regex-only (no LLM roundtrip): image payloads win first, then
    translation/summarization phrasings, code fences/definitions and finally
    coding keywords (two or more). Everything else is general chat; a wrong
    guess is harmless because the fallback chain still serves the request.
    """
    if image_generation:
        return 'image_gen'
    if has_images:
        return 'vision'
    text = prompt or ''
    if _RE_TRANSLATE.search(text):
        return 'translation'
    if _RE_SUMMARIZE.search(text):
        return 'summarize'
    if (_RE_CODE_FENCE.search(text) or _RE_CODE_ASK.search(text)
            or _RE_EXPLAIN_LANG.search(text)
            or len(_RE_CODE_HINTS.findall(text)) >= 2):
        return 'coding'
    return 'general'


class Router:
    """Registry of providers and routes with retry/fallback orchestration.

    Routes are (re)built dynamically from each provider's ``list_models``;
    see ``refresh_models``.
    """

    def __init__(self) -> None:
        self.providers: Dict[str, Provider] = {
            name: getattr(importlib.import_module(module, __package__), cls)()
            for name, module, cls in PROVIDER_MODULES
            if provider_enabled(name)
        }
        self.routes: Dict[str, Route] = {}
        self._lock = threading.Lock()
        self._refreshed_at = 0.0
        # The DeepSeek modes are configuration-derived (no network involved),
        # so the registry is never empty, even before web discovery completes.
        try:
            if 'deepseek' in self.providers:
                self._apply_provider_models(
                    'deepseek', self.providers['deepseek'].list_models())
            self._apply_fallbacks()
        except Exception as e:  # pragma: no cover - defensive
            logger.warning('deepseek route bootstrap failed: %s', e)
        self._auto_route()

    # -------------------------------------------------------------- discovery
    def refresh_models(self, auth_key: Optional[str] = None,
                       force: bool = False) -> bool:
        """(Re-)discover models from every provider. Thread-safe, TTL-cached.

        Per-provider failures are tolerated: a provider that fails discovery
        keeps its previously known routes. Returns True when the registry
        changed.
        """
        with self._lock:
            if not force and time.time() - self._refreshed_at < MODELS_TTL:
                return False
            changed = False
            for name, provider in self.providers.items():
                try:
                    # Only DeepSeek can authenticate per-request (userToken as
                    # API key); the web providers use operator cookies.
                    models = provider.list_models(
                        auth_key if name == 'deepseek' else None)
                except ProviderAuthError as e:
                    logger.info('%s: no credentials for model discovery (%s)',
                                name, e)
                    continue
                except ProviderError as e:
                    logger.warning('%s model discovery failed: %s', name, e)
                    continue
                except Exception as e:  # never let discovery kill the registry
                    logger.warning('%s model discovery crashed: %s', name, e)
                    continue
                if self._apply_provider_models(name, models):
                    changed = True
            self._apply_fallbacks()
            self._refreshed_at = time.time()
            if changed:
                logger.info('model registry updated: %d models available',
                            len(self.routes))
            return changed

    def _apply_provider_models(self, name: str,
                               models: List[Dict[str, Any]]) -> bool:
        """Replace one provider's routes with its discovered models."""
        changed = False
        wanted = set()
        for entry in models:
            model_id = str(entry.get('id') or '').strip()
            if not model_id:
                continue
            wanted.add(model_id)
            route = Route(
                model_id=model_id,
                provider_name=name,
                upstream_model=str(entry.get('upstream_model') or model_id),
                thinking_enabled=bool(entry.get('thinking_enabled')),
                search_enabled=bool(entry.get('search_enabled')),
                vision=bool(entry.get('vision')),
                image_gen=bool(entry.get('image_gen')),
                context_length=int(entry.get('context_length') or 131072),
                max_output_tokens=int(entry.get('max_output_tokens') or 32768),
                extra=dict(entry.get('extra') or {}),
            )
            if self.routes.get(model_id) != route:
                changed = True
            self.routes[model_id] = route
        # Drop models of this provider that disappeared upstream.
        for model_id in [m for m, r in self.routes.items()
                         if r.provider_name == name and m not in wanted]:
            del self.routes[model_id]
            changed = True
        return changed

    def _apply_fallbacks(self) -> None:
        """Attach the configured fallback chains to every route.

        Fallback ids are kept raw: unknown targets are skipped at serve time,
        so chains may reference providers whose discovery has not completed
        yet.
        """
        explicit = _parse_fallbacks()
        default_chain = _csv_env('DSF_DEFAULT_FALLBACKS', '')
        for model_id, route in self.routes.items():
            if model_id == AUTO_MODEL_ID:
                continue  # dynamic chain, rebuilt per request
            chain = explicit.get(model_id) or default_chain
            route.fallbacks = []
            for fallback in chain:
                if fallback != model_id and fallback not in route.fallbacks:
                    route.fallbacks.append(fallback)

    # ----------------------------------------------------------- auto routing
    def _auto_route(self) -> Route:
        """Return (registering on first use) the synthetic 'auto' route."""
        route = self.routes.get(AUTO_MODEL_ID)
        if route is None:
            route = Route(
                model_id=AUTO_MODEL_ID,
                provider_name='router',
                upstream_model='auto',
                vision=True, image_gen=True,
            )
            self.routes[AUTO_MODEL_ID] = route
        return route

    def _auto_chain(self, category: str,
                    thinking_override: Optional[bool] = None,
                    search_override: Optional[bool] = None) -> List[str]:
        """Build the ordered model chain for an 'auto' request.

        Preferred targets for the classified category come first (a provider
        entry expands to every discovered model of that provider), then every
        other discovered model as a safety net. Providers without usable
        credentials (``available() == False``) are demoted out of the front;
        when nothing is healthy the full ordered list is kept — the stream
        loop still probes every target and falls back on auth/rate/offline
        errors. Vision/image-gen requests are restricted to capable targets;
        explicit thinking/search requests to routes supporting the mode.
        """
        ordered: List[str] = []

        def _expand(pref: str) -> None:
            for model_id, route in self.routes.items():
                if route.model_id == AUTO_MODEL_ID or model_id in ordered:
                    continue
                if (route.provider_name == pref
                        or route.model_id.startswith(pref + '-')):
                    ordered.append(model_id)

        for pref in AUTO_CATEGORIES.get(category, AUTO_CATEGORIES['general']):
            _expand(pref)
        for model_id in self.routes:
            if model_id != AUTO_MODEL_ID and model_id not in ordered:
                ordered.append(model_id)

        for flag, requested in (('thinking_enabled', thinking_override),
                                ('search_enabled', search_override)):
            if requested:
                kept = [mid for mid in ordered
                        if getattr(self.routes[mid], flag)]
                if kept:
                    ordered = kept

        def _healthy(model_id: str) -> bool:
            route = self.routes.get(model_id)
            provider = self.providers.get(route.provider_name) if route else None
            if provider is None:
                return False
            try:
                return bool(provider.available())
            except Exception:  # noqa: BLE001 — a broken probe means unproven
                return False

        flags = [(mid, _healthy(mid)) for mid in ordered]
        # Healthy targets first (category order preserved); credential-less
        # or offline providers stay at the very back so the stream loop
        # still probes them — credentials can appear at any moment (the
        # renewal bot runs continuously).
        chain = ([mid for mid, ok in flags if ok]
                 + [mid for mid, ok in flags if not ok])
        if category in ('vision', 'image_gen'):
            cap = 'image_gen' if category == 'image_gen' else 'vision'
            capable = [mid for mid in chain if getattr(self.routes[mid], cap)]
            if capable:
                chain = capable
        return [mid for mid in chain if mid != AUTO_MODEL_ID]

    def register(self, route: Route) -> None:
        """Add/replace a route (used by tests and custom setups)."""
        self.routes[route.model_id] = route

    def reload_providers(self) -> bool:
        """Re-import provider modules and rebuild every provider instance.

        Used by the self-healing layer after patching a provider module on
        disk (upstream web-app changed). Thread-safe: swaps the instances
        under the lock, then re-bootstraps DeepSeek's config-derived routes
        and forces a full re-discovery of the web providers.
        """
        with self._lock:
            new_providers = {}
            for name, module_name, class_name in PROVIDER_MODULES:
                module = importlib.import_module(module_name, __package__)
                importlib.reload(module)
                new_providers[name] = getattr(module, class_name)()
            self.providers = new_providers
            self._refreshed_at = 0.0
        changed = False
        try:
            changed = self._apply_provider_models(
                'deepseek', self.providers['deepseek'].list_models())
            self._apply_fallbacks()
        except Exception as e:  # pragma: no cover - defensive
            logger.warning('deepseek route bootstrap failed after reload: %s', e)
        threading.Thread(target=self.refresh_models, kwargs={'force': True},
                         name='model-re-discovery', daemon=True).start()
        return changed

    # ------------------------------------------------------------- inspection
    def resolve(self, model_id: str, auth_key: Optional[str] = None) -> Route:
        """Resolve a model id to its route.

        Unknown ids trigger a best-effort re-discovery (the upstream may have
        added models since the last refresh); ids that are still unknown fall
        back to the default fast route, so clients sending an arbitrary name
        still get served. An empty id or ``'auto'`` selects the smart router:
        its serving chain is built per request from live provider state.
        """
        model_id = (model_id or '').strip().lower()
        if not model_id or model_id == AUTO_MODEL_ID:
            return self._auto_route()
        route = self.routes.get(model_id)
        if route is not None:
            return route
        try:
            self.refresh_models(auth_key=auth_key)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning('re-discovery on unknown model failed: %s', e)
        route = self.routes.get(model_id)
        if route is not None:
            return route
        fast_id = os.getenv('DSF_MODEL_FAST', 'deepseek-chat').strip()
        return self.routes.get(fast_id) or next(iter(self.routes.values()))

    def available(self, route: Route, auth_key: Optional[str] = None) -> bool:
        provider = self.providers.get(route.provider_name)
        return bool(provider and provider.available(auth_key))

    def list_models(self) -> List[Dict[str, Any]]:
        """OpenAI-style /v1/models payload with agent-tooling metadata."""
        auto = self.routes.get(AUTO_MODEL_ID)
        entries: List[Dict[str, Any]] = []
        if auto is not None:
            # Listed first: the smart router handles every capability (it
            # re-routes to a capable model at serve time), so clients must
            # not pre-gate vision/image requests on its behalf.
            entries.append({
                'id': auto.model_id,
                'object': 'model',
                'created': 1700000000,
                'owned_by': 'deepseek4free',
                'context_length': 131072,
                'max_model_len': 131072,
                'max_completion_tokens': 32768,
                'max_tokens': 32768,
                'thinking_enabled': True,
                'search_enabled': True,
                'vision': True,
                'image_gen': True,
                'fallbacks': [],
            })
        entries.extend(
            {
                'id': r.model_id,
                'object': 'model',
                'created': 1700000000,
                'owned_by': OWNED_BY.get(r.provider_name,
                                         r.provider_name),
                'context_length': r.context_length,
                'max_model_len': r.context_length,
                'max_completion_tokens': r.max_output_tokens,
                'max_tokens': r.max_output_tokens,
                'thinking_enabled': r.thinking_enabled,
                'search_enabled': r.search_enabled,
                'vision': r.vision,
                'image_gen': r.image_gen,
                'fallbacks': list(r.fallbacks),
            }
            for r in self.routes.values() if r.model_id != AUTO_MODEL_ID
        )
        return entries

    # ---------------------------------------------------------------- serving
    def stream(self, route: Route, prompt: str, *, temperature: Optional[float] = None,
               max_tokens: Optional[int] = None, auth_key: Optional[str] = None,
               thinking_override: Optional[bool] = None,
               search_override: Optional[bool] = None,
               images: Optional[List[Dict[str, Any]]] = None,
               image_generation: bool = False,
               no_proxy: bool = False,
               ) -> Generator[Dict[str, Any], None, None]:
        """Yield unified chunks, retrying rate limits/network errors and
        falling back through the route's chain when a provider keeps failing.

        Fallbacks happen on request-level failures. Errors raised mid-stream
        (after content was already emitted) are surfaced as-is to avoid
        duplicating partial output.

        ``images``/``image_generation`` restrict the fallback chain to
        vision/image-gen capable targets and are forwarded to the provider.
        """
        thinking = route.thinking_enabled if thinking_override is None else thinking_override
        search = route.search_enabled if search_override is None else search_override

        if route.model_id == AUTO_MODEL_ID:
            # Smart router: classify the request and build the chain from
            # live provider state (preferred category models first, the rest
            # as safety net). The loop below still handles rate limits, auth
            # failures (with inline renewal), outages and blocks.
            category = classify_request(prompt, bool(images), image_generation)
            chain = self._auto_chain(category, thinking_override,
                                     search_override)
            if not chain:
                raise ProviderError(
                    'auto router found no available model — providers are '
                    'still discovering or credentials are being renewed')
            logger.info('auto router: category=%s chain=%s', category,
                        ' -> '.join(chain[:5]) + ('…' if len(chain) > 5 else ''))
        else:
            chain = [route.model_id] + [f for f in route.fallbacks
                                        if f != route.model_id]
        last_error: Optional[ProviderError] = None

        if images or image_generation:
            # Vision/image-gen requests may only be served by capable targets.
            capability = 'image_gen' if image_generation else 'vision'
            capable = [mid for mid in chain
                       if (target := self.routes.get(mid)) and getattr(target, capability)]
            if not capable:
                what = 'image generation' if image_generation else 'vision'
                raise ProviderError(
                    f'No model in the fallback chain of {route.model_id} '
                    f'supports {what} (vision/image-capable providers: '
                    f'chatgpt, gemini — they need credentials; the bot '
                    f'creates them automatically when possible)')
            chain = capable

        for position, model_id in enumerate(chain):
            target = self.routes.get(model_id)
            if target is None:
                continue
            provider = self.providers.get(target.provider_name)
            if provider is None:
                continue
            served_by = f'{target.provider_name}/{target.upstream_model}'
            attempt = 0
            while True:
                attempt += 1
                emitted = False
                try:
                    gen = provider.stream(
                        prompt, model=target.upstream_model,
                        thinking_enabled=thinking, search_enabled=search,
                        temperature=temperature, max_tokens=max_tokens,
                        images=images, image_generation=image_generation,
                        no_proxy=no_proxy,
                        auth_key=auth_key,
                    )
                    for chunk in gen:
                        emitted = True
                        if isinstance(chunk, dict):
                            chunk.setdefault('served_by', served_by)
                        yield chunk
                    return
                except ProviderRateLimitError as e:
                    last_error = e
                    if emitted:
                        raise  # mid-stream failure: fallback would duplicate output
                    if attempt <= MAX_RETRIES:
                        wait = min(e.retry_after if e.retry_after
                                   else RETRY_BACKOFF * (2 ** (attempt - 1)), RETRY_CAP)
                        logger.warning('%s rate limited (attempt %d/%d), retrying in %.1fs: %s',
                                       served_by, attempt, MAX_RETRIES + 1, wait, e)
                        time.sleep(wait)
                        continue
                    logger.warning('%s rate limited after %d attempts: %s',
                                   served_by, attempt - 1, e)
                    break
                except ProviderUnavailableError as e:
                    last_error = e
                    if emitted:
                        raise  # mid-stream failure: fallback would duplicate output
                    if attempt <= MAX_RETRIES:
                        wait = min(RETRY_BACKOFF * (2 ** (attempt - 1)), RETRY_CAP)
                        logger.warning('%s unavailable (attempt %d/%d), retrying in %.1fs: %s',
                                       served_by, attempt, MAX_RETRIES + 1, wait, e)
                        time.sleep(wait)
                        continue
                    logger.warning('%s unavailable after %d attempts: %s',
                                   served_by, attempt - 1, e)
                    break
                except ProviderAuthError as e:
                    # Credentials rejected/missing: retrying cannot help.
                    last_error = e
                    if emitted:
                        raise  # mid-stream failure: fallback would duplicate output
                    logger.warning('%s auth failed, skipping to fallback: %s', served_by, e)
                    # Request-path remediation: fire a background renewal
                    # ladder so the NEXT request can use fresh credentials.
                    try:
                        from dsk import refresher as _refresher
                        _refresher.renew_inline(target.provider_name, str(e)[:120])
                    except Exception:  # noqa: BLE001 — never break the request
                        pass
                    break
                except ProviderError as e:
                    last_error = e
                    if emitted:
                        raise  # mid-stream failure: fallback would duplicate output
                    logger.warning('%s failed, skipping to fallback: %s', served_by, e)
                    break
                except Exception as e:  # noqa: BLE001 — unclassified provider
                    # crash (upstream format change, provider bug): keep the
                    # request alive by falling through the chain. Mid-stream
                    # failures still re-raise: output was already emitted.
                    if emitted:
                        raise
                    last_error = ProviderError(f'{type(e).__name__}: {e}')
                    logger.warning('%s crashed, skipping to fallback: %s',
                                   served_by, e)
                    break

            if position < len(chain) - 1:
                logger.info('falling back: %s -> %s', route.model_id, chain[position + 1])

        if last_error is not None:
            raise last_error
        raise ProviderError(f'No provider available for model {route.model_id}')
