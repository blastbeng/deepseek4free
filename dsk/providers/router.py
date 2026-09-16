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
"""

import importlib
import json
import logging
import os
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
            chain = explicit.get(model_id) or default_chain
            route.fallbacks = []
            for fallback in chain:
                if fallback != model_id and fallback not in route.fallbacks:
                    route.fallbacks.append(fallback)

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
        still get served.
        """
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
        return [
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
            for r in self.routes.values()
        ]

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

        chain = [route.model_id] + [f for f in route.fallbacks if f != route.model_id]
        last_error: Optional[ProviderError] = None

        if images or image_generation:
            # Vision/image-gen requests may only be served by capable targets.
            capability = 'image_gen' if image_generation else 'vision'
            capable = [mid for mid in chain
                       if (target := self.routes.get(mid)) and getattr(target, capability)]
            if not capable:
                what = 'image generation' if image_generation else 'vision'
                raise ProviderError(
                    f'No model in the fallback chain of {route.model_id} supports {what}')
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
                    break
                except ProviderError as e:
                    last_error = e
                    if emitted:
                        raise  # mid-stream failure: fallback would duplicate output
                    logger.warning('%s failed, skipping to fallback: %s', served_by, e)
                    break

            if position < len(chain) - 1:
                logger.info('falling back: %s -> %s', route.model_id, chain[position + 1])

        if last_error is not None:
            raise last_error
        raise ProviderError(f'No provider available for model {route.model_id}')
