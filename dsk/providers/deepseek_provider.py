"""DeepSeek provider adapter.

Wraps the existing reverse-engineered DeepSeekAPI client so it can participate
in the shared router (retries, fallback chains) alongside Gemini and ChatGPT.
"""

import os
import threading
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from ..api import (
    DeepSeekAPI,
    AuthenticationError,
    RateLimitError,
    NetworkError,
    APIError,
)
from .base import (
    Provider,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderUnavailableError,
)


def _bot_token_file() -> Path:
    """Bot-managed token written by dsk.refresher (login/signup renewal)."""
    base = os.getenv('COOKIES_DIR') or os.getenv('DSF_SELFHEAL_DIR')
    directory = Path(base) if base else Path(__file__).resolve().parent.parent
    return directory / 'deepseek_token'


def _resolve_token(provided_key: Optional[str] = None) -> Optional[str]:
    """DeepSeek auth token resolution order:
    1. bot-managed token file (<COOKIES_DIR>/deepseek_token, written by the
       refresher bot — wins over a stale env value so renewals take effect)
    2. DEEPSEEK_AUTH_TOKEN env var
    3. key sent by the client (Authorization Bearer)
    """
    try:
        bot_token = _bot_token_file().read_text(encoding='utf-8').strip()
        if bot_token:
            return bot_token
    except OSError:
        pass
    env_token = os.getenv('DEEPSEEK_AUTH_TOKEN', '').strip()
    if env_token:
        return env_token
    if provided_key and provided_key.strip():
        return provided_key.strip()
    return None


def _cookies_file() -> Path:
    cookies_dir = os.getenv('COOKIES_DIR')
    if cookies_dir and Path(cookies_dir).is_dir():
        return Path(cookies_dir) / 'cookies.json'
    return Path(__file__).resolve().parent.parent / 'cookies.json'


class DeepSeekProvider(Provider):
    name = 'deepseek'

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._api: Optional[DeepSeekAPI] = None
        self._api_token: Optional[str] = None

    def available(self, auth_key: Optional[str] = None) -> bool:
        if _resolve_token(auth_key):
            return True
        try:
            return _cookies_file().is_file()
        except OSError:
            return False

    def _get_api(self, auth_key: Optional[str] = None) -> DeepSeekAPI:
        token = _resolve_token(auth_key)
        if not token:
            raise ProviderAuthError(
                'No DeepSeek auth token. Set DEEPSEEK_AUTH_TOKEN, provide a '
                'cookies.json, or send your userToken as the API key.'
            )
        with self._lock:
            if self._api is None or token != self._api_token:
                self._api = DeepSeekAPI(token)
                self._api_token = token
            return self._api

    def stream(self, prompt: str, *, model: str, thinking_enabled: bool = False,
               search_enabled: bool = False, temperature: Optional[float] = None,
               max_tokens: Optional[int] = None,
               auth_key: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
        # Temperature/max_tokens are not supported by the DeepSeek web API.
        try:
            api = self._get_api(auth_key)
            session_id = api.create_chat_session()
            return api.chat_completion(
                session_id, prompt,
                thinking_enabled=thinking_enabled,
                search_enabled=search_enabled,
            )
        except AuthenticationError as e:
            raise ProviderAuthError(str(e))
        except RateLimitError as e:
            raise ProviderRateLimitError(str(e))
        except NetworkError as e:
            raise ProviderUnavailableError(str(e))
        except (APIError, ValueError) as e:
            raise ProviderError(str(e))

    def list_models(self, auth_key: Optional[str] = None) -> List[Dict[str, Any]]:
        """Expose the DeepSeek web app's chat modes as models.

        The chat.deepseek.com web app has exactly three fixed modes (plain
        chat, thinking, search) — there is no model-list endpoint to scrape,
        so "dynamic" here means the exposed ids follow the operator's
        DSF_MODEL_* configuration instead of being frozen in code. The web
        providers (Gemini, ChatGPT) discover their models live instead.
        """
        thinking_id = os.getenv('DSF_MODEL_THINKER', 'deepseek-reasoner').strip()
        fast_id = os.getenv('DSF_MODEL_FAST', 'deepseek-chat').strip()
        search_id = os.getenv('DSF_MODEL_SEARCH', 'deepseek-search').strip()
        context_length = int(os.getenv('DSF_CONTEXT_LENGTH', '131072'))
        max_thinking = int(os.getenv('DSF_MAX_OUTPUT_THINKING', '65536'))
        max_output = int(os.getenv('DSF_MAX_OUTPUT', '32768'))

        models: List[Dict[str, Any]] = []
        seen = set()
        for model_id in (fast_id, thinking_id, search_id):
            if not model_id or model_id in seen:
                continue
            seen.add(model_id)
            models.append({
                'id': model_id,
                'upstream_model': model_id,
                'thinking_enabled': model_id == thinking_id,
                'search_enabled': model_id == search_id,
                'context_length': context_length,
                'max_output_tokens': (max_thinking if model_id == thinking_id
                                      else max_output),
                'extra': {},
            })
        return models
