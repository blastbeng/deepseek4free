from curl_cffi import requests
from typing import Optional, Dict, Any, Generator, Literal
import json
from .pow import DeepSeekPOW
try:  # optional outbound proxy rotation (dsk/proxies.py)
    from . import proxies as _proxies
except ImportError:  # pragma: no cover - standalone use
    import proxies as _proxies
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path
import subprocess
import time
import os

ThinkingMode = Literal['detailed', 'simple', 'disabled']
SearchMode = Literal['enabled', 'disabled']

class DeepSeekError(Exception):
    """Base exception for all DeepSeek API errors"""
    pass

class AuthenticationError(DeepSeekError):
    """Raised when authentication fails"""
    pass

class RateLimitError(DeepSeekError):
    """Raised when API rate limit is exceeded"""
    pass

class NetworkError(DeepSeekError):
    """Raised when network communication fails"""
    pass

class CloudflareError(DeepSeekError):
    """Raised when Cloudflare blocks the request"""
    pass

class APIError(DeepSeekError):
    """Raised when API returns an error response"""
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code

class DeepSeekAPI:
    BASE_URL = "https://chat.deepseek.com/api/v0"

    def __init__(self, auth_token: str):
        if not auth_token or not isinstance(auth_token, str):
            # Fall back to environment variable (used by the OpenAI server & Docker)
            auth_token = os.getenv('DEEPSEEK_AUTH_TOKEN', '').strip()
            if not auth_token:
                raise AuthenticationError("Invalid auth token provided")

        try:
            try:
                curl_cffi_version = importlib_metadata.version('curl_cffi')
            except importlib_metadata.PackageNotFoundError:
                curl_cffi_version = importlib_metadata.version('curl-cffi')
            if curl_cffi_version != '0.8.1b9':
                print("\033[93mWarning: DeepSeek API requires curl-cffi version 0.8.1b9", file=sys.stderr)
                print("Please install the correct version using: pip install curl-cffi==0.8.1b9\033[0m", file=sys.stderr)
        except Exception:
            print("\033[93mWarning: curl-cffi not found. Please install version 0.8.1b9:", file=sys.stderr)
            print("pip install curl-cffi==0.8.1b9\033[0m", file=sys.stderr)

        self.auth_token = auth_token
        self.pow_solver = DeepSeekPOW()
        # Track the last JSON-Patch path so events that omit "p" can be
        # attributed to the right field (DeepSeek stream format)
        self._last_patch_path = ''

        # Load cookies from JSON file (override location with COOKIES_DIR,
        # e.g. a mounted Docker volume at /data)
        cookies_dir = os.getenv('COOKIES_DIR')
        if cookies_dir and Path(cookies_dir).is_dir():
            cookies_path = Path(cookies_dir) / 'cookies.json'
        else:
            cookies_path = Path(__file__).parent / 'cookies.json'
        try:
            with open(cookies_path, 'r') as f:
                cookie_data = json.load(f)
                self.cookies = cookie_data.get('cookies', {})
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"\033[93mWarning: Could not load cookies from {cookies_path}: {e}\033[0m", file=sys.stderr)
            self.cookies = {}

    def _get_headers(self, pow_response: Optional[str] = None) -> Dict[str, str]:
        headers = {
            'accept': '*/*',
            'accept-language': 'en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3',
            'authorization': f'Bearer {self.auth_token}',
            'content-type': 'application/json',
            'origin': 'https://chat.deepseek.com',
            'referer': 'https://chat.deepseek.com/',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36',
            'x-app-version': '20241129.1',
            'x-client-locale': 'en_US',
            'x-client-platform': 'web',
            'x-client-version': '1.0.0-always',
        }

        if pow_response:
            headers['x-ds-pow-response'] = pow_response

        return headers

    def _refresh_cookies(self) -> None:
        """Run the cookie refresh script and reload cookies"""
        try:
            # Get path to bypass.py
            script_path = Path(__file__).parent / 'bypass.py'

            # Run the script
            subprocess.run([sys.executable, script_path], check=True)

            # Wait briefly for cookies file to be written
            time.sleep(2)

            # Reload cookies
            cookies_dir = os.getenv('COOKIES_DIR')
            if cookies_dir and Path(cookies_dir).is_dir():
                cookies_path = Path(cookies_dir) / 'cookies.json'
            else:
                cookies_path = Path(__file__).parent / 'cookies.json'
            with open(cookies_path, 'r') as f:
                cookie_data = json.load(f)
                self.cookies = cookie_data.get('cookies', {})

        except Exception as e:
            print(f"\033[93mWarning: Failed to refresh cookies: {e}\033[0m", file=sys.stderr)

    def _make_request(self, method: str, endpoint: str, json_data: Dict[str, Any], pow_required: bool = False, no_proxy: bool = False) -> Any:
        url = f"{self.BASE_URL}{endpoint}"

        retry_count = 0
        max_retries = 2

        while retry_count < max_retries:
            try:
                headers = self._get_headers()
                if pow_required:
                    challenge = self._get_pow_challenge()
                    pow_response = self.pow_solver.solve_challenge(challenge)
                    headers = self._get_headers(pow_response)

                proxy_kw = _proxies.proxies_kwargs(url=url, no_proxy=no_proxy)
                response = requests.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=json_data,
                    cookies=self.cookies,
                    impersonate='chrome120',
                    timeout=None,
                    **proxy_kw
                )

                # Check if we hit Cloudflare protection
                if "<!DOCTYPE html>" in response.text and "Just a moment" in response.text:
                    print("\033[93mWarning: Cloudflare protection detected. Bypassing...\033[0m", file=sys.stderr)
                    if retry_count < max_retries - 1:
                        self._refresh_cookies()  # Refresh cookies
                        retry_count += 1
                        continue

                # Handle other response codes
                if response.status_code == 401:
                    raise AuthenticationError("Invalid or expired authentication token")
                elif response.status_code == 429:
                    raise RateLimitError("API rate limit exceeded")
                elif response.status_code >= 500:
                    raise APIError(f"Server error occurred: {response.text}", response.status_code)
                elif response.status_code != 200:
                    raise APIError(f"API request failed: {response.text}", response.status_code)

                try:
                    payload = response.json()
                except ValueError:
                    raise APIError("Empty or non-JSON response from server", response.status_code)

                # DeepSeek signals business errors inside a 200 body:
                # {"code": <int != 0>, "msg": "...", "data": null}
                # Surface the real upstream message as a typed error instead
                # of letting callers mask it as a response-format problem.
                if isinstance(payload, dict):
                    biz_code = payload.get('code')
                    biz_msg = payload.get('msg') or payload.get('message')
                    if isinstance(biz_code, int) and biz_code != 0:
                        detail = f"{biz_msg or 'unknown upstream error'} (code {biz_code})"
                        if biz_code in (40001, 40003, 40100) or (
                                biz_msg and 'authorization' in str(biz_msg).lower()):
                            raise AuthenticationError(f"DeepSeek upstream: {detail}")
                        raise APIError(f"DeepSeek upstream: {detail}", biz_code)
                return payload

            except requests.exceptions.RequestException as e:
                _proxies.mark_failure(proxy_kw.get('proxies', {}).get('https'))
                raise NetworkError(f"Network error occurred: {str(e)}")
            except json.JSONDecodeError:
                raise APIError("Invalid JSON response from server")

        raise APIError("Failed to bypass Cloudflare protection after multiple attempts")

    def _get_pow_challenge(self, no_proxy: bool = False) -> Dict[str, Any]:
        try:
            response = self._make_request(
                'POST',
                '/chat/create_pow_challenge',
                {'target_path': '/api/v0/chat/completion'},
                no_proxy=no_proxy,
            )
            return response['data']['biz_data']['challenge']
        except (KeyError, TypeError):
            raise APIError("Invalid challenge response format from server")

    def create_chat_session(self, no_proxy: bool = False) -> str:
        """Creates a new chat session and returns the session ID"""
        try:
            response = self._make_request(
                'POST',
                '/chat_session/create',
                {'character_id': None},
                no_proxy=no_proxy,
            )
            return response['data']['biz_data']['id']
        except (KeyError, TypeError):
            raise APIError("Invalid session creation response format from server")

    def chat_completion(self,
                    chat_session_id: str,
                    prompt: str,
                    parent_message_id: Optional[str] = None,
                    thinking_enabled: bool = True,
                    search_enabled: bool = False,
                    no_proxy: bool = False) -> Generator[Dict[str, Any], None, None]:
        """
        Send a message and get streaming response

        Args:
            chat_session_id (str): The ID of the chat session
            prompt (str): The message to send
            parent_message_id (Optional[str]): ID of the parent message for threading
            thinking_enabled (bool): Whether to show the thinking process
            search_enabled (bool): Whether to enable web search for up-to-date information

        Returns:
            Generator[Dict[str, Any], None, None]: Yields message chunks with content and type

        Raises:
            AuthenticationError: If the authentication token is invalid
            RateLimitError: If the API rate limit is exceeded
            NetworkError: If a network error occurs
            APIError: If any other API error occurs
        """
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string")
        if not chat_session_id or not isinstance(chat_session_id, str):
            raise ValueError("Chat session ID must be a non-empty string")

        json_data = {
            'chat_session_id': chat_session_id,
            'parent_message_id': parent_message_id,
            'prompt': prompt,
            'ref_file_ids': [],
            'thinking_enabled': thinking_enabled,
            'search_enabled': search_enabled,
        }

        try:
            headers = self._get_headers(
                pow_response=self.pow_solver.solve_challenge(
                    self._get_pow_challenge(no_proxy=no_proxy)
                )
            )

            proxy_kw = _proxies.proxies_kwargs(
                url=f"{self.BASE_URL}/chat/completion", no_proxy=no_proxy)
            response = requests.post(
                f"{self.BASE_URL}/chat/completion",
                headers=headers,
                json=json_data,
                cookies=self.cookies,  # Add cookies
                impersonate='chrome120',
                stream=True,
                timeout=None,
                **proxy_kw
            )

            if response.status_code != 200:
                error_text = next(response.iter_lines(), b'').decode('utf-8', 'ignore')
                if response.status_code == 401:
                    raise AuthenticationError("Invalid or expired authentication token")
                elif response.status_code == 429:
                    raise RateLimitError("API rate limit exceeded")
                else:
                    raise APIError(f"API request failed: {error_text}", response.status_code)

            for chunk in response.iter_lines():
                try:
                    parsed = self._parse_chunk(chunk)
                    if parsed:
                        yield parsed
                        if parsed.get('finish_reason') == 'stop':
                            break
                except APIError:
                    raise
                except Exception as e:
                    raise APIError(f"Error parsing response chunk: {str(e)}")

        except requests.exceptions.RequestException as e:
            raise NetworkError(f"Network error occurred during streaming: {str(e)}")

    def _parse_chunk(self, chunk: bytes) -> Optional[Dict[str, Any]]:
        """Parse a SSE chunk from the API response.

        Supports the current DeepSeek stream format (JSON-Patch style events
        like {"p": "response/content", "o": "APPEND", "v": "..."}) as well as
        the legacy OpenAI-style {"choices": [{"delta": ...}]} format.
        """
        if not chunk:
            return None

        try:
            if chunk.startswith(b'data: '):
                data = json.loads(chunk[6:])

                if isinstance(data, dict) and 'v' in data:
                    return self._parse_patch_chunk(data)

                if 'choices' in data and data['choices']:
                    choice = data['choices'][0]
                    if 'delta' in choice:
                        delta = choice['delta']

                        return {
                            'content': delta.get('content', ''),
                            'type': delta.get('type', ''),
                            'finish_reason': choice.get('finish_reason')
                        }
        except json.JSONDecodeError:
            raise APIError("Invalid JSON in response chunk")
        except APIError:
            raise
        except Exception as e:
            raise APIError(f"Error parsing chunk: {str(e)}")

        return None

    def _parse_patch_chunk(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse a JSON-Patch style stream event.

        Observed events:
          {"v": {...response object...}}                       -> initial message state
          {"p": "response/content", "o": "APPEND", "v": "..."} -> content delta
          {"v": "..."} (right after a content event)           -> implicit content delta
          {"p": "response/status", "v": "FINISHED"}            -> completion
          {"p": "response/thinking_content", ...}              -> thinking delta
        """
        path = data.get('p', '')
        value = data.get('v')

        # Initial message snapshot: nothing to stream
        if isinstance(value, dict):
            return None

        # Completion signal
        if 'status' in path and value == 'FINISHED':
            return {'content': '', 'type': 'text', 'finish_reason': 'stop'}

        # Content delta: explicit path or implicit continuation of the last one
        if path.endswith('/content') or ('/content' in self._last_patch_path and not path):
            self._last_patch_path = path or self._last_patch_path
            if isinstance(value, str) and value:
                return {'content': value, 'type': 'text', 'finish_reason': None}
            return None

        # Thinking content delta
        if 'thinking_content' in path or ('thinking_content' in self._last_patch_path and not path):
            self._last_patch_path = path or self._last_patch_path
            if isinstance(value, str) and value:
                return {'content': value, 'type': 'thinking', 'finish_reason': None}
            return None

        # Any other path (token usage, tips, search results...) is ignored,
        # but remember it in case the next event omits the path
        self._last_patch_path = path
        return None
