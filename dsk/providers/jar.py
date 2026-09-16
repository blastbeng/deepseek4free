"""Shared bot-managed cookie-jar helpers for the web providers.

Every reverse-engineered provider stores its operator credentials in a small
JSON file (``<name>_cookies.json``) under ``COOKIES_DIR`` (or the project root
when unset). ``dsk.refresher`` keeps those files fresh; the providers only
read/write them through the helpers below.
"""

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict

_LOCKS: Dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _jar_lock(name: str) -> threading.Lock:
    """One lock per jar name: the refresher daemon and request threads
    (e.g. copilot identity rotation) save concurrently — without it the
    load→merge→write sequence loses updates."""
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(name, threading.Lock())


def jar_path(name: str) -> Path:
    cookies_dir = os.getenv('COOKIES_DIR')
    if cookies_dir and Path(cookies_dir).is_dir():
        return Path(cookies_dir) / f'{name}_cookies.json'
    return Path(__file__).resolve().parent.parent / f'{name}_cookies.json'


def _normalize(data: Any) -> Dict[str, str]:
    """Accept both {name: value} dicts and [{name, value}, ...] lists."""
    if isinstance(data, list):
        data = {e.get('name'): e.get('value') for e in data
                if isinstance(e, dict) and e.get('name')}
    if not isinstance(data, dict):
        return {}
    if isinstance(data.get('cookies'), dict):
        data = data['cookies']  # deepseek bypass format
    return {str(k): str(v) for k, v in data.items()
            if k and v is not None and str(v).strip()}


def load_jar(name: str) -> Dict[str, str]:
    path = jar_path(name)
    try:
        if path.is_file():
            return _normalize(json.loads(path.read_text(encoding='utf-8')))
    except (OSError, ValueError):
        pass
    return {}


def env_cookies(name: str) -> Dict[str, str]:
    """``<NAME>_COOKIES`` env var (JSON dict or list) as a fallback source."""
    raw = (os.getenv(f'{name.upper()}_COOKIES', '') or '').strip()
    if not raw:
        return {}
    try:
        return _normalize(json.loads(raw))
    except ValueError:
        return {}


def save_jar(name: str, updates: Dict[str, Any]) -> None:
    """Merge updates into the jar file (atomic replace, best-effort)."""
    with _jar_lock(name):
        jar = load_jar(name)
        jar.update({str(k): str(v) for k, v in updates.items() if v})
        path = jar_path(name)
        try:
            tmp = path.with_suffix('.tmp')
            tmp.write_text(json.dumps(jar, indent=2), encoding='utf-8')
            tmp.replace(path)
        except OSError:
            pass
