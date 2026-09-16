"""LLM CALL -> llmtrim -> proxy rotator -> LLM response.

Always-on inbound payload trimmer: before a chat request reaches the
router/proxy layer, the message list is measured against the resolved
route's context budget. Stale history is dropped oldest-first (system
messages and the latest turns are always kept) and single oversized
messages get middle-out truncation. Trimming the payload BEFORE the proxy
rotator keeps upstream tokens (and therefore time-to-first-byte and
per-IP rate-limit pressure) minimal on every egress route.

The stage is ALWAYS working by default; ``DSF_LLMTRIM=false`` disables it.
"""

import logging
import os
from typing import Any, Dict, List, Tuple

logger = logging.getLogger('dsk.llmtrim')

CHARS_PER_TOKEN = 4.0
_SAFETY_TOKENS = 1024
_MIN_BUDGET_CHARS = 4000.0
_MARKER = '\n\u2026[llmtrim removed {n} chars]\u2026\n'


def enabled() -> bool:
    return os.getenv('DSF_LLMTRIM', 'true').strip().lower() \
        not in ('0', 'false', 'no', 'off')


def _role(msg: Any) -> str:
    return str(getattr(msg, 'role', None)
               or (msg.get('role') if isinstance(msg, dict) else '') or '')


def _content(msg: Any) -> Any:
    return getattr(msg, 'content', None) if not isinstance(msg, dict) \
        else msg.get('content')


def _text_of(content: Any) -> str:
    """Flat text of a message content (string or OpenAI parts list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                if p.get('type') == 'text':
                    parts.append(str(p.get('text', '')))
                elif p.get('type') == 'image_url':
                    parts.append('[image]')
        return '\n'.join(parts)
    return ''


def _image_weight(msg: Any) -> int:
    """Bytes carried by inline image data-URIs (base64 blobs are huge but
    render as a 7-char '[image]' placeholder — without this the trimmer
    underweights vision conversations by megabytes)."""
    content = _content(msg)
    if not isinstance(content, list):
        return 0
    n = 0
    for p in content:
        if isinstance(p, dict) and p.get('type') == 'image_url':
            url = p.get('image_url')
            url = url.get('url') if isinstance(url, dict) else url
            if isinstance(url, str) and url.startswith('data:'):
                n += len(url)
    return n


def _chars(msg: Any) -> int:
    """Message weight in characters (content + rendered tool calls + image data)."""
    n = len(_text_of(_content(msg))) + _image_weight(msg)
    calls = getattr(msg, 'tool_calls', None) if not isinstance(msg, dict) \
        else msg.get('tool_calls')
    if calls:
        try:
            n += len(str(calls))
        except Exception:  # noqa: BLE001 — weight is advisory
            pass
    return n


def _copy_with_content(msg: Any, new_content: Any) -> Any:
    if isinstance(msg, dict):
        out = dict(msg)
        out['content'] = new_content
        return out
    for attr in ('model_copy', 'copy'):
        fn = getattr(msg, attr, None)
        if callable(fn):
            try:
                return fn(update={'content': new_content})
            except TypeError:
                try:
                    return fn(content=new_content)
                except TypeError:
                    pass
            except Exception:  # noqa: BLE001 — fall through to mutation
                pass
    try:
        msg.content = new_content
    except Exception:  # noqa: BLE001 — read-only object: leave untouched
        return msg
    return msg


def _truncate_text(text: str, keep_head: int, keep_tail: int) -> str:
    removed = max(0, len(text) - keep_head - keep_tail)
    if removed <= 0:
        return text
    return (text[:keep_head] + _MARKER.format(n=removed) + text[-keep_tail:]
            if keep_tail else text[:keep_head] + _MARKER.format(n=removed))


def _truncate_msg(msg: Any, keep_chars: int) -> Any:
    """Middle-out truncate one message down to ~keep_chars characters."""
    content = _content(msg)
    if isinstance(content, str):
        return _copy_with_content(msg, _truncate_text(content, int(keep_chars * 0.8),
                                                      int(keep_chars * 0.1)))
    if isinstance(content, list) and content:
        # truncate the largest text part in place (images are untouched)
        texts = [(i, p) for i, p in enumerate(content)
                 if isinstance(p, dict) and p.get('type') == 'text']
        if not texts:
            return msg
        i, part = max(texts, key=lambda t: len(str(t[1].get('text', ''))))
        new_parts = list(content)
        new_parts[i] = dict(part)
        new_parts[i]['text'] = _truncate_text(str(part.get('text', '')),
                                              int(keep_chars * 0.8),
                                              int(keep_chars * 0.1))
        return _copy_with_content(msg, new_parts)
    return msg


def trim_messages(messages: List[Any], context_tokens: float,
                  max_output_tokens: float = 0
                  ) -> Tuple[List[Any], Dict[str, Any]]:
    """Trim an OpenAI-style message list to the route's context budget.

    Returns (messages, stats). stats['trimmed'] is True when anything was
    dropped or truncated, so the server can log the llmtrim stage.
    """
    stats: Dict[str, Any] = {'trimmed': False, 'dropped': 0, 'truncated': 0,
                             'in_chars': 0, 'out_chars': 0}
    if not enabled() or not messages:
        stats['skipped'] = True
        return messages, stats

    budget = max(_MIN_BUDGET_CHARS,
                 (float(context_tokens or 32768)
                  - float(max_output_tokens or 0) - _SAFETY_TOKENS)
                 * CHARS_PER_TOKEN)

    sizes = [_chars(m) for m in messages]
    total = sum(sizes)
    stats['in_chars'] = total
    if total <= budget:
        stats['out_chars'] = total
        return messages, stats

    # 1) drop OLDEST droppable groups (system + the final message are kept).
    #    An assistant message with tool_calls and its immediately-following
    #    tool results form an ATOMIC group: dropping only part leaves orphan
    #    tool results (or calls without results) that upstreams reject.
    last = len(messages) - 1

    def _has_calls(m: Any) -> bool:
        calls = getattr(m, 'tool_calls', None) if not isinstance(m, dict) \
            else m.get('tool_calls')
        return bool(calls)

    groups: List[Tuple[int, int]] = []  # inclusive (start, end) indexes
    gi = 0
    while gi <= last:
        gj = gi
        if _role(messages[gi]) == 'assistant' and _has_calls(messages[gi]):
            while gj + 1 <= last and _role(messages[gj + 1]) == 'tool':
                gj += 1
        groups.append((gi, gj))
        gi = gj + 1

    drop: set = set()
    cur = total
    for gs, ge in groups:  # oldest first
        if cur <= budget:
            break
        if ge >= last:  # group contains the protected final message
            continue
        if any(_role(messages[k]) == 'system' for k in range(gs, ge + 1)):
            continue
        for k in range(gs, ge + 1):
            drop.add(k)
        cur -= sum(sizes[gs:ge + 1])
    kept = [(i, m) for i, m in enumerate(messages) if i not in drop]
    stats['dropped'] = len(drop)

    # 2) still over budget: middle-out truncate, biggest/newest first
    if cur > budget:
        over = cur - budget
        order = sorted(range(len(kept)),
                       key=lambda k: -sizes[kept[k][0]])
        for k in order:
            if over <= 0:
                break
            i, m = kept[k]
            c = sizes[i]
            if c < 400:            # small turns are not worth mangling
                continue
            keep_chars = max(200, c - over)
            new_msg = _truncate_msg(m, keep_chars)
            new_c = _chars(new_msg)
            kept[k] = (i, new_msg)
            # credit only what was ACTUALLY removed — tool_calls weight is
            # part of c but never truncatable, so c - keep_chars can
            # overstate progress on call-heavy, content-light messages.
            over -= max(0, c - new_c)
            stats['truncated'] += 1
            cur = sum(_chars(m2) for _, m2 in kept)

    out = [m for _, m in kept]
    stats['out_chars'] = sum(_chars(m) for m in out)
    stats['trimmed'] = bool(drop) or stats['truncated'] > 0
    stats['budget_chars'] = int(budget)
    if stats['trimmed']:
        logger.info('[llmtrim] %d -> %d chars (dropped %d messages, '
                    'truncated %d, budget %d)', stats['in_chars'],
                    stats['out_chars'], stats['dropped'],
                    stats['truncated'], int(budget))
    return out, stats
