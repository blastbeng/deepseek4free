"""Multi-provider support: DeepSeek (reverse API), Gemini (AI Studio), ChatGPT (web).

Every provider exposes the same streaming contract: a generator of chunk dicts
    {'content': str, 'type': 'text' | 'thinking', 'finish_reason': None | 'stop'}
which mirrors the DeepSeek stream format, so the OpenAI-compatible server can
consume all providers through a single code path (streaming bridge + tool-call
emulation).
"""
