"""DeepSeek provider package exports."""
from __future__ import annotations

from v2.providers.deepseek.adapter import DeepSeekAdapter
from v2.providers.deepseek.wire import (
    send_message,
    create_new_chat,
    upload_file,
    get_file_content,
)
from v2.providers.deepseek.pow import (
    solve_create_pow,
    find_pow_answer,
    create_challenge_pow,
    get_cookies,
)
from v2.providers.deepseek.cookies import get_cookies as get_deepseek_cookies, _regenerate_cookies

__all__ = [
    "DeepSeekAdapter",
    "send_message",
    "create_new_chat",
    "upload_file",
    "get_file_content",
    "solve_create_pow",
    "find_pow_answer",
    "create_challenge_pow",
    "get_cookies",
    "get_deepseek_cookies",
    "_regenerate_cookies",
]