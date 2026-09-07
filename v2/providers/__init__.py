"""Provider package exports."""
from __future__ import annotations

from v2.providers.base import (
    ProviderAdapter,
    Capabilities,
    SessionRef,
    SendOptions,
    ProviderEvent,
    ProviderText,
    ProviderThinking,
    ProviderFinished,
    ProviderError,
    ProviderHealth,
    RenderedPrompt,
)
from v2.providers.fake import FakeProvider
from v2.providers.registry import ProviderRegistry, get_registry
from v2.providers.deepseek import DeepSeekAdapter

# Qwen adapter may not be installed; guard the import
try:
    from v2.providers.qwen.adapter import QwenAdapter  # type: ignore
except ImportError:  # pragma: no cover
    QwenAdapter = None  # type: ignore

__all__ = [
    "ProviderAdapter",
    "Capabilities",
    "SessionRef",
    "SendOptions",
    "ProviderEvent",
    "ProviderText",
    "ProviderThinking",
    "ProviderFinished",
    "ProviderError",
    "ProviderHealth",
    "RenderedPrompt",
    "FakeProvider",
    "DeepSeekAdapter",
    "QwenAdapter",
    "ProviderRegistry",
    "get_registry",
]