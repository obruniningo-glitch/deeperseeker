"""Provider registry — maps model aliases to adapters."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from v2.providers.base import ProviderAdapter, RenderedPrompt
from v2.providers.fake import FakeProvider
from v2.providers.deepseek import DeepSeekAdapter


@dataclass(frozen=True, slots=True)
class ModelConfig:
    alias: str
    provider: str
    model_name: str
    capabilities_override: dict | None = None


class ProviderRegistry:
    """Maps model aliases to provider adapters."""

    def __init__(self):
        self._adapters: dict[str, ProviderAdapter] = {}
        self._models: dict[str, ModelConfig] = {}
        self._default_provider = "fake"

        # Register built-in adapters
        self.register(FakeProvider())
        self.register(DeepSeekAdapter())

    def register(self, provider: ProviderAdapter) -> "ProviderRegistry":
        self._adapters[provider.name] = provider
        return self

    def get(self, name: str) -> ProviderAdapter:
        if name not in self._adapters:
            raise KeyError(f"Provider '{name}' not registered")
        return self._adapters[name]

    def register_model(
        self,
        alias: str,
        provider: str,
        model_name: str,
        capabilities_override: dict | None = None,
    ) -> "ProviderRegistry":
        self._models[alias] = ModelConfig(
            alias=alias,
            provider=provider,
            model_name=model_name,
            capabilities_override=capabilities_override,
        )
        return self

    def resolve(self, alias: str) -> tuple[ProviderAdapter, ModelConfig]:
        """Resolve an alias to (adapter, model_config)."""
        if alias not in self._models:
            # Fallback: assume it's a provider name
            return self.get(alias), ModelConfig(
                alias=alias, provider=alias, model_name=alias
            )
        cfg = self._models[alias]
        return self.get(cfg.provider), cfg

    @property
    def default_provider(self) -> str:
        return self._default_provider

    @default_provider.setter
    def default_provider(self, name: str) -> None:
        if name not in self._adapters:
            raise KeyError(f"Default provider '{name}' not registered")
        self._default_provider = name


_REGISTRY: ProviderRegistry | None = None


def get_registry() -> ProviderRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = ProviderRegistry()
        # Register built-in fake provider
        _REGISTRY.register(FakeProvider())
        _REGISTRY.register_model("fake", "fake", "fake")
        # Register DeepSeek model aliases
        _REGISTRY.register_model("instant", "deepseek", "deepseek-v4-flash")
        _REGISTRY.register_model("expert", "deepseek", "deepseek-v4-pro")
        _REGISTRY.register_model("vision", "deepseek", "deepseek-v4-pro")
        _REGISTRY.default_provider = "fake"
    return _REGISTRY


def configure_registry(registry: ProviderRegistry) -> None:
    """Replace the global registry (for tests)."""
    global _REGISTRY
    _REGISTRY = registry