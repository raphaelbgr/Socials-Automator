"""Provider configuration loading and validation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# Load .env file
load_dotenv()


class ProviderSettings(BaseModel):
    """Global provider settings."""

    timeout_seconds: int = 60
    max_retries: int = 3
    retry_delay_seconds: int = 2
    fallback_on_error: bool = True


class TextProviderConfig(BaseModel):
    """Configuration for a text provider."""

    priority: int
    enabled: bool = True
    litellm_model: str
    base_url: str | None = None
    base_url_env: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    timeout: int = 60
    models: dict[str, str] | None = None

    def get_api_key(self) -> str | None:
        """Get API key from config or environment."""
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.getenv(self.api_key_env)
        return None

    def get_base_url(self) -> str | None:
        """Get base URL from config or environment."""
        if self.base_url:
            return self.base_url
        if self.base_url_env:
            return os.getenv(self.base_url_env)
        return None


class ImageProviderConfig(BaseModel):
    """Configuration for an image provider."""

    priority: int
    enabled: bool = True
    type: str  # fal, replicate, openai, comfyui
    model: str
    api_key_env: str | None = None
    base_url_env: str | None = None
    timeout: int = 90
    cost_per_image: float = 0.0
    settings: dict[str, Any] = Field(default_factory=dict)

    def get_api_key(self) -> str | None:
        """Get API key from environment."""
        if self.api_key_env:
            return os.getenv(self.api_key_env)
        return None

    def get_base_url(self) -> str | None:
        """Get base URL from environment."""
        if self.base_url_env:
            return os.getenv(self.base_url_env)
        return None


class TaskOverride(BaseModel):
    """Task-specific provider override."""

    text_provider: str | None = None
    text_model: str | None = None
    image_provider: str | None = None
    temperature: float | None = None


class LLMFallbackConfig(BaseModel):
    """LLM fallback and retry configuration."""

    local_max_retries: int = 10
    external_max_retries: int = 5
    local_providers: list[str] = Field(default_factory=lambda: ["lmstudio", "ollama"])
    fallback_priority: list[str] = Field(default_factory=lambda: ["zai", "gemini", "groq", "openai"])


class ProviderConfig(BaseModel):
    """Full provider configuration."""

    provider_settings: ProviderSettings = Field(default_factory=ProviderSettings)
    llm_fallback: LLMFallbackConfig = Field(default_factory=LLMFallbackConfig)
    text_providers: dict[str, TextProviderConfig] = Field(default_factory=dict)
    image_providers: dict[str, ImageProviderConfig] = Field(default_factory=dict)
    text_priority_chain: list[str] = Field(default_factory=list)
    image_priority_chain: list[str] = Field(default_factory=list)
    task_overrides: dict[str, TaskOverride] = Field(default_factory=dict)

    def get_enabled_text_providers(self) -> list[tuple[str, TextProviderConfig]]:
        """Get enabled text providers sorted by priority."""
        enabled = [
            (name, config)
            for name, config in self.text_providers.items()
            if config.enabled
        ]
        return sorted(enabled, key=lambda x: x[1].priority)

    def get_enabled_image_providers(self) -> list[tuple[str, ImageProviderConfig]]:
        """Get enabled image providers sorted by priority."""
        enabled = [
            (name, config)
            for name, config in self.image_providers.items()
            if config.enabled
        ]
        return sorted(enabled, key=lambda x: x[1].priority)

    def get_text_provider_for_task(self, task: str) -> tuple[str, TextProviderConfig] | None:
        """Get the preferred text provider for a specific task."""
        override = self.task_overrides.get(task)
        if override and override.text_provider:
            provider_name = override.text_provider
            if provider_name in self.text_providers:
                return (provider_name, self.text_providers[provider_name])

        # Fall back to priority chain
        for name, config in self.get_enabled_text_providers():
            return (name, config)
        return None

    def get_image_provider_for_task(self, task: str) -> tuple[str, ImageProviderConfig] | None:
        """Get the preferred image provider for a specific task."""
        override = self.task_overrides.get(task)
        if override and override.image_provider:
            provider_name = override.image_provider
            if provider_name in self.image_providers:
                return (provider_name, self.image_providers[provider_name])

        # Fall back to priority chain
        for name, config in self.get_enabled_image_providers():
            return (name, config)
        return None


def load_provider_config(config_path: Path | None = None) -> ProviderConfig:
    """Load provider configuration from YAML file."""
    if config_path is None:
        # Default to config/providers.yaml relative to project root
        config_path = Path(__file__).parent.parent.parent.parent / "config" / "providers.yaml"

    if not config_path.exists():
        # Return default config if file doesn't exist
        return ProviderConfig()

    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return ProviderConfig(**data)
