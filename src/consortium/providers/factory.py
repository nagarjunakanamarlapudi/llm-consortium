"""Provider factory: creates LLMProvider instances from model configuration."""

from __future__ import annotations

import structlog

from consortium.config.models import ModelConfig
from consortium.providers.base import LLMProvider

logger = structlog.get_logger()

# Lazy imports to avoid requiring all SDKs to be installed
_PROVIDER_REGISTRY: dict[str, type[LLMProvider]] = {}


def _ensure_registry() -> None:
    """Populate the registry on first use (lazy to avoid import errors)."""
    if _PROVIDER_REGISTRY:
        return

    try:
        from consortium.providers.anthropic import AnthropicProvider

        _PROVIDER_REGISTRY["anthropic"] = AnthropicProvider
    except ImportError:
        logger.debug("anthropic_sdk_not_available")

    try:
        from consortium.providers.openai import OpenAIProvider

        _PROVIDER_REGISTRY["openai"] = OpenAIProvider
    except ImportError:
        logger.debug("openai_sdk_not_available")

    try:
        from consortium.providers.google import GoogleProvider

        _PROVIDER_REGISTRY["google"] = GoogleProvider
        _PROVIDER_REGISTRY["google_vertex"] = GoogleProvider
    except ImportError:
        logger.debug("google_sdk_not_available")

    try:
        from consortium.providers.ollama import OllamaProvider

        _PROVIDER_REGISTRY["ollama"] = OllamaProvider
    except ImportError:
        logger.debug("ollama_provider_not_available")

    try:
        from consortium.providers.vertex_openai import VertexOpenAIProvider

        _PROVIDER_REGISTRY["google_vertex_openai"] = VertexOpenAIProvider
    except ImportError:
        logger.debug("vertex_openai_provider_not_available")


def create_provider(
    model_config: ModelConfig,
    *,
    api_key: str | None = None,
) -> LLMProvider:
    """Create an LLMProvider instance from a ModelConfig.

    Args:
        model_config: The model configuration specifying provider, model, etc.
        api_key: Optional API key override. If not provided, the provider
            will use the appropriate environment variable.

    Returns:
        An initialized LLMProvider instance.

    Raises:
        ValueError: If the provider type is not supported.
    """
    _ensure_registry()

    provider_name = model_config.provider.lower()
    provider_cls = _PROVIDER_REGISTRY.get(provider_name)

    if provider_cls is None:
        available = ", ".join(sorted(_PROVIDER_REGISTRY.keys()))
        msg = (
            f"Provider '{provider_name}' not available. "
            f"Registered providers: {available}. "
            f"Check that the required SDK is installed."
        )
        raise ValueError(msg)

    if provider_name == "ollama":
        # Ollama doesn't need an API key
        return provider_cls(model_config)  # type: ignore[call-arg]

    if provider_name == "google_vertex":
        # Vertex AI uses ADC (gcloud auth), not an API key
        return provider_cls(model_config, vertexai=True)  # type: ignore[call-arg]

    if provider_name == "google_vertex_openai":
        # Vertex OpenAI-compatible endpoint uses gcloud auth internally
        return provider_cls(model_config)  # type: ignore[call-arg]

    if api_key is not None:
        return provider_cls(model_config, api_key=api_key)  # type: ignore[call-arg]

    return provider_cls(model_config)  # type: ignore[call-arg]


def list_available_providers() -> list[str]:
    """Return names of all available provider types."""
    _ensure_registry()
    return sorted(_PROVIDER_REGISTRY.keys())
