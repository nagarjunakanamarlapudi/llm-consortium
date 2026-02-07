"""Global registry of models, variants, and tasks."""

from __future__ import annotations

from consortium.config.models import FullConfig, ModelConfig, TaskConfig, VariantConfig

# Module-level singleton — set once at startup via `init_registry()`
_config: FullConfig | None = None


def init_registry(config: FullConfig) -> None:
    """Initialize the global config registry."""
    global _config  # noqa: PLW0603
    _config = config


def get_config() -> FullConfig:
    """Get the global configuration. Raises if not initialized."""
    if _config is None:
        msg = "Config registry not initialized. Call init_registry() first."
        raise RuntimeError(msg)
    return _config


def get_model(model_id: str) -> ModelConfig:
    """Shortcut to get a model config from the global registry."""
    return get_config().get_model(model_id)


def get_variant(variant_id: str) -> VariantConfig:
    """Shortcut to get a variant config from the global registry."""
    return get_config().get_variant(variant_id)


def get_task(task_id: str) -> TaskConfig:
    """Shortcut to get a task config from the global registry."""
    return get_config().get_task(task_id)
