"""DigitalOcean serverless inference provider (OpenAI-compatible).

DigitalOcean's GenAI platform exposes an OpenAI-compatible Chat Completions
endpoint (``https://inference.do-ai.run/v1``) that fronts a large model catalog
spanning multiple families — Anthropic Claude, OpenAI GPT-4.1/GPT-OSS, Llama,
DeepSeek, and more. Routing every model in the heterogeneous consortium through
one endpoint keeps the provider layer uniform and the experiment reproducible:
the only thing that changes between conditions is ``ModelConfig.api_model``.

This provider is a thin subclass of :class:`OpenAIProvider` that points the SDK
at the DO endpoint and authenticates with ``DO_INFERENCE_API_KEY`` (falling back
to ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` so a globally-configured env also
works). All request building, batching, and cost accounting are inherited.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import structlog

from consortium.providers.openai import OpenAIProvider

if TYPE_CHECKING:
    from consortium.config.models import ModelConfig

logger = structlog.get_logger(__name__)

#: Default DigitalOcean serverless-inference base URL (OpenAI-compatible).
DEFAULT_DO_BASE_URL = "https://inference.do-ai.run/v1"


class DOInferenceProvider(OpenAIProvider):
    """OpenAI-compatible provider targeting DigitalOcean serverless inference.

    Args:
        config: Model configuration; ``api_model`` is the DO catalog id
            (e.g. ``anthropic-claude-4.5-sonnet``, ``openai-gpt-oss-120b``).
        api_key: Explicit key override; otherwise ``DO_INFERENCE_API_KEY`` then
            ``OPENAI_API_KEY`` are read from the environment.
        base_url: Explicit endpoint override; otherwise ``DO_INFERENCE_BASE_URL``
            then :data:`DEFAULT_DO_BASE_URL`.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        resolved_key = (
            api_key
            or os.environ.get("DO_INFERENCE_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        resolved_url = (
            base_url
            or os.environ.get("DO_INFERENCE_BASE_URL")
            or DEFAULT_DO_BASE_URL
        )
        if not resolved_key:
            msg = (
                "DigitalOcean inference requires an API key: set DO_INFERENCE_API_KEY "
                "(or OPENAI_API_KEY) in the environment / .env."
            )
            raise RuntimeError(msg)
        super().__init__(config, api_key=resolved_key, base_url=resolved_url)
        logger.debug("do_inference_provider_init", api_model=config.api_model)
