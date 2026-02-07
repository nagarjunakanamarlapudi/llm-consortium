"""Tests for provider protocol and factory."""

from __future__ import annotations

from pathlib import Path

import pytest

from consortium.config.loader import load_model_config
from consortium.providers.base import LLMProvider, LLMRequest, LLMResponse
from consortium.providers.factory import create_provider, list_available_providers


class TestLLMRequest:
    def test_create_request(self) -> None:
        req = LLMRequest(
            system_prompt="You are a helpful assistant.",
            messages=[{"role": "user", "content": "Hello"}],
            model_config_id="gpt-4.1",
        )
        assert req.system_prompt == "You are a helpful assistant."
        assert len(req.messages) == 1
        assert req.parameters == {}
        assert req.metadata == {}

    def test_request_immutable(self) -> None:
        req = LLMRequest(
            system_prompt="test",
            messages=[],
            model_config_id="test",
        )
        with pytest.raises(AttributeError):
            req.system_prompt = "changed"  # type: ignore[misc]


class TestProviderFactory:
    def test_list_available_providers(self) -> None:
        available = list_available_providers()
        # At minimum, these should be available since their SDKs are installed
        assert "anthropic" in available
        assert "openai" in available

    def test_create_anthropic_provider(self, configs_dir: Path) -> None:
        mc = load_model_config(configs_dir / "models" / "anthropic_sonnet.yaml")
        provider = create_provider(mc, api_key="sk-ant-test-key")
        assert isinstance(provider, LLMProvider)
        assert provider.supports_batch()

    def test_create_openai_provider(self, configs_dir: Path) -> None:
        mc = load_model_config(configs_dir / "models" / "openai_gpt4_1.yaml")
        provider = create_provider(mc, api_key="sk-test-key")
        assert isinstance(provider, LLMProvider)
        assert provider.supports_batch()

    def test_create_unknown_provider_fails(self) -> None:
        from consortium.config.models import ModelConfig

        mc = ModelConfig(id="test", provider="nonexistent", api_model="test")
        with pytest.raises(ValueError, match="not available"):
            create_provider(mc)

    def test_estimate_cost_openai(self, configs_dir: Path) -> None:
        mc = load_model_config(configs_dir / "models" / "openai_gpt4_1.yaml")
        provider = create_provider(mc, api_key="sk-test-key")
        cost = provider.estimate_cost(1_000_000, 1_000_000, batch=False)
        # $2/M input + $8/M output = $10
        assert cost == pytest.approx(10.0, abs=0.01)

    def test_estimate_cost_batch_discount(self, configs_dir: Path) -> None:
        mc = load_model_config(configs_dir / "models" / "openai_gpt4_1.yaml")
        provider = create_provider(mc, api_key="sk-test-key")
        cost_realtime = provider.estimate_cost(1_000_000, 1_000_000, batch=False)
        cost_batch = provider.estimate_cost(1_000_000, 1_000_000, batch=True)
        assert cost_batch < cost_realtime

    def test_estimate_cost_ollama_free(self, configs_dir: Path) -> None:
        mc = load_model_config(configs_dir / "models" / "ollama_gptoss.yaml")
        provider = create_provider(mc)
        cost = provider.estimate_cost(1_000_000, 1_000_000)
        assert cost == 0.0
