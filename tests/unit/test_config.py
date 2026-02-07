"""Tests for configuration loading and validation."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from consortium.config.loader import (
    _interpolate_env,
    load_evaluator_config,
    load_experiment_config,
    load_full_config,
    load_model_config,
    load_rubric_config,
    load_task_config,
    load_variant_config,
)
from consortium.config.models import (
    ExperimentConfig,
    FullConfig,
    ModelConfig,
    VariantConfig,
)
from consortium.config.registry import get_config, init_registry


class TestEnvInterpolation:
    def test_simple_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_VAR", "hello")
        assert _interpolate_env("${TEST_VAR}") == "hello"

    def test_var_with_default(self) -> None:
        # Ensure var is NOT set
        os.environ.pop("NONEXISTENT_VAR_12345", None)
        assert _interpolate_env("${NONEXISTENT_VAR_12345:-fallback}") == "fallback"

    def test_var_set_overrides_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_VAR", "actual")
        assert _interpolate_env("${MY_VAR:-default}") == "actual"

    def test_unresolved_var_unchanged(self) -> None:
        os.environ.pop("MISSING_VAR_99999", None)
        assert _interpolate_env("${MISSING_VAR_99999}") == "${MISSING_VAR_99999}"

    def test_multiple_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("A", "1")
        monkeypatch.setenv("B", "2")
        assert _interpolate_env("${A}-${B}") == "1-2"


class TestModelConfig:
    def test_load_openai_model(self, configs_dir: Path) -> None:
        mc = load_model_config(configs_dir / "models" / "openai_gpt4_1.yaml")
        assert mc.id == "gpt-4.1"
        assert mc.provider == "openai"
        assert mc.pricing.input == 2.00
        assert mc.pricing.output == 8.00
        assert mc.supports_batch is True

    def test_load_anthropic_model(self, configs_dir: Path) -> None:
        mc = load_model_config(configs_dir / "models" / "anthropic_sonnet.yaml")
        assert mc.id == "sonnet-4.5"
        assert mc.provider == "anthropic"
        assert mc.context_window == 200_000

    def test_load_ollama_model(self, configs_dir: Path) -> None:
        mc = load_model_config(configs_dir / "models" / "ollama_gptoss.yaml")
        assert mc.id == "gpt-oss-120b"
        assert mc.provider == "ollama"
        assert mc.pricing.input == 0.0
        assert mc.ollama is not None
        assert mc.ollama.concurrency == 2

    def test_model_frozen(self, configs_dir: Path) -> None:
        mc = load_model_config(configs_dir / "models" / "openai_gpt4_1.yaml")
        with pytest.raises(Exception):  # noqa: B017 (ValidationError in frozen model)
            mc.id = "changed"  # type: ignore[misc]


class TestVariantConfig:
    def test_load_v1_baseline(self, configs_dir: Path) -> None:
        vc = load_variant_config(configs_dir / "variants" / "v1_baseline.yaml")
        assert vc.id == "v1"
        assert vc.name == "Baseline"

    def test_load_v3_parallel(self, configs_dir: Path) -> None:
        vc = load_variant_config(configs_dir / "variants" / "v3_parallel_merge.yaml")
        assert vc.id == "v3"
        assert vc.workflow.merge_strategy is not None

    def test_load_all_variants(self, configs_dir: Path) -> None:
        variants_dir = configs_dir / "variants"
        for f in sorted(variants_dir.glob("*.yaml")):
            vc = load_variant_config(f)
            assert vc.id, f"Variant in {f.name} has no id"
            assert vc.name, f"Variant in {f.name} has no name"


class TestTaskConfig:
    def test_load_t1(self, configs_dir: Path) -> None:
        tc = load_task_config(configs_dir / "tasks" / "t1_url_shortener.yaml")
        assert tc.id == "t1"
        assert tc.complexity == "simple"
        assert tc.design_type == "system"

    def test_load_t5(self, configs_dir: Path) -> None:
        tc = load_task_config(configs_dir / "tasks" / "t5_trading_platform.yaml")
        assert tc.id == "t5"
        assert tc.complexity == "complex"

    def test_load_all_tasks(self, configs_dir: Path) -> None:
        tasks_dir = configs_dir / "tasks"
        for f in sorted(tasks_dir.glob("*.yaml")):
            tc = load_task_config(f)
            assert tc.id, f"Task in {f.name} has no id"
            assert tc.complexity in ("simple", "medium", "complex")


class TestRubricConfig:
    def test_load_system_rubric(self, configs_dir: Path) -> None:
        rc = load_rubric_config(configs_dir / "rubrics" / "system_design.yaml")
        assert rc.id == "system_design"
        assert len(rc.dimensions) >= 10

    def test_rubric_dimensions_have_anchors(self, configs_dir: Path) -> None:
        rc = load_rubric_config(configs_dir / "rubrics" / "system_design.yaml")
        for dim in rc.dimensions:
            assert dim.id, "Dimension has no id"
            assert dim.weight > 0


class TestExperimentConfig:
    def test_load_experiment(self, configs_dir: Path) -> None:
        ec = load_experiment_config(configs_dir / "experiment.yaml")
        assert ec.name == "llm-consortium-v1"
        assert ec.repetitions == 5
        assert len(ec.variants) == 8
        assert len(ec.tasks) == 8
        assert ec.limits.max_total_cost_usd == 500.0

    def test_load_evaluator(self, configs_dir: Path) -> None:
        ev = load_evaluator_config(configs_dir / "evaluator.yaml")
        assert ev.runs_per_design == 3
        assert ev.aggregation == "median"
        assert ev.coherence_check.enabled is True


class TestFullConfig:
    def test_load_full_config(self, configs_dir: Path) -> None:
        full = load_full_config(configs_dir)
        assert isinstance(full, FullConfig)
        assert len(full.models) >= 4
        assert len(full.variants) >= 8
        assert len(full.tasks) >= 8
        assert len(full.rubrics) >= 2

    def test_get_model(self, configs_dir: Path) -> None:
        full = load_full_config(configs_dir)
        m = full.get_model("gpt-4.1")
        assert m.provider == "openai"

    def test_get_model_not_found(self, configs_dir: Path) -> None:
        full = load_full_config(configs_dir)
        with pytest.raises(KeyError, match="not found"):
            full.get_model("nonexistent")


class TestRegistry:
    def test_init_and_get(self, configs_dir: Path) -> None:
        full = load_full_config(configs_dir)
        init_registry(full)
        cfg = get_config()
        assert cfg is full
