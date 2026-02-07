"""YAML configuration loading with environment variable interpolation."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from consortium.config.models import (
    EvaluatorConfig,
    ExperimentConfig,
    FullConfig,
    ModelConfig,
    RubricConfig,
    RubricDimensionConfig,
    TaskConfig,
    VariantConfig,
)

# Pattern: ${VAR_NAME} or ${VAR_NAME:-default_value}
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _interpolate_env(value: str) -> str:
    """Replace ${VAR} and ${VAR:-default} patterns with environment values."""

    def _replace(match: re.Match[str]) -> str:
        var_name = match.group(1)
        default = match.group(2)
        env_val = os.environ.get(var_name)
        if env_val is not None:
            return env_val
        if default is not None:
            return default
        return match.group(0)  # leave unresolved

    return _ENV_PATTERN.sub(_replace, value)


def _interpolate_recursive(data: Any) -> Any:
    """Recursively interpolate environment variables in a data structure."""
    if isinstance(data, str):
        return _interpolate_env(data)
    if isinstance(data, dict):
        return {k: _interpolate_recursive(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_interpolate_recursive(item) for item in data]
    return data


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file with environment variable interpolation."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    if raw is None:
        return {}
    return _interpolate_recursive(raw)


def load_model_config(path: Path) -> ModelConfig:
    """Load a model config from a YAML file."""
    data = load_yaml(path)
    model_data = data.get("model", data)
    return ModelConfig(**model_data)


def load_variant_config(path: Path) -> VariantConfig:
    """Load a variant config from a YAML file."""
    data = load_yaml(path)
    variant_data = data.get("variant", data)
    return VariantConfig(**variant_data)


def load_task_config(path: Path) -> TaskConfig:
    """Load a task config from a YAML file."""
    data = load_yaml(path)
    task_data = data.get("task", data)
    return TaskConfig(**task_data)


def load_rubric_config(path: Path) -> RubricConfig:
    """Load a rubric config from a YAML file."""
    data = load_yaml(path)
    rubric_data = data.get("rubric", data)
    # Parse dimensions list
    if "dimensions" in rubric_data:
        rubric_data["dimensions"] = [
            RubricDimensionConfig(**d) if isinstance(d, dict) else d
            for d in rubric_data["dimensions"]
        ]
    return RubricConfig(**rubric_data)


def load_evaluator_config(path: Path) -> EvaluatorConfig:
    """Load evaluator config from a YAML file."""
    data = load_yaml(path)
    eval_data = data.get("evaluator", data)
    return EvaluatorConfig(**eval_data)


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Load experiment config from a YAML file."""
    data = load_yaml(path)
    exp_data = data.get("experiment", data)
    return ExperimentConfig(**exp_data)


def load_full_config(config_dir: str | Path) -> FullConfig:
    """Load all configuration from a directory into a single FullConfig.

    Expected directory structure:
        config_dir/
            experiment.yaml
            evaluator.yaml
            models/*.yaml
            variants/*.yaml
            tasks/*.yaml
            rubrics/*.yaml
    """
    config_dir = Path(config_dir)

    # Experiment config
    exp_path = config_dir / "experiment.yaml"
    if not exp_path.exists():
        msg = f"Experiment config not found: {exp_path}"
        raise FileNotFoundError(msg)
    experiment = load_experiment_config(exp_path)

    # Models
    models: dict[str, ModelConfig] = {}
    models_dir = config_dir / "models"
    if models_dir.exists():
        for f in sorted(models_dir.glob("*.yaml")):
            mc = load_model_config(f)
            models[mc.id] = mc

    # Variants
    variants: dict[str, VariantConfig] = {}
    variants_dir = config_dir / "variants"
    if variants_dir.exists():
        for f in sorted(variants_dir.glob("*.yaml")):
            vc = load_variant_config(f)
            variants[vc.id] = vc

    # Tasks
    tasks: dict[str, TaskConfig] = {}
    tasks_dir = config_dir / "tasks"
    if tasks_dir.exists():
        for f in sorted(tasks_dir.glob("*.yaml")):
            tc = load_task_config(f)
            tasks[tc.id] = tc

    # Rubrics
    rubrics: dict[str, RubricConfig] = {}
    rubrics_dir = config_dir / "rubrics"
    if rubrics_dir.exists():
        for f in sorted(rubrics_dir.glob("*.yaml")):
            rc = load_rubric_config(f)
            rubrics[rc.id] = rc

    # Evaluator
    eval_path = config_dir / "evaluator.yaml"
    evaluator = load_evaluator_config(eval_path) if eval_path.exists() else EvaluatorConfig()

    return FullConfig(
        experiment=experiment,
        models=models,
        variants=variants,
        tasks=tasks,
        rubrics=rubrics,
        evaluator=evaluator,
    )
