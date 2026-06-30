"""YAML configuration loading with environment variable interpolation."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import copy

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


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Deep-merge *overrides* into a copy of *base*.

    - Dict values are recursively merged.
    - Lists and scalars in *overrides* replace the base value entirely.
    """
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_variant_config(
    path: Path,
    *,
    variants_dir: Path | None = None,
) -> VariantConfig:
    """Load a variant config from a YAML file.

    If the YAML contains an ``extends`` key, the referenced parent
    variant is loaded first and the current file's values are
    deep-merged on top.  This lets sub-variant files specify only
    the fields that differ from the parent, avoiding duplication.

    Example sub-variant YAML::

        variant:
          extends: v2_leader_reviewers.yaml
          id: v2a
          name: "Leader + Reviewers (Same Model)"
          sub_variant: same_model
          agents:
            reviewers:
              model: vertex-gemini-2.5-pro
    """
    data = load_yaml(path)
    variant_data = data.get("variant", data)

    extends = variant_data.pop("extends", None)
    if extends is not None:
        # Resolve parent path relative to the same directory
        parent_dir = variants_dir or path.parent
        parent_path = parent_dir / extends
        if not parent_path.exists():
            msg = f"Parent variant not found: {parent_path} (extends: {extends})"
            raise FileNotFoundError(msg)
        parent_data = load_yaml(parent_path)
        parent_variant = parent_data.get("variant", parent_data)
        # Drop parent's extends to prevent accidental chaining
        parent_variant.pop("extends", None)
        variant_data = _deep_merge(parent_variant, variant_data)

    _normalize_named_agents(variant_data)
    return VariantConfig(**variant_data)


def _normalize_named_agents(variant_data: dict) -> None:
    """Convert named ``participant_N`` / ``debater_N`` keys into lists.

    This allows YAML authors to write individually-named agent
    blocks (each with a different model) while producing the
    ``list[AgentConfig]`` structure that Pydantic expects.
    """
    agents = variant_data.get("agents")
    if not isinstance(agents, dict):
        return

    # --- participants ---
    if "participants" not in agents:
        named: dict[str, dict] = {}
        for key in list(agents.keys()):
            if key.startswith("participant_"):
                named[key] = agents.pop(key)
        if named:
            participants_list = []
            for key in sorted(named.keys()):
                entry = named[key]
                entry.setdefault("id", key)
                participants_list.append(entry)
            agents["participants"] = participants_list

    # --- debaters ---
    if "debaters" not in agents:
        named_d: dict[str, dict] = {}
        for key in list(agents.keys()):
            if key.startswith("debater_"):
                named_d[key] = agents.pop(key)
        if named_d:
            debaters_list = []
            for key in sorted(named_d.keys()):
                entry = named_d[key]
                entry.setdefault("id", key)
                debaters_list.append(entry)
            agents["debaters"] = debaters_list


def load_task_config(path: Path) -> TaskConfig:
    """Load a task config from a YAML file.

    The nested ``coding`` block (present only for ``task_type: coding``) is
    coerced into a :class:`CodingProblemConfig` automatically by Pydantic when
    constructing :class:`TaskConfig`.
    """
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

    # Variants (two-pass: files without 'extends' first, then sub-variants)
    variants: dict[str, VariantConfig] = {}
    variants_dir = config_dir / "variants"
    if variants_dir.exists():
        all_variant_files = sorted(variants_dir.glob("*.yaml"))
        for f in all_variant_files:
            vc = load_variant_config(f, variants_dir=variants_dir)
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
