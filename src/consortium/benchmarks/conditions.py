"""Experimental conditions as in-memory ``VariantConfig`` objects.

A *condition* is one arm of the experiment. Single-model baselines reproduce
leaderboard pass@1; consortium conditions are heterogeneous topologies that
reuse the existing v1-v8 orchestrators (the ``id`` prefix ``vN`` selects the
orchestrator class; the suffix keeps each condition a distinct DB row).

Roster (all via DigitalOcean serverless inference):
    sonnet -> do-sonnet (anthropic-claude-4.6-sonnet)
    gpt52  -> do-gpt52  (openai-gpt-5.2)
    gptoss -> do-gptoss (openai-gpt-oss-120b)
"""

from __future__ import annotations

from consortium.config.models import (
    AgentConfig,
    VariantAgentsConfig,
    VariantConfig,
    WorkflowConfig,
)

#: Friendly roster name -> model config id (see configs/models/do_*.yaml).
ROSTER = {
    "sonnet": "do-sonnet",
    "gpt52": "do-gpt52",
    "gptoss": "do-gptoss",
}

# Prompt templates (function-level coding).
_SOLVE = "generation/solve_coding.j2"
_CODE_REVIEW = "review/code_review.j2"
_CODE_ADVERSARIAL = "review/structural_code_adversarial.j2"


# ── Baselines (single model, single shot) ─────────────────────────────────────


def _baseline(label: str, model_id: str) -> VariantConfig:
    """v1a single-shot baseline for one model (reproduces leaderboard pass@1)."""
    return VariantConfig(
        id=f"v1a_{label}",
        name=f"base-{label}",
        description=f"Single-shot baseline ({model_id})",
        sub_variant="single_shot",
        agents=VariantAgentsConfig(
            designer=AgentConfig(role="designer", model=model_id, system_prompt_template=_SOLVE),
        ),
        workflow=WorkflowConfig(max_rounds=0, include_rubric_in_prompt=False),
    )


# ── Consortium topologies (heterogeneous) ─────────────────────────────────────


def _xreview(writer: str, reviewer: str) -> VariantConfig:
    """c-xreview (v2 cross-model review): writer drafts, a different-family model
    reviews, writer revises once."""
    return VariantConfig(
        id="v2b_xreview",
        name="c-xreview",
        description=f"Cross-model review: {writer} writes, {reviewer} reviews",
        sub_variant="cross_model",
        agents=VariantAgentsConfig(
            leader=AgentConfig(role="leader", model=writer, system_prompt_template=_SOLVE),
            reviewers=AgentConfig(
                role="reviewer", model=reviewer, system_prompt_template=_CODE_REVIEW, count=1
            ),
        ),
        workflow=WorkflowConfig(
            max_rounds=2,  # round 0 draft, round 1 review->revise
            include_rubric_in_prompt=False,
            reviewer_sees_other_reviews=False,
            review_template=_CODE_REVIEW,
        ),
    )


def _adversarial(writer: str, adversary: str) -> VariantConfig:
    """c-adv (v4 structural-adversarial): writer drafts, a different model demands
    rewrites, writer revises until accepted or rounds exhausted."""
    return VariantConfig(
        id="v4b_adv",
        name="c-adv",
        description=f"Structural-adversarial: {writer} writes, {adversary} attacks",
        sub_variant="structural_adversarial",
        agents=VariantAgentsConfig(
            leader=AgentConfig(role="leader", model=writer, system_prompt_template=_SOLVE),
            adversarial_reviewer=AgentConfig(
                role="adversarial_reviewer",
                model=adversary,
                system_prompt_template=_CODE_ADVERSARIAL,
            ),
        ),
        workflow=WorkflowConfig(
            max_rounds=3,
            include_rubric_in_prompt=False,
            review_template=_CODE_ADVERSARIAL,
        ),
    )


def _all_condition_builders() -> dict[str, VariantConfig]:
    """Construct every known condition keyed by its friendly name."""
    s, g, o = ROSTER["sonnet"], ROSTER["gpt52"], ROSTER["gptoss"]
    return {
        "base-sonnet": _baseline("sonnet", s),
        "base-gpt52": _baseline("gpt52", g),
        "base-gptoss": _baseline("gptoss", o),
        # Heterogeneous consortium: strong writer + different-family critic.
        "c-xreview": _xreview(writer=s, reviewer=o),
        "c-adv": _adversarial(writer=s, adversary=g),
    }


#: All condition names in canonical reporting order.
ALL_CONDITIONS = (
    "base-sonnet",
    "base-gpt52",
    "base-gptoss",
    "c-xreview",
    "c-adv",
)

#: Baseline-only subset (used for the calibration gate).
BASELINE_CONDITIONS = ("base-sonnet", "base-gpt52", "base-gptoss")


def build_conditions(names: list[str] | tuple[str, ...]) -> dict[str, VariantConfig]:
    """Build the requested conditions as ``{variant_id: VariantConfig}``.

    Args:
        names: Friendly condition names (see :data:`ALL_CONDITIONS`).

    Returns:
        Mapping from the resulting ``variant_id`` to its config.

    Raises:
        KeyError: If a name is unknown.
    """
    builders = _all_condition_builders()
    out: dict[str, VariantConfig] = {}
    for name in names:
        if name not in builders:
            available = ", ".join(sorted(builders))
            msg = f"unknown condition {name!r} (available: {available})"
            raise KeyError(msg)
        vc = builders[name]
        out[vc.id] = vc
    return out
