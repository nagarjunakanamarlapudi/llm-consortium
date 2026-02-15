"""Variant registry and agent instantiation factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from consortium.orchestrator.variants.base import VariantOrchestrator
from consortium.orchestrator.variants.v1_baseline import V1BaselineOrchestrator
from consortium.orchestrator.variants.v2_leader_reviewers import V2LeaderReviewersOrchestrator
from consortium.orchestrator.variants.v3_parallel_merge import V3ParallelMergeOrchestrator
from consortium.orchestrator.variants.v4_adversarial import V4AdversarialOrchestrator
from consortium.orchestrator.variants.v5_specialist_panel import V5SpecialistPanelOrchestrator
from consortium.orchestrator.variants.v6_rotating_leader import V6RotatingLeaderOrchestrator
from consortium.orchestrator.variants.v7_consensus import V7ConsensusOrchestrator
from consortium.orchestrator.variants.v8_structured_debate import V8StructuredDebateOrchestrator

if TYPE_CHECKING:
    from consortium.agents.base import BaseAgent
    from consortium.config.models import (
        FullConfig,
        LimitsConfig,
        RubricConfig,
        VariantConfig,
    )
    from consortium.prompts.renderer import PromptRenderer
    from consortium.providers.base import LLMProvider

logger = structlog.get_logger()

# ── Variant Registry ──────────────────────────────────────────────────────────

_VARIANT_REGISTRY: dict[str, type[VariantOrchestrator]] = {
    "v1": V1BaselineOrchestrator,
    "v2": V2LeaderReviewersOrchestrator,
    "v3": V3ParallelMergeOrchestrator,
    "v4": V4AdversarialOrchestrator,
    "v5": V5SpecialistPanelOrchestrator,
    "v6": V6RotatingLeaderOrchestrator,
    "v7": V7ConsensusOrchestrator,
    "v8": V8StructuredDebateOrchestrator,
}


def _resolve_variant_class(variant_id: str) -> type[VariantOrchestrator] | None:
    """Resolve a variant ID to its orchestrator class.

    Sub-variants (e.g. ``v1a``, ``v2b``, ``v3c``) share the same orchestrator
    as their parent variant. The behavioral differences come from the YAML
    config (different models, prompt templates, workflow parameters), not
    from different orchestrator code.

    Resolution: try exact match first (``v2``), then strip trailing
    letter to get the parent (``v2a`` → ``v2``).
    """
    # Exact match
    if variant_id in _VARIANT_REGISTRY:
        return _VARIANT_REGISTRY[variant_id]

    # Sub-variant: strip trailing letter(s) to find parent.
    # Handles v1a → v1, v2a → v2, v2b → v2, v3c → v3, etc.
    import re

    parent_match = re.match(r"^(v\d+)", variant_id)
    if parent_match:
        parent_id = parent_match.group(1)
        return _VARIANT_REGISTRY.get(parent_id)

    return None


def create_variant_orchestrator(
    variant_config: VariantConfig,
    agents: dict[str, BaseAgent | list[BaseAgent]],
) -> VariantOrchestrator:
    """Create a variant orchestrator from config and pre-built agents.

    Sub-variants (v1a, v2a-c, v3a-c) are resolved to their parent
    orchestrator class. The behavioral differences between sub-variants
    come from the YAML config (model assignments, prompt templates,
    workflow parameters) rather than from different orchestrator code.

    Args:
        variant_config: The variant's configuration.
        agents: Instantiated agents keyed by role name.

    Returns:
        An initialized VariantOrchestrator subclass instance.

    Raises:
        ValueError: If the variant ID is not registered.
    """
    variant_cls = _resolve_variant_class(variant_config.id)
    if variant_cls is None:
        available = ", ".join(sorted(_VARIANT_REGISTRY))
        msg = f"Unknown variant '{variant_config.id}'. Available: {available}"
        raise ValueError(msg)

    logger.info(
        "variant_orchestrator_resolved",
        variant_id=variant_config.id,
        sub_variant=variant_config.sub_variant,
        orchestrator=variant_cls.__name__,
    )

    return variant_cls(config=variant_config, agents=agents)


# ── Agent Instantiation ──────────────────────────────────────────────────────


def instantiate_agents(
    variant_config: VariantConfig,
    full_config: FullConfig,
    renderer: PromptRenderer,
    limits: LimitsConfig | None = None,
    *,
    _provider_cache: dict[str, LLMProvider] | None = None,
    _provider_factory: object | None = None,
) -> dict[str, BaseAgent | list[BaseAgent]]:
    """Create all agents defined in a variant config.

    Each agent is constructed with its model config, provider, prompt template,
    and parameters. Providers are cached by model ID so identical models share
    a single provider instance.

    Args:
        variant_config: Variant configuration with agent definitions.
        full_config: Full experiment config (for resolving model configs).
        renderer: Shared prompt renderer.
        limits: Optional safety limits passed to each agent.
        _provider_cache: Optional shared provider cache across calls.
        _provider_factory: Optional callable ``(model_id: str) -> LLMProvider``
            used instead of the default ``create_provider()``.  Enables
            sharing providers across runs via a ProviderRegistry.

    Returns:
        Dict mapping agent key names to agent instances (or lists of agents).
    """
    from consortium.agents.adversary import AdversarialReviewer
    from consortium.agents.designer import DesignerAgent
    from consortium.agents.judge import JudgeAgent
    from consortium.agents.merger import MergerAgent
    from consortium.agents.reviewer import ReviewerAgent
    from consortium.agents.specialist import SpecialistReviewer
    from consortium.providers.factory import create_provider

    providers: dict[str, LLMProvider] = _provider_cache or {}
    agents_map: dict[str, BaseAgent | list[BaseAgent]] = {}
    va = variant_config.agents

    def _get_provider(model_id: str) -> LLMProvider:
        if model_id not in providers:
            if _provider_factory is not None:
                providers[model_id] = _provider_factory(model_id)
            else:
                model_cfg = full_config.get_model(model_id)
                providers[model_id] = create_provider(model_cfg)
        return providers[model_id]

    # ── Single-agent fields ──────────────────────────────────────────────

    # designer (v1)
    if va.designer is not None:
        agents_map["designer"] = _make_agent(
            DesignerAgent,
            "designer",
            va.designer,
            full_config,
            renderer,
            _get_provider,
            limits,
        )

    # leader (v2, v4, v5)
    if va.leader is not None:
        agents_map["leader"] = _make_agent(
            DesignerAgent,
            "leader",
            va.leader,
            full_config,
            renderer,
            _get_provider,
            limits,
        )

    # merger (v3)
    if va.merger is not None:
        agents_map["merger"] = _make_agent(
            MergerAgent,
            "merger",
            va.merger,
            full_config,
            renderer,
            _get_provider,
            limits,
        )

    # adversarial_reviewer (v4)
    if va.adversarial_reviewer is not None:
        agents_map["adversarial_reviewer"] = _make_agent(
            AdversarialReviewer,
            "adversarial_reviewer",
            va.adversarial_reviewer,
            full_config,
            renderer,
            _get_provider,
            limits,
        )

    # judge (v8)
    if va.judge is not None:
        agents_map["judge"] = _make_agent(
            JudgeAgent,
            "judge",
            va.judge,
            full_config,
            renderer,
            _get_provider,
            limits,
        )

    # ── Multi-agent fields (count-based) ─────────────────────────────────

    # reviewers (v2)
    if va.reviewers is not None:
        agents_map["reviewers"] = _make_agents_counted(
            ReviewerAgent,
            "reviewer",
            va.reviewers,
            full_config,
            renderer,
            _get_provider,
            limits,
        )

    # parallel_leaders (v3)
    if va.parallel_leaders is not None:
        agents_map["parallel_leaders"] = _make_agents_counted(
            DesignerAgent,
            "designer",
            va.parallel_leaders,
            full_config,
            renderer,
            _get_provider,
            limits,
        )

    # participants (v6, v7) — individually defined agents
    if va.participants:
        participant_agents: list[BaseAgent] = []
        for agent_cfg in va.participants:
            agent_id = agent_cfg.id or f"participant_{len(participant_agents)}"
            participant_agents.append(
                _make_agent(
                    DesignerAgent,
                    agent_id,
                    agent_cfg,
                    full_config,
                    renderer,
                    _get_provider,
                    limits,
                )
            )
        agents_map["participants"] = participant_agents

    # debaters (v8)
    if va.debaters:
        debater_agents: list[BaseAgent] = []
        for agent_cfg in va.debaters:
            debater_agents.append(
                _make_agent(
                    DesignerAgent,
                    agent_cfg.id or f"debater_{len(debater_agents)}",
                    agent_cfg,
                    full_config,
                    renderer,
                    _get_provider,
                    limits,
                )
            )
            # Store perspective on the agent instance
            debater_agents[-1].perspective = agent_cfg.perspective
        agents_map["debaters"] = debater_agents

    # ── Specialists (v5) — list of SpecialistConfig ──────────────────────

    if va.specialists:
        specialist_agents: list[BaseAgent] = []
        for spec_cfg in va.specialists:
            model_cfg = full_config.get_model(spec_cfg.model)
            specialist_agents.append(
                SpecialistReviewer(
                    agent_id=spec_cfg.id,
                    role=spec_cfg.role,
                    model_config=model_cfg,
                    provider=_get_provider(spec_cfg.model),
                    renderer=renderer,
                    prompt_template=spec_cfg.system_prompt_template,
                    limits=limits,
                    specialty=spec_cfg.specialty,
                    focus_dimensions=list(spec_cfg.focus_dimensions),
                )
            )
        agents_map["specialists"] = specialist_agents

    logger.info(
        "agents_instantiated",
        variant=variant_config.id,
        agent_keys=sorted(agents_map.keys()),
    )
    return agents_map


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_agent(
    cls,
    agent_id,
    agent_config,
    full_config,
    renderer,
    get_provider,
    limits,
):
    """Create a single agent instance from an AgentConfig."""
    model_cfg = full_config.get_model(agent_config.model)
    return cls(
        agent_id=agent_id,
        role=agent_config.role,
        model_config=model_cfg,
        provider=get_provider(agent_config.model),
        renderer=renderer,
        prompt_template=agent_config.system_prompt_template,
        application_prompt_template=agent_config.application_prompt_template,
        parameters=agent_config.parameters,
        limits=limits,
    )


def _make_agents_counted(
    cls,
    role,
    agent_config,
    full_config,
    renderer,
    get_provider,
    limits,
):
    """Create a list of agents from a count-based AgentConfig."""
    model_cfg = full_config.get_model(agent_config.model)
    return [
        cls(
            agent_id=f"{role}_{i}",
            role=agent_config.role,
            model_config=model_cfg,
            provider=get_provider(agent_config.model),
            renderer=renderer,
            prompt_template=agent_config.system_prompt_template,
            parameters=agent_config.parameters,
            limits=limits,
        )
        for i in range(agent_config.count)
    ]
