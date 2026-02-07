"""Abstract base class for variant orchestrators."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from consortium.agents.base import BaseAgent
    from consortium.config.models import TaskConfig, VariantConfig
    from consortium.orchestrator.context import DesignArtifact, RunContext

logger = structlog.get_logger()


class VariantOrchestrator(abc.ABC):
    """Base class for all variant orchestrators.

    Each variant subclass implements ``execute()`` to define its multi-agent
    workflow topology (centralized/decentralized, homogeneous/specialized,
    cooperative/adversarial).
    """

    def __init__(
        self,
        config: VariantConfig,
        agents: dict[str, BaseAgent | list[BaseAgent]],
    ) -> None:
        self.config = config
        self.agents = agents
        self._log = logger.bind(variant_id=config.id, variant_name=config.name)

    @abc.abstractmethod
    async def execute(
        self,
        task: TaskConfig,
        context: RunContext,
    ) -> DesignArtifact:
        """Run the variant's complete workflow and return the final design.

        Args:
            task: The design task configuration.
            context: Mutable run context for this execution.

        Returns:
            The final design artifact (with ``is_final=True``).
        """
        ...

    def _get_agent(self, key: str) -> BaseAgent:
        """Get a single agent by key."""
        agent = self.agents.get(key)
        if agent is None:
            msg = f"Agent '{key}' not found. Available: {sorted(self.agents)}"
            raise KeyError(msg)
        if isinstance(agent, list):
            msg = f"Agent '{key}' is a list; expected a single agent"
            raise TypeError(msg)
        return agent

    def _get_agents(self, key: str) -> list[BaseAgent]:
        """Get a list of agents by key."""
        agents = self.agents.get(key)
        if agents is None:
            msg = f"Agents '{key}' not found. Available: {sorted(self.agents)}"
            raise KeyError(msg)
        if not isinstance(agents, list):
            msg = f"Agents '{key}' is not a list; got {type(agents).__name__}"
            raise TypeError(msg)
        return agents
