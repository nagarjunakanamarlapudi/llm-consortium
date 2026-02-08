"""Base agent class for all consortium agents."""

from __future__ import annotations

import abc
import hashlib
import uuid
from datetime import UTC, datetime
from typing import Any

import structlog

from consortium.config.models import LimitsConfig, ModelConfig, ModelParametersConfig
from consortium.orchestrator.context import LLMCallTrace, RunContext
from consortium.prompts.renderer import PromptRenderer
from consortium.providers.base import LLMProvider, LLMRequest, LLMResponse

logger = structlog.get_logger()


class BaseAgent(abc.ABC):
    """Base class for all agent types in the consortium.

    Each agent has a role, a unique ID, an LLM provider, a prompt template,
    and access to a shared prompt renderer. The ``_call_llm`` helper handles
    request construction, provider invocation, trace recording, and limit
    checking so that subclasses only need to assemble template variables.
    """

    def __init__(
        self,
        *,
        agent_id: str,
        role: str,
        model_config: ModelConfig,
        provider: LLMProvider,
        renderer: PromptRenderer,
        prompt_template: str,
        parameters: ModelParametersConfig | None = None,
        limits: LimitsConfig | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.role = role
        self.model_config = model_config
        self.provider = provider
        self.renderer = renderer
        self.prompt_template = prompt_template
        self.parameters = parameters or model_config.parameters
        self.limits = limits
        self._log = logger.bind(agent_id=agent_id, role=role, model=model_config.id)

    async def _call_llm(
        self,
        *,
        template: str,
        template_vars: dict[str, Any],
        context: RunContext,
        step: str,
        round_num: int,
    ) -> LLMResponse:
        """Render a prompt template, call the LLM, and record a trace.

        Args:
            template: Path to the Jinja2 template (e.g. ``generation/design_system.j2``).
            template_vars: Variables to inject into the template.
            context: The current run context (traces are appended here).
            step: Workflow step label for tracing.
            round_num: Current round number.

        Returns:
            The raw ``LLMResponse`` from the provider.

        Raises:
            RuntimeError: If token or cost limits have been exceeded.
        """
        # Check limits before calling
        if self.limits is not None:
            within, msg = context.check_limits(self.limits)
            if not within:
                self._log.error("limit_exceeded_before_call", error=msg)
                raise RuntimeError(f"Run aborted: {msg}")

        # Render prompt content from template
        prompt_content = self.renderer.render(template, **template_vars)

        request = LLMRequest(
            system_prompt="",
            messages=[{"role": "user", "content": prompt_content}],
            model_config_id=self.model_config.id,
            parameters={
                "temperature": self.parameters.temperature,
                "max_tokens": self.parameters.max_tokens,
                "top_p": self.parameters.top_p,
            },
            metadata={
                "agent_id": self.agent_id,
                "role": self.role,
                "step": step,
                "round": str(round_num),
            },
        )

        self._log.debug("calling_llm", step=step, round=round_num, template=template)
        started_at = datetime.now(UTC)
        response = await self.provider.complete(request)
        ended_at = datetime.now(UTC)

        # Build and record trace
        trace = LLMCallTrace(
            trace_id=uuid.uuid4().hex,
            run_id=context.run_id,
            variant_id=context.variant_id,
            task_id=context.task_id,
            repetition=context.repetition,
            agent_role=self.role,
            agent_id=self.agent_id,
            step=step,
            round=round_num,
            model_config_id=self.model_config.id,
            api_model=response.model,
            provider=self.model_config.provider,
            system_prompt_hash=hashlib.sha256(
                prompt_content.encode()
            ).hexdigest()[:16],
            prompt_template=template,
            prompt_text=prompt_content,
            response_text=response.content,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cached_input_tokens=response.cached_input_tokens,
            cost_usd=response.cost_usd,
            latency_ms=response.latency_ms,
            started_at=started_at,
            ended_at=ended_at,
            batch_id=response.batch_id,
        )
        context.record_trace(trace)

        self._log.info(
            "llm_call_complete",
            step=step,
            round=round_num,
            tokens_in=response.input_tokens,
            tokens_out=response.output_tokens,
            cost=f"${response.cost_usd:.4f}",
            latency_ms=f"{response.latency_ms:.0f}",
        )

        # Warn if approaching limits
        if self.limits is not None:
            total_tokens = context.total_input_tokens + context.total_output_tokens
            token_pct = total_tokens / max(self.limits.max_tokens_per_run, 1) * 100
            cost_pct = context.total_cost_usd / max(self.limits.max_cost_per_run_usd, 0.01) * 100
            if token_pct >= 80 or cost_pct >= 80:
                self._log.warning(
                    "approaching_limit",
                    token_pct=f"{token_pct:.1f}%",
                    cost_pct=f"{cost_pct:.1f}%",
                )

        return response

    @abc.abstractmethod
    async def act(
        self,
        *,
        context: RunContext,
        round_num: int,
        **kwargs: Any,
    ) -> Any:
        """Execute the agent's primary action.

        Subclasses implement role-specific logic and return the appropriate
        artifact type (``DesignArtifact`` or ``ReviewArtifact``).
        """
        ...
