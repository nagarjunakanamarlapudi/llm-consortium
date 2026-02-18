"""Run context and artifact dataclasses for orchestration state."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from consortium.config.models import LimitsConfig

if TYPE_CHECKING:
    from consortium.config.models import RubricDimensionConfig


@dataclass
class ConversationTurn:
    """A single turn in the conversation history."""

    round: int
    agent_role: str
    agent_id: str
    step: str  # "generation", "review", "revision", "merge", "rebuttal", "judge"
    content: str
    timestamp: datetime


@dataclass
class DesignArtifact:
    """A design document produced by an agent."""

    design_id: str
    run_id: str
    round: int
    agent_role: str
    agent_id: str
    full_text: str
    token_count: int
    is_final: bool
    created_at: datetime


@dataclass
class ReviewArtifact:
    """A review or critique of a design."""

    review_id: str
    run_id: str
    design_id: str
    round: int
    agent_role: str
    agent_id: str
    review_text: str
    verdict: str | None = None  # "accept"/"reject" for adversarial, None for others
    created_at: datetime | None = None


@dataclass
class LLMCallTrace:
    """Complete trace of a single LLM API call."""

    trace_id: str
    run_id: str
    variant_id: str
    task_id: str
    repetition: int

    agent_role: str
    agent_id: str
    step: str
    round: int

    model_config_id: str
    api_model: str
    provider: str

    system_prompt_hash: str
    prompt_template: str

    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    cost_usd: float
    latency_ms: float

    started_at: datetime
    ended_at: datetime

    batch_id: str | None = None
    status: str = "success"  # "success", "retried", "failed"
    error: str | None = None
    retry_count: int = 0
    prompt_text: str | None = None      # full rendered prompt sent to LLM
    response_text: str | None = None    # raw LLM response content


@dataclass
class RunContext:
    """Mutable shared state for a single experiment run."""

    run_id: str
    variant_id: str
    task_id: str
    repetition: int

    # Conversation & artifacts
    conversation_history: list[ConversationTurn] = field(default_factory=list)
    designs: list[DesignArtifact] = field(default_factory=list)
    reviews: list[ReviewArtifact] = field(default_factory=list)

    # Accounting
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0

    # Rubric dimensions for this run's design type
    rubric_dimensions: list[RubricDimensionConfig] = field(default_factory=list)

    # Trace (every LLM call recorded here)
    traces: list[LLMCallTrace] = field(default_factory=list)

    # Timing
    started_at: datetime | None = None
    ended_at: datetime | None = None

    # Status
    status: str = "pending"  # "pending", "running", "completed", "failed", "aborted"
    error: str | None = None

    # Extensible metadata (variant-specific info for traces/analysis)
    metadata: dict[str, str] = field(default_factory=dict)

    # Checkpoint (for resume after crash)
    checkpoint_step: str | None = None  # e.g. "round:2:review"
    checkpoint_data: str | None = None  # JSON-serialized state for resume

    def record_trace(self, trace: LLMCallTrace) -> None:
        """Append a traced LLM call and update accounting."""
        self.traces.append(trace)
        self.total_input_tokens += trace.input_tokens
        self.total_output_tokens += trace.output_tokens
        self.total_cost_usd += trace.cost_usd

    def check_limits(self, limits: LimitsConfig) -> tuple[bool, str | None]:
        """Check if limits have been exceeded.

        Returns:
            (within_limits, error_message) — True if OK, False with reason if exceeded.
        """
        total_tokens = self.total_input_tokens + self.total_output_tokens
        if total_tokens > limits.max_tokens_per_run:
            return False, (
                f"Token limit exceeded: {total_tokens:,} > "
                f"{limits.max_tokens_per_run:,}"
            )
        if self.total_cost_usd > limits.max_cost_per_run_usd:
            return False, (
                f"Cost limit exceeded: ${self.total_cost_usd:.4f} > "
                f"${limits.max_cost_per_run_usd:.2f}"
            )
        return True, None

    def get_latest_design(self) -> DesignArtifact | None:
        """Get the most recent design artifact."""
        return self.designs[-1] if self.designs else None

    def get_reviews_for_round(self, round_num: int) -> list[ReviewArtifact]:
        """Get all reviews for a specific round."""
        return [r for r in self.reviews if r.round == round_num]

    def compute_seed(self, agent_id: str, round_num: int) -> int:
        """Compute a deterministic seed for an LLM call.

        Uses the stable identifiers (variant, task, repetition, agent, round)
        rather than the timestamp-containing ``run_id`` so that re-running the
        same configuration produces identical seeds.
        """
        import hashlib

        combined = (
            f"{self.variant_id}_{self.task_id}_{self.repetition}"
            f"_{agent_id}_{round_num}"
        )
        return int(hashlib.md5(combined.encode()).hexdigest(), 16) % (2**31)
