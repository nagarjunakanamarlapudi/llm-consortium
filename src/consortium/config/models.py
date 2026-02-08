"""Pydantic models for all configuration types."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Model Config ─────────────────────────────────────────────────────────────


class PricingConfig(BaseModel, frozen=True):
    """Per-million-token pricing for a model."""

    input: float = 0.0
    output: float = 0.0
    batch_discount: float = 0.0
    cached_input: float = 0.0


class RateLimitConfig(BaseModel, frozen=True):
    """Rate limits for a model provider."""

    requests_per_minute: int = 500
    tokens_per_minute: int = 2_000_000


class OllamaConfig(BaseModel, frozen=True):
    """Ollama-specific settings."""

    host: str = "http://localhost:11434"
    concurrency: int = 2
    keep_alive: str = "30m"


class ModelParametersConfig(BaseModel, frozen=True):
    """Default generation parameters for a model."""

    temperature: float = 0.7
    max_tokens: int = 16384
    top_p: float = 1.0


class ModelConfig(BaseModel, frozen=True):
    """Configuration for a single LLM model."""

    id: str
    provider: str  # "openai", "anthropic", "google", "ollama"
    api_model: str  # exact API model string

    pricing: PricingConfig = PricingConfig()
    parameters: ModelParametersConfig = ModelParametersConfig()
    context_window: int = 128_000
    supports_batch: bool = False
    supports_caching: bool = False
    rate_limits: RateLimitConfig = RateLimitConfig()
    ollama: OllamaConfig | None = None


# ── Agent & Variant Config ───────────────────────────────────────────────────


class DiversityConfig(BaseModel, frozen=True):
    """Diversity mechanism for parallel agents."""

    enabled: bool = False
    strategy: str = "perspective_prompts"
    perspectives: list[str] = Field(default_factory=list)


class AgentConfig(BaseModel, frozen=True):
    """Configuration for a single agent within a variant."""

    role: str
    model: str  # references a ModelConfig.id
    system_prompt_template: str = ""
    parameters: ModelParametersConfig = ModelParametersConfig()
    count: int = 1
    diversity: DiversityConfig | None = None


class SpecialistConfig(BaseModel, frozen=True):
    """Configuration for a specialist reviewer agent."""

    id: str
    role: str = "specialist_reviewer"
    specialty: str = ""
    model: str = ""
    system_prompt_template: str = ""
    focus_dimensions: list[str] = Field(default_factory=list)


class MatrixPositionConfig(BaseModel, frozen=True):
    """Position in the 2x2x2 variant taxonomy."""

    authority: str = ""  # "centralized" | "decentralized"
    roles: str = ""  # "homogeneous" | "specialized"
    dynamics: str = ""  # "cooperative" | "adversarial"


class WorkflowConfig(BaseModel, frozen=True):
    """Workflow parameters for a variant."""

    max_rounds: int = 1
    include_rubric_in_prompt: bool = True
    reviewer_sees_other_reviews: bool = False
    merge_strategy: str | None = None
    post_merge_review: bool = False
    specialists_review_in_parallel: bool = True
    leader_sees_all_specialist_reports: bool = True
    rotation_order: list[str] = Field(default_factory=list)
    review_template: str | None = None
    convergence_template: str | None = None
    rebuttal_template: str | None = None
    synthesis_template: str | None = None


class TokenBudgetConfig(BaseModel, frozen=True):
    """Estimated token budget for a variant."""

    estimated_input: int = 50_000
    estimated_output: int = 30_000


class VariantAgentsConfig(BaseModel, frozen=True):
    """All agents in a variant (flexible structure per variant type)."""

    leader: AgentConfig | None = None
    designer: AgentConfig | None = None
    reviewers: AgentConfig | None = None
    parallel_leaders: AgentConfig | None = None
    merger: AgentConfig | None = None
    adversarial_reviewer: AgentConfig | None = None
    specialists: list[SpecialistConfig] = Field(default_factory=list)
    participants: AgentConfig | None = None
    debaters: AgentConfig | None = None
    judge: AgentConfig | None = None
    evaluator: AgentConfig | None = None


class VariantConfig(BaseModel, frozen=True):
    """Configuration for a single consortium variant."""

    id: str
    name: str
    description: str = ""
    matrix_position: MatrixPositionConfig = MatrixPositionConfig()
    sub_variant: str | None = None
    agents: VariantAgentsConfig = VariantAgentsConfig()
    workflow: WorkflowConfig = WorkflowConfig()
    token_budget: TokenBudgetConfig = TokenBudgetConfig()


# ── Task Config ──────────────────────────────────────────────────────────────


class TaskVariables(BaseModel, frozen=True):
    """Template variables for task prompts."""

    system_name: str = ""
    problem_statement: str = ""
    hard_constraints: list[str] = Field(default_factory=list)
    use_cases: list[str] = Field(default_factory=list)
    complexity_drivers: list[str] = Field(default_factory=list)


class TaskConfig(BaseModel, frozen=True):
    """Configuration for a single design task."""

    id: str
    name: str
    complexity: str = "medium"  # "simple" | "medium" | "complex"
    design_type: str = "system"  # "system" | "application"
    rubric: str = ""
    prompt_template: str = ""
    variables: TaskVariables = TaskVariables()


# ── Rubric Config ────────────────────────────────────────────────────────────


class RubricDimensionConfig(BaseModel, frozen=True):
    """A single dimension in an evaluation rubric."""

    id: str
    name: str
    weight: float = 1.0
    description: str = ""
    anchors: dict[str, str] = Field(default_factory=dict)  # score -> description


class RubricConfig(BaseModel, frozen=True):
    """Evaluation rubric with weighted dimensions."""

    id: str
    name: str
    description: str = ""
    coherence_pairs: list[list[str]] = Field(default_factory=list)
    dimensions: list[RubricDimensionConfig] = Field(default_factory=list)


# ── Evaluator Config ─────────────────────────────────────────────────────────


class CoherenceCheckConfig(BaseModel, frozen=True):
    """Configuration for coherence checking."""

    enabled: bool = True
    prompt_template: str = "evaluation/coherence_check.j2"
    section_pairs: list[list[str]] = Field(default_factory=list)


class ReliabilityConfig(BaseModel, frozen=True):
    """Reliability validation thresholds."""

    min_krippendorff_alpha: float = 0.7
    human_validation_sample_size: int = 20


class EvaluatorBatchConfig(BaseModel, frozen=True):
    """Batch settings for evaluation."""

    enabled: bool = True
    batch_size: int = 50


class EvaluatorConfig(BaseModel, frozen=True):
    """Configuration for the evaluation pipeline."""

    model: str = "sonnet-4.5"
    runs_per_design: int = 3
    aggregation: str = "median"
    prompt_template: str = "evaluation/evaluate_design.j2"
    parameters: ModelParametersConfig = ModelParametersConfig(temperature=0.2, max_tokens=4096)
    coherence_check: CoherenceCheckConfig = CoherenceCheckConfig()
    reliability: ReliabilityConfig = ReliabilityConfig()
    batch: EvaluatorBatchConfig = EvaluatorBatchConfig()


# ── Experiment Config (top-level) ────────────────────────────────────────────


class BatchConfig(BaseModel, frozen=True):
    """Batch API settings."""

    enabled: bool = True
    max_concurrent_batches: int = 5
    poll_interval_seconds: int = 60
    fallback_to_realtime: bool = True


class LimitsConfig(BaseModel, frozen=True):
    """Safety limits for the experiment."""

    max_tokens_per_run: int = 500_000
    max_cost_per_run_usd: float = 2.00
    max_total_cost_usd: float = 500.00
    max_retries: int = 3
    retry_backoff_base: float = 2.0
    timeout_seconds: int = 300
    max_concurrent_runs: int = 5


class DefaultsConfig(BaseModel, frozen=True):
    """Default model assignments."""

    designer_model: str = "gpt-4.1"
    reviewer_model: str = "gpt-4.1"
    evaluator_model: str = "sonnet-4.5"


class ExperimentConfig(BaseModel, frozen=True):
    """Top-level experiment configuration."""

    name: str = "llm-consortium-v1"
    seed: int = 42
    output_dir: str = "data/"
    database: str = "data/consortium.db"

    variants: list[str] = Field(default_factory=list)
    tasks: list[str] = Field(default_factory=list)
    repetitions: int = 5

    defaults: DefaultsConfig = DefaultsConfig()
    batch: BatchConfig = BatchConfig()
    limits: LimitsConfig = LimitsConfig()


# ── Full Config (all configs merged) ─────────────────────────────────────────


class FullConfig(BaseModel, frozen=True):
    """Complete merged configuration for the experiment."""

    experiment: ExperimentConfig
    models: dict[str, ModelConfig] = Field(default_factory=dict)
    variants: dict[str, VariantConfig] = Field(default_factory=dict)
    tasks: dict[str, TaskConfig] = Field(default_factory=dict)
    rubrics: dict[str, RubricConfig] = Field(default_factory=dict)
    evaluator: EvaluatorConfig = EvaluatorConfig()

    def get_model(self, model_id: str) -> ModelConfig:
        """Get a model config by ID, raising if not found."""
        if model_id not in self.models:
            available = ", ".join(sorted(self.models.keys()))
            msg = f"Model '{model_id}' not found. Available: {available}"
            raise KeyError(msg)
        return self.models[model_id]

    def get_variant(self, variant_id: str) -> VariantConfig:
        """Get a variant config by ID, raising if not found."""
        if variant_id not in self.variants:
            available = ", ".join(sorted(self.variants.keys()))
            msg = f"Variant '{variant_id}' not found. Available: {available}"
            raise KeyError(msg)
        return self.variants[variant_id]

    def get_task(self, task_id: str) -> TaskConfig:
        """Get a task config by ID, raising if not found."""
        if task_id not in self.tasks:
            available = ", ".join(sorted(self.tasks.keys()))
            msg = f"Task '{task_id}' not found. Available: {available}"
            raise KeyError(msg)
        return self.tasks[task_id]

    def get_rubric(self, rubric_id: str) -> RubricConfig:
        """Get a rubric config by ID, raising if not found."""
        if rubric_id not in self.rubrics:
            available = ", ".join(sorted(self.rubrics.keys()))
            msg = f"Rubric '{rubric_id}' not found. Available: {available}"
            raise KeyError(msg)
        return self.rubrics[rubric_id]
