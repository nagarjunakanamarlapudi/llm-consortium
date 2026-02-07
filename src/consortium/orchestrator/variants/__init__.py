"""Variant orchestrator implementations (v1–v8)."""

from consortium.orchestrator.variants.base import VariantOrchestrator
from consortium.orchestrator.variants.v1_baseline import V1BaselineOrchestrator
from consortium.orchestrator.variants.v2_leader_reviewers import V2LeaderReviewersOrchestrator
from consortium.orchestrator.variants.v3_parallel_merge import V3ParallelMergeOrchestrator
from consortium.orchestrator.variants.v4_adversarial import V4AdversarialOrchestrator
from consortium.orchestrator.variants.v5_specialist_panel import V5SpecialistPanelOrchestrator
from consortium.orchestrator.variants.v6_rotating_leader import V6RotatingLeaderOrchestrator
from consortium.orchestrator.variants.v7_consensus import V7ConsensusOrchestrator
from consortium.orchestrator.variants.v8_structured_debate import V8StructuredDebateOrchestrator

__all__ = [
    "V1BaselineOrchestrator",
    "V2LeaderReviewersOrchestrator",
    "V3ParallelMergeOrchestrator",
    "V4AdversarialOrchestrator",
    "V5SpecialistPanelOrchestrator",
    "V6RotatingLeaderOrchestrator",
    "V7ConsensusOrchestrator",
    "V8StructuredDebateOrchestrator",
    "VariantOrchestrator",
]
