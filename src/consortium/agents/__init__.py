"""Agent layer — role-specific LLM agents for consortium workflows."""

from consortium.agents.adversary import AdversarialReviewer
from consortium.agents.base import BaseAgent
from consortium.agents.designer import DesignerAgent
from consortium.agents.judge import JudgeAgent
from consortium.agents.merger import MergerAgent
from consortium.agents.reviewer import ReviewerAgent
from consortium.agents.specialist import SpecialistReviewer

__all__ = [
    "AdversarialReviewer",
    "BaseAgent",
    "DesignerAgent",
    "JudgeAgent",
    "MergerAgent",
    "ReviewerAgent",
    "SpecialistReviewer",
]
