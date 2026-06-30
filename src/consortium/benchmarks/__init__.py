"""Coding-benchmark dataset loaders and consortium condition builders (Paper #2).

A *loader* turns an official benchmark dataset into a list of coding
:class:`~consortium.config.models.TaskConfig` objects (``task_type="coding"``)
that the existing orchestrator can run. A *condition* is an in-memory
:class:`~consortium.config.models.VariantConfig` describing one experimental arm
(a single-model baseline or a heterogeneous consortium topology) over the DO
inference model roster.

Keeping conditions in code (rather than a combinatorial explosion of YAML files)
makes the model roster a parameter and keeps baselines and consortium arms in
one place.
"""

from __future__ import annotations

from consortium.benchmarks.conditions import (
    ALL_CONDITIONS,
    ROSTER,
    build_conditions,
)
from consortium.benchmarks.loaders import load_benchmark

__all__ = ["ALL_CONDITIONS", "ROSTER", "build_conditions", "load_benchmark"]
