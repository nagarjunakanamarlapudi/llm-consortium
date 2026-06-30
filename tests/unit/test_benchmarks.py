"""Tests for benchmark condition building and loader dispatch."""

from __future__ import annotations

import pytest

from consortium.benchmarks import build_conditions, load_benchmark
from consortium.benchmarks.conditions import ALL_CONDITIONS


def test_build_baseline_condition() -> None:
    conds = build_conditions(["base-sonnet"])
    assert set(conds) == {"v1a_sonnet"}
    vc = conds["v1a_sonnet"]
    assert vc.agents.designer is not None
    assert vc.agents.designer.model == "do-sonnet"
    assert vc.workflow.max_rounds == 0  # single shot


def test_build_consortium_conditions_use_distinct_models() -> None:
    conds = build_conditions(["c-xreview", "c-adv"])
    assert set(conds) == {"v2b_xreview", "v4b_adv"}
    xr = conds["v2b_xreview"]
    # cross-model: writer and reviewer must differ
    assert xr.agents.leader.model != xr.agents.reviewers.model
    adv = conds["v4b_adv"]
    assert adv.agents.leader.model != adv.agents.adversarial_reviewer.model


def test_variant_ids_resolve_to_orchestrators() -> None:
    # Every condition's variant_id must start with vN so the factory resolves it.
    conds = build_conditions(list(ALL_CONDITIONS))
    import re

    for vid in conds:
        assert re.match(r"^v\d+", vid), vid


def test_unknown_condition_raises() -> None:
    with pytest.raises(KeyError):
        build_conditions(["does-not-exist"])


def test_unknown_benchmark_raises() -> None:
    with pytest.raises(KeyError):
        load_benchmark("nonexistent-benchmark")
