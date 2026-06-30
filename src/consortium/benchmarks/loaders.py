"""Dataset loaders: official benchmarks -> coding ``TaskConfig`` lists.

Each loader returns ``list[TaskConfig]`` with ``task_type="coding"`` and an
embedded :class:`CodingProblemConfig`. The benchmark's *hidden* grading tests
are never put in the task — they live in the official harness (scored later by
``bench score``). For function-level benchmarks the example tests are already in
the problem prompt/docstring, so ``visible_tests`` stays empty.

Loaders are intentionally lazy about heavy imports (``evalplus`` etc.) so that
importing this module never requires the benchmark packages to be installed.
"""

from __future__ import annotations

from consortium.config.models import CodingProblemConfig, TaskConfig

#: Benchmarks with a registered loader (function-level land first).
SUPPORTED = ("humanevalplus", "mbppplus")


def _safe_task_id(problem_id: str) -> str:
    """Turn an official problem id (``HumanEval/0``) into a DB-safe task id."""
    return problem_id.replace("/", "_").replace(" ", "_").lower()


def load_humanevalplus(limit: int | None = None) -> list[TaskConfig]:
    """Load HumanEval+ (EvalPlus) as coding tasks.

    The prompt is the function signature + docstring (which already contains the
    public doctest examples); the official base+plus tests stay hidden in the
    EvalPlus harness.
    """
    from evalplus.data import get_human_eval_plus

    data = get_human_eval_plus()
    tasks: list[TaskConfig] = []
    for problem_id, p in data.items():
        tasks.append(
            TaskConfig(
                id=_safe_task_id(problem_id),
                name=problem_id,
                task_type="coding",
                coding=CodingProblemConfig(
                    id=problem_id,
                    benchmark="humanevalplus",
                    prompt=p["prompt"],
                    entry_point=p.get("entry_point", ""),
                ),
            )
        )
    tasks.sort(key=lambda t: t.id)
    return tasks[:limit] if limit else tasks


def load_mbppplus(limit: int | None = None) -> list[TaskConfig]:
    """Load MBPP+ (EvalPlus) as coding tasks."""
    from evalplus.data import get_mbpp_plus

    data = get_mbpp_plus()
    tasks: list[TaskConfig] = []
    for problem_id, p in data.items():
        tasks.append(
            TaskConfig(
                id=_safe_task_id(problem_id),
                name=problem_id,
                task_type="coding",
                coding=CodingProblemConfig(
                    id=problem_id,
                    benchmark="mbppplus",
                    prompt=p["prompt"],
                    entry_point=p.get("entry_point", ""),
                ),
            )
        )
    tasks.sort(key=lambda t: t.id)
    return tasks[:limit] if limit else tasks


_LOADERS = {
    "humanevalplus": load_humanevalplus,
    "mbppplus": load_mbppplus,
}


def load_benchmark(name: str, limit: int | None = None) -> list[TaskConfig]:
    """Load a benchmark's problems as coding tasks.

    Args:
        name: Benchmark key (see :data:`SUPPORTED`).
        limit: Optional cap on the number of problems (for pilot subsets).

    Raises:
        KeyError: If *name* has no registered loader.
    """
    try:
        loader = _LOADERS[name]
    except KeyError:
        available = ", ".join(sorted(_LOADERS))
        msg = f"no loader for benchmark {name!r} (available: {available})"
        raise KeyError(msg) from None
    return loader(limit)
