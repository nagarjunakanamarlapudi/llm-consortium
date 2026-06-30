"""Benchmark harness runners and their registry.

A :class:`HarnessRunner` wraps a benchmark's grading protocol: it drives a
:class:`~consortium.execution.base.Sandbox` and maps the raw outcome onto an
:class:`~consortium.execution.base.ExecResult`. Runners are looked up by
benchmark name via :func:`get_runner`; a runner *instance* is registered with
:func:`register_runner`.

Only :class:`FunctionTestRunner` (benchmark ``"function-smoke"``) ships here —
a minimal assert-based smoke runner used by tests and pipeline scaffolding.
The official benchmark runners (LiveCodeBench, BigCodeBench, EvalPlus,
SWE-bench) land in later phases and will register themselves under their own
names.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, ClassVar

import structlog

from consortium.execution.base import ErrorType, ExecLimits, ExecResult

if TYPE_CHECKING:
    from consortium.config.models import CodingProblemConfig
    from consortium.execution.base import Sandbox

logger = structlog.get_logger()

# ── Driver exit codes ────────────────────────────────────────────────────────
# The smoke driver communicates the outcome class purely via its exit code so
# the runner can map it without parsing free-form output. Values avoid 1 and 2
# (used by the interpreter for uncaught exceptions / CLI errors) so that an
# unexpected driver crash falls through to the "runtime" bucket.
EXIT_PASS = 0
EXIT_COMPILE = 11
EXIT_WRONG_ANSWER = 12
EXIT_RUNTIME = 13


# The smoke-test driver. Written verbatim to the sandbox; reads the solution
# and tests from sibling files (so arbitrary user code never needs escaping)
# and exits with a distinct code per outcome class.
_SMOKE_DRIVER = """\
import sys

EXIT_PASS = 0
EXIT_COMPILE = 11
EXIT_WRONG_ANSWER = 12
EXIT_RUNTIME = 13


def _read(path):
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def main():
    solution_src = _read("solution.py")
    test_src = _read("tests.py")
    namespace = {"__name__": "__sandbox_solution__"}

    try:
        solution_code = compile(solution_src, "solution.py", "exec")
    except SyntaxError:
        return EXIT_COMPILE

    try:
        exec(solution_code, namespace)
    except Exception:
        import traceback
        traceback.print_exc()
        return EXIT_RUNTIME

    try:
        test_code = compile(test_src, "tests.py", "exec")
    except SyntaxError:
        return EXIT_COMPILE

    try:
        exec(test_code, namespace)
    except AssertionError:
        import traceback
        traceback.print_exc()
        return EXIT_WRONG_ANSWER
    except Exception:
        import traceback
        traceback.print_exc()
        return EXIT_RUNTIME

    return EXIT_PASS


if __name__ == "__main__":
    sys.exit(main())
"""


#: Max characters of captured output retained in :attr:`ExecResult.stdout_tail`.
_STDOUT_TAIL_CHARS = 4000


def _tail(text: str, limit: int) -> str:
    """Return the trailing *limit* characters of *text*."""
    if limit <= 0 or len(text) <= limit:
        return text
    return text[-limit:]


class HarnessRunner(abc.ABC):
    """Maps a graded solution to an :class:`ExecResult` for one benchmark.

    Subclasses set :attr:`benchmark` and implement :meth:`evaluate`, which
    drives a :class:`Sandbox` and interprets the raw outcome (exit code /
    timeout / output) into the canonical pass-fail result.
    """

    #: Benchmark name this runner grades (used as its registry key).
    benchmark: ClassVar[str] = ""

    #: Whether this runner's results may be reported as official leaderboard
    #: pass@1. Official benchmark harnesses set this ``True``; scaffolding and
    #: smoke runners leave it ``False`` so their results can never be mistaken
    #: for a published number.
    reportable: ClassVar[bool] = False

    @abc.abstractmethod
    def evaluate(
        self,
        problem: CodingProblemConfig,
        solution_code: str,
        sandbox: Sandbox,
        limits: ExecLimits | None = None,
    ) -> ExecResult:
        """Grade *solution_code* against *problem* inside *sandbox*."""
        ...


# ── Registry ─────────────────────────────────────────────────────────────────

_RUNNERS: dict[str, HarnessRunner] = {}


def register_runner(r: HarnessRunner) -> None:
    """Register runner instance *r* under its :attr:`~HarnessRunner.benchmark`."""
    _RUNNERS[r.benchmark] = r


def get_runner(benchmark: str) -> HarnessRunner:
    """Return the runner instance registered for *benchmark*.

    Raises:
        KeyError: If no runner is registered for *benchmark*.
    """
    try:
        return _RUNNERS[benchmark]
    except KeyError:
        available = ", ".join(sorted(_RUNNERS)) or "<none>"
        msg = f"no harness runner registered for benchmark {benchmark!r} (available: {available})"
        raise KeyError(msg) from None


def available_runners() -> tuple[str, ...]:
    """Return the sorted tuple of registered benchmark names."""
    return tuple(sorted(_RUNNERS))


# ── Runners ──────────────────────────────────────────────────────────────────


class FunctionTestRunner(HarnessRunner):
    """Assert-based smoke runner: exec the solution, then run the tests.

    The solution and the test snippet share one namespace, so tests reference
    the solution's names directly (e.g. ``assert add(1, 2) == 3``). The whole
    submission counts as a single test case: ``n_total == 1`` and ``n_pass``
    is ``1`` on success else ``0``. Outcomes map to :class:`ErrorType` as:

    * exit 0 -> ``none`` (passed)
    * sandbox timeout -> ``timeout``
    * ``SyntaxError`` in the solution or tests -> ``compile``
    * ``AssertionError`` from the tests -> ``wrong_answer``
    * any other exception (or unexpected exit) -> ``runtime``
    """

    benchmark = "function-smoke"

    #: Smoke runner — its results are never a reportable leaderboard pass@1.
    reportable = False

    def evaluate(
        self,
        problem: CodingProblemConfig,
        solution_code: str,
        sandbox: Sandbox,
        limits: ExecLimits | None = None,
    ) -> ExecResult:
        limits = limits or ExecLimits()
        result = sandbox.run(
            files={
                "solution.py": solution_code,
                "tests.py": problem.visible_tests,
                "_driver.py": _SMOKE_DRIVER,
            },
            command=[sandbox.python_cmd, "_driver.py"],
            limits=limits,
        )

        if result.timed_out:
            error_type: ErrorType = "timeout"
        elif result.exit_code == EXIT_PASS:
            error_type = "none"
        elif result.exit_code == EXIT_COMPILE:
            error_type = "compile"
        elif result.exit_code == EXIT_WRONG_ANSWER:
            error_type = "wrong_answer"
        else:
            # EXIT_RUNTIME, signal deaths (negative codes), or any unexpected
            # nonzero exit all fall through to runtime.
            error_type = "runtime"

        passed = error_type == "none"
        combined = result.stdout
        if result.stderr:
            combined = f"{combined}\n{result.stderr}" if combined else result.stderr

        return ExecResult(
            passed=passed,
            n_pass=1 if passed else 0,
            n_total=1,
            error_type=error_type,
            exec_ms=result.duration_ms,
            stdout_tail=_tail(combined, _STDOUT_TAIL_CHARS),
        )


register_runner(FunctionTestRunner())
