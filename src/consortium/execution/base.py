"""Core data types and the abstract :class:`Sandbox` contract.

The execution module runs *model-generated code* to obtain an **objective**
pass/fail signal (Paper #2). It is split into two layers:

* A :class:`Sandbox` — a low-level primitive that materialises a set of files,
  runs a command under resource limits, and returns the *raw* outcome
  (:class:`SandboxResult`). It knows nothing about benchmarks or pass/fail.
* A ``HarnessRunner`` (see :mod:`consortium.execution.harness`) — wraps a
  benchmark's official protocol, drives a :class:`Sandbox`, and maps the raw
  outcome onto a benchmark-level :class:`ExecResult`.

This separation keeps the security-sensitive execution primitive small and
auditable while letting each benchmark plug in its own scoring logic.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

#: Canonical execution error taxonomy persisted to ``code_results.error_type``.
ErrorType = Literal["none", "compile", "wrong_answer", "timeout", "runtime"]


@dataclass(frozen=True)
class ExecLimits:
    """Resource caps applied to a single sandboxed execution.

    * ``timeout_s`` — wall-clock budget; enforced by every backend.
    * ``memory_mb`` — memory cap (a hard container limit; best-effort POSIX
      ``RLIMIT_AS`` locally).
    * ``cpus`` — fractional CPU cap (a hard container limit).

    :class:`~consortium.execution.local_sandbox.LocalSandbox` enforces the
    wall-clock ``timeout_s`` and applies ``memory_mb`` as a best-effort POSIX
    ``rlimit``; :class:`~consortium.execution.docker_sandbox.DockerSandbox`
    enforces all three as hard container limits.
    """

    timeout_s: float = 30.0
    memory_mb: int = 1024
    cpus: float = 1.0


@dataclass(frozen=True)
class SandboxResult:
    """Raw outcome of running a command inside a :class:`Sandbox`."""

    exit_code: int
    stdout: str
    stderr: str
    duration_ms: float
    timed_out: bool


@dataclass(frozen=True)
class ExecResult:
    """Benchmark-level result for a single graded solution.

    Produced by a ``HarnessRunner`` after interpreting a
    :class:`SandboxResult`. Mirrors the ``code_results`` table (schema v4).
    """

    passed: bool
    n_pass: int
    n_total: int
    error_type: ErrorType
    exec_ms: float
    stdout_tail: str = ""
    detail: str = ""


class Sandbox(abc.ABC):
    """Abstract isolated execution environment for untrusted code.

    Implementations materialise *files* into an ephemeral working directory,
    execute *command* there under *limits*, and return the raw outcome. The
    working directory is the process ``cwd``, so relative paths in *command*
    and *files* resolve against it.
    """

    #: How to invoke the Python interpreter inside this sandbox. Subclasses
    #: override (e.g. the host ``sys.executable`` locally, ``"python"`` in a
    #: container image) so callers can stay backend-agnostic.
    python_cmd: str = "python3"

    @abc.abstractmethod
    def run(
        self,
        *,
        files: Mapping[str, str],
        command: Sequence[str],
        limits: ExecLimits | None = None,
    ) -> SandboxResult:
        """Run *command* with *files* written into a fresh working directory.

        Args:
            files: Mapping of relative path -> file contents to materialise in
                the working directory before execution. Paths must stay within
                the working directory.
            command: argv of the process to execute (no shell).
            limits: Resource caps; ``None`` uses :class:`ExecLimits` defaults.

        Returns:
            The raw :class:`SandboxResult`.
        """
        ...
