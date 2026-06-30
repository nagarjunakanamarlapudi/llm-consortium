"""Sandboxed execution of model-generated code for objective grading.

Public surface:

* :class:`Sandbox` / :class:`SandboxResult` — the execution primitive and its
  raw outcome.
* :class:`LocalSandbox` — host subprocess sandbox (DEV/TEST ONLY).
* :class:`DockerSandbox` — hardened container sandbox (default for grading).
* :class:`ExecLimits` / :class:`ExecResult` / :data:`ErrorType` — shared
  data types.
* :class:`HarnessRunner` plus :func:`register_runner` / :func:`get_runner` /
  :func:`available_runners` — the benchmark-runner registry.
* :class:`FunctionTestRunner` — the built-in ``"function-smoke"`` runner
  (importing this package registers it).
"""

from __future__ import annotations

from consortium.execution.base import (
    ErrorType,
    ExecLimits,
    ExecResult,
    Sandbox,
    SandboxResult,
)
from consortium.execution.docker_sandbox import DockerSandbox
from consortium.execution.harness import (
    FunctionTestRunner,
    HarnessRunner,
    available_runners,
    get_runner,
    register_runner,
)
from consortium.execution.local_sandbox import LocalSandbox

__all__ = [
    "DockerSandbox",
    "ErrorType",
    "ExecLimits",
    "ExecResult",
    "FunctionTestRunner",
    "HarnessRunner",
    "LocalSandbox",
    "Sandbox",
    "SandboxResult",
    "available_runners",
    "get_runner",
    "register_runner",
]
