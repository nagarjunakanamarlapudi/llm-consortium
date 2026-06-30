"""Local subprocess sandbox — **DEV/TEST ONLY**.

.. danger::

   :class:`LocalSandbox` runs untrusted, model-generated code as a child
   process on the *host*. It is **not a security boundary**. It provides a
   wall-clock timeout and *best-effort* POSIX ``rlimit``\\ s (CPU time and
   address space) via ``preexec_fn``, but the code shares the
   host filesystem, network, and kernel. The rlimits are advisory: some
   (notably ``RLIMIT_AS`` on macOS) cannot be set and are silently skipped.

   Use it only for trusted/known-safe code and for the test suite. For grading
   real model output, use :class:`~consortium.execution.docker_sandbox.DockerSandbox`.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from consortium.execution.base import ExecLimits, Sandbox, SandboxResult

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

try:  # POSIX-only; absent on Windows.
    import resource as _resource
except ImportError:  # pragma: no cover - non-POSIX platforms
    _resource = None  # type: ignore[assignment]

logger = structlog.get_logger()

_MB = 1024 * 1024


def _materialize(root: Path, files: Mapping[str, str]) -> None:
    """Write *files* (relative path -> contents) under *root*.

    Rejects paths that would escape *root* (defence-in-depth against ``..``).
    """
    root_resolved = root.resolve()
    for rel, content in files.items():
        dest = (root / rel).resolve()
        if dest != root_resolved and root_resolved not in dest.parents:
            msg = f"refusing to write file outside sandbox: {rel!r}"
            raise ValueError(msg)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")


def _try_setrlimit(res: int, want: int) -> None:
    """Lower a single ``rlimit`` soft limit, preserving the hard limit.

    Best-effort: any failure (unsupported resource, insufficient privileges,
    platform quirks) is swallowed — this is not a security boundary.
    """
    if _resource is None:  # pragma: no cover - non-POSIX platforms
        return
    try:
        _soft, hard = _resource.getrlimit(res)
        new_soft = want if hard == _resource.RLIM_INFINITY else min(want, hard)
        _resource.setrlimit(res, (new_soft, hard))
    except (ValueError, OSError):  # pragma: no cover - platform dependent
        pass


def _build_preexec(limits: ExecLimits) -> Callable[[], None]:
    """Build a ``preexec_fn`` that applies best-effort rlimits in the child."""

    def _apply() -> None:  # runs post-fork, pre-exec, in the child process
        if _resource is None:  # pragma: no cover - non-POSIX platforms
            return
        # Derive a CPU-time ceiling from the wall-clock budget (+1s slack) and
        # cap address space from the memory budget; both are best-effort.
        _try_setrlimit(_resource.RLIMIT_CPU, int(limits.timeout_s) + 1)
        _try_setrlimit(_resource.RLIMIT_AS, limits.memory_mb * _MB)

    return _apply


def _kill_process_group(proc: subprocess.Popen[str]) -> None:
    """Best-effort kill of the child's whole process group."""
    if os.name == "posix":
        try:
            os.killpg(os.getpgid(proc.pid), 9)  # SIGKILL
            return
        except (ProcessLookupError, PermissionError, OSError):  # pragma: no cover
            pass
    with contextlib.suppress(ProcessLookupError):
        proc.kill()


def _restricted_env(workdir: str) -> dict[str, str]:
    """A minimal environment that hides host secrets but keeps Python working."""
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "HOME": workdir,
        "TMPDIR": workdir,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }


class LocalSandbox(Sandbox):
    """Run code in a temp directory as a host subprocess (DEV/TEST ONLY).

    Enforces a wall-clock timeout (the whole process group is killed on
    expiry) and applies best-effort POSIX rlimits. Not a security boundary —
    see the module docstring.
    """

    def __init__(self, *, python_cmd: str | None = None) -> None:
        # Default to the running interpreter so the sandbox is self-contained
        # and does not depend on a ``python`` entry being on PATH.
        self.python_cmd = python_cmd or sys.executable

    def run(
        self,
        *,
        files: Mapping[str, str],
        command: Sequence[str],
        limits: ExecLimits | None = None,
    ) -> SandboxResult:
        limits = limits or ExecLimits()
        log = logger.bind(sandbox="local", command=list(command))

        with tempfile.TemporaryDirectory(prefix="consortium-sbx-") as tmp:
            _materialize(Path(tmp), files)

            preexec = _build_preexec(limits) if os.name == "posix" else None
            start = time.monotonic()
            proc = subprocess.Popen(  # argv list, shell=False
                list(command),
                cwd=tmp,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=_restricted_env(tmp),
                preexec_fn=preexec,
                start_new_session=(os.name == "posix"),
            )

            timed_out = False
            try:
                stdout, stderr = proc.communicate(timeout=limits.timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                _kill_process_group(proc)
                try:
                    stdout, stderr = proc.communicate(timeout=5.0)
                except subprocess.TimeoutExpired:
                    # A double-forked grandchild can survive the killpg and hold
                    # the pipes open; don't hang the caller — drop captured output.
                    stdout, stderr = "", ""
                log.warning("sandbox_timeout", timeout_s=limits.timeout_s)

            duration_ms = (time.monotonic() - start) * 1000.0
            returncode = proc.returncode if proc.returncode is not None else -1

        return SandboxResult(
            exit_code=returncode,
            stdout=stdout or "",
            stderr=stderr or "",
            duration_ms=duration_ms,
            timed_out=timed_out,
        )
