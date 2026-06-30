"""Hardened Docker sandbox — the default for grading model-generated code.

Each execution runs in a fresh, named container with: no network, a read-only
root filesystem (the code is bind-mounted read-only; ``/tmp`` is a writable
tmpfs), an unprivileged user, and hard CPU / memory / PID caps. A wall-clock
timeout triggers a forced teardown (``docker kill`` + ``docker rm -f``) so no
container is leaked.

The exact ``docker run`` argv is produced by the pure
:meth:`DockerSandbox._build_command` helper, which has no side effects and can
be unit-tested without Docker installed.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from consortium.execution.base import ExecLimits, Sandbox, SandboxResult
from consortium.execution.local_sandbox import _materialize

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

logger = structlog.get_logger()

#: Default container image. ``python`` is on PATH inside the official images.
#: TODO(phase-1): pin by digest (python:3.11-slim@sha256:...) before routing any
#: model-generated code through Docker, so a moved/compromised tag can't be run.
DEFAULT_IMAGE = "python:3.11-slim"

#: Read-only mount point for the materialised code (also the container workdir).
_CONTAINER_SRC = "/work"

#: Hard cap on the number of processes/threads allowed inside the container.
_PIDS_LIMIT = 128

#: Short timeout (seconds) for the teardown ``docker`` calls themselves.
_TEARDOWN_TIMEOUT = 30.0


def _as_text(value: str | bytes | None) -> str:
    """Normalise captured stream output (possibly bytes/None) to ``str``."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value


class DockerSandbox(Sandbox):
    """Run code in a hardened, ephemeral Docker container.

    Args:
        image: Container image providing a Python interpreter on PATH.
        docker_path: Explicit path to the ``docker`` binary (defaults to the
            first ``docker`` found on PATH).

    Raises:
        RuntimeError: If no ``docker`` executable can be located.
    """

    python_cmd = "python"

    def __init__(
        self,
        *,
        image: str = DEFAULT_IMAGE,
        docker_path: str | None = None,
    ) -> None:
        resolved = docker_path or shutil.which("docker")
        if resolved is None:
            msg = (
                "docker executable not found on PATH; DockerSandbox requires "
                "Docker to be installed and runnable."
            )
            raise RuntimeError(msg)
        self._docker = resolved
        self.image = image

    @staticmethod
    def _build_command(
        *,
        image: str,
        name: str,
        host_dir: str,
        command: Sequence[str],
        limits: ExecLimits,
        docker: str = "docker",
        container_dir: str = _CONTAINER_SRC,
    ) -> list[str]:
        """Build the hardened ``docker run`` argv (pure, no side effects).

        The container is locked down with: ``--network none`` (no network),
        ``--read-only`` root FS with a writable ``/tmp`` tmpfs, an
        unprivileged ``--user``, swap disabled, and ``--memory`` / ``--cpus``
        / ``--pids-limit`` hard caps. The host code directory is bind-mounted
        read-only at *container_dir*, which is also the working directory.
        """
        mem = f"{limits.memory_mb}m"
        return [
            docker,
            "run",
            "--rm",
            "--name",
            name,
            "--network",
            "none",
            "--cap-drop",
            "ALL",  # drop all Linux capabilities (the smoke driver needs none)
            "--security-opt",
            "no-new-privileges",  # block setuid-root escalation inside the container
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,nodev,size=64m",
            "--user",
            "65534:65534",  # nobody:nogroup
            "--memory",
            mem,
            "--memory-swap",
            mem,  # equal to --memory disables swap
            "--cpus",
            str(limits.cpus),
            "--pids-limit",
            str(_PIDS_LIMIT),
            "--workdir",
            container_dir,
            "--volume",
            f"{host_dir}:{container_dir}:ro",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            "HOME=/tmp",
            image,
            *command,
        ]

    def _teardown(self, name: str) -> None:
        """Force-stop and remove a named container (best-effort)."""
        for argv in (["kill", name], ["rm", "-f", name]):
            try:
                subprocess.run(  # fixed argv, shell=False
                    [self._docker, *argv],
                    capture_output=True,
                    timeout=_TEARDOWN_TIMEOUT,
                    check=False,
                )
            except (subprocess.SubprocessError, OSError):  # pragma: no cover - best effort
                logger.warning("docker_teardown_failed", container=name, action=argv[0])

    def run(
        self,
        *,
        files: Mapping[str, str],
        command: Sequence[str],
        limits: ExecLimits | None = None,
    ) -> SandboxResult:
        limits = limits or ExecLimits()
        name = f"consortium-sbx-{uuid.uuid4().hex}"
        log = logger.bind(sandbox="docker", container=name, image=self.image)

        with tempfile.TemporaryDirectory(prefix="consortium-dkr-") as tmp:
            _materialize(Path(tmp), files)
            argv = self._build_command(
                image=self.image,
                name=name,
                host_dir=tmp,
                command=command,
                limits=limits,
                docker=self._docker,
            )

            start = time.monotonic()
            timed_out = False
            try:
                completed = subprocess.run(  # argv list, shell=False
                    argv,
                    capture_output=True,
                    text=True,
                    timeout=limits.timeout_s,
                    check=False,
                )
                returncode = completed.returncode
                stdout = completed.stdout
                stderr = completed.stderr
            except subprocess.TimeoutExpired as exc:
                timed_out = True
                returncode = -1
                stdout = _as_text(exc.stdout)
                stderr = _as_text(exc.stderr)
                log.warning("sandbox_timeout", timeout_s=limits.timeout_s)
                self._teardown(name)
            except BaseException:
                # Any other failure (OSError, KeyboardInterrupt, ...) leaves the
                # daemon-managed container running despite --rm; tear it down
                # before propagating so no container is leaked.
                self._teardown(name)
                raise

            duration_ms = (time.monotonic() - start) * 1000.0

        return SandboxResult(
            exit_code=returncode,
            stdout=stdout or "",
            stderr=stderr or "",
            duration_ms=duration_ms,
            timed_out=timed_out,
        )
