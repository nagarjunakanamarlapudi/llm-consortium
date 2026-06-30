"""Tests for the sandboxed execution module."""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import pytest

from consortium.config.models import CodingProblemConfig
from consortium.execution import (
    DockerSandbox,
    ExecLimits,
    ExecResult,
    FunctionTestRunner,
    LocalSandbox,
    Sandbox,
    SandboxResult,
    get_runner,
)
from consortium.execution.docker_sandbox import _CONTAINER_SRC, _PIDS_LIMIT
from consortium.execution.local_sandbox import _materialize

if TYPE_CHECKING:
    from pathlib import Path

_HAS_DOCKER = shutil.which("docker") is not None


def _flag_value(argv: list[str], flag: str) -> str:
    """Return the token immediately following *flag* in *argv*."""
    return argv[argv.index(flag) + 1]


class TestLocalSandbox:
    def test_runs_trivial_safe_code(self) -> None:
        sandbox = LocalSandbox()
        result = sandbox.run(
            files={"main.py": "print('hello-sandbox')"},
            command=[sandbox.python_cmd, "main.py"],
        )
        assert isinstance(result, SandboxResult)
        assert result.timed_out is False
        assert result.exit_code == 0
        assert "hello-sandbox" in result.stdout

    def test_wall_clock_timeout(self) -> None:
        sandbox = LocalSandbox()
        result = sandbox.run(
            files={"loop.py": "while True:\n    pass\n"},
            command=[sandbox.python_cmd, "loop.py"],
            limits=ExecLimits(timeout_s=1.0),
        )
        assert result.timed_out is True
        # Killed shortly after the 1s wall-clock budget, not hanging forever.
        assert result.duration_ms < 20_000


class TestFunctionTestRunner:
    def _grade(self, code: str, tests: str) -> ExecResult:
        runner = get_runner("function-smoke")
        assert isinstance(runner, FunctionTestRunner)
        problem = CodingProblemConfig(
            id="smoke",
            benchmark="function-smoke",
            prompt="",
            visible_tests=tests,
        )
        return runner.evaluate(problem, code, LocalSandbox(), ExecLimits(timeout_s=15.0))

    def test_pass(self) -> None:
        result = self._grade("def add(a, b):\n    return a + b\n", "assert add(1, 2) == 3\n")
        assert result.passed is True
        assert result.error_type == "none"
        assert result.n_pass == 1
        assert result.n_total == 1

    def test_compile_error(self) -> None:
        # Missing colon -> SyntaxError at compile time.
        result = self._grade("def add(a, b)\n    return a + b\n", "assert add(1, 2) == 3\n")
        assert result.passed is False
        assert result.error_type == "compile"
        assert result.n_pass == 0
        assert result.n_total == 1

    def test_wrong_answer(self) -> None:
        result = self._grade("def add(a, b):\n    return a - b\n", "assert add(1, 2) == 3\n")
        assert result.passed is False
        assert result.error_type == "wrong_answer"
        assert result.n_pass == 0

    def test_runtime_error(self) -> None:
        result = self._grade("def add(a, b):\n    return a / 0\n", "assert add(1, 2) == 3\n")
        assert result.passed is False
        assert result.error_type == "runtime"
        assert result.n_pass == 0


    def test_timeout_maps_to_timeout_error(self) -> None:
        """A sandbox that reports a timeout maps to error_type 'timeout', not runtime."""

        class _TimeoutSandbox(Sandbox):
            python_cmd = "python"

            def run(self, **_: object) -> SandboxResult:
                return SandboxResult(
                    exit_code=-1, stdout="", stderr="", duration_ms=1234.0, timed_out=True
                )

        runner = get_runner("function-smoke")
        problem = CodingProblemConfig(
            id="smoke", benchmark="function-smoke", prompt="", visible_tests="assert True"
        )
        result = runner.evaluate(problem, "def f():\n    pass\n", _TimeoutSandbox(), ExecLimits())
        assert result.passed is False
        assert result.error_type == "timeout"
        assert result.n_pass == 0
        assert result.exec_ms == pytest.approx(1234.0)


class TestMaterialize:
    def test_rejects_path_escape(self, tmp_path: Path) -> None:
        """_materialize refuses to write outside the sandbox root (``..`` guard)."""
        with pytest.raises(ValueError, match="outside sandbox"):
            _materialize(tmp_path, {"../escape.py": "x = 1"})

    def test_writes_nested_files(self, tmp_path: Path) -> None:
        _materialize(tmp_path, {"pkg/mod.py": "y = 2"})
        assert (tmp_path / "pkg" / "mod.py").read_text(encoding="utf-8") == "y = 2"


class TestRunnerRegistry:
    def test_unknown_benchmark_raises_keyerror(self) -> None:
        with pytest.raises(KeyError):
            get_runner("livecodebench")


class TestDockerSandboxBuildCommand:
    def test_hardened_flags_present(self) -> None:
        limits = ExecLimits(memory_mb=256, cpus=1.0)
        argv = DockerSandbox._build_command(
            image="python:3.11-slim",
            name="consortium-sbx-test",
            host_dir="/host/work",
            command=["python", "_driver.py"],
            limits=limits,
        )

        assert argv[:2] == ["docker", "run"]
        assert _flag_value(argv, "--network") == "none"
        assert "--read-only" in argv
        assert "--tmpfs" in argv
        assert _flag_value(argv, "--user").startswith("65534")
        assert _flag_value(argv, "--memory") == "256m"
        # Swap equal to memory disables swap entirely.
        assert _flag_value(argv, "--memory-swap") == "256m"
        # All Linux capabilities dropped and privilege escalation blocked.
        assert _flag_value(argv, "--cap-drop") == "ALL"
        assert _flag_value(argv, "--security-opt") == "no-new-privileges"
        assert _flag_value(argv, "--cpus") == "1.0"
        assert _flag_value(argv, "--pids-limit") == str(_PIDS_LIMIT)
        assert _flag_value(argv, "--name") == "consortium-sbx-test"
        # Code is bind-mounted read-only at the container workdir.
        assert _flag_value(argv, "--volume") == f"/host/work:{_CONTAINER_SRC}:ro"
        # Image and command come last, in order.
        assert argv[-3:] == ["python:3.11-slim", "python", "_driver.py"]

    def test_missing_docker_raises_runtimeerror(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "consortium.execution.docker_sandbox.shutil.which",
            lambda _name: None,
        )
        with pytest.raises(RuntimeError):
            DockerSandbox()


@pytest.mark.skipif(not _HAS_DOCKER, reason="docker not installed")
class TestDockerSandboxLive:
    def test_runs_trivial_code_in_container(self) -> None:
        sandbox = DockerSandbox()
        result = sandbox.run(
            files={"main.py": "print('hello-docker')"},
            command=[sandbox.python_cmd, "main.py"],
            limits=ExecLimits(timeout_s=120.0),
        )
        assert result.timed_out is False
        assert result.exit_code == 0
        assert "hello-docker" in result.stdout
