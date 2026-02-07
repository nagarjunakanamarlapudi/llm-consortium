"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"
PROMPTS_DIR = REPO_ROOT / "prompts"


@pytest.fixture
def configs_dir() -> Path:
    return CONFIGS_DIR


@pytest.fixture
def prompts_dir() -> Path:
    return PROMPTS_DIR


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "test.db"
