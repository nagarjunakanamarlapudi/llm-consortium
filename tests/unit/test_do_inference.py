"""Tests for DO-inference sampling-parameter normalization."""

from __future__ import annotations

from consortium.providers.openai import _clean_sampling_params


def test_drops_top_p_one_when_temperature_present() -> None:
    # Some DO-backed models reject temperature+top_p together; top_p=1.0 is a no-op.
    out = _clean_sampling_params({"temperature": 0.0, "top_p": 1.0, "max_tokens": 10})
    assert "top_p" not in out
    assert out["temperature"] == 0.0
    assert out["max_tokens"] == 10


def test_keeps_non_unit_top_p() -> None:
    out = _clean_sampling_params({"temperature": 0.5, "top_p": 0.9})
    assert out["top_p"] == 0.9


def test_keeps_top_p_when_no_temperature() -> None:
    # Only drop the redundant pairing; a lone top_p is left untouched.
    out = _clean_sampling_params({"top_p": 1.0})
    assert out["top_p"] == 1.0
