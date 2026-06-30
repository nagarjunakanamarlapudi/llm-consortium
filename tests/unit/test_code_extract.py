"""Tests for code/diff extraction from LLM responses."""

from __future__ import annotations

from consortium.agents.code_extract import extract_code, extract_diff


def fence(body: str, language: str = "") -> str:
    """Build a markdown code fence with an optional language tag."""
    return f"```{language}\n{body}\n```"


class TestExtractCode:
    def test_single_block(self) -> None:
        text = "Here is the code:\n" + fence("print('hi')", "python")
        assert extract_code(text) == "print('hi')"

    def test_largest_block_wins_over_position(self) -> None:
        # The largest block comes first; the small block comes last.
        text = fence("the largest block of code here", "python") + "\n\n" + fence("tiny", "python")
        assert extract_code(text) == "the largest block of code here"

    def test_tie_breaks_to_last_block(self) -> None:
        # Equal-length bodies: the last one wins.
        text = fence("AAAA") + "\n\n" + fence("BBBB")
        assert extract_code(text) == "BBBB"

    def test_language_hint_overrides_size(self) -> None:
        text = (
            fence("a very long javascript snippet here", "javascript")
            + "\n"
            + fence("x = 1", "python")
        )
        # Without a hint, the larger javascript block wins.
        assert extract_code(text) == "a very long javascript snippet here"
        # With the hint, the smaller python block is selected.
        assert extract_code(text, prefer_language="python") == "x = 1"

    def test_language_hint_picks_largest_last_among_matches(self) -> None:
        text = (
            fence("short", "python")
            + "\n"
            + fence("a longer python block", "python")
            + "\n"
            + fence("some other code", "ruby")
        )
        assert extract_code(text, prefer_language="python") == "a longer python block"

    def test_language_hint_is_case_insensitive(self) -> None:
        text = fence("py code", "Python") + "\n" + fence("a longer go program", "go")
        assert extract_code(text, prefer_language="python") == "py code"

    def test_language_hint_no_match_falls_back(self) -> None:
        text = fence("aaa", "python") + "\n" + fence("bbbb longer", "ruby")
        # No block matches the hint, so the largest/last block is returned.
        assert extract_code(text, prefer_language="go") == "bbbb longer"

    def test_no_fence_returns_stripped_text(self) -> None:
        assert extract_code("just some plain text") == "just some plain text"
        assert extract_code("  spaced out  \n") == "spaced out"

    def test_empty_input(self) -> None:
        assert extract_code("") == ""
        assert extract_code("", prefer_language="python") == ""


class TestExtractDiff:
    diff_body = "diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old\n+new"
    unified_body = "--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old\n+new"

    def test_fenced_diff_label(self) -> None:
        text = "Here is the patch:\n" + fence(self.diff_body, "diff")
        assert extract_diff(text) == self.diff_body

    def test_fenced_patch_label(self) -> None:
        text = fence(self.diff_body, "patch")
        assert extract_diff(text) == self.diff_body

    def test_fenced_unlabelled_diff_body(self) -> None:
        text = fence(self.diff_body)
        assert extract_diff(text) == self.diff_body

    def test_raw_diff_git_to_end(self) -> None:
        text = "Here is the patch:\n\n" + self.diff_body
        assert extract_diff(text) == self.diff_body

    def test_raw_unified_header(self) -> None:
        text = "Changes below:\n" + self.unified_body
        assert extract_diff(text) == self.unified_body

    def test_fenced_takes_precedence_over_raw(self) -> None:
        body = "diff --git a/x b/x\n+added"
        text = "intro\n" + fence(body, "diff") + "\ntrailing prose"
        # The fenced body is returned, not the raw scan through the trailing prose.
        assert extract_diff(text) == body

    def test_horizontal_rule_is_not_a_diff(self) -> None:
        # A markdown rule ("---", no trailing space) must not match "--- ".
        assert extract_diff("title\n---\nmore text") == ""

    def test_no_diff_returns_empty(self) -> None:
        assert extract_diff("just prose, nothing to patch") == ""
        assert extract_diff(fence("print('hi')", "python")) == ""

    def test_empty_input(self) -> None:
        assert extract_diff("") == ""
