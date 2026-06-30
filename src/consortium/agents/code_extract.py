"""Code extraction — pull code and diffs out of free-form LLM responses.

Mirrors the markdown-fence parsing style of
:mod:`consortium.evaluation.pipeline` (see ``_JSON_FENCE``) but generalises it
to arbitrary languages and to unified diffs.
"""

from __future__ import annotations

import re

# Pattern to extract code from markdown code fences. Group 1 is the fence info
# string (the language tag); group 2 is the fenced body. Mirrors the fence
# style of consortium.evaluation.pipeline._JSON_FENCE.
_CODE_FENCE = re.compile(r"```([^\n]*)\n(.*?)\n```", re.DOTALL)

# Line prefixes that mark the start of a unified diff / patch.
_DIFF_PREFIXES = ("diff --git", "--- ")


def _fenced_blocks(text: str) -> list[tuple[str, str]]:
    """Return ``(language, body)`` for each fenced code block in ``text``.

    ``language`` is the lower-cased first token of the fence info string, or
    ``""`` when the fence is unlabelled.
    """
    blocks: list[tuple[str, str]] = []
    for match in _CODE_FENCE.finditer(text):
        info = match.group(1).split()
        language = info[0].lower() if info else ""
        blocks.append((language, match.group(2)))
    return blocks


def _looks_like_diff(text: str) -> bool:
    """Return True when ``text`` begins with a unified-diff marker."""
    return text.lstrip().startswith(_DIFF_PREFIXES)


def extract_code(text: str, *, prefer_language: str | None = None) -> str:
    """Extract a code block from an LLM response.

    Returns the body of the largest fenced ``` block, breaking ties in favour
    of the last (latest) block. When ``prefer_language`` is supplied and at
    least one fenced block declares that language, selection is restricted to
    those blocks before the largest/last rule is applied. When the response
    contains no fenced block, the whole stripped text is returned.

    Args:
        text: The raw LLM response.
        prefer_language: Optional language hint (e.g. ``"python"``) used to
            disambiguate when several fenced blocks are present. Matching is
            case-insensitive.

    Returns:
        The extracted code, stripped of surrounding whitespace.
    """
    if not text:
        return ""

    blocks = _fenced_blocks(text)
    if not blocks:
        return text.strip()

    candidates = blocks
    if prefer_language:
        wanted = prefer_language.strip().lower()
        preferred = [block for block in blocks if block[0] == wanted]
        if preferred:
            candidates = preferred

    # Largest body wins; iterating in order with ">=" means a later block of
    # equal size overwrites an earlier one, so ties resolve to the last block.
    best_body = ""
    best_len = -1
    for _language, body in candidates:
        if len(body) >= best_len:
            best_len = len(body)
            best_body = body

    return best_body.strip()


def extract_diff(text: str) -> str:
    """Extract a unified-diff / patch block from an LLM response.

    Resolution order:

    1. The first fenced ``` block tagged ``diff``/``patch`` or whose body
       begins with a diff marker (``diff --git`` or ``--- ``).
    2. Otherwise, the first raw line that begins with a diff marker, through to
       the end of the text.
    3. Otherwise, ``""``.

    Args:
        text: The raw LLM response.

    Returns:
        The extracted diff, stripped of surrounding whitespace, or ``""`` when
        no diff is present.
    """
    if not text:
        return ""

    for language, body in _fenced_blocks(text):
        if language in {"diff", "patch"} or _looks_like_diff(body):
            return body.strip()

    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith(_DIFF_PREFIXES):
            return "\n".join(lines[index:]).strip()

    return ""
