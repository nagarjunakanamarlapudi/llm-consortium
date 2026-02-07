"""Jinja2 template loading and rendering for prompts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jinja2
import structlog

logger = structlog.get_logger()


class PromptRenderer:
    """Renders Jinja2 prompt templates from a directory.

    Templates are loaded from a base directory (typically ``prompts/``) and
    referenced by relative path, e.g. ``generation/design_system.j2``.
    """

    def __init__(self, template_dir: str | Path) -> None:
        self._template_dir = Path(template_dir)
        self._env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self._template_dir)),
            autoescape=False,  # prompts are plain text, not HTML
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
            undefined=jinja2.StrictUndefined,  # fail fast on missing variables
        )
        # Register custom filters
        self._env.filters["bullet_list"] = _bullet_list_filter
        self._env.filters["numbered_list"] = _numbered_list_filter

    def render(self, template_path: str, **variables: Any) -> str:
        """Render a template with the given variables.

        Args:
            template_path: Relative path to the template, e.g.
                ``generation/design_system.j2``.
            **variables: Template variables to inject.

        Returns:
            Rendered prompt string.

        Raises:
            jinja2.TemplateNotFound: If the template file doesn't exist.
            jinja2.UndefinedError: If a required variable is missing.
        """
        template = self._env.get_template(template_path)
        rendered = template.render(**variables)
        logger.debug(
            "prompt_rendered",
            template=template_path,
            length=len(rendered),
        )
        return rendered

    def list_templates(self) -> list[str]:
        """List all available template paths."""
        return sorted(self._env.list_templates(extensions=["j2"]))

    def template_exists(self, template_path: str) -> bool:
        """Check if a template exists."""
        try:
            self._env.get_template(template_path)
        except jinja2.TemplateNotFound:
            return False
        return True


def _bullet_list_filter(items: list[str]) -> str:
    """Jinja2 filter: render a list as a bulleted list."""
    return "\n".join(f"- {item}" for item in items)


def _numbered_list_filter(items: list[str]) -> str:
    """Jinja2 filter: render a list as a numbered list."""
    return "\n".join(f"{i}. {item}" for i, item in enumerate(items, 1))
