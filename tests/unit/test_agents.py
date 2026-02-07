"""Tests for prompt renderer."""

from __future__ import annotations

from pathlib import Path

import pytest

from consortium.prompts.renderer import PromptRenderer


class TestPromptRenderer:
    def test_list_templates(self, prompts_dir: Path) -> None:
        renderer = PromptRenderer(prompts_dir)
        templates = renderer.list_templates()
        assert len(templates) > 0
        assert "generation/design_system.j2" in templates
        assert "evaluation/evaluate_design.j2" in templates

    def test_render_design_system(self, prompts_dir: Path) -> None:
        renderer = PromptRenderer(prompts_dir)
        rendered = renderer.render(
            "generation/design_system.j2",
            system_name="Test System",
            problem_statement="Design a test system.",
            hard_constraints=["Must be fast", "Must be reliable"],
            use_cases=["Use case 1", "Use case 2"],
            complexity_drivers=["Driver 1"],
            rubric_dimensions=None,
            review_feedback=None,
            previous_design=None,
        )
        assert "Test System" in rendered
        assert "Must be fast" in rendered
        assert "Use case 1" in rendered

    def test_render_evaluation(self, prompts_dir: Path) -> None:
        renderer = PromptRenderer(prompts_dir)
        rendered = renderer.render(
            "evaluation/evaluate_design.j2",
            design_text="This is a sample design.",
            system_name="Test",
            complexity="medium",
            design_type="system",
            rubric_dimensions=[
                {
                    "name": "Scalability",
                    "weight": 1.0,
                    "description": "How well it scales",
                    "anchors": {"1": "Bad", "5": "Great"},
                }
            ],
        )
        assert "Scalability" in rendered
        assert "sample design" in rendered

    def test_template_exists(self, prompts_dir: Path) -> None:
        renderer = PromptRenderer(prompts_dir)
        assert renderer.template_exists("generation/design_system.j2")
        assert not renderer.template_exists("nonexistent/template.j2")

    def test_missing_var_raises(self, prompts_dir: Path) -> None:
        renderer = PromptRenderer(prompts_dir)
        with pytest.raises(Exception):  # jinja2.UndefinedError
            renderer.render(
                "generation/design_system.j2",
                # Missing required variables
            )

    def test_bullet_list_filter(self, prompts_dir: Path) -> None:
        renderer = PromptRenderer(prompts_dir)
        # The filter is registered; test it renders constraints as bullets
        rendered = renderer.render(
            "generation/design_system.j2",
            system_name="Test",
            problem_statement="Test.",
            hard_constraints=["C1", "C2"],
            use_cases=["U1"],
            complexity_drivers=[],
            rubric_dimensions=None,
            review_feedback=None,
            previous_design=None,
        )
        assert "- C1" in rendered
        assert "- C2" in rendered

    def test_render_coherence_check(self, prompts_dir: Path) -> None:
        renderer = PromptRenderer(prompts_dir)
        rendered = renderer.render(
            "evaluation/coherence_check.j2",
            section_a_name="Data Architecture",
            section_a_text="We use Cassandra for eventual consistency.",
            section_b_name="Scalability Strategy",
            section_b_text="We rely on strong consistency for joins.",
        )
        assert "Data Architecture" in rendered
        assert "Cassandra" in rendered
