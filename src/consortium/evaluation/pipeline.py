"""Evaluation pipeline — scores final designs using LLM evaluator."""

from __future__ import annotations

import json
import re
import statistics
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from consortium.config.models import FullConfig, RubricDimensionConfig
from consortium.prompts.renderer import PromptRenderer
from consortium.providers.base import LLMRequest, LLMResponse
from consortium.providers.factory import create_provider
from consortium.storage.database import Database

logger = structlog.get_logger()

# Pattern to extract JSON from markdown code fences
_JSON_FENCE = re.compile(r"```(?:json)?\s*\n(.*?)\n```", re.DOTALL)


def _extract_json(text: str) -> dict[str, Any]:
    """Extract JSON from LLM response, handling markdown fences."""
    # Try raw parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from code fence
    match = _JSON_FENCE.search(text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Last resort: find first { to last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    msg = f"Could not parse JSON from evaluator response (length={len(text)})"
    raise ValueError(msg)


class EvaluationPipeline:
    """Scores final designs from completed runs.

    For each final design:
        1. Calls the evaluator LLM N times (default 3)
        2. Parses dimension scores from JSON response
        3. Computes median across runs for each dimension
        4. Detects disagreement (flags dimensions with high variance)
        5. Persists evaluations + median scores to database
    """

    def __init__(
        self,
        config: FullConfig,
        database: Database,
        prompts_dir: str | Path,
    ) -> None:
        self.config = config
        self.database = database
        self.renderer = PromptRenderer(prompts_dir)
        self.evaluator_config = config.evaluator

        # Create evaluator provider
        model_config = config.get_model(self.evaluator_config.model)
        self.provider = create_provider(model_config)
        self.model_config = model_config

    def _get_unevaluated_designs(self, run_id: str | None = None) -> list[dict]:
        """Get final designs that haven't been fully evaluated yet."""
        conn = self.database.conn

        if run_id:
            rows = conn.execute(
                """SELECT d.design_id, d.run_id, d.full_text, r.task_id, r.variant_id
                   FROM designs d
                   JOIN runs r ON d.run_id = r.run_id
                   WHERE d.is_final = TRUE AND r.run_id = ?
                   AND d.design_id NOT IN (SELECT design_id FROM scores_median)""",
                (run_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT d.design_id, d.run_id, d.full_text, r.task_id, r.variant_id
                   FROM designs d
                   JOIN runs r ON d.run_id = r.run_id
                   WHERE d.is_final = TRUE AND r.status = 'completed'
                   AND d.design_id NOT IN (SELECT design_id FROM scores_median)"""
            ).fetchall()

        return [dict(row) for row in rows]

    def _get_all_final_designs(self, run_id: str | None = None) -> list[dict]:
        """Get all final designs (for --force mode)."""
        conn = self.database.conn

        if run_id:
            rows = conn.execute(
                """SELECT d.design_id, d.run_id, d.full_text, r.task_id, r.variant_id
                   FROM designs d
                   JOIN runs r ON d.run_id = r.run_id
                   WHERE d.is_final = TRUE AND r.run_id = ?""",
                (run_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT d.design_id, d.run_id, d.full_text, r.task_id, r.variant_id
                   FROM designs d
                   JOIN runs r ON d.run_id = r.run_id
                   WHERE d.is_final = TRUE AND r.status = 'completed'"""
            ).fetchall()

        return [dict(row) for row in rows]

    async def evaluate(
        self,
        *,
        run_id: str | None = None,
        force: bool = False,
        dry_run: bool = False,
    ) -> dict[str, int]:
        """Evaluate final designs.

        Args:
            run_id: Evaluate only this run's designs.
            force: Re-evaluate already scored designs.
            dry_run: Report what would be evaluated.

        Returns:
            Stats dict: evaluated, skipped, failed counts.
        """
        if force:
            designs = self._get_all_final_designs(run_id)
        else:
            designs = self._get_unevaluated_designs(run_id)

        log = logger.bind(designs_to_evaluate=len(designs))
        log.info("evaluation_start")

        if dry_run:
            return {"total": len(designs), "evaluated": 0, "failed": 0}

        stats = {"total": len(designs), "evaluated": 0, "failed": 0}

        for design_row in designs:
            design_id = design_row["design_id"]
            task_id = design_row["task_id"]
            design_text = design_row["full_text"]
            d_run_id = design_row["run_id"]

            d_log = log.bind(design_id=design_id, task=task_id)

            try:
                task_config = self.config.get_task(task_id)
                rubric_config = self.config.get_rubric(task_config.rubric)

                # Clear existing evaluations if force
                if force:
                    self._clear_evaluations(design_id)

                # Run N evaluator calls
                evaluations = []
                for eval_run in range(self.evaluator_config.runs_per_design):
                    d_log.info("evaluator_call", eval_run=eval_run)
                    result = await self._evaluate_single(
                        design_text=design_text,
                        task_config=task_config,
                        rubric_config=rubric_config,
                    )
                    evaluations.append(result)

                    # Persist individual evaluation
                    self._persist_evaluation(
                        design_id=design_id,
                        evaluator_run=eval_run,
                        result=result,
                    )

                # Compute and persist median scores
                self._compute_and_persist_medians(
                    design_id=design_id,
                    run_id=d_run_id,
                    evaluations=evaluations,
                    rubric_config=rubric_config,
                )

                stats["evaluated"] += 1
                d_log.info("design_evaluated", overall=evaluations[0]["overall_score"])

            except Exception as e:
                stats["failed"] += 1
                d_log.error("evaluation_failed", error=str(e))

        log.info("evaluation_complete", **stats)
        return stats

    async def _evaluate_single(
        self,
        *,
        design_text: str,
        task_config,
        rubric_config,
    ) -> dict[str, Any]:
        """Call the evaluator LLM once and parse the JSON response."""
        template_vars = {
            "design_text": design_text,
            "system_name": task_config.variables.system_name,
            "complexity": task_config.complexity,
            "design_type": task_config.design_type,
            "rubric_dimensions": [
                d.model_dump() for d in rubric_config.dimensions
            ],
        }

        prompt_content = self.renderer.render(
            self.evaluator_config.prompt_template, **template_vars,
        )

        request = LLMRequest(
            system_prompt="",
            messages=[{"role": "user", "content": prompt_content}],
            model_config_id=self.model_config.id,
            parameters={
                "temperature": self.evaluator_config.parameters.temperature,
                "max_tokens": self.evaluator_config.parameters.max_tokens,
                "top_p": 1.0,
            },
            metadata={"step": "evaluation"},
        )

        response: LLMResponse = await self.provider.complete(request)
        result = _extract_json(response.content)

        # Attach token/cost info
        result["_input_tokens"] = response.input_tokens
        result["_output_tokens"] = response.output_tokens
        result["_cost_usd"] = response.cost_usd
        result["_batch_id"] = response.batch_id

        return result

    def _persist_evaluation(
        self,
        *,
        design_id: str,
        evaluator_run: int,
        result: dict[str, Any],
    ) -> None:
        """Write a single evaluator result to the evaluations table."""
        conn = self.database.conn
        conn.execute(
            """INSERT OR REPLACE INTO evaluations (
                evaluation_id, design_id, evaluator_run, evaluator_model,
                dimension_scores, overall_score, qualitative_summary,
                input_tokens, output_tokens, cost_usd, batch_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                uuid.uuid4().hex,
                design_id,
                evaluator_run,
                self.model_config.api_model,
                json.dumps(result.get("dimension_scores", [])),
                result.get("overall_score", 0.0),
                result.get("qualitative_summary", ""),
                result.get("_input_tokens", 0),
                result.get("_output_tokens", 0),
                result.get("_cost_usd", 0.0),
                result.get("_batch_id"),
            ),
        )
        conn.commit()

    def _compute_and_persist_medians(
        self,
        *,
        design_id: str,
        run_id: str,
        evaluations: list[dict[str, Any]],
        rubric_config,
    ) -> None:
        """Compute median scores and disagreement flags across evaluator runs."""
        # Collect per-dimension scores across all evaluator runs
        dim_scores: dict[str, list[float]] = {}
        overall_scores: list[float] = []

        for evaluation in evaluations:
            overall_scores.append(evaluation.get("overall_score", 0.0))
            for ds in evaluation.get("dimension_scores", []):
                dim_id = ds.get("dimension", "")
                score = ds.get("score", 0.0)
                dim_scores.setdefault(dim_id, []).append(score)

        # Compute medians
        dim_medians = {}
        disagreement_flags = {}
        for dim_id, scores in dim_scores.items():
            median = statistics.median(scores)
            dim_medians[dim_id] = median
            # Flag if range > 1.5 (significant disagreement)
            if len(scores) >= 2 and (max(scores) - min(scores)) > 1.5:
                disagreement_flags[dim_id] = {
                    "range": max(scores) - min(scores),
                    "scores": scores,
                }

        overall_median = statistics.median(overall_scores) if overall_scores else 0.0

        # Compute Krippendorff's alpha (simplified: use correlation-based proxy)
        # Full implementation would use the krippendorff library
        alpha = self._compute_alpha_proxy(dim_scores)

        conn = self.database.conn
        conn.execute(
            """INSERT OR REPLACE INTO scores_median (
                design_id, run_id, dimension_medians, overall_median,
                disagreement_flags, krippendorff_alpha
            ) VALUES (?, ?, ?, ?, ?, ?)""",
            (
                design_id,
                run_id,
                json.dumps(dim_medians),
                overall_median,
                json.dumps(disagreement_flags) if disagreement_flags else None,
                alpha,
            ),
        )
        conn.commit()

        logger.info(
            "medians_computed",
            design_id=design_id,
            overall_median=overall_median,
            disagreements=len(disagreement_flags),
            alpha=f"{alpha:.3f}" if alpha else "N/A",
        )

    @staticmethod
    def _compute_alpha_proxy(dim_scores: dict[str, list[float]]) -> float | None:
        """Compute a simplified inter-rater reliability metric.

        This is a variance-based proxy for Krippendorff's alpha.
        alpha = 1 - (observed_disagreement / expected_disagreement)

        For the full experiment analysis, use the krippendorff package.
        """
        if not dim_scores:
            return None

        all_scores = []
        within_var_sum = 0.0
        n_dims = 0

        for scores in dim_scores.values():
            if len(scores) < 2:
                continue
            all_scores.extend(scores)
            within_var_sum += statistics.variance(scores)
            n_dims += 1

        if n_dims == 0 or len(all_scores) < 3:
            return None

        within_var = within_var_sum / n_dims
        total_var = statistics.variance(all_scores)

        if total_var == 0:
            return 1.0  # Perfect agreement

        alpha = 1.0 - (within_var / total_var)
        return max(0.0, min(1.0, alpha))

    def _clear_evaluations(self, design_id: str) -> None:
        """Remove existing evaluations for a design (for --force)."""
        conn = self.database.conn
        conn.execute("DELETE FROM evaluations WHERE design_id = ?", (design_id,))
        conn.execute("DELETE FROM scores_median WHERE design_id = ?", (design_id,))
        conn.commit()
