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

    @property
    def _use_batch(self) -> bool:
        """Check if batch evaluation should be used."""
        batch_cfg = getattr(self.evaluator_config, "batch", None)
        batch_enabled = batch_cfg is not None and getattr(batch_cfg, "enabled", False)
        provider_supports = getattr(self.provider, "supports_batch", False)
        if callable(provider_supports):
            provider_supports = provider_supports()
        return batch_enabled and provider_supports

    def _build_eval_request(
        self,
        *,
        design_text: str,
        task_config,
        rubric_config,
    ) -> LLMRequest:
        """Build an LLMRequest for a single evaluation call."""
        template_vars = {
            "design_text": design_text,
            "system_name": task_config.variables.system_name,
            "complexity": task_config.complexity,
            "design_type": task_config.design_type,
            "rubric_dimensions": [d.model_dump() for d in rubric_config.dimensions],
        }

        prompt_content = self.renderer.render(
            self.evaluator_config.prompt_template,
            **template_vars,
        )

        return LLMRequest(
            system_prompt="",
            messages=[{"role": "user", "content": prompt_content}],
            model_config_id=self.model_config.id,
            parameters={
                "temperature": self.evaluator_config.parameters.temperature,
                "max_tokens": self.evaluator_config.parameters.max_tokens,
            },
            metadata={"step": "evaluation"},
        )

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

        if self._use_batch and designs:
            return await self._evaluate_batch(designs, force=force, stats=stats)

        # Sequential fallback
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

    async def _evaluate_batch(
        self,
        designs: list[dict],
        *,
        force: bool,
        stats: dict[str, int],
    ) -> dict[str, int]:
        """Evaluate all designs using batch API for cost savings.

        Collects all (design x runs_per_design) requests, submits as a single
        batch, then parses and persists results. Falls back to sequential for
        any designs that fail in the batch.
        """
        runs_per_design = self.evaluator_config.runs_per_design
        batch_cfg = self.evaluator_config.batch
        batch_size = getattr(batch_cfg, "batch_size", 100)

        log = logger.bind(
            mode="batch",
            designs=len(designs),
            runs_per_design=runs_per_design,
            total_requests=len(designs) * runs_per_design,
            batch_size=batch_size,
        )
        log.info("batch_evaluation_start")

        # Phase 1: Build all requests and track metadata for response mapping
        all_requests: list[LLMRequest] = []
        request_metadata: list[dict] = []  # parallel to all_requests

        for design_row in designs:
            design_id = design_row["design_id"]
            task_id = design_row["task_id"]
            design_text = design_row["full_text"]
            d_run_id = design_row["run_id"]

            try:
                task_config = self.config.get_task(task_id)
                rubric_config = self.config.get_rubric(task_config.rubric)
            except Exception as e:
                log.error("batch_config_error", design_id=design_id, error=str(e))
                stats["failed"] += 1
                continue

            if force:
                self._clear_evaluations(design_id)

            for eval_run in range(runs_per_design):
                request = self._build_eval_request(
                    design_text=design_text,
                    task_config=task_config,
                    rubric_config=rubric_config,
                )
                all_requests.append(request)
                request_metadata.append(
                    {
                        "design_id": design_id,
                        "task_id": task_id,
                        "run_id": d_run_id,
                        "eval_run": eval_run,
                        "rubric_config": rubric_config,
                    }
                )

        if not all_requests:
            log.info("batch_evaluation_no_requests")
            return stats

        # Phase 2: Submit in batch chunks
        all_responses: list[LLMResponse | None] = []
        for chunk_start in range(0, len(all_requests), batch_size):
            chunk = all_requests[chunk_start : chunk_start + batch_size]
            chunk_num = chunk_start // batch_size + 1
            total_chunks = (len(all_requests) + batch_size - 1) // batch_size
            log.info(
                "batch_chunk_submit",
                chunk=chunk_num,
                total_chunks=total_chunks,
                requests=len(chunk),
            )
            try:
                responses = await self.provider.complete_batch(chunk)
                all_responses.extend(responses)
            except Exception as e:
                log.error("batch_chunk_failed", chunk=chunk_num, error=str(e))
                # Mark remaining as None for fallback
                all_responses.extend([None] * len(chunk))

        # Phase 3: Parse responses, persist, compute medians
        # Group responses by design_id
        design_evaluations: dict[str, list[dict]] = {}
        design_meta: dict[str, dict] = {}

        for idx, (response, meta) in enumerate(zip(all_responses, request_metadata, strict=True)):
            design_id = meta["design_id"]
            eval_run = meta["eval_run"]
            d_log = log.bind(design_id=design_id, eval_run=eval_run)

            if response is None:
                d_log.error("batch_response_missing", idx=idx)
                continue

            try:
                result = _extract_json(response.content)
                result["_input_tokens"] = response.input_tokens
                result["_output_tokens"] = response.output_tokens
                result["_cost_usd"] = response.cost_usd
                result["_batch_id"] = response.batch_id

                self._persist_evaluation(
                    design_id=design_id,
                    evaluator_run=eval_run,
                    result=result,
                )

                design_evaluations.setdefault(design_id, []).append(result)
                design_meta[design_id] = meta
                d_log.debug("batch_eval_parsed", overall=result.get("overall_score"))

            except Exception as e:
                d_log.error("batch_parse_failed", error=str(e))

        # Phase 4: Compute medians for each design that got all evaluations
        for design_id, evaluations in design_evaluations.items():
            meta = design_meta[design_id]
            if len(evaluations) < runs_per_design:
                log.warning(
                    "batch_incomplete_evaluations",
                    design_id=design_id,
                    got=len(evaluations),
                    expected=runs_per_design,
                )

            if evaluations:
                try:
                    self._compute_and_persist_medians(
                        design_id=design_id,
                        run_id=meta["run_id"],
                        evaluations=evaluations,
                        rubric_config=meta["rubric_config"],
                    )
                    stats["evaluated"] += 1
                except Exception as e:
                    stats["failed"] += 1
                    log.error("batch_median_failed", design_id=design_id, error=str(e))
            else:
                stats["failed"] += 1

        log.info("batch_evaluation_complete", **stats)
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
            "rubric_dimensions": [d.model_dump() for d in rubric_config.dimensions],
        }

        prompt_content = self.renderer.render(
            self.evaluator_config.prompt_template,
            **template_vars,
        )

        request = LLMRequest(
            system_prompt="",
            messages=[{"role": "user", "content": prompt_content}],
            model_config_id=self.model_config.id,
            parameters={
                "temperature": self.evaluator_config.parameters.temperature,
                "max_tokens": self.evaluator_config.parameters.max_tokens,
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
