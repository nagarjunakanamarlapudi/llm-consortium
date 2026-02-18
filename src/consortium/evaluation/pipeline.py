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

    preview = text[:300].replace("\n", " ")
    msg = f"Could not parse JSON from evaluator response (length={len(text)}): {preview}"
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
        registry: object | None = None,
    ) -> None:
        self.config = config
        self.database = database
        self.renderer = PromptRenderer(prompts_dir)
        self.evaluator_config = config.evaluator

        # Create evaluator provider (use registry if provided for shared batching)
        model_config = config.get_model(self.evaluator_config.model)
        if registry is not None:
            self.provider = registry.get(model_config)
        else:
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
        quiet: bool = False,
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
            return await self._evaluate_batch(designs, force=force, stats=stats, quiet=quiet)

        # Sequential evaluation with progress bar
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.fields[current]}"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TextColumn("[green]ok:{task.fields[ok]}[/green]"),
            TextColumn("[red]err:{task.fields[fail]}[/red]"),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("eta"),
            TimeRemainingColumn(),
            disable=quiet,
        )

        with progress:
            ptask = progress.add_task(
                "Evaluating",
                total=len(designs),
                current="starting...",
                ok=0,
                fail=0,
            )

            for design_row in designs:
                design_id = design_row["design_id"]
                task_id = design_row["task_id"]
                design_text = design_row["full_text"]
                d_run_id = design_row["run_id"]

                short_id = design_id[:20] if len(design_id) > 20 else design_id
                progress.update(ptask, current=f"{short_id} ({task_id})")

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
                    d_log.error(
                        "evaluation_failed",
                        error=str(e),
                        design_id=design_id,
                        task=task_id,
                    )

                progress.update(
                    ptask,
                    advance=1,
                    ok=stats["evaluated"],
                    fail=stats["failed"],
                )

        log.info("evaluation_complete", **stats)
        return stats

    async def _evaluate_batch(
        self,
        designs: list[dict],
        *,
        force: bool,
        stats: dict[str, int],
        quiet: bool = False,
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

        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.fields[phase]}"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TextColumn("[green]ok:{task.fields[ok]}[/green]"),
            TextColumn("[red]err:{task.fields[fail]}[/red]"),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("eta"),
            TimeRemainingColumn(),
            disable=quiet,
        )

        with progress:
            total_chunks = (len(all_requests) + batch_size - 1) // batch_size

            # Phase 2: Submit in batch chunks
            all_responses: list[LLMResponse | None] = []
            ptask = progress.add_task(
                "Batch submit",
                total=total_chunks,
                phase="Submitting batches",
                ok=0,
                fail=0,
            )
            for chunk_start in range(0, len(all_requests), batch_size):
                chunk = all_requests[chunk_start : chunk_start + batch_size]
                chunk_num = chunk_start // batch_size + 1
                progress.update(
                    ptask, phase=f"Batch {chunk_num}/{total_chunks} ({len(chunk)} reqs)"
                )
                log.info(
                    "batch_chunk_submit",
                    chunk=chunk_num,
                    total_chunks=total_chunks,
                    requests=len(chunk),
                )
                try:
                    responses = await self.provider.complete_batch(chunk)
                    all_responses.extend(responses)
                    progress.update(ptask, advance=1, ok=chunk_num)
                except Exception as e:
                    log.error("batch_chunk_failed", chunk=chunk_num, error=str(e))
                    all_responses.extend([None] * len(chunk))
                    progress.update(ptask, advance=1, fail=chunk_num)

            # Phase 3: Parse responses, persist, compute medians
            design_evaluations: dict[str, list[dict]] = {}
            design_meta: dict[str, dict] = {}

            progress.update(ptask, phase="Parsing responses", completed=0, total=len(all_responses))
            for idx, (response, meta) in enumerate(
                zip(all_responses, request_metadata, strict=True)
            ):
                design_id = meta["design_id"]
                eval_run = meta["eval_run"]
                d_log = log.bind(design_id=design_id, eval_run=eval_run)

                if response is None:
                    d_log.error("batch_response_missing", idx=idx)
                    progress.update(ptask, advance=1)
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

                progress.update(ptask, advance=1)

            # Phase 4: Compute medians for each design that got all evaluations
            progress.update(
                ptask,
                phase="Computing medians",
                completed=0,
                total=len(design_evaluations),
                ok=0,
                fail=0,
            )
            for design_id, evaluations in design_evaluations.items():
                meta = design_meta[design_id]
                if len(evaluations) < runs_per_design:
                    log.warning(
                        "batch_incomplete_evaluations",
                        design_id=design_id,
                        got=len(evaluations),
                        expected=runs_per_design,
                    )
                    # Don't compute medians from incomplete data — count as
                    # failed so the design can be re-evaluated later.
                    stats["failed"] += 1
                elif evaluations:
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

                progress.update(ptask, advance=1, ok=stats["evaluated"], fail=stats["failed"])

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
        """Compute median scores, disagreement flags, and blocker counts across evaluator runs."""
        # Collect per-dimension scores across all evaluator runs.
        # Use a dict keyed by (rater_index, dimension) so that missing
        # dimensions in one rater don't misalign other raters' scores.
        dim_scores: dict[str, list[float]] = {}
        dim_scores_by_rater: dict[str, dict[int, float]] = {}
        overall_scores: list[float] = []

        # Extract critical blockers from evaluations
        all_blockers: list[str] = []
        for rater_idx, evaluation in enumerate(evaluations):
            overall_scores.append(evaluation.get("overall_score", 0.0))
            for ds in evaluation.get("dimension_scores", []):
                dim_id = ds.get("dimension", "")
                score = ds.get("score", 0.0)
                dim_scores.setdefault(dim_id, []).append(score)
                dim_scores_by_rater.setdefault(dim_id, {})[rater_idx] = score
                # Extract blockers from each dimension score
                blockers = ds.get("blockers", [])
                if isinstance(blockers, list):
                    all_blockers.extend(blockers)
            # Also check top-level blockers
            top_blockers = evaluation.get("blockers", [])
            if isinstance(top_blockers, list):
                all_blockers.extend(top_blockers)

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

        # Compute Krippendorff's alpha — use properly indexed scores so
        # partial dimension failures don't misalign the reliability matrix.
        alpha = self._compute_krippendorff_alpha(
            dim_scores_by_rater, n_raters=len(evaluations),
        )

        # Persist with blocker information
        blocker_count = len(all_blockers)
        blockers_json = json.dumps(all_blockers) if all_blockers else None

        conn = self.database.conn
        conn.execute(
            """INSERT OR REPLACE INTO scores_median (
                design_id, run_id, dimension_medians, overall_median,
                disagreement_flags, krippendorff_alpha,
                blocker_count, blockers_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                design_id,
                run_id,
                json.dumps(dim_medians),
                overall_median,
                json.dumps(disagreement_flags) if disagreement_flags else None,
                alpha,
                blocker_count,
                blockers_json,
            ),
        )
        conn.commit()

        logger.info(
            "medians_computed",
            design_id=design_id,
            overall_median=overall_median,
            disagreements=len(disagreement_flags),
            alpha=f"{alpha:.3f}" if alpha else "N/A",
            blocker_count=blocker_count,
        )

    @staticmethod
    def _compute_krippendorff_alpha(
        dim_scores_by_rater: dict[str, dict[int, float]],
        n_raters: int,
    ) -> float | None:
        """Compute Krippendorff's alpha for inter-rater reliability.

        Uses the ``krippendorff`` package if available, otherwise falls back
        to a variance-based proxy.

        *dim_scores_by_rater* maps dimension_id → {rater_index: score}.
        Using explicit rater indices ensures the reliability matrix stays
        aligned even when some dimensions are missing for a rater (partial
        parse failures).
        """
        if not dim_scores_by_rater:
            return None

        if n_raters < 2 or len(dim_scores_by_rater) < 2:
            return None

        try:
            import krippendorff

            # Build reliability data matrix: raters × items (dimensions)
            # Each row is a rater, each column is a dimension
            dim_ids = sorted(dim_scores_by_rater.keys())
            reliability_data = []
            for rater_idx in range(n_raters):
                row = []
                for dim_id in dim_ids:
                    rater_map = dim_scores_by_rater[dim_id]
                    if rater_idx in rater_map:
                        row.append(rater_map[rater_idx])
                    else:
                        row.append(float("nan"))  # missing value
                reliability_data.append(row)

            alpha = krippendorff.alpha(
                reliability_data=reliability_data,
                level_of_measurement="ordinal",
            )
            return float(alpha)

        except ImportError:
            # Graceful fallback to variance-based proxy
            all_scores = []
            within_var_sum = 0.0
            n_dims = 0

            for rater_map in dim_scores_by_rater.values():
                scores = list(rater_map.values())
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

    # ── Coherence Checking ───────────────────────────────────────────────────

    @staticmethod
    def _parse_coherence_response(text: str) -> dict[str, Any]:
        """Parse coherence check response, with regex fallback."""
        try:
            return _extract_json(text)
        except ValueError:
            # Fallback: look for CONTRADICTS/CONSISTENT keyword
            if re.search(r"CONTRADICTS", text, re.IGNORECASE):
                return {"contradicts": True, "explanation": text, "confidence": 0.5}
            return {"contradicts": False, "explanation": text, "confidence": 0.5}

    def _resolve_section_pairs(self, task_config: Any) -> list[tuple[str, str]]:
        """Resolve section pairs from evaluator config or rubric."""
        # Priority 1: evaluator config section_pairs
        cc = self.evaluator_config.coherence_check
        if cc.section_pairs:
            return [(p[0], p[1]) for p in cc.section_pairs if len(p) >= 2]

        # Priority 2: rubric coherence_pairs
        rubric_config = self.config.get_rubric(task_config.rubric)
        if rubric_config.coherence_pairs:
            return [(p[0], p[1]) for p in rubric_config.coherence_pairs if len(p) >= 2]

        # No pairs configured — skip
        return []

    async def run_coherence_checks(
        self,
        design_id: str,
        design_text: str,
        section_pairs: list[tuple[str, str]],
        design_type: str = "system",
    ) -> list[dict[str, Any]]:
        """Check internal consistency between design sections.

        For each (section_a, section_b) pair:
        1. Render coherence_check.j2 with full design + section names
        2. Call evaluator LLM (it locates the sections itself)
        3. Parse JSON response with defensive fallback
        4. Persist to coherence_checks table
        """
        results: list[dict[str, Any]] = []

        for section_a_name, section_b_name in section_pairs:
            log = logger.bind(
                design_id=design_id,
                section_a=section_a_name,
                section_b=section_b_name,
            )
            log.info("coherence_check_start")

            try:
                # Pass the full design text — let the LLM locate sections.
                # No regex extraction needed; section names in the YAML
                # match the heading names from our prompt templates.
                template_vars = {
                    "design_text": design_text,
                    "design_type": design_type,
                    "section_a_name": section_a_name,
                    "section_b_name": section_b_name,
                }

                prompt_content = self.renderer.render(
                    self.evaluator_config.coherence_check.prompt_template,
                    **template_vars,
                )

                request = LLMRequest(
                    system_prompt="",
                    messages=[{"role": "user", "content": prompt_content}],
                    model_config_id=self.model_config.id,
                    parameters={
                        "temperature": 0.1,
                        "max_tokens": 1024,
                    },
                    metadata={"step": "coherence_check"},
                )

                response: LLMResponse = await self.provider.complete(request)
                parsed = self._parse_coherence_response(response.content)

                # Persist
                check_id = uuid.uuid4().hex
                conn = self.database.conn
                conn.execute(
                    """INSERT OR REPLACE INTO coherence_checks
                       (check_id, design_id, section_pair, contradicts, explanation)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        check_id,
                        design_id,
                        f"{section_a_name}|{section_b_name}",
                        parsed.get("contradicts", False),
                        parsed.get("explanation", ""),
                    ),
                )
                conn.commit()

                result = {
                    "check_id": check_id,
                    "design_id": design_id,
                    "section_a": section_a_name,
                    "section_b": section_b_name,
                    "contradicts": parsed.get("contradicts", False),
                    "explanation": parsed.get("explanation", ""),
                    "confidence": parsed.get("confidence", 1.0),
                    "_input_tokens": response.input_tokens,
                    "_output_tokens": response.output_tokens,
                    "_cost_usd": response.cost_usd,
                }
                results.append(result)

                log.info(
                    "coherence_check_done",
                    contradicts=result["contradicts"],
                )

            except Exception as e:
                log.error("coherence_check_failed", error=str(e))

        return results

    async def run_all_coherence_checks(
        self,
        *,
        run_id: str | None = None,
        force: bool = False,
        dry_run: bool = False,
        quiet: bool = False,
    ) -> dict[str, int]:
        """Run coherence checks on all final designs.

        Args:
            run_id: Only check this run's designs.
            force: Re-check already checked designs.
            dry_run: Report what would be checked.

        Returns:
            Stats dict: checked, skipped, failed counts.
        """
        if not self.evaluator_config.coherence_check.enabled:
            logger.info("coherence_checks_disabled")
            return {"total": 0, "checked": 0, "failed": 0}

        if force:
            designs = self._get_all_final_designs(run_id)
        else:
            designs = self._get_unchecked_designs(run_id)

        log = logger.bind(designs_to_check=len(designs))
        log.info("coherence_check_start_all")

        if dry_run:
            return {"total": len(designs), "checked": 0, "failed": 0}

        stats = {"total": len(designs), "checked": 0, "failed": 0}

        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.fields[phase]}"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TextColumn("[green]ok:{task.fields[ok]}[/green]"),
            TextColumn("[red]err:{task.fields[fail]}[/red]"),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("eta"),
            TimeRemainingColumn(),
            disable=quiet,
        )

        # Pre-resolve section pairs and build work items
        work_items: list[dict] = []
        for design_row in designs:
            design_id = design_row["design_id"]
            task_id = design_row["task_id"]
            design_text = design_row["full_text"]

            try:
                task_config = self.config.get_task(task_id)
                section_pairs = self._resolve_section_pairs(task_config)
                if not section_pairs:
                    log.debug("coherence_no_pairs", design_id=design_id, task=task_id)
                    continue

                if force:
                    self.database.conn.execute(
                        "DELETE FROM coherence_checks WHERE design_id = ?",
                        (design_id,),
                    )
                    self.database.conn.commit()

                work_items.append(
                    {
                        "design_id": design_id,
                        "task_id": task_id,
                        "design_text": design_text,
                        "design_type": task_config.design_type,
                        "section_pairs": section_pairs,
                    }
                )
            except Exception as e:
                stats["failed"] += 1
                log.error("coherence_config_failed", design_id=design_id, error=str(e))

        if not work_items:
            log.info("coherence_check_no_work")
            return stats

        if self._use_batch:
            return await self._coherence_batch(
                work_items,
                stats=stats,
                progress=progress,
                log=log,
            )

        # Sequential fallback
        with progress:
            ptask = progress.add_task(
                "Coherence",
                total=len(work_items),
                phase="starting...",
                ok=0,
                fail=0,
            )

            for item in work_items:
                design_id = item["design_id"]
                short_id = design_id[:20] if len(design_id) > 20 else design_id
                progress.update(ptask, phase=f"{short_id} ({item['task_id']})")

                try:
                    await self.run_coherence_checks(
                        design_id=design_id,
                        design_text=item["design_text"],
                        section_pairs=item["section_pairs"],
                        design_type=item["design_type"],
                    )
                    stats["checked"] += 1
                except Exception as e:
                    stats["failed"] += 1
                    log.error("coherence_design_failed", design_id=design_id, error=str(e))

                progress.update(
                    ptask,
                    advance=1,
                    ok=stats["checked"],
                    fail=stats["failed"],
                )

        log.info("coherence_check_complete", **stats)
        return stats

    async def _coherence_batch(
        self,
        work_items: list[dict],
        *,
        stats: dict[str, int],
        progress,
        log,
    ) -> dict[str, int]:
        """Run coherence checks using batch API.

        Collects all (design x section_pair) requests, submits in chunks
        via complete_batch, then parses and persists results.
        """
        batch_cfg = self.evaluator_config.batch
        batch_size = getattr(batch_cfg, "batch_size", 100)

        # Phase 1: Build all requests
        all_requests: list[LLMRequest] = []
        request_meta: list[dict] = []

        for item in work_items:
            design_id = item["design_id"]
            design_text = item["design_text"]
            design_type = item["design_type"]

            for section_a, section_b in item["section_pairs"]:
                template_vars = {
                    "design_text": design_text,
                    "design_type": design_type,
                    "section_a_name": section_a,
                    "section_b_name": section_b,
                }
                prompt_content = self.renderer.render(
                    self.evaluator_config.coherence_check.prompt_template,
                    **template_vars,
                )
                request = LLMRequest(
                    system_prompt="",
                    messages=[{"role": "user", "content": prompt_content}],
                    model_config_id=self.model_config.id,
                    parameters={"temperature": 0.1, "max_tokens": 1024},
                    metadata={"step": "coherence_check"},
                )
                all_requests.append(request)
                request_meta.append(
                    {
                        "design_id": design_id,
                        "section_a": section_a,
                        "section_b": section_b,
                    }
                )

        log.info(
            "coherence_batch_start",
            total_requests=len(all_requests),
            batch_size=batch_size,
        )

        if not all_requests:
            return stats

        total_chunks = (len(all_requests) + batch_size - 1) // batch_size

        with progress:
            # Phase 2: Submit in batch chunks
            all_responses: list[LLMResponse | None] = []
            ptask = progress.add_task(
                "Coherence batch",
                total=total_chunks,
                phase="Submitting batches",
                ok=0,
                fail=0,
            )

            for chunk_start in range(0, len(all_requests), batch_size):
                chunk = all_requests[chunk_start : chunk_start + batch_size]
                chunk_num = chunk_start // batch_size + 1
                progress.update(
                    ptask,
                    phase=f"Batch {chunk_num}/{total_chunks} ({len(chunk)} reqs)",
                )
                try:
                    responses = await self.provider.complete_batch(chunk)
                    all_responses.extend(responses)
                    progress.update(ptask, advance=1, ok=chunk_num)
                except Exception as e:
                    log.error("coherence_batch_chunk_failed", chunk=chunk_num, error=str(e))
                    all_responses.extend([None] * len(chunk))
                    progress.update(ptask, advance=1, fail=chunk_num)

            # Phase 3: Parse responses and persist
            progress.update(
                ptask,
                phase="Parsing responses",
                completed=0,
                total=len(all_responses),
            )

            design_ok: set[str] = set()
            design_fail: set[str] = set()
            conn = self.database.conn

            for response, meta in zip(
                all_responses,
                request_meta,
                strict=True,
            ):
                design_id = meta["design_id"]

                if response is None:
                    design_fail.add(design_id)
                    progress.update(ptask, advance=1)
                    continue

                try:
                    parsed = self._parse_coherence_response(response.content)
                    check_id = uuid.uuid4().hex
                    conn.execute(
                        """INSERT OR REPLACE INTO coherence_checks
                           (check_id, design_id, section_pair,
                            contradicts, explanation)
                           VALUES (?, ?, ?, ?, ?)""",
                        (
                            check_id,
                            design_id,
                            f"{meta['section_a']}|{meta['section_b']}",
                            parsed.get("contradicts", False),
                            parsed.get("explanation", ""),
                        ),
                    )
                    design_ok.add(design_id)
                except Exception as e:
                    log.error(
                        "coherence_batch_parse_failed",
                        design_id=design_id,
                        error=str(e),
                    )
                    design_fail.add(design_id)

                progress.update(ptask, advance=1)

            conn.commit()

        # Count designs that succeeded (appeared in ok but not fail)
        stats["checked"] = len(design_ok - design_fail)
        stats["failed"] += len(design_fail)

        log.info("coherence_batch_complete", **stats)
        return stats

    def _get_unchecked_designs(self, run_id: str | None = None) -> list[dict]:
        """Get final designs that haven't had coherence checks."""
        conn = self.database.conn

        if run_id:
            rows = conn.execute(
                """SELECT d.design_id, d.run_id, d.full_text, r.task_id, r.variant_id
                   FROM designs d
                   JOIN runs r ON d.run_id = r.run_id
                   WHERE d.is_final = TRUE AND r.run_id = ?
                   AND d.design_id NOT IN (SELECT DISTINCT design_id FROM coherence_checks)""",
                (run_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT d.design_id, d.run_id, d.full_text, r.task_id, r.variant_id
                   FROM designs d
                   JOIN runs r ON d.run_id = r.run_id
                   WHERE d.is_final = TRUE AND r.status = 'completed'
                   AND d.design_id NOT IN (SELECT DISTINCT design_id FROM coherence_checks)"""
            ).fetchall()

        return [dict(row) for row in rows]
