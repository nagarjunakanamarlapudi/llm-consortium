# Coherence Integration Plan

## Problem

Coherence checks are collected (`coherence_checks` table) but never consumed by the analysis pipeline. The data is a dead end — it's not factored into rankings, the decision framework, predictions, or any CLI output.

## Current State

| Component | Coherence? | Notes |
|---|---|---|
| `pipeline.py` → `coherence_checks` table | ✅ writes | `run_coherence_checks()` stores per-pair results |
| `cli/run.py coherence` | ✅ triggers + prints count | Dead-end: only shows "N contradictions found" |
| `analysis/loader.py` | ❌ | No `load_coherence_dataframe()` |
| `analysis/statistics.py` | ❌ | No coherence comparisons |
| `analysis/framework.py` | ❌ | Decision table ignores coherence |
| `analysis/heatmaps.py` | ❌ | No coherence heatmap |
| `cli/analyze.py` | ❌ | No `coherence` subcommand, `all` doesn't include it |
| `check_completeness()` | ❌ | Doesn't report missing coherence checks |

## Schema (already exists — no migration needed)

```sql
CREATE TABLE coherence_checks (
    check_id    TEXT PRIMARY KEY,
    design_id   TEXT NOT NULL REFERENCES designs(design_id),
    section_pair TEXT NOT NULL,       -- "Section A|Section B"
    contradicts BOOLEAN NOT NULL,
    explanation TEXT,
    created_at  TEXT DEFAULT (datetime('now'))
);
```

**Note:** The schema lacks a `severity` column, even though the prompt template asks for it and the JSON response includes it. Consider adding `severity TEXT` via migration, but it's optional — the boolean `contradicts` is sufficient for the core metric.

---

## Implementation

### 1. `analysis/loader.py` — Add `load_coherence_dataframe()`

**What:** Query `coherence_checks` joined with `designs` and `runs` to produce a per-design coherence rate.

```python
def load_coherence_dataframe(db: Database) -> pd.DataFrame:
    """Load coherence data: one row per design with coherence rate.

    Returns DataFrame with columns:
        variant_id, task_id, repetition, design_id, run_id,
        pairs_checked, contradictions, coherence_rate
    """
    query = """
        SELECT
            r.variant_id,
            r.task_id,
            r.repetition,
            d.design_id,
            d.run_id,
            COUNT(*) as pairs_checked,
            SUM(CASE WHEN cc.contradicts = TRUE THEN 1 ELSE 0 END) as contradictions
        FROM coherence_checks cc
        JOIN designs d ON cc.design_id = d.design_id
        JOIN runs r ON d.run_id = r.run_id
        WHERE d.is_final = TRUE
          AND r.status = 'completed'
        GROUP BY d.design_id
        ORDER BY r.variant_id, r.task_id, r.repetition
    """
```

Compute derived column: `coherence_rate = 1 - (contradictions / pairs_checked)`

**Also update** `check_completeness()` to report how many designs have coherence checks vs how many should.

---

### 2. `analysis/statistics.py` — Add coherence comparison functions

**2a. `rank_variants_by_coherence(coherence_df)`**

Same pattern as `rank_variants_overall()` but on `coherence_rate` instead of `overall_median`. Friedman test + Nemenyi post-hoc. This answers: "Do some architectures produce systematically more contradictions?"

**2b. `compare_coherence_to_baseline(coherence_df, baseline="v1")`**

Same pattern as `compare_to_baseline()`. Wilcoxon signed-rank test + Bonferroni correction. Tests whether each consortium variant has better/worse coherence than the single-LLM baseline.

**2c. `coherence_by_complexity(coherence_df)`**

Group by task complexity (simple/medium/complex), compare coherence rates. This tests a natural hypothesis: more complex designs may have more contradictions regardless of architecture.

**2d. `coherence_quality_correlation(scores_df, coherence_df)`**

Spearman rank correlation between `overall_median` and `coherence_rate` per design. This answers: "Do higher-quality designs also tend to be more internally consistent?"

```python
@dataclass
class CoherenceCorrelation:
    spearman_r: float
    p_value: float
    n_designs: int
    significant: bool
```

---

### 3. `analysis/framework.py` — Wire coherence into decision table + predictions

**3a. Decision framework (`generate_decision_framework`)**

Add to `FrameworkRow`:
```python
expected_coherence_rate: float  # mean coherence rate for this bucket
coherence_warning: str | None   # e.g. "High contradictions on complex tasks"
```

In the generation logic, compute mean coherence rate per variant-complexity bucket. Flag any row where coherence_rate < 0.8 (configurable threshold). This surfaces: "v4 scores well but contradicts itself on 30% of section pairs."

**3b. Add P7 prediction**

```python
"P7": "Consortium variants (v2-v8) have higher coherence than single-LLM baseline (v1)"
```

Rationale: multi-agent review rounds should catch contradictions that a single pass misses. If this is NOT supported, that's an interesting finding (more agents = more incoherence).

Implementation: `_validate_p7(coherence_df)` — Mann-Whitney U, consortium vs v1.

**3c. Optional: Add P8 prediction**

```python
"P8": "Quality and coherence are positively correlated (r > 0.3)"
```

Implementation: `_validate_p8(scores_df, coherence_df)` — Spearman correlation test.

---

### 4. `analysis/heatmaps.py` — Add coherence heatmap

**`coherence_variant_heatmap(coherence_df, output_path)`**

Heatmap: rows = variants, columns = tasks (or complexity levels). Cell value = mean coherence rate. Color scale: green (1.0) → red (0.0). Same pattern as `task_variant_heatmap()`.

---

### 5. `cli/analyze.py` — Add `coherence` subcommand + update `all`

**5a. New command: `consortium analyze coherence`**

```
@app.command()
def coherence(
    config_dir, database, output_dir
) -> None:
    """Analyze coherence check results across variants."""
```

Output (Rich table):
- Per-variant: mean coherence rate, contradiction count, pairs checked
- Friedman test result
- Baseline comparison summary
- Quality-coherence correlation

Export: `data/exports/reports/coherence_analysis.md` + `data/exports/coherence_summary.csv`

**5b. Update `all` command**

Add between step 11 (predictions) and the final message:
```python
# 12. Coherence
coherence_df = load_coherence_dataframe(db)
if not coherence_df.empty:
    coh_ranking = rank_variants_by_coherence(coherence_df)
    console.print(f"Coherence Friedman: χ²={coh_ranking.statistic:.2f}")
    coherence_variant_heatmap(coherence_df, fig_dir / "heatmap_coherence.png")
```

---

### 6. `check_completeness()` update

Add to `CompletenessReport`:
```python
coherence_checked: int = 0
missing_coherence: list[str] = field(default_factory=list)
```

Query designs that have evaluations but no coherence checks, and report them.

---

## File Change Summary

| File | Change |
|---|---|
| `analysis/loader.py` | Add `load_coherence_dataframe()`, update `check_completeness()` |
| `analysis/statistics.py` | Add `rank_variants_by_coherence()`, `compare_coherence_to_baseline()`, `coherence_by_complexity()`, `coherence_quality_correlation()` |
| `analysis/framework.py` | Add coherence to `FrameworkRow`, add P7 (+ optional P8), update `generate_decision_framework()` |
| `analysis/heatmaps.py` | Add `coherence_variant_heatmap()` |
| `cli/analyze.py` | Add `coherence` command, update `all` command |
| `tests/unit/test_statistics.py` | Add tests for new coherence functions |
| `tests/unit/test_framework.py` | Add test for P7 prediction |

## Optional / Future

- **Schema migration:** Add `severity TEXT` column to `coherence_checks` — the prompt already returns it but `pipeline.py` doesn't persist it. Would enable severity-weighted coherence rates.
- **Pareto with 3 axes:** Quality × Cost × Coherence. Currently Pareto is 2D (quality vs cost). A 3D frontier would identify variants that are good, cheap, AND consistent. Complex to visualize but could use bubble charts (size = coherence).
- **Per-section-pair analysis:** Which section pairs are most commonly contradictory? This could inform prompt template improvements.

## Ordering

Implement in this order:
1. `loader.py` (data foundation)
2. `statistics.py` (analysis functions)
3. `heatmaps.py` (visualization)
4. `framework.py` (predictions + decision table)
5. `cli/analyze.py` (user-facing)
6. Tests
