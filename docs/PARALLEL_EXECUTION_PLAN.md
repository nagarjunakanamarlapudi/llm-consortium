# Parallel Experiment Execution Across Two Machines

## Approach

Split the remaining 427 runs by **variant** between two machines. No code changes needed — existing CLI `--variants` filter and `db export`/`db load` handle everything.

Both machines run the same experiment (`llm-consortium-v1`). There is no per-machine experiment ID — `run_id` is scoped to `{variant}-{task}-rep{N}-{timestamp}`, so both machines just contribute rows to the same logical experiment.

---

## Workload-Balanced Split

Remaining work per variant (each has 8 tasks × 5 reps = 40 target):

| Variant | Completed | Remaining | Variant | Completed | Remaining |
|---------|-----------|-----------|---------|-----------|-----------|
| v7 | 0 | **40** | v4 | 5 | **35** |
| v6 | 1 | **39** | v3b | 7 | **33** |
| v5 | 6 | **34** | v1a | 9 | **31** |
| v2c | 7 | **33** | v2a | 9 | **31** |
| v1 | 9 | **31** | v3c | 10 | **30** |
| v8 | 9 | **31** | v3a | 10 | **30** |
| | | | v2b | 11 | **29** |

| | Machine A (Your Mac) | Machine B |
|---|---|---|
| **Variants** | v7, v6, v5, v2c, v1, v8 | v4, v3b, v1a, v2a, v3c, v3a, v2b |
| **Remaining** | **208 runs** | **219 runs** |

> **Tip:** v6 and v7 have the most failures (Vertex AI rate limits). Put these on the machine with the better connection or higher rate limits.

---

## Step-by-Step

### 1. Prepare Machine B

```bash
git clone <repo-url> && cd llm-consortium
pip install -e .
scp machine-a:.env .env
scp machine-a:llm-consortium/data/consortium.db data/consortium.db
```

### 2. Run Simultaneously

**Machine A:**
```bash
consortium run full-experiment \
  --variants v7,v6,v5,v2c,v1,v8 --skip-analysis
```

**Machine B:**
```bash
consortium run full-experiment \
  --variants v4,v3b,v1a,v2a,v3c,v3a,v2b --skip-analysis
```

Both auto-skip completed runs. Use `--skip-analysis` — run analysis only after merge.

### 3. Merge Results

```bash
# Machine B: export
consortium db export -o data/exports_machine_b

# Transfer to Machine A
scp -r machine-b:llm-consortium/data/exports_machine_b data/exports_machine_b

# Machine A: merge
consortium db load data/exports_machine_b
```

### 4. Verify & Analyze

```bash
consortium run status
# Then run analysis on the merged DB
```

---

## Collision Safety — Verified ✅

| Concern | Why It's Safe |
|---|---|
| **run_id PK** | Format: `{variant}-{task}-rep{N}-{timestamp}` — different variants = different IDs |
| **design_id, review_id, trace_id, check_id** | All `uuid.uuid4().hex` — globally unique, zero collision risk |
| **UNIQUE(variant_id, task_id, repetition)** | Each machine owns different variants — constraint can never fire |
| **Shared baseline rows** | `INSERT OR IGNORE` in `db load` silently skips existing rows |
| **FK ordering** | `db load` imports in dependency order: `runs` → `designs` → `reviews` → `evaluations` → `scores_median` → `coherence_checks` → `traces` → `batches` |
| **batches table** | Provider-generated IDs, different runs = different batches |
| **Experiment identity** | Both machines share the same `experiment.yaml` (`name: llm-consortium-v1`, `seed: 42`). No per-machine experiment ID exists — the merged DB looks as if one machine ran everything. |
