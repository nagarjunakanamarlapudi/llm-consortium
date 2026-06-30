# LLM Consortium for Software Engineering (Paper #2) — benchmark harness.
#
# Pipeline: generate code via topologies (DO inference) -> score with the
# official harness in Docker -> report pass@1 / McNemar.
#
# Quickstart:
#   cp .env.example .env   # add DO_INFERENCE_API_KEY (OpenAI-compatible)
#   make setup             # deps + EvalPlus Docker harness
#   make validate-harness  # sanity: canonical solutions ~99% (no LLM calls)
#   make pilot             # baselines + consortium + report on HumanEval+
#
# Override any variable:  make pilot BENCH=mbppplus REPS=2 DB=data/run.db
.DEFAULT_GOAL := help

DB        ?= data/consortium_code.db
BENCH     ?= humanevalplus
LIMIT     ?= 0
REPS      ?= 1
MAXCOST   ?= 50
BASELINES ?= base-sonnet,base-gpt52,base-gptoss
CONSORTIUM ?= c-xreview,c-adv
RUN        = uv run consortium bench
LOGDIR    ?= data/logs

.PHONY: help setup harness-image validate-harness gen-baselines gen-consortium \
        grade report baselines consortium pilot test lint clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

setup: ## Install deps (incl. evalplus) and build the harness image
	uv sync --extra dev
	uv pip install evalplus
	$(MAKE) harness-image

harness-image: ## Build the EvalPlus scoring Docker image
	docker build -f docker/evalplus.Dockerfile -t consortium-evalplus .

validate-harness: ## Score canonical solutions in Docker (expect ~99%, no LLM)
	uv run python3 -c "from evalplus.data import get_human_eval_plus as g; from consortium.evaluation.evalplus_scorer import EvalPlusScorer as S; d=g(); st=S(database=None)._run_harness('humaneval','humanevalplus',{k:v['prompt']+v['canonical_solution'] for k,v in d.items()}); n=len(st); p=sum(1 for s in st.values() if s['plus_status']=='pass'); print(f'canonical plus pass@1 = {p}/{n} = {p/n*100:.1f}%')"

gen-baselines: ## Generate single-model baselines (parallel) for $(BENCH)
	@mkdir -p $(LOGDIR)
	@for c in $$(echo $(BASELINES) | tr ',' ' '); do \
	  $(RUN) run $(BENCH) --conditions $$c --limit $(LIMIT) --reps $(REPS) \
	    --database $(DB) --max-cost $(MAXCOST) > $(LOGDIR)/gen_$$c.log 2>&1 & \
	done; wait; echo "baselines generated"

gen-consortium: ## Generate consortium conditions for $(BENCH)
	@mkdir -p $(LOGDIR)
	@for c in $$(echo $(CONSORTIUM) | tr ',' ' '); do \
	  $(RUN) run $(BENCH) --conditions $$c --limit $(LIMIT) --reps $(REPS) \
	    --database $(DB) --max-cost $(MAXCOST) > $(LOGDIR)/gen_$$c.log 2>&1 & \
	done; wait; echo "consortium generated"

grade: ## Score all generated artifacts with the official harness
	$(RUN) grade $(BENCH) --database $(DB)

report: ## Print pass@1 / McNemar for $(BENCH)
	$(RUN) report $(BENCH) --database $(DB)

baselines: gen-baselines grade report ## Baselines: generate + grade + report

consortium: gen-consortium grade report ## Consortium: generate + grade + report

pilot: gen-baselines gen-consortium grade report ## Full pilot for $(BENCH)

test: ## Run the test suite
	uv run --extra dev pytest -q

lint: ## Ruff + mypy on the new modules
	uv run ruff check src/consortium tests
	uv run mypy src/consortium/benchmarks src/consortium/evaluation/evalplus_scorer.py \
	  src/consortium/analysis/code_stats.py src/consortium/providers/do_inference.py

clean: ## Remove pilot databases and logs
	rm -f data/*.db data/*.db-wal data/*.db-shm
	rm -rf $(LOGDIR)
