# Official EvalPlus harness (HumanEval+/MBPP+) for objective pass@1 scoring.
#
# Built once, run per scoring batch as an ephemeral, network-disabled container:
#   docker run --rm --network none -v <workdir>:/work consortium-evalplus \
#       bash -lc "python -m evalplus.sanitize --samples /work/samples.jsonl && \
#                 python -m evalplus.evaluate --dataset humaneval \
#                        --samples /work/samples-sanitized.jsonl"
#
# Running on Linux (this image) avoids the macOS reliability_guard setrlimit
# failure, and `--network none` contains the model-generated code under test.
# Datasets are cached at build time so evaluation works fully offline.
FROM python:3.11-slim

RUN pip install --no-cache-dir evalplus

# Pre-cache the HumanEval+/MBPP+ ground truth so runtime needs no network.
RUN python -c "from evalplus.data import get_human_eval_plus, get_mbpp_plus; get_human_eval_plus(); get_mbpp_plus()"

WORKDIR /work
