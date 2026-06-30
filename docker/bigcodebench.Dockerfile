# Official BigCodeBench harness for objective pass@1 scoring (headroom benchmark).
#
# Run per scoring batch as an ephemeral, network-disabled container:
#   docker run --rm --network none -v <workdir>:/work consortium-bigcodebench \
#       bash -lc "bigcodebench.evaluate --subset hard --split complete \
#                 --samples /work/samples.jsonl --no_gt"
#
# BigCodeBench executes library-heavy solutions, so the image is large.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir bigcodebench

# Pre-cache the dataset so evaluation can run offline (--network none).
RUN python -c "from bigcodebench.data import get_bigcodebench; get_bigcodebench(subset='hard')" || true

# BLOCKER (deferred): BigCodeBench-Hard solutions import a large scientific
# stack (matplotlib, numpy, pandas, scikit-learn, scipy, seaborn, flask,
# requests, bs4, pillow, ...). This slim image lacks them, so even the
# canonical solutions fail with ModuleNotFoundError. To finish Phase 2, either
#   (a) base this image on the official `bigcodebench/bigcodebench-evaluate`,
#       which bundles the full execution environment, OR
#   (b) `pip install` BigCodeBench's eval requirements list on top of slim.
# Also note: `bigcodebench.evaluate` defaults to execution='gradio' (remote) and
# loads the dataset from HF at eval time, so runs need either network access or a
# fully-populated cache; pass `--execution=local` for in-container execution.

WORKDIR /work
