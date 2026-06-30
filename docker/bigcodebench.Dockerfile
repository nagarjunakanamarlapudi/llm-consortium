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

WORKDIR /work
