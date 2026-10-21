#!/usr/bin/env bash
# Run h3_diffusers_bench.py for each "TPxU" config in a fresh lmsysorg/sglang:dev container with diffusers
# installed (--no-deps, editable) from DFX, a checkout of PR #14609 with PR #14725 merged in.
#   CONFIGS="1x8 8x1 2x4" ./run_diffusers.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DFX=${DFX:-/opt/dlami/nvme/dfx/diffusers}; HF_CACHE=${HF_CACHE:-/opt/dlami/nvme/hf}
OUT=${OUT:-$HERE/diffusers_out}; CONFIGS=${CONFIGS:-"1x8 8x1 2x4"}
mkdir -p "$OUT"
for c in $CONFIGS; do
  tp=${c%x*}; u=${c#*x}
  echo "== $(date -u +%T) TP=$tp U=$u" | tee -a "$OUT/runs.log"
  docker rm -f h3-diffusers >/dev/null 2>&1 || true
  docker run --name h3-diffusers --rm --gpus all --ipc=host --shm-size 32g \
    -v "$HF_CACHE:/root/.cache/huggingface" -v "$DFX:/dfx" -v "$OUT:/out" -v "$HERE:/bench:ro" \
    -e HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} -e TP=$tp -e U=$u -e N=${N:-2} -e WARM=${WARM:-1} -e ATTN=${ATTN:-} \
    --entrypoint bash lmsysorg/sglang:dev -c \
    "pip install -q --no-deps -e /dfx >/dev/null 2>&1 && git -c safe.directory=/dfx -C /dfx log --oneline -1; \
     torchrun --nproc_per_node=8 /bench/h3_diffusers_bench.py" \
    > "$OUT/tp${tp}_u${u}.log" 2>&1 || echo "FAILED TP=$tp U=$u" | tee -a "$OUT/runs.log"
  grep -E "^\[load\]|^(warmup|run)[0-9]" "$OUT/tp${tp}_u${u}.log" | tee -a "$OUT/runs.log" || true
done
