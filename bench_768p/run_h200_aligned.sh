#!/usr/bin/env bash
# 8x H200, the original bench_768p method (API serving, client wall, 1 warmup + N back to back,
# 1344x768 / 124 frames / 50 steps / shifts 12+3), every stack on the same requests:
#   t2va         the original reference request (Night-vision prompt, seed 1101)
#   t2va_parrot  vllm-omni-neuron's own E2E request (parrot prompt, seed 42)
#   t2va_atnight the vllm-omni reference prompt (seed 1101), which the 2026-10 vllm-omni / Trn2 API rows used
#   fl2va        vllm-omni-neuron #436's HF model card request (seed 0)
# sglang lmsysorg/sglang:dev (Ulysses 8, exact) and vllm-omni 0.29 (Ulysses 8).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
N=${N:-5}; HF_CACHE=${HF_CACHE:-/opt/dlami/nvme/hf}; ASSETS=${ASSETS:-/opt/dlami/nvme/assets}
OUT=${OUT:-$HERE/h200_aligned_$(date +%Y%m%d)}; mkdir -p "$OUT" "$ASSETS"
cp "$HERE/assets_fl2va/fl2va_input.png" "$ASSETS/"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$OUT/gpus.csv"
wait_up() { for i in $(seq 180); do curl -sf "localhost:$1/health" >/dev/null && return 0; docker ps -q -f "name=$2" | grep -q . || return 1; sleep 10; done; return 1; }

HF_CACHE=$HF_CACHE ASSETS=$ASSETS "$HERE/serve_sglang.sh" > "$OUT/sglang_image.txt" 2>&1
if wait_up 30010 h3-sglang; then
  docker exec h3-sglang bash -c 'pip show sglang | sed -n 2p' > "$OUT/sglang_version.txt" 2>&1
  for task in t2va t2va_parrot t2va_atnight fl2va; do
    python3 "$HERE/req_matrix.py" "$task" "$OUT/sglang" --n "$N" --warm 1 --quality exact 2>&1 | tee -a "$OUT/sglang.log"
  done
else echo "SGLANG FAILED TO START" | tee -a "$OUT/sglang.log"; fi
docker logs h3-sglang > "$OUT/sglang_server.log" 2>&1; docker rm -f h3-sglang >/dev/null 2>&1

SNAP=$(ls "$HF_CACHE/hub/models--MiniMaxAI--MiniMax-H3/snapshots/" | head -1)
IMAGE=h3-vomni:0.29.0rc1-vllm0.29.0 SNAP=$SNAP HF_CACHE=$HF_CACHE EXTRA="--enable-diffusion-pipeline-profiler" \
  "$HERE/serve_vomni.sh" > "$OUT/vomni_image.txt" 2>&1
if wait_up 8000 h3-vomni; then
  docker exec h3-vomni pip show vllm vllm-omni 2>/dev/null | grep -E '^(Name|Version)' > "$OUT/vomni_versions.txt"
  for task in t2va t2va_parrot t2va_atnight; do
    TASK=$task ASPECT=16:9 N=$N WARM=1 OUT="$OUT/vomni" "$HERE/req_vomni.sh" 2>&1 | tee -a "$OUT/vomni.log"
  done
  TASK=fl2va N=$N WARM=1 OUT="$OUT/vomni" "$HERE/req_vomni.sh" 2>&1 | tee -a "$OUT/vomni.log"
else echo "VOMNI FAILED TO START" | tee -a "$OUT/vomni.log"; fi
docker logs h3-vomni > "$OUT/vomni_server.log" 2>&1; docker rm -f h3-vomni >/dev/null 2>&1
echo DONE
