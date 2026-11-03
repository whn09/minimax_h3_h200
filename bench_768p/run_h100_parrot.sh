#!/usr/bin/env bash
# t2va on vllm-omni-neuron's own E2E request (parrot prompt, seed 42) on 8x H100, after run_h100_h3.sh:
# sglang TP2 x U4 (port 30010) and vllm-omni 0.29 Ulysses 8 (port 8010), 1 warmup + N each.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
N=${N:-5}; HF_CACHE=${HF_CACHE:-/opt/dlami/nvme/h3/hf}; ASSETS=${ASSETS:-/opt/dlami/nvme/h3/assets}
OUT=${OUT:-$HERE/h100_$(date +%Y%m%d)}; mkdir -p "$OUT"
wait_up() { for i in $(seq 180); do curl -sf "localhost:$1/health" >/dev/null && return 0; docker ps -q -f "name=$2" | grep -q . || return 1; sleep 10; done; return 1; }
TP=2 ULYSSES=4 HF_CACHE=$HF_CACHE ASSETS=$ASSETS PORT=30010 "$HERE/serve_sglang.sh" > "$OUT/sglang_parrot_image.txt" 2>&1
if wait_up 30010 h3-sglang; then
  for task in t2va_parrot t2va_atnight; do
    python3 "$HERE/req_matrix.py" $task "$OUT/sglang_tp2u4" --n "$N" --warm 1 --quality exact 2>&1 | tee -a "$OUT/sglang_parrot.log"
  done
else echo "SGLANG FAILED TO START" | tee -a "$OUT/sglang_parrot.log"; fi
docker logs h3-sglang > "$OUT/sglang_parrot_server.log" 2>&1; docker rm -f h3-sglang >/dev/null 2>&1
SNAP=$(ls "$HF_CACHE/hub/models--MiniMaxAI--MiniMax-H3/snapshots/" | head -1)
IMAGE=h3-vomni:0.29.0rc1-vllm0.29.0 SNAP=$SNAP HF_CACHE=$HF_CACHE PORT=8010 EXTRA="--enable-diffusion-pipeline-profiler" \
  "$HERE/serve_vomni.sh" > "$OUT/vomni_parrot_image.txt" 2>&1
if wait_up 8010 h3-vomni; then
  TASK=t2va_parrot ASPECT=16:9 PORT=8010 N=$N WARM=1 OUT="$OUT/vomni" "$HERE/req_vomni.sh" 2>&1 | tee -a "$OUT/vomni_parrot.log"
  TASK=t2va_atnight ASPECT=16:9 PORT=8010 N=$N WARM=1 OUT="$OUT/vomni" "$HERE/req_vomni.sh" 2>&1 | tee -a "$OUT/vomni_parrot.log"
else echo "VOMNI FAILED TO START" | tee -a "$OUT/vomni_parrot.log"; fi
docker logs h3-vomni > "$OUT/vomni_parrot_server.log" 2>&1; docker rm -f h3-vomni >/dev/null 2>&1
echo PARROT_DONE
