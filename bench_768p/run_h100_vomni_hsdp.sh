#!/usr/bin/env bash
# vllm-omni 0.29 on 8x H100 with the DiT TP-sharded: TP2 x Ulysses4 (Ulysses 8 alone OOMs at load on
# 80 GB: the replicated 62 GB DiT). Same four requests as the H200 aligned run. Port 8010.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
N=${N:-5}; HF_CACHE=${HF_CACHE:-/opt/dlami/nvme/h3/hf}
OUT=${OUT:-$HERE/h100_$(date +%Y%m%d)}; mkdir -p "$OUT"
wait_up() { for i in $(seq 180); do curl -sf "localhost:$1/health" >/dev/null && return 0; docker ps -q -f "name=$2" | grep -q . || return 1; sleep 10; done; return 1; }
SNAP=$(ls "$HF_CACHE/hub/models--MiniMaxAI--MiniMax-H3/snapshots/" | head -1)
IMAGE=h3-vomni:0.29.0rc1-vllm0.29.0 SNAP=$SNAP HF_CACHE=$HF_CACHE PORT=8010 \
  EXTRA="--use-hsdp --hsdp-shard-size 8 --enable-diffusion-pipeline-profiler" \
  "$HERE/serve_vomni.sh" > "$OUT/vomni_hsdp8_image.txt" 2>&1
if wait_up 8010 h3-vomni; then
  for task in t2va t2va_parrot t2va_atnight; do
    TASK=$task ASPECT=16:9 PORT=8010 N=$N WARM=1 OUT="$OUT/vomni_hsdp8" "$HERE/req_vomni.sh" 2>&1 | tee -a "$OUT/vomni_hsdp8.log"
  done
  TASK=fl2va PORT=8010 N=$N WARM=1 OUT="$OUT/vomni_hsdp8" "$HERE/req_vomni.sh" 2>&1 | tee -a "$OUT/vomni_hsdp8.log"
else echo "VOMNI HSDP8 FAILED TO START" | tee -a "$OUT/vomni_hsdp8.log"; fi
docker logs h3-vomni > "$OUT/vomni_hsdp8_server.log" 2>&1; docker rm -f h3-vomni >/dev/null 2>&1
echo VOMNI_HSDP_DONE
