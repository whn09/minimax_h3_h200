#!/usr/bin/env bash
# FL2VA 768p / 124-frame / 50-step GPU baseline on vllm-omni-neuron #436's fl2va case (the HF model card's
# reproducible 768p fl2va request: its first keyframe, prompt and seed 0), with t2va re-measured on the
# same image the same day so the two stay comparable. 8xH200, Ulysses=8, no patches / cache.
#   (1) stock sglang (lmsysorg/sglang:dev, quality exact): t2va then fl2va, 1 warmup + N each
#   (2) vllm-omni 0.29 (Dockerfile.vomni029, the version the Trainium2 side runs), pipeline profiler on
#   N=5 ./run_fl2va_hf.sh      # results in ./fl2va_hf_<date>/
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
N=${N:-5}
HF_CACHE=${HF_CACHE:-/opt/dlami/nvme/hf}
ASSETS=${ASSETS:-/opt/dlami/nvme/assets}
OUT=${OUT:-$HERE/fl2va_hf_$(date +%Y%m%d)}
mkdir -p "$OUT" "$ASSETS"
cp "$HERE/assets_fl2va/fl2va_input.png" "$ASSETS/"   # req_matrix.py sends it as a file:// URI
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$OUT/gpus.csv"

wait_up() {  # wait_up <port> <container>
  for i in $(seq 180); do
    curl -sf "localhost:$1/health" >/dev/null && return 0
    docker ps -q -f "name=$2" | grep -q . || return 1
    sleep 10
  done
  return 1
}

# (1) sglang
HF_CACHE=$HF_CACHE ASSETS=$ASSETS "$HERE/serve_sglang.sh" > "$OUT/sglang_image.txt" 2>&1
if wait_up 30010 h3-sglang; then
  docker exec h3-sglang bash -c 'pip show sglang | sed -n 2p; cd /sgl-workspace/sglang 2>/dev/null && git rev-parse --short HEAD' \
    > "$OUT/sglang_version.txt" 2>&1
  for task in t2va fl2va; do
    python3 "$HERE/req_matrix.py" "$task" "$OUT/sglang" --n "$N" --warm 1 --quality exact 2>&1 | tee -a "$OUT/sglang.log"
  done
else
  echo "SGLANG SERVER FAILED" | tee -a "$OUT/sglang.log"
fi
docker logs h3-sglang > "$OUT/sglang_server.log" 2>&1
docker rm -f h3-sglang >/dev/null 2>&1

# (2) vllm-omni 0.29
SNAP=$(ls "$HF_CACHE/hub/models--MiniMaxAI--MiniMax-H3/snapshots/" | head -1)
IMAGE=h3-vomni:0.29.0rc1-vllm0.29.0 SNAP=$SNAP HF_CACHE=$HF_CACHE EXTRA="--enable-diffusion-pipeline-profiler" \
  "$HERE/serve_vomni.sh" > "$OUT/vomni_image.txt" 2>&1
if wait_up 8000 h3-vomni; then
  docker exec h3-vomni pip show vllm vllm-omni 2>/dev/null | grep -E '^(Name|Version)' > "$OUT/vomni_versions.txt"
  for task in t2va fl2va; do
    TASK=$task N=$N WARM=1 OUT="$OUT/vomni" "$HERE/req_vomni.sh" 2>&1 | tee -a "$OUT/vomni.log"
  done
else
  echo "VOMNI SERVER FAILED" | tee -a "$OUT/vomni.log"
fi
docker logs h3-vomni > "$OUT/vomni_server.log" 2>&1
docker rm -f h3-vomni >/dev/null 2>&1
echo DONE
