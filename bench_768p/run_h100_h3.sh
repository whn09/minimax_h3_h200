#!/usr/bin/env bash
# MiniMax-H3 768p / 124-frame / 50-step on 8x H100 80GB, same requests as the 8x H200 run
# (run_fl2va_hf.sh: t2va three-cats seed 1101, fl2va = vllm-omni-neuron #436's HF example, seed 0).
# The 62 GB DiT plus activations peak at 82.5 GB/GPU under Ulysses 8 on H200, so here sglang runs
# TP2 x Ulysses4 first and falls back to TP4 x Ulysses2. vllm-omni 0.29 has no DiT TP: Ulysses 8, and
# the row is reported as failed if it does not fit. Ports stay off 8000 (shared box).
#   N=5 ./run_h100_h3.sh        # results in ./h100_<date>/
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
N=${N:-5}
HF_CACHE=${HF_CACHE:-/opt/dlami/nvme/h3/hf}
ASSETS=${ASSETS:-/opt/dlami/nvme/h3/assets}
OUT=${OUT:-$HERE/h100_$(date +%Y%m%d)}
mkdir -p "$OUT" "$ASSETS"
cp "$HERE/assets_fl2va/fl2va_input.png" "$ASSETS/"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$OUT/gpus.csv"

wait_up() {  # wait_up <port> <container>
  for i in $(seq 180); do
    curl -sf "localhost:$1/health" >/dev/null && return 0
    docker ps -q -f "name=$2" | grep -q . || return 1
    sleep 10
  done
  return 1
}

# (1) sglang: TP2 x U4, else TP4 x U2
for arm in "2 4" "4 2"; do
  set -- $arm; tag=tp${1}u${2}
  TP=$1 ULYSSES=$2 HF_CACHE=$HF_CACHE ASSETS=$ASSETS PORT=30010 "$HERE/serve_sglang.sh" > "$OUT/sglang_${tag}_image.txt" 2>&1
  if wait_up 30010 h3-sglang; then
    docker exec h3-sglang bash -c 'pip show sglang | sed -n 2p' > "$OUT/sglang_version.txt" 2>&1
    ok=1
    for task in t2va fl2va; do
      python3 "$HERE/req_matrix.py" "$task" "$OUT/sglang_$tag" --n "$N" --warm 1 --quality exact 2>&1 \
        | tee -a "$OUT/sglang_$tag.log" || ok=0
    done
    docker logs h3-sglang > "$OUT/sglang_${tag}_server.log" 2>&1; docker rm -f h3-sglang >/dev/null 2>&1
    grep -q "SUMMARY fl2va" "$OUT/sglang_$tag.log" && break
  else
    echo "SGLANG $tag FAILED TO START" | tee -a "$OUT/sglang_$tag.log"
    docker logs h3-sglang > "$OUT/sglang_${tag}_server.log" 2>&1; docker rm -f h3-sglang >/dev/null 2>&1
  fi
done

# (2) vllm-omni 0.29, Ulysses 8
SNAP=$(ls "$HF_CACHE/hub/models--MiniMaxAI--MiniMax-H3/snapshots/" | head -1)
IMAGE=h3-vomni:0.29.0rc1-vllm0.29.0 SNAP=$SNAP HF_CACHE=$HF_CACHE PORT=8010 \
  EXTRA="--enable-diffusion-pipeline-profiler" "$HERE/serve_vomni.sh" > "$OUT/vomni_image.txt" 2>&1
if wait_up 8010 h3-vomni; then
  docker exec h3-vomni pip show vllm vllm-omni 2>/dev/null | grep -E '^(Name|Version)' > "$OUT/vomni_versions.txt"
  TASK=t2va ASPECT=16:9 PORT=8010 N=$N WARM=1 OUT="$OUT/vomni" "$HERE/req_vomni.sh" 2>&1 | tee -a "$OUT/vomni.log"
  TASK=fl2va PORT=8010 N=$N WARM=1 OUT="$OUT/vomni" "$HERE/req_vomni.sh" 2>&1 | tee -a "$OUT/vomni.log"
else
  echo "VOMNI FAILED TO START" | tee -a "$OUT/vomni.log"
fi
docker logs h3-vomni > "$OUT/vomni_server.log" 2>&1; docker rm -f h3-vomni >/dev/null 2>&1
echo DONE
