#!/usr/bin/env bash
# Steady-state 768p (1344x768) / 124-frame / 50-step benchmark against serve_sglang.sh.
#   (1) the reference bench_serving command: 1 warmup at 50 steps, then N vbench prompts, conc=1
#   (2) the reference curl payload (fixed prompt, seed 1101) N times back to back
# Results land in ./results_<timestamp>/ on the host.
#   N=5 ./bench_sglang.sh
set -euo pipefail
NAME=${NAME:-h3-sglang}
N=${N:-5}
HERE=$(cd "$(dirname "$0")" && pwd)
RES=${RES:-$HERE/results_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$RES"
docker inspect --format '{{.Image}}' "$NAME" > "$RES/image_digest.txt"
docker exec "$NAME" bash -c 'pip show sglang | sed -n 2p' > "$RES/sglang_version.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$RES/gpus.csv"

docker exec "$NAME" mkdir -p /out
docker exec "$NAME" python3 -m sglang.multimodal_gen.benchmarks.bench_serving \
  --host 127.0.0.1 --port 30010 --model MiniMaxAI/MiniMax-H3 \
  --dataset vbench --task text-to-video \
  --num-prompts "$N" --max-concurrency 1 \
  --warmup-requests 1 --warmup-inference-steps 50 \
  --extra-body '{"task":"t2va","conditions":[],"target":{"short_edge":768,"aspect_ratio":"16:9","duration_seconds":5.0},"seconds":5,"flow_shift":12.0,"audio_flow_shift":3.0}' \
  --output-file /out/bench_serving.json --disable-tqdm 2>&1 | tee "$RES/bench_serving.log"
docker cp "$NAME:/out/bench_serving.json" "$RES/"

python3 "$HERE/req_768p.py" "$N" "$RES/fixed_prompt" 2>&1 | tee "$RES/fixed_prompt.log"

# Per-stage timings straight from the server (denoise / decode) for every request above.
docker logs --since 2h "$NAME" 2>&1 | tr '\r' '\n' | grep -E 'Stage\] finished in' > "$RES/stage_times.log" || true
grep -E 'DenoisingStage\] finished' "$RES/stage_times.log" | tail -n $((2*N+1)) | awk '{print $(NF-1)}' \
  | python3 -c 'import sys,statistics as s;x=[float(l) for l in sys.stdin];print(f"denoise n={len(x)} mean={s.mean(x):.2f}s min={min(x):.2f}s max={max(x):.2f}s -> {s.mean(x)/50:.4f} s/step")' \
  | tee "$RES/denoise_summary.txt"
echo "results in $RES"
