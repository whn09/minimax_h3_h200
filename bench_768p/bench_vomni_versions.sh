#!/usr/bin/env bash
# T2VA 768p / 124-frame / 50-step across vllm-omni images, same 8-GPU flags (U8 + VAE tile-parallel 8,
# pipeline profiler on), FL2VA partition passed as its subdirectory (the repo root's
# modular_model_index.json makes nightly misdetect the official checkpoint as a FastH3 release).
#   TAGS="latest v0.29.0rc1" ./bench_vomni_versions.sh
# Per image: 1 excluded warmup + N sync requests (client wall incl. MP4 encode), then NA async
# requests for the per-component stage_durations. Results in vomni_<tag>/.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TAGS=${TAGS:-"latest v0.29.0rc1"}; N=${N:-5}; NA=${NA:-3}
for tag in $TAGS; do
  out="$HERE/vomni_${tag//./_}"; mkdir -p "$out"
  echo "== $(date -u +%T) vllm/vllm-omni:$tag" | tee -a "$HERE/vomni_versions.log"
  IMAGE=vllm/vllm-omni:$tag EXTRA="--task-type fl2va --enable-diffusion-pipeline-profiler" \
    "$HERE/serve_vomni.sh" > "$out/image.txt" 2>&1
  ok=0
  for i in $(seq 120); do
    curl -sf localhost:8000/health >/dev/null && { ok=1; break; }
    docker ps -q -f name=h3-vomni | grep -q . || break
    sleep 10
  done
  docker logs h3-vomni > "$out/server_start.log" 2>&1 || true
  if [ $ok != 1 ]; then
    # older CLIs may not know --task-type: retry once without it
    IMAGE=vllm/vllm-omni:$tag EXTRA="--enable-diffusion-pipeline-profiler" "$HERE/serve_vomni.sh" >> "$out/image.txt" 2>&1
    for i in $(seq 120); do
      curl -sf localhost:8000/health >/dev/null && { ok=1; break; }
      docker ps -q -f name=h3-vomni | grep -q . || break
      sleep 10
    done
    docker logs h3-vomni > "$out/server_start.log" 2>&1 || true
  fi
  if [ $ok != 1 ]; then echo "SERVER FAILED $tag" | tee -a "$HERE/vomni_versions.log"; continue; fi
  docker exec h3-vomni pip show vllm vllm-omni 2>/dev/null | grep -E '^(Name|Version)' > "$out/versions.txt" || true
  # newer servers require aspect_ratio for t2va, older ones may reject it: probe with 2 steps
  aspect=16:9
  code=$(curl -s -o /dev/null -w '%{http_code}' -X POST localhost:8000/v1/videos/sync -F 'prompt=probe' \
    -F width=1344 -F height=768 -F aspect_ratio=16:9 -F fps=24 -F num_inference_steps=2 -F flow_shift=12 \
    -F seed=1 -F 'extra_params={"task":"t2va","duration":5.0,"audio_flow_shift":3.0}')
  [ "$code" = 200 ] || aspect=
  echo "aspect_probe http=$code aspect=[${aspect}]" | tee -a "$out/versions.txt"
  ASPECT=$aspect TASK=t2va N=$N WARM=1 OUT=$out "$HERE/req_vomni.sh" 2>&1 | tail -4 | tee -a "$HERE/vomni_versions.log"
  python3 "$HERE/req_vomni_async.py" "$out" --n "$NA" --aspect "$aspect" 2>&1 | tail -1 \
    | sed "s/^/$tag /" | tee -a "$HERE/vomni_versions.log"
done
docker rm -f h3-vomni >/dev/null 2>&1 || true
echo "== done $(date -u +%T)" | tee -a "$HERE/vomni_versions.log"
