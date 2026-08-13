#!/usr/bin/env bash
# Bring up MiniMax-H3 on p5e.48xlarge (8xH200) with the 480p short-edge patch applied.
#
# This is the exact sequence that produced RESULTS.md, wrapped in a script. It creates a long
# lived `sleep infinity` container, applies the patch to the source inside it, then starts the
# server detached -- so restarting the server with different flags does not re-pull the image or
# re-apply the patch.
#
# THE DEFAULTS ARE THE RECOMMENDED H200 CONFIG: 8 GPUs, TP=1, Ulysses=8, encoder-parallel auto,
# 480p enabled and warmed, nothing else warmed. That is the measured 10.05 s / 6.2 vid/min /
# 95.9 GiB-per-GPU shape. Bare `./serve.sh` needs no arguments and no env vars.
#
# Usage:
#   ./serve.sh                          # <- recommended H200 config, 864x480 @ 40 steps -> 10.05 s
#   GPUS=4 ./serve.sh                   # the cookbook's 4xH200 recipe
#   TP=2 ULYSSES=4 ./serve.sh           # shard the DiT: 63.9 GiB/GPU instead of 95.9
#   OFFLOAD=1 GPUS=1 ULYSSES=1 ./serve.sh   # the g7e / 96 GB shape (79.4 GiB/GPU)
#   SHORT_EDGES= ./serve.sh             # unpatched policy (768 only), patch stays inert
#   MODEL=MiniMaxAI/MiniMax-H3 ./serve.sh   # pull weights from HF instead of a local dir
#   ./serve.sh stop | logs | status
#
# Both patches in patches/ are applied: the short-edge one enables 480p, and the cpu-offload one
# is REQUIRED whenever OFFLOAD=1 (unpatched, any *-cpu-offload flag dies during warmup at
# decoding.py:92). Applying both unconditionally is safe -- the offload patch is a no-op when no
# offload flag is passed.
#
# Local weights MUST live in a directory *named* MiniMax-H3: registry.py:1199 resolves the
# pipeline class from the --model-path basename, not --model-id. A differently named dir reads the
# root model_index.json and dies with `module diffusers has no attribute
# MiniMaxH3ModularPipeline`.
set -euo pipefail

IMAGE=${IMAGE:-lmsysorg/sglang:dev}
NAME=${NAME:-h3}
GPUS=${GPUS:-8}
TP=${TP:-1}
# Ulysses defaults to whatever is left after TP takes its slice.
ULYSSES=${ULYSSES:-$((GPUS / TP))}
ENCODER_PARALLEL=${ENCODER_PARALLEL:-auto}
# OFFLOAD=1 moves text_encoder + VAEs to CPU: -52.8 GiB per GPU for +7.9% latency. Needed to fit
# a 1-GPU replica under a 96 GB cap; requires the cpu-offload patch, which this script applies.
OFFLOAD=${OFFLOAD:-}
PORT=${PORT:-30010}
VARIANT=${VARIANT:-fl2va}
# Every resolution served must be warmed, or the first request at a cold shape pays about 10 s.
# parse_size takes raw WxH and bypasses the canonical short-edge validator, so 864x480 warms up
# even on an unpatched server. Default warms ONLY 864x480 -- the recommended shape -- because
# warming a resolution you never serve just adds startup time. Serving 768p too:
#   WARMUP="1344x768 864x480" ./serve.sh
WARMUP=${WARMUP:-"864x480"}
# Comma-separated extra short edges. Empty leaves the released 768-only policy untouched.
SHORT_EDGES=${SHORT_EDGES-480}

# Local weights dir on the host (mounted as /models/MiniMax-H3), or an HF repo id via MODEL=.
WEIGHTS=${WEIGHTS:-/opt/dlami/nvme/h3-fl2va}
OUTDIR=${OUTDIR:-/opt/dlami/nvme/out}
MODEL=${MODEL:-/models/MiniMax-H3}
HERE=$(cd "$(dirname "$0")" && pwd)
PATCHDIR=$HERE/patches

case "${1:-start}" in
  stop)
    # Kill the server but keep the container, so the patched source survives.
    docker exec "$NAME" bash -lc 'pkill -f "sglang serve" || true; sleep 8; pkill -9 -f sglang || true; sleep 5'
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
    exit 0 ;;
  logs)   exec docker exec "$NAME" bash -lc "tail -f /out/serve.log" ;;
  status)
    printf 'health=%s\n' "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health")"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
    exit 0 ;;
  start) ;;
  *) echo "usage: $0 [start|stop|logs|status]" >&2; exit 2 ;;
esac

mkdir -p "$OUTDIR"

if ! docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "== creating container $NAME"
  args=(--name "$NAME" -d --gpus all --ipc=host --network host --shm-size 32g
        -v "$OUTDIR:/out" -v "$PATCHDIR:/patches:ro" -e HF_HOME=/out/hf)
  # Only mount weights when serving from a local directory.
  [ "$MODEL" = "/models/MiniMax-H3" ] && args+=(-v "$WEIGHTS:/models/MiniMax-H3:ro")
  docker run "${args[@]}" "$IMAGE" sleep infinity
fi

echo "== applying patches (idempotent)"
docker exec "$NAME" bash -lc '
  set -e
  cd /sgl-workspace/sglang
  for p in /patches/*.patch; do
    n=$(basename "$p")
    if git apply -p1 --check "$p" 2>/dev/null; then
      git apply -p1 "$p" && echo "APPLIED       $n"
    elif git apply -p1 -R --check "$p" 2>/dev/null; then
      echo "ALREADY       $n"
    else
      echo "DOES_NOT_APPLY $n -- image moved off c7c03ec53b, re-diff before trusting it" >&2
      exit 1
    fi
  done'

echo "== starting server: gpus=$GPUS tp=$TP ulysses=$ULYSSES encoder-parallel=$ENCODER_PARALLEL" \
     "offload=${OFFLOAD:-0} short_edges='${SHORT_EDGES:-<none>}' warmup='$WARMUP'"
docker exec -d "$NAME" bash -lc "
  cd /sgl-workspace/sglang
  ${SHORT_EDGES:+SGLANG_MINIMAX_H3_EXTRA_SHORT_EDGES=$SHORT_EDGES} \
  sglang serve \
    --model-path $MODEL \
    --model-variant $VARIANT \
    --num-gpus $GPUS \
    --tp-size $TP \
    --ulysses-degree $ULYSSES \
    --performance-mode speed \
    --encoder-parallel $ENCODER_PARALLEL \
    ${OFFLOAD:+--text-encoder-cpu-offload --vae-cpu-offload} \
    --warmup-resolutions $WARMUP \
    --host 0.0.0.0 --port $PORT > /out/serve.log 2>&1"

echo "== waiting for readiness (weights load + warmup; measured ~90 s at 8 GPUs)"
for i in $(seq 1 60); do
  sleep 10
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" || true)
  if [ "$code" = "200" ]; then
    echo "ready after $((i*10))s"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
    echo "submit the recommended request with: python3 $HERE/h3req.py"
    echo "  (or override: h3req.py [short_edge [steps [duration_s [out-prefix]]]])"
    exit 0
  fi
  if ! docker exec "$NAME" bash -lc 'pgrep -f "sglang serve" >/dev/null'; then
    echo "server died -- last log:" >&2
    docker exec "$NAME" bash -lc 'tail -40 /out/serve.log' >&2
    exit 1
  fi
done
echo "timed out waiting for health" >&2
docker exec "$NAME" bash -lc 'tail -40 /out/serve.log' >&2
exit 1
