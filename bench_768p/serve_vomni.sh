#!/usr/bin/env bash
# Stock vllm-omni MiniMax-H3 server (vllm/vllm-omni:minimax-h3) -- the reference 4-GPU command with
# every "4" raised to 8: Ulysses=8, ring 1, native tiled VAE patch parallel 8. No other changes.
# The image has no entrypoint, so `vllm serve` is spelled out. The model path must be a partition
# subdirectory (FL2VA serves t2va+fl2va, Ref2VA serves ref2va); the repo root's model_index.json is
# the diffusers *modular* pipeline, which the registry does not know.
#   ./serve_vomni.sh                  # FL2VA, port 8000
#   PART=Ref2VA ./serve_vomni.sh
#   EXTRA="--text-encoder-tp-size 8" ./serve_vomni.sh
set -euo pipefail
IMAGE=${IMAGE:-vllm/vllm-omni:minimax-h3}
HF_CACHE=${HF_CACHE:-/opt/dlami/nvme/hf}
SNAP=${SNAP:-42ed227ee7df40d41602854ae760620d6eb651fe}
PART=${PART:-FL2VA}
NGPU=${NGPU:-8}
NAME=${NAME:-h3-vomni}
PORT=${PORT:-8000}
MODEL=/root/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-H3/snapshots/$SNAP/$PART

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus all \
  --privileged --ipc=host -p "$PORT:$PORT" \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
  ${ENVS:-} \
  "$IMAGE" \
  vllm serve "$MODEL" \
    --trust-remote-code \
    --host 0.0.0.0 \
    --num-gpus "$NGPU" \
    --usp "$NGPU" \
    --ring 1 \
    --vae-patch-parallel-size "$NGPU" \
    --vae-parallel-mode tile \
    --vae-use-tiling \
    --port "$PORT" \
    --omni ${EXTRA:-}
docker inspect --format '{{.Image}}' "$NAME"
