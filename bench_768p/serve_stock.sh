#!/usr/bin/env bash
# Stock MiniMax-H3 sglang server for the 768p / 124-frame / 50-step Trainium2 comparison.
# Exactly the reference command (lmsysorg/sglang:dev, 8 GPUs, Ulysses=8, fl2va, --performance-mode
# speed) -- no patches, no quantization, no cache. Only the HF cache path is adapted: weights live on
# the instance NVMe, mounted at the container's default HF cache.
#   HF_CACHE=/opt/dlami/nvme/hf ./serve_stock.sh        # then: docker logs -f h3-stock
set -euo pipefail
IMAGE=${IMAGE:-lmsysorg/sglang:dev}
HF_CACHE=${HF_CACHE:-/opt/dlami/nvme/hf}
NAME=${NAME:-h3-stock}
PORT=${PORT:-30010}

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus all \
  --shm-size 32g --ipc=host \
  -p "$PORT:$PORT" \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  ${HF_TOKEN:+--env HF_TOKEN=$HF_TOKEN} \
  "$IMAGE" \
  sglang serve \
    --model-path MiniMaxAI/MiniMax-H3 \
    --num-gpus 8 \
    --ulysses-degree 8 \
    --host 0.0.0.0 \
    --port "$PORT" \
    --model-variant fl2va \
    --performance-mode speed
docker inspect --format '{{.Image}}' "$NAME"
