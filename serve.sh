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
# THREE DEPLOYMENT MODES:
#
#   (1) fl2va on all or some GPUs        -- serves t2va + fl2va, port 30010
#         ./serve.sh                                  all 8
#         GPUS=4 ./serve.sh                           4 GPUs, sglang picks which
#         CUDA_VISIBLE_DEVICES=0,1,2,3 ./serve.sh     those 4 exactly (GPUS is inferred)
#
#   (2) ref2va on all or some GPUs       -- serves ref2va, port 30030
#         VARIANT=ref2va ./serve.sh
#         VARIANT=ref2va CUDA_VISIBLE_DEVICES=4,5,6,7 ./serve.sh
#
#   (3) both at once, isolated by CUDA_VISIBLE_DEVICES -- all three tasks on one box
#         ./serve.sh both                             4 + 4
#         GPUS_A=2 GPUS_B=6 ./serve.sh both           uneven split (ref2va is 3.3x/step slower)
#
# Add DRYRUN=1 to any of these to print the resolved placement without starting anything.
#
# Usage:
#   ./serve.sh                          # <- recommended H200 config, 864x480 @ 40 steps -> 10.05 s
#   ./serve.sh both                     # ALL THREE TASKS: 2 replicas (see below)
#   GPUS=4 ./serve.sh                   # the cookbook's 4xH200 recipe
#   VARIANT=ref2va ./serve.sh           # ref2va only, on port 30030
#   TP=2 ULYSSES=4 ./serve.sh           # shard the DiT: 63.9 GiB/GPU instead of 95.9
#   OFFLOAD=1 GPUS=1 ULYSSES=1 ./serve.sh   # the g7e / 96 GB shape (79.4 GiB/GPU)
#   SHORT_EDGES= ./serve.sh             # unpatched policy (768 only), patch stays inert
#   MODEL=MiniMaxAI/MiniMax-H3 ./serve.sh   # pull weights from HF instead of a local dir
#   OUTPATH= ./serve.sh                 # do not persist videos at all, HTTP fetch only
#   ./serve.sh stop | logs | status     # stop/logs/status act on ALL replicas unless VARIANT is set
#
# TASK COVERAGE. --model-variant selects which DiT is loaded out of the checkpoint, and the
# task -> partition map is a hard gate, not a preference:
#
#   --model-variant fl2va   (default)  serves t2va AND fl2va
#   --model-variant ref2va             serves ref2va only
#
# So ONE process can never serve all three. Asking an fl2va server for ref2va fails with
#   "task 'ref2va' is not served by MiniMax H3 partition 'fl2va'; supported tasks: ['t2va','fl2va']"
# `./serve.sh both` splits the box into two replicas (fl2va on 30010, ref2va on 30030) and the
# client routes by task. Per-variant PORT/MASTER/SCHED defaults below are what keep them from
# colliding; GPUs are isolated with CUDA_VISIBLE_DEVICES because --base-gpu-id is ignored.
#
# Finished videos land on the HOST in $OUTDIR/videos (default /opt/dlami/nvme/out/videos), because
# this script passes --output-path. Without it the server writes a relative `outputs/` inside the
# container, which nothing mounts, and the videos vanish with the container.
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
# GPUs this replica may touch, as a device list. CUDA_VISIBLE_DEVICES is accepted under its own
# name because that is what anyone would reach for; DEVICES is the same knob with a shorter name.
# Empty = every GPU on the box. --base-gpu-id does NOT do this, it is silently ignored, so pinning
# has to go through the environment variable.
DEVICES=${DEVICES:-${CUDA_VISIBLE_DEVICES:-}}
# With a device list and no explicit GPUS, the list decides how many GPUs to use -- otherwise
# `CUDA_VISIBLE_DEVICES=0,1 ./serve.sh` would ask for 8 GPUs out of the 2 it can see and die.
if [ -n "$DEVICES" ] && [ -z "${GPUS:-}" ]; then
  GPUS=$(printf '%s' "$DEVICES" | tr ',' '\n' | grep -c .)
fi
GPUS=${GPUS:-8}
TP=${TP:-1}
# Ulysses defaults to whatever is left after TP takes its slice.
ULYSSES=${ULYSSES:-$((GPUS / TP))}
ENCODER_PARALLEL=${ENCODER_PARALLEL:-auto}
# OFFLOAD=1 moves text_encoder + VAEs to CPU: -52.8 GiB per GPU for +7.9% latency. Needed to fit
# a 1-GPU replica under a 96 GB cap; requires the cpu-offload patch, which this script applies.
OFFLOAD=${OFFLOAD:-}
# Was VARIANT set by the caller, or are we falling back to the default? `stop`/`logs`/`status` use
# this to act on one replica when asked and on all of them otherwise. Must be captured BEFORE the
# default is applied.
VARIANT_EXPLICIT=${VARIANT+yes}
VARIANT=${VARIANT:-fl2va}
# Per-variant defaults, so a second replica needs no extra flags to coexist with the first. The
# scheduler/master ports must be spaced by more than 1 -- sglang derives further ports from them
# and adjacent values collide silently into a hang.
case $VARIANT in
  fl2va)  PORT=${PORT:-30010}; MASTER=${MASTER:-30100}; SCHED=${SCHED:-5700} ;;
  ref2va) PORT=${PORT:-30030}; MASTER=${MASTER:-30120}; SCHED=${SCHED:-5720} ;;
  *) echo "VARIANT must be fl2va (serves t2va+fl2va) or ref2va, got '$VARIANT'" >&2; exit 2 ;;
esac
LOG=/out/serve_$VARIANT.log
# Every resolution served must be warmed, or the first request at a cold shape pays about 10 s.
# parse_size takes raw WxH and bypasses the canonical short-edge validator, so 864x480 warms up
# even on an unpatched server. Default warms ONLY 864x480 -- the recommended shape -- because
# warming a resolution you never serve just adds startup time. Serving 768p too:
#   WARMUP="1344x768 864x480" ./serve.sh
WARMUP=${WARMUP:-"864x480"}
# Comma-separated extra short edges. Empty leaves the released 768-only policy untouched.
SHORT_EDGES=${SHORT_EDGES-480}

# Local weights dir on the host (mounted as /models/MiniMax-H3), or an HF repo id via MODEL=.
# It holds BOTH partitions -- FL2VA/ (serves t2va and fl2va) and Ref2VA/ -- so it is named for the
# model, not for one partition. For VARIANT=ref2va or `both` the Ref2VA/ side must be filled in;
# see fill_ref2va.sh, which downloads only Ref2VA/transformer (62 GiB) and hardlinks the other 16
# files, which are bit-identical to FL2VA's.
# Falls back to the old h3-fl2va name so a box provisioned before the rename keeps working; the
# container's bind mount is created from whichever path wins here.
WEIGHTS=${WEIGHTS:-/opt/dlami/nvme/h3}
if [ ! -d "$WEIGHTS" ] && [ -d /opt/dlami/nvme/h3-fl2va ]; then
  WEIGHTS=/opt/dlami/nvme/h3-fl2va
  echo "== note: using legacy weights dir $WEIGHTS (rename it to /opt/dlami/nvme/h3 when idle)"
fi
OUTDIR=${OUTDIR:-/opt/dlami/nvme/out}
# Where the server persists finished videos. Without this it writes to a relative `outputs/` INSIDE
# the container, which is not mounted, so every generated video dies with the container. Pointing
# it at the already-mounted /out makes them show up on the host in $OUTDIR/videos. Set OUTPATH=""
# to turn persistent saving off entirely and fetch only over HTTP.
OUTPATH=${OUTPATH-/out/videos}
MODEL=${MODEL:-/models/MiniMax-H3}
HERE=$(cd "$(dirname "$0")" && pwd)
PATCHDIR=$HERE/patches

case "${1:-start}" in
  stop)
    # Kill the server but keep the container, so the patched source survives. The [s]glang bracket
    # matters: a plain `pkill -f sglang` also matches this docker exec shell's own command line and
    # kills the killer. With VARIANT set, only that replica dies -- the pattern is the flag itself.
    pat='[s]glang serve'
    [ -n "$VARIANT_EXPLICIT" ] && pat="model-variant $VARIANT"
    docker exec "$NAME" bash -lc "pkill -f '$pat' || true; sleep 8; pkill -9 -f '$pat' || true; sleep 5"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
    exit 0 ;;
  logs)   exec docker exec "$NAME" bash -lc "tail -f $LOG" ;;
  status)
    # Report every port a replica could be on, not just this invocation's, so one command tells you
    # whether all three tasks are actually being served.
    for p in 30010 30030; do
      printf 'port %s health=%s\n' "$p" \
        "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$p/health" || true)"
    done
    docker exec "$NAME" bash -lc 'pgrep -af "[s]glang serve" | grep -o "model-variant [a-z0-9]*" | sort | uniq -c' || true
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
    exit 0 ;;
  both)
    # All three tasks on one box: fl2va replica (t2va+fl2va) + ref2va replica, isolated from each
    # other with CUDA_VISIBLE_DEVICES. Split defaults to half/half but does not have to be even --
    # ref2va costs 3.489 s/step against fl2va's 1.045, so giving it more GPUs is often the right
    # call:  GPUS_A=2 GPUS_B=6 ./serve.sh both
    # Started SEQUENTIALLY -- the second waits for the first to be healthy -- because two replicas
    # initialising at once race on runtime init. Slower to come up (~90 s each), but it either works
    # or tells you which replica failed instead of leaving both half-dead.
    A=${GPUS_A:-$((GPUS / 2))}
    B=${GPUS_B:-$((GPUS - A))}
    [ "$A" -ge "$TP" ] && [ "$B" -ge "$TP" ] \
      || { echo "GPUS_A=$A / GPUS_B=$B: each replica needs at least TP=$TP GPUs" >&2; exit 2; }
    [ $((A + B)) -le "$GPUS" ] \
      || { echo "GPUS_A=$A + GPUS_B=$B exceeds GPUS=$GPUS" >&2; exit 2; }
    # ULYSSES/PORT/MASTER/SCHED are passed explicitly rather than inherited: an explicit
    # ULYSSES=8 in the caller's env would otherwise be handed to a 4-GPU replica. Empty values fall
    # back to the per-variant defaults, since those use ${VAR:-...} and not ${VAR-...}.
    # Re-invoked via an absolute path: `bash serve.sh both` leaves $0 relative, and "." is not on
    # PATH, so `"$0" start` would die with `serve.sh: command not found`.
    SELF="$HERE/$(basename "$0")"
    VARIANT=fl2va  GPUS=$A ULYSSES=$((A / TP)) PORT= MASTER= SCHED= \
      DEVICES="$(seq -s, 0 $((A - 1)))" bash "$SELF" start
    VARIANT=ref2va GPUS=$B ULYSSES=$((B / TP)) PORT= MASTER= SCHED= \
      DEVICES="$(seq -s, $A $((A + B - 1)))" bash "$SELF" start
    echo "== all three tasks served: t2va+fl2va -> :30010 ($A GPUs), ref2va -> :30030 ($B GPUs)"
    echo "   route by task, e.g. h3gen.py --task ref2va --port 30030 ..."
    exit 0 ;;
  start) ;;
  *) echo "usage: $0 [start|both|stop|logs|status]" >&2; exit 2 ;;
esac

# DRYRUN=1 resolves every knob and prints the command that would run, touching neither docker nor
# the GPUs. Worth having because the alternative way to check a placement is a 90 s launch, and the
# thing most likely to be wrong -- which GPUs each replica gets -- is visible in one line.
# It propagates into `both`, so `DRYRUN=1 ./serve.sh both` shows the whole layout.
DRYRUN=${DRYRUN:-}
if [ -n "$DRYRUN" ]; then
  echo "DRYRUN variant=$VARIANT port=$PORT gpus=$GPUS devices=${DEVICES:-<all>} tp=$TP" \
       "ulysses=$ULYSSES master=$MASTER sched=$SCHED log=$LOG"
  echo "  ${DEVICES:+CUDA_VISIBLE_DEVICES=$DEVICES }${SHORT_EDGES:+SGLANG_MINIMAX_H3_EXTRA_SHORT_EDGES=$SHORT_EDGES }" \
       "sglang serve --model-path $MODEL --model-variant $VARIANT --num-gpus $GPUS --tp-size $TP" \
       "--ulysses-degree $ULYSSES ${OFFLOAD:+--text-encoder-cpu-offload --vae-cpu-offload}" \
       "${OUTPATH:+--output-path $OUTPATH} --port $PORT"
  exit 0
fi

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
# Order matters: target-width-height is diffed on top of short-edge, because both edit
# _validate_target in request_validation.py. Listed explicitly rather than left to the glob so the
# dependency is stated, not inherited from alphabetical luck.
docker exec "$NAME" bash -lc '
  set -e
  cd /sgl-workspace/sglang
  STAMPS=/sgl-workspace/.h3-patches; mkdir -p $STAMPS
  for n in minimax-h3-cpu-offload-inplace.patch \
           minimax-h3-short-edge.patch \
           minimax-h3-target-width-height.patch; do
    p=/patches/$n
    [ -f "$p" ] || { echo "MISSING       $n" >&2; exit 1; }
    if git apply -p1 --check "$p" 2>/dev/null; then
      git apply -p1 "$p" && touch "$STAMPS/$n" && echo "APPLIED       $n"
    elif [ -f "$STAMPS/$n" ]; then
      # A stamp is as durable as the patched source itself: both live in this container filesystem
      # and are created together. Needed because `git apply -R --check` stops working as an
      # already-applied test as soon as a LATER patch touches the same hunk context -- which
      # target-width-height does to short-edge.
      echo "ALREADY       $n"
    else
      echo "DOES_NOT_APPLY $n -- does not apply and was never applied by this script;" >&2
      echo "  image likely moved off c7c03ec53b, re-diff with patches/make_patch.sh" >&2
      exit 1
    fi
  done'

echo "== starting server: variant=$VARIANT port=$PORT gpus=$GPUS${DEVICES:+ devices=$DEVICES}" \
     "tp=$TP ulysses=$ULYSSES encoder-parallel=$ENCODER_PARALLEL" \
     "offload=${OFFLOAD:-0} short_edges='${SHORT_EDGES:-<none>}' warmup='$WARMUP'"
docker exec -d "$NAME" bash -lc "
  cd /sgl-workspace/sglang
  ${DEVICES:+CUDA_VISIBLE_DEVICES=$DEVICES} \
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
    ${OUTPATH:+--output-path $OUTPATH} \
    --warmup-resolutions $WARMUP \
    --master-port $MASTER --scheduler-port $SCHED \
    --host 0.0.0.0 --port $PORT > $LOG 2>&1"

echo "== waiting for readiness (weights load + warmup; measured ~90 s at 8 GPUs)"
for i in $(seq 1 60); do
  sleep 10
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" || true)
  if [ "$code" = "200" ]; then
    echo "ready after $((i*10))s"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
    echo "submit the recommended request with: python3 $HERE/h3req.py"
    echo "  (or override: h3req.py [short_edge [steps [duration_s [out-prefix]]]])"
    if [ -n "$OUTPATH" ]; then
      echo "finished videos appear on the host in $OUTDIR/${OUTPATH#/out/}"
    else
      echo "videos are NOT saved (OUTPATH empty); fetch them with GET /v1/videos/<id>/content"
    fi
    exit 0
  fi
  # Match on --model-variant, not on "sglang serve": under `both` the other replica is also alive,
  # and a bare process check would call a dead replica healthy.
  if ! docker exec "$NAME" bash -lc "pgrep -f 'model-variant $VARIANT' >/dev/null"; then
    echo "$VARIANT replica died -- last log:" >&2
    docker exec "$NAME" bash -lc "tail -40 $LOG" >&2
    exit 1
  fi
done
echo "timed out waiting for health on $VARIANT (port $PORT)" >&2
docker exec "$NAME" bash -lc "tail -40 $LOG" >&2
exit 1
