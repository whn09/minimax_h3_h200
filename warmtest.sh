#!/usr/bin/env bash
# Does --warmup-resolutions actually warm the shapes you name, and what does adding 1344x768 cost?
#
#   bash warmtest.sh "1344x768 864x480" both      # the new default
#   bash warmtest.sh "864x480"          only480   # the old default, for the paired comparison
#
# Open question this answers: a run with warmup_resolutions=["864x480"] in server_args logged its
# only warmup request as `server warmup req (1344x768x124f, 2/50 steps)`. Either the flag is
# ignored for MiniMax-H3 (then 480p never gets warmed and the fix is a patch, not a flag), or that
# run simply predates something. This prints what the scheduler actually saw.
#
# Runs on GPUs 4-7 by default and REFUSES to start if they are not free -- a hand-rolled launch
# that ignored CUDA_VISIBLE_DEVICES already collided with a live replica on GPU 0 and died with
# `Failed to load customized transformer` (really a CUDA OOM two frames up). Launching through
# serve.sh keeps the isolation logic that is actually validated.
set -u
WARMUP=${WARMUP:-${1:-"1344x768 864x480"}}
TAG=${2:-warm}
DEVS=${DEVS:-4,5,6,7}
PORT=${PORT:-30050}
HERE=$(cd "$(dirname "$0")" && pwd)

echo "== checking GPUs $DEVS are free"
busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
       awk -F', ' -v d="$DEVS" 'index(","d",", ","$1",")>0 && $2>1024 {print $1":"$2"MiB"}')
[ -n "$busy" ] && { echo "REFUSING: $busy in use. Stop that replica or set DEVS=."; exit 1; }

echo "== launching via serve.sh, warmup='$WARMUP'"
T0=$(date +%s)
CUDA_VISIBLE_DEVICES=$DEVS WARMUP="$WARMUP" PORT=$PORT MASTER=30140 SCHED=5740 \
  LOG=/out/serve_wt_$TAG.log "$HERE/serve.sh" || exit 1
echo "== serve.sh returned after $(( $(date +%s) - T0 ))s (its own poll is 10 s granular)"

echo "== which shapes did warmup actually run?"
docker exec h3 bash -lc \
  "tr '\r' '\n' < /out/serve_wt_$TAG.log | grep -o '[a-z ]*warmup req ([^)]*)' | sort | uniq -c"

echo "== per-GPU memory"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

# Cold first, then a warm repeat of each: the gap is what warming that shape is worth.
cd /opt/dlami/nvme/out || exit 1
for pass in cold warm; do
  for se in 768 480; do
    steps=$([ "$se" = 768 ] && echo 12 || echo 40)
    echo "== $pass ${se}p / $steps steps"
    python3 h3gen.py --port $PORT --short-edge $se --aspect 16:9 --steps $steps --duration 5 \
      --out "${TAG}_${se}_$pass" --seed 1101 2>&1 | tail -3
  done
done

echo "== peak memory after both shapes"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
# NOT `serve.sh stop`: its pattern is `model-variant fl2va`, which would also kill a main fl2va
# replica on the other 4 GPUs. Match this replica's own port instead.
echo "== stop this replica only with:"
echo "   docker exec h3 bash -lc \"pkill -f 'port $PORT' ; sleep 8; pkill -9 -f 'port $PORT' || true\""
