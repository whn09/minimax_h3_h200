#!/usr/bin/env bash
# The reference vllm-omni t2va request (/v1/videos/sync), 1344x768 / 5 s / 50 steps / seed 1101,
# sent WARM excluded warmups + N measured times back to back. Prints curl time_total per request;
# keeps the last mp4 and checks it with ffprobe/ffmpeg inside the server container.
#   N=5 WARM=1 OUT=./out_vomni ./req_vomni.sh
set -euo pipefail
N=${N:-5}; WARM=${WARM:-1}; OUT=${OUT:-out_vomni}; PORT=${PORT:-8000}
WIDTH=${WIDTH:-1344}; HEIGHT=${HEIGHT:-768}; NAME=${NAME:-h3-vomni}
mkdir -p "$OUT"
for i in $(seq 1 $((WARM + N))); do
  tag=$([ "$i" -le "$WARM" ] && echo "warmup$i" || echo "run$((i - WARM))")
  t=$(curl --fail-with-body -sS -o "$OUT/$tag.mp4" -w '%{time_total}' \
    -X POST "http://127.0.0.1:$PORT/v1/videos/sync" \
    -F 'prompt=At night, three cats march into a bedroom playing tiny brass instruments, then abruptly file out, with synchronized room ambience.' \
    -F "width=$WIDTH" -F "height=$HEIGHT" -F 'fps=24' \
    -F 'num_inference_steps=50' -F 'flow_shift=12' -F 'seed=1101' \
    -F 'extra_params={"task":"t2va","duration":5.0,"audio_flow_shift":3.0}')
  echo "$tag wall=${t}s md5=$(md5sum < "$OUT/$tag.mp4" | cut -c1-12)" | tee -a "$OUT/latency.log"
done
docker cp "$OUT/$tag.mp4" "$NAME:/tmp/check.mp4"
docker exec "$NAME" ffprobe -v error -show_entries \
  stream=index,codec_name,width,height,r_frame_rate,sample_rate,channels,nb_frames -of compact /tmp/check.mp4 \
  | tee "$OUT/ffprobe.txt"
docker exec "$NAME" ffmpeg -v error -i /tmp/check.mp4 -map 0:v:0 -map 0:a:0 -f null - && echo "decode OK"
grep '^run' "$OUT/latency.log" | sed 's/.*wall=\([0-9.]*\)s.*/\1/' | python3 -c \
  'import sys,statistics as s;x=[float(l) for l in sys.stdin];print(f"n={len(x)} mean={s.mean(x):.2f}s min={min(x):.2f}s max={max(x):.2f}s stdev={s.pstdev(x):.3f}s")' \
  | tee "$OUT/summary.txt"
