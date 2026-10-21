#!/usr/bin/env bash
# vllm-omni requests (/v1/videos/sync), 1344x768 / 5 s / 50 steps / shifts 12+3, sent WARM excluded
# warmups + N measured times back to back. Prints curl time_total per request and, when the server
# runs with --enable-diffusion-pipeline-profiler, that request's stage times (rank-0 lines only).
#   N=5 WARM=1 OUT=./out_vomni ./req_vomni.sh                  # TASK=t2va (the reference curl)
#   TASK=fl2va|ref2va_image|ref2va_video ...                    # same inputs/prompts/seeds as req_matrix.py
# ASSETS is the host dir with first.png / ref5s.mp4 (uploaded as multipart files).
# ASPECT=16:9 adds the aspect_ratio field, which newer images (nightly 0.32) require for t2va and the
# minimax-h3 image predates; leave it unset there.
set -euo pipefail
N=${N:-5}; WARM=${WARM:-1}; OUT=${OUT:-out_vomni}; PORT=${PORT:-8000}; TASK=${TASK:-t2va}
WIDTH=${WIDTH:-1344}; HEIGHT=${HEIGHT:-768}; NAME=${NAME:-h3-vomni}
ASSETS=${ASSETS:-/opt/dlami/nvme/assets}
ASPECT=${ASPECT:-}
mkdir -p "$OUT"
case $TASK in
  t2va) task=t2va; seed=1101; files=()
    prompt='At night, three cats march into a bedroom playing tiny brass instruments, then abruptly file out, with synchronized room ambience.' ;;
  fl2va) task=fl2va; seed=2101; files=(-F "input_reference=@$ASSETS/first.png;type=image/png")
    prompt='A dim bedroom at night. Three kittens keep tumbling and batting at each other on the patterned rug in front of the radiator, tails flicking, while the person in the bed stays asleep under the blanket. The bedside lamp keeps its steady warm glow, the tall curtains stir faintly, night sky outside the windows. Warm low-key lamp light, static cinematic camera, quiet room tone with soft paw pats and faint fabric rustle.' ;;
  ref2va_image) task=ref2va; seed=3101; files=(-F "input_reference=@$ASSETS/first.png;type=image/png")
    prompt='Use <Picture 1> as the visual subject. The same dim night bedroom: the three kittens go on tussling on the rug by the radiator, the sleeping person does not stir, the bedside lamp holds its warm glow and the curtains move slightly. Preserve the layout, colours and low-key lighting of the reference, realistic coherent motion, static cinematic camera, synchronized quiet ambience.' ;;
  ref2va_video) task=ref2va; seed=4101; files=(-F "input_references=@$ASSETS/ref5s.mp4;type=video/mp4")
    prompt='Follow the motion and appearance of <Video 1>, changing the setting to a moonlit bedroom while preserving coherent timing.' ;;
  *) echo "unknown TASK $TASK" >&2; exit 1 ;;
esac
for i in $(seq 1 $((WARM + N))); do
  tag=$TASK-$([ "$i" -le "$WARM" ] && echo "warmup$i" || echo "run$((i - WARM))")
  since=$(date +%s)
  t=$(curl --fail-with-body -sS -o "$OUT/$tag.mp4" -w '%{time_total}' \
    -X POST "http://127.0.0.1:$PORT/v1/videos/sync" \
    -F "prompt=$prompt" -F "width=$WIDTH" -F "height=$HEIGHT" ${ASPECT:+-F "aspect_ratio=$ASPECT"} -F 'fps=24' \
    -F 'num_inference_steps=50' -F 'flow_shift=12' -F "seed=$seed" \
    -F "extra_params={\"task\":\"$task\",\"duration\":5.0,\"audio_flow_shift\":3.0}" "${files[@]}")
  sleep 1
  st=$(docker logs --since "$since" "$NAME" 2>&1 | tr '\r' '\n' \
    | grep -aoE 'MiniMaxH3Pipeline\.[a-z_]+ took [0-9.]+s|e2e_total_ms \| [0-9,.]+|MP4 bytes\): [0-9.]+ ms' \
    | awk '!seen[$1]++' | sed 's/MiniMaxH3Pipeline\.//; s/ took /=/; s/e2e_total_ms | /engine_ms=/; s/MP4 bytes): /mp4_ms=/; s/ ms$//' | tr '\n' ' ' | tr -d ',')
  echo "$tag wall=${t}s md5=$(md5sum < "$OUT/$tag.mp4" | cut -c1-12) $st" | tee -a "$OUT/latency.log"
done
docker cp "$OUT/$tag.mp4" "$NAME:/tmp/check.mp4"
docker exec "$NAME" ffprobe -v error -show_entries \
  stream=index,codec_name,width,height,r_frame_rate,sample_rate,channels,nb_frames -of compact /tmp/check.mp4 \
  | tee "$OUT/$TASK-ffprobe.txt"
docker exec "$NAME" ffmpeg -v error -i /tmp/check.mp4 -map 0:v:0 -map 0:a:0 -f null - && echo "decode OK"
grep "^$TASK-run" "$OUT/latency.log" | sed 's/.*wall=\([0-9.]*\)s.*/\1/' | python3 -c \
  'import sys,statistics as s;x=[float(l) for l in sys.stdin];print(f"n={len(x)} mean={s.mean(x):.2f}s min={min(x):.2f}s max={max(x):.2f}s stdev={s.pstdev(x):.3f}s")' \
  | sed "s/^/$TASK /" | tee -a "$OUT/summary.txt"
