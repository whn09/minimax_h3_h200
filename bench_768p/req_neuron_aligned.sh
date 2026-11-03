#!/usr/bin/env bash
# The aligned bench_768p requests against the Neuron server (/v1/videos/sync), 1344x768 / 124 frames /
# 50 steps / shifts 12+3, WARM excluded warmups + N measured back to back, client wall (curl time_total).
# Prompts and seeds are minimax_h3_h200/bench_768p's req_vomni.sh / req_matrix.py, character for character:
#   TASK=t2va         the original reference request (Night-vision prompt, seed 1101)
#   TASK=t2va_parrot  vllm-omni-neuron's own E2E request (parrot prompt, seed 42)
#   TASK=t2va_atnight the vllm-omni reference prompt (seed 1101), which the 2026-10 Trn2 API rows used
#   TASK=fl2va        vllm-omni-neuron #436's HF model card request (first keyframe + prompt, seed 0);
#                     FL2VA_DIR holds bench_768p/assets_fl2va/{fl2va_input.png,fl2va_prompt.txt}
#   N=5 WARM=1 TASK=t2va OUT=out LOG=serve.log ./req_neuron_aligned.sh     # GAP=90 idles between requests
set -euo pipefail
N=${N:-5}; WARM=${WARM:-1}; OUT=${OUT:-out_api}; PORT=${PORT:-8000}; LOG=${LOG:-serve.log}; TASK=${TASK:-t2va}
mkdir -p "$OUT"
FL2VA_DIR=${FL2VA_DIR:-$(dirname "$0")/assets_fl2va}
task=t2va; files=()
case $TASK in
  t2va) seed=1101
    prompt='Night-vision bedroom footage: while the owner sleeps, three cats burst in playing tiny brass instruments at full volume, freeze, then march out as if nothing happened.' ;;
  t2va_parrot) seed=42
    prompt='A sun conure parrot springs off a pale perch, opens its orange and green wings and flies toward a glass office tower, sunlight flashing off the windows' ;;
  t2va_atnight) seed=1101
    prompt='At night, three cats march into a bedroom playing tiny brass instruments, then abruptly file out, with synchronized room ambience.' ;;
  fl2va) seed=0; task=fl2va; prompt=$(cat "$FL2VA_DIR/fl2va_prompt.txt")
    files=(-F "input_reference=@$FL2VA_DIR/fl2va_input.png;type=image/png") ;;
  *) echo "unknown TASK $TASK" >&2; exit 1 ;;
esac
for i in $(seq 1 $((WARM + N))); do
  tag=$TASK-$([ "$i" -le "$WARM" ] && echo "warmup$i" || echo "run$((i - WARM))")
  lines=$(wc -l < "$LOG")
  t=$(curl --fail-with-body -sS -o "$OUT/$tag.mp4" -w '%{time_total}' \
    -X POST "http://127.0.0.1:$PORT/v1/videos/sync" \
    --form-string "prompt=$prompt" -F 'width=1344' -F 'height=768' -F 'fps=24' -F 'num_frames=124' \
    -F 'num_inference_steps=50' -F 'flow_shift=12' -F "seed=$seed" \
    -F "extra_params={\"task\":\"$task\",\"duration\":5.0,\"audio_flow_shift\":3.0}" "${files[@]}")
  sleep 1
  st=$(tail -n +"$((lines + 1))" "$LOG" | tr '\r' '\n' | grep -aoE 'MiniMax-H3 [0-9x]+, [0-9]+ steps: .*' | tail -1 || true)
  echo "$tag wall=${t}s md5=$(md5sum < "$OUT/$tag.mp4" | cut -c1-12) | $st" | tee -a "$OUT/latency.log"
  [ "${GAP:-0}" -gt 0 ] && sleep "$GAP"  # idle between requests (Trn2 throttles on back-to-back load)
done
grep "^$TASK-run" "$OUT/latency.log" | sed 's/.*wall=\([0-9.]*\)s.*/\1/' | python3 -c \
  'import sys,statistics as s;x=[float(l) for l in sys.stdin];print(f"n={len(x)} mean={s.mean(x):.2f}s min={min(x):.2f}s max={max(x):.2f}s stdev={s.pstdev(x):.3f}s")' \
  | sed "s/^/$TASK /" | tee -a "$OUT/summary.txt"
