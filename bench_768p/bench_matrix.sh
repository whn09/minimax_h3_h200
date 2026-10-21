#!/usr/bin/env bash
# 768p / 124-frame / 50-step optimization ladder x {T2VA, FL2VA, Ref2VA-image, Ref2VA-video} on
# 8xH200 Ulysses=8, with a per-stage breakdown of every request (see req_matrix.py).
#
#   ./bench_matrix.sh                                  # everything (~2.5 h)
#   PARTS=fl2va ARMS="exact lossless" N=3 ./bench_matrix.sh
#
# One server per (partition, arm); each arm differs from the previous one by one lever:
#   exact         stock reference path, request quality="exact" (bit-exact vs reference)
#   lossless      quality="lossless" (fused RMSNorm+AdaLN, VAE QK-norm/RoPE fusion + cuDNN SDPA;
#                 rounding-level only) + --warmup-resolutions 1344x768
#   compile       lossless + --enable-torch-compile true (sglang: changes numerics, not default)
#   fp8           lossless + --quantization fp8 (online w8a8 DiT)                        [lossy]
#   fp8sage       fp8 + SageAttention for the DiT (h3-sglang-sage:dev = stock + sage)     [lossy]
#   fp8sagecache  fp8sage + generic Cache-DiT RDT=$RDT; request carries NO quality field,
#                 because any explicit quality takes H3 off the generic Cache-DiT path    [lossy]
# quality="high" (audited Cache-DiT) is not an arm: sglang fail-closes it outside 4xH200 fl2va.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PARTS=${PARTS:-"fl2va ref2va"}
ARMS=${ARMS:-"exact lossless compile fp8 fp8sage fp8sagecache"}
N=${N:-3}; WARM=${WARM:-1}; RDT=${RDT:-0.24}
OUT=${OUT:-$HERE/matrix_$(date +%Y%m%d_%H%M%S)}
STOCK=lmsysorg/sglang:dev; SAGE=h3-sglang-sage:dev
STAGE_ENVS="-e SGLANG_DIFFUSION_STAGE_LOGGING=1 -e SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=1"
WR="--warmup-resolutions 1344x768"
# sage handles only the DiT shapes; --attention-backend is global, so pin the others to sdpa.
SAGE_FLAGS="--attention-backend sage_attn --component-attention-backends text_encoder=torch_sdpa,audio_vae=torch_sdpa,video_vae=torch_sdpa"
CACHE_ENVS="-e SGLANG_CACHE_DIT_ENABLED=1 -e SGLANG_CACHE_DIT_RDT=$RDT -e SGLANG_CACHE_DIT_SECONDARY_RDT=$RDT"
mkdir -p "$OUT"

tasks_of() { case $1 in fl2va) echo "t2va fl2va";; ref2va) echo "ref2va_image ref2va_video";; esac; }

arm() {  # -> IMAGE|EXTRA|ENVS|QUALITY ('|' not tab: tab is IFS whitespace, so empty fields would collapse)
  case $1 in
    exact)        printf '%s|%s|%s|%s' "$STOCK" "" "" exact ;;
    lossless)     printf '%s|%s|%s|%s' "$STOCK" "$WR" "" lossless ;;
    compile)      printf '%s|%s|%s|%s' "$STOCK" "$WR --enable-torch-compile true" "" lossless ;;
    fp8)          printf '%s|%s|%s|%s' "$STOCK" "$WR --quantization fp8" "" lossless ;;
    fp8sage)      printf '%s|%s|%s|%s' "$SAGE" "$WR --quantization fp8 $SAGE_FLAGS" "" lossless ;;
    fp8sagecache) printf '%s|%s|%s|%s' "$SAGE" "$WR --quantization fp8 $SAGE_FLAGS" "$CACHE_ENVS" none ;;
    *) echo "unknown arm $1" >&2; return 1 ;;
  esac
}

for part in $PARTS; do
  for a in $ARMS; do
    IFS='|' read -r image extra envs quality <<<"$(arm "$a")"
    cell="$OUT/${part}_$a"; mkdir -p "$cell"
    echo "== $(date -u +%T) $part/$a image=$image quality=$quality extra=[$extra] envs=[$envs]" | tee -a "$OUT/matrix.log"
    IMAGE=$image VARIANT=$part EXTRA="$extra" ENVS="$STAGE_ENVS $envs" "$HERE/serve_sglang.sh" > "$cell/image.txt"
    ok=0
    for i in $(seq 120); do
      curl -sf localhost:30010/health >/dev/null && { ok=1; break; }
      docker ps -q -f name=h3-sglang | grep -q . || break
      sleep 10
    done
    docker logs h3-sglang > "$cell/server_start.log" 2>&1 || true
    if [ $ok != 1 ]; then echo "SERVER FAILED $part/$a" | tee -a "$OUT/matrix.log"; continue; fi
    # what the server actually resolved -- the arm is only real if these lines say so
    grep -aoiE 'Using [a-z0-9_]+ attention backend|Attention backends for [a-z_]+: [a-z0-9_, ]+|quantiz[a-z]*[^,]{0,60}|cache-dit enabled[^$]{0,80}|torch\.compile[^$]{0,60}|performance_mode=[a-z]+' \
      "$cell/server_start.log" | sort | uniq -c > "$cell/resolved.txt" || true
    for t in $(tasks_of "$part"); do
      python3 "$HERE/req_matrix.py" "$t" "$cell" --n "$N" --warm "$WARM" --quality "$quality" \
        --tag "$t" 2>&1 | tee -a "$cell/client.log" | grep SUMMARY | sed "s#^#$part/$a #" | tee -a "$OUT/matrix.log" \
        || echo "CLIENT FAILED $part/$a/$t" | tee -a "$OUT/matrix.log"
    done
    docker logs h3-sglang 2>&1 | tr '\r' '\n' | grep -aiE 'cache-dit|cached steps|skipped' | tail -20 > "$cell/cache_dit.log" || true
  done
done
docker rm -f h3-sglang >/dev/null 2>&1 || true
echo "== done $(date -u +%T) -> $OUT" | tee -a "$OUT/matrix.log"
