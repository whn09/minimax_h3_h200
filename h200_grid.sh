#!/usr/bin/env bash
# H200（p5e/p5en.48xlarge，8×H200 141 GB）上的四臂网格：BF16 → FP8 → +SageAttention → +Cache-DiT。
#
#   ./h200_grid.sh                       # 默认 GPUS="1 8" × 四臂 × 四档 = 32 条
#   GPUSETS="1" ARMS="bf16 fp8sagecache" ./h200_grid.sh
#   CASES="768_30" GPUSETS="1 2 4 8" ARMS=fp8sagecache ./h200_grid.sh    # 加卡曲线
#   TASK=ref2va ./h200_grid.sh           # 需要 Ref2VA 分区（serve.sh 会自己选 30030 端口）
#
# 为什么四臂而不是直接量交付配置：g7e 那边这三层各自的收益是**分开量过**的（FP8 1.073×、
# +sage 1.26–1.29×、+Cache-DiT 2.1–2.3×），H200 上只报一个总数没法回答"哪一层在这张卡上还成立"。
# 尤其 sage：它在 sm_120 上值 1.776×（孤立），而 sm_90 的 fp8 tensor core 本来就快，收益可能被吃掉。
#
# 三条口径，别改：
#   1. **每个 gpus 档自己带 bf16 分母。** 跨机器/跨重启的分母是错的（同配置跨重启实测漂 5.4%）。
#   2. **臂之间只差该差的那一项**，包括 text_encoder 的 layerwise offload —— 它对所有臂都开。
#      理由不是显存（141 GB 装得下），是 g7e 那条线一直开着，比较时少一个变量。text encoder 只
#      花 0.65 s，offload 的延迟代价在噪声里。
#   3. **Cache-DiT 走 env（通用路径），请求里不能带 `quality`** —— 带了就把通用 Cache-DiT 关掉
#      （`super()._cache_dit_requested() and "quality" not in explicit_fields`）。h3gen.py 不传
#      --quality 就不会带。
#
# 输出一行一条 `REPRO <tag> rc= inference_time_s= wall_s= mem=`，用 tee 落到 $OUT/grid_*.log。
set -u
cd "${WORKDIR:-$(cd "$(dirname "$0")" && pwd)}"
IMAGE=${IMAGE:-h3-h200:local}
NAME=${NAME:-h3}
OUT=${OUT:-/opt/dlami/nvme/out}
TASK=${TASK:-fl2va}
GPUSETS=${GPUSETS:-"1 8"}
ARMS=${ARMS:-"bf16 fp8 fp8sage fp8sagecache"}
CASES=${CASES:-"480_20 480_30 768_20 768_30"}
# Cache-DiT 的残差阈值。0.24 是 g7e 交付档（30 步 2.1–2.3×，画质在地板之上）。
RDT=${RDT:-0.24}
# 追加到 tag 末尾。**扫 RDT 时必须传**（例如 TAGSUF=_R16）—— tag 里没有 RDT，不传就静默覆盖
# 同名 mp4 和 status.json，只剩最后一档的画质。
TAGSUF=${TAGSUF:-}
# 换 transformer 权重（LoRA 合并后的 bf16 目录）。空 = 用 checkpoint 自带的那份。
# H200 上这个目录是 **bf16**：`--quantization fp8` 是在线量化，量的就是这份权重，
# 所以不需要像 g7e 那样再离线量化一次。见 h200_turbo.sh。
CKPT=${CKPT:-}
IMG=${IMG:-$([ -f assets/input_cat.jpg ] && echo assets/input_cat.jpg || echo assets/first.png)}
mkdir -p "$OUT"
. assets/prompts.sh

case $TASK in
  fl2va)  PORT=30010; SEED=${SEED:-6201}; TAGMID=""; REFENV=""
          PROMPT=${PROMPT:-$FL2VA_PROMPT} ;;
  ref2va) PORT=30030; SEED=${SEED:-8201}; REF_SHORT_EDGE=${REF_SHORT_EDGE:-1024}
          TAGMID="_r${REF_SHORT_EDGE}"; REFENV="SGLANG_MINIMAX_H3_REF_IMAGE_SHORT_EDGE=$REF_SHORT_EDGE"
          PROMPT=${PROMPT:-$REF2VA_PROMPT} ;;
  *) echo "TASK must be fl2va|ref2va" >&2; exit 2 ;;
esac

# 一个臂 = 一套 EXTRA/ENVX。sage 一开就必须把 text_encoder / 两个 VAE 摘出去走 torch_sdpa：
# `--attention-backend` 是**全局**的（issue #35743），不摘会让 sage 去处理它接不了的形状。
arm_flags() {  # arm_flags <arm> -> 打印 "EXTRA\tENVX"
  local base_extra="--layerwise-offload-components text_encoder${CKPT:+ --transformer-weights-path $CKPT}"
  local sage_extra="--attention-backend sage_attn --component-attention-backends text_encoder=torch_sdpa,audio_vae=torch_sdpa,video_vae=torch_sdpa"
  local envx="SGLANG_USE_RUNAI_MODEL_STREAMER=0 $REFENV"
  case $1 in
    bf16)         printf '%s\t%s' "$base_extra" "$envx" ;;
    fp8)          printf '%s\t%s' "$base_extra --quantization fp8" "$envx" ;;
    fp8sage)      printf '%s\t%s' "$base_extra --quantization fp8 $sage_extra" "$envx" ;;
    fp8sagecache) printf '%s\t%s' "$base_extra --quantization fp8 $sage_extra" \
                    "$envx SGLANG_CACHE_DIT_ENABLED=1 SGLANG_CACHE_DIT_RDT=$RDT SGLANG_CACHE_DIT_SECONDARY_RDT=$RDT" ;;
    *) echo "unknown arm $1" >&2; return 2 ;;
  esac
}

run_arm() {  # run_arm <gpus> <arm>
  local gpus=$1 arm=$2 extra envx
  IFS=$'\t' read -r extra envx <<<"$(arm_flags "$arm")" || return 2
  echo "=== ARM task=$TASK gpus=$gpus arm=$arm $(date -u +%FT%TZ)"
  (unset VARIANT; ./serve.sh stop) >/dev/null 2>&1
  VARIANT=$TASK GPUS=$gpus IMAGE=$IMAGE NAME=$NAME \
    ENVX="$envx" EXTRA="$extra" \
    LOG=/out/serve_grid_${TASK}_${arm}_g$gpus.log ./serve.sh start \
      > "$OUT/start_grid_${TASK}_${arm}_g$gpus.log" 2>&1 \
    || { echo "SERVER_FAILED gpus=$gpus arm=$arm (看 $OUT/serve_grid_${TASK}_${arm}_g$gpus.log)"; return 1; }

  local c se st tag t0 t1 rc inf
  for c in $CASES; do
    se=${c%_*}; st=${c#*_}; tag="${TASK}_${se}_${st}${TAGMID}_${arm}_g${gpus}${TAGSUF}"
    t0=$(date +%s)
    python3 h3gen.py --task "$TASK" --image "$IMG" --inline \
      --short-edge "$se" --aspect 16:9 --duration 5.0 --steps "$st" \
      --seed "$SEED" --flow-shift 12.0 --audio-flow-shift 3.0 \
      --prompt "$PROMPT" --port "$PORT" --out "$tag" > "${tag}_client.log" 2>&1
    rc=$?; t1=$(date +%s)
    inf=$(python3 - "$tag" <<'PY' 2>/dev/null
import json, sys
def dig(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == "inference_time_s": return v
            r = dig(v)
            if r is not None: return r
    elif isinstance(o, list):
        for v in o:
            r = dig(v)
            if r is not None: return r
print(round(dig(json.load(open(sys.argv[1] + "_status.json"))), 3))
PY
)
    echo "REPRO $tag rc=$rc inference_time_s=${inf:-NA} wall_s=$((t1-t0)) mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | paste -sd/ -)"
  done

  # 回读真正落地的东西，**在跑完之后**：attention 后端会静默降级，Cache-DiT 挂不上也只是不打印
  # 加速。放在启动之后立刻读会漏 —— cache-dit 的那行是 warmup 时才打的，和 /health 变绿是同一秒级。
  docker exec "$NAME" bash -lc "tr '\r' '\n' < /out/serve_grid_${TASK}_${arm}_g$gpus.log \
    | grep -oiE 'cache-dit enabled on transformer[^$]{0,60}|Acceleration hooks is disabled[^$]{0,40}|sageattn[a-z0-9_]*|\"attention_backend\": [^,]*|\"quantization\": [^,]*' \
    | sort -u | head -8" 2>/dev/null | sed 's/^/    landed: /'
}

for g in $GPUSETS; do
  for a in $ARMS; do
    run_arm "$g" "$a"
  done
done
(unset VARIANT; ./serve.sh stop) >/dev/null 2>&1
echo "H200_GRID_DONE task=$TASK gpusets='$GPUSETS' arms='$ARMS' $(date -u +%FT%TZ)"
