#!/usr/bin/env bash
# Turbo LoRA（`larryvrh/MiniMax-H3-Turbo-Lora`，v4_step600_ema）在 H200 上：能不能和
# `--quantization fp8` / sage / Cache-DiT 叠起来，8 步值多少，画质掉到哪。
#
#   nohup ./h200_turbo.sh > /opt/dlami/nvme/out/h200_turbo.log 2>&1 &
#   PHASES=tc ./h200_turbo.sh          # 只补一个 phase
#
# 前置（一次，CPU，别和计时请求并跑）：
#   mkdir -p /opt/dlami/nvme/out/lora && curl -sL --retry 5 \
#     -o /opt/dlami/nvme/out/lora/minimax_h3_turbo_v4_step600_ema.safetensors \
#     https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/resolve/main/minimax_h3_turbo_v4_step600_ema.safetensors
#   docker cp lora_merge_transformer.py h3:/tmp/
#   docker exec -e SRC=/models/MiniMax-H3/FL2VA/transformer \
#     -e LORA=/out/lora/minimax_h3_turbo_v4_step600_ema.safetensors \
#     -e DST=/out/turbo_v4_600_bf16 h3 python3 /tmp/lora_merge_transformer.py   # 须 259/259
#
# **和 g7e 那条线只差一步：H200 不需要离线量化。** g7e 上要 `nvfp4_quantize_transformer.py` 把
# 合并后的权重转成 NVFP4 文件，因为 NVFP4 checkpoint 是离线格式；H200 的量化档是 `--quantization
# fp8`（在线，load 后 `process_weights_after_loading` 里量），所以直接把 `--transformer-weights-path`
# 指向**合并后的 bf16 目录**就行。`resolve_transformer_safetensors_to_load` 对目录和单文件都接
# （目录走 `_list_safetensors_files`），而 `_resolve_quant_config` 里显式 `--quantization` 优先级
# 最高，所以 override 权重 + 在线 fp8 两件事不打架。
#
# 为什么不用 `--lora-path`：见 lora_merge_transformer.py 的注释（运行时合并在量化权重上形状对不上）。
# 在 H200 上这条同样成立 —— fp8 把权重转置存。
#
# 口径同 h200_grid.sh：fl2va / seed 6201 / 5.175 s 成片 / `inference_time_s`。
set -u
cd "${WORKDIR:-$(cd "$(dirname "$0")" && pwd)}"
TURBO=${TURBO:-/out/turbo_v4_600_bf16}
TURBO_REF=${TURBO_REF:-/out/turbo_ref2va_bf16}
OUT=${OUT:-/opt/dlami/nvme/out}
PHASES=${PHASES:-"t0 to tc tg tr"}

has() { case " $PHASES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

echo "===== H200_TURBO phases='$PHASES' turbo=$TURBO $(date -u +%FT%TZ)"

# t0：stock 20 步的参考，**在这一轮里重跑**而不是引用 RESULTS_QUANT.md 的数。
# 理由：跨重启同配置实测漂 5.4%，倍数和 SSIM 的分母必须和分子同一次开机。
has t0 && { echo "##### t0 (stock FP8 / FP8+sage, 20 步, 1 卡, 参考)"
  GPUSETS=1 ARMS="fp8 fp8sage" CASES="480_20 768_20" TAGSUF=_s20 ./h200_grid.sh; }

# to：turbo 的步数曲线。模型卡说 4–8 步是有用区间、8 步最好。两个臂都量，因为 H200 的交付配置
# 是 480p 不带 sage / 768p 带 sage（sage 在 sm_90 上 480p 是负收益）。
has to && { echo "##### to (turbo, 4/6/8 步, 1 卡)"
  GPUSETS=1 ARMS="fp8 fp8sage" CASES="480_4 480_6 480_8 768_4 768_6 768_8" \
    CKPT="$TURBO" TAGSUF=_turbo ./h200_grid.sh; }

# tc：turbo 8 步再叠 Cache-DiT，扫 RDT。stock 那轮定的推荐档（480p 0.16 / 768p 0.24）是按 20/30 步
# 定的，步数少 → 每步 residual 变化大 → 低 RDT 可能一次都不触发（g7e 上 0.16 就是逐位相同）。
# g7e 的 turbo 膝点是 0.24，这里从 0.16 扫到 0.32 看 H200 是否同一个膝点。
has tc && { for r in 0.16 0.24 0.32; do
    echo "##### tc RDT=$r (turbo, 8 步, 1 卡, +Cache-DiT)"
    GPUSETS=1 ARMS=fp8sagecache CASES="480_8 768_8" RDT=$r \
      CKPT="$TURBO" TAGSUF=_turbo_R${r#0.} ./h200_grid.sh
  done; }

# tg：加卡。turbo 把每步压薄，通信占比升高，加卡效率会不会被稀释 —— H200 卖的正是延迟天花板，
# 所以这一档必须量。g7e 上没被稀释（甚至微升）。
has tg && { echo "##### tg (turbo, 8 步, 2/4/8 卡, ±Cache-DiT R24)"
  GPUSETS="2 4 8" ARMS=fp8sage CASES="480_8 768_8" CKPT="$TURBO" TAGSUF=_turbo ./h200_grid.sh
  GPUSETS="2 4 8" ARMS=fp8sagecache CASES="480_8 768_8" RDT=0.24 \
    CKPT="$TURBO" TAGSUF=_turbo_R24 ./h200_grid.sh; }

# tr：ref2va。**这个 LoRA 库里没有 ref2va 专用权重**（整库就一份 t2v 命名的 LoRA），但同一份
# LoRA 合进 Ref2VA 分区可用（g7e 实测 259/259，8 步倍数比 fl2va 还高一点，因为参考图编码那份
# 固定开销不随步数缩）。**ref2va 的 SSIM 不能当画质判据**（只给主体图不给首帧，构图本身允许不同），
# 只能目视。前置（一次，CPU）：
#   docker exec -e SRC=/models/MiniMax-H3/Ref2VA/transformer \
#     -e LORA=/out/lora/minimax_h3_turbo_v4_step600_ema.safetensors \
#     -e DST=/out/turbo_ref2va_bf16 h3 python3 /tmp/lora_merge_transformer.py
has tr && { echo "##### tr (ref2va: stock 20 步参考 + turbo 8 步 ±cache, 1/8 卡)"
  TASK=ref2va GPUSETS=1 ARMS=fp8sage CASES="480_20 768_20" TAGSUF=_s20 ./h200_grid.sh
  TASK=ref2va GPUSETS="1 8" ARMS=fp8sage CASES="480_8 768_8" \
    CKPT="$TURBO_REF" TAGSUF=_turbo ./h200_grid.sh
  TASK=ref2va GPUSETS="1 8" ARMS=fp8sagecache CASES="480_8 768_8" RDT=0.24 \
    CKPT="$TURBO_REF" TAGSUF=_turbo_R24 ./h200_grid.sh; }

echo "===== H200_TURBO_DONE $(date -u +%FT%TZ)"
