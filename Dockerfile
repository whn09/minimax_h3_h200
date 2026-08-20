# H3-on-H200 交付镜像。和 minimax_h3_g7e 的 Dockerfile 同一套骨架，**base digest 故意用同一个**
# —— H200 这条线存在的意义就是和 g7e 比性价比，分母和分子必须是同一版 sglang。
#
# 相对 g7e 版本的四处差异，都是平台决定的：
#   1. **SAGE_ARCH=9.0**（H200 是 sm_90）。而且 sm_90 有**自己的**扩展 `_qattn_sm90`，不像 sm_120
#      那样复用 sm89 的源码，所以自检查的文件名不同（见 SAGE_EXT / SAGE_CUBIN）。
#   2. **不打两个 NVFP4 补丁**。sm_90 没有 FP4 tensor core，NVFP4 在这张卡上不是路径；H200 的
#      量化档是 **FP8（w8a8）**，走 sglang 自带的 `--quantization fp8`，不需要离线量化文件。
#   3. **不打 patch_sol_attn_dense_sage.py**。它只在 `--attention-backend sol_attn` 时才走到，
#      而稀疏 attention 不在 H200 的交付路径里。
#   4. **不设 H3_FP4_* env**（同 2）。`SGLANG_USE_RUNAI_MODEL_STREAMER=0` 留着：g7e 上它是必需的
#      （128 GB 主机被 streamer 的匿名内存打爆），p5en 有 2 TB 不需要，但关掉走 mmap 没有代价，
#      而"两台机器只差该差的那几项"比省一次 mmap 值钱。
#
# 保留 `minimax-h3-short-edge.patch`：这个 base（c0b6474b）还没放开非 768 短边，而 480p 是客户的
# 主力档。`minimax-h3-mark-missing-params-required.patch` 也留着 —— 它把 DiT 的 "缺参数就报错"
# 从无条件盖章改成只盖没声明过策略的参数，量化方法自己合成的 scale 参数才不会被误伤。
#
# 不烤进去的：**权重**（269 GiB，bind mount 到 /models/MiniMax-H3）、GPU 数/并行度（serve.sh 入参）。
ARG BASE=lmsysorg/sglang@sha256:51e576f02368480c055c7aadb67590d82b172e2392123ce4cf4cc8251b2d8caf
FROM ${BASE}

ARG PATCHES="minimax-h3-short-edge.patch minimax-h3-mark-missing-params-required.patch"
ARG SAGE_REPO=https://github.com/thu-ml/SageAttention
# d1a57a5 = g7e 那张 FP8+sage 表的出处，同 commit 才能跨平台比。
ARG SAGE_REF=d1a57a5
ARG SAGE_ARCH=9.0
# sm_90 的 qattn 是独立扩展（sm_120 是复用 sm89 源码只多编一份 cubin），所以这两项要显式给。
ARG SAGE_EXT=_qattn_sm90
ARG SAGE_CUBIN=sm_90a
# nvcc 并发。192 vCPU，16 很稳。
ARG JOBS=16

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

# ---- 1) .patch 文件 + 印戳。
# 印戳（/sgl-workspace/.h3-patches/<名字>）是给 serve.sh 的 apply 循环看的：烤进镜像的补丁当然
# 打不上第二次，没有印戳它会在**已经正确**的镜像上以 DOES_NOT_APPLY 拒绝启动。
COPY patches/ /patches/
RUN cd /sgl-workspace/sglang \
 && mkdir -p /sgl-workspace/.h3-patches \
 && for n in ${PATCHES}; do \
      test -f "/patches/$n" || { echo "MISSING $n" >&2; exit 1; }; \
      git apply -p1 "/patches/$n"; \
      touch "/sgl-workspace/.h3-patches/$n"; \
      echo "BAKED         $n"; \
    done \
 && printf '%s\n' "${PATCHES}" > /sgl-workspace/.h3-image-patches \
 && bash /patches/inplace_ref_short_edge.sh \
 && python3 -m compileall -q /sgl-workspace/sglang/python/sglang/multimodal_gen >/dev/null

# ---- 2) SageAttention 从源码编，并断言真的产出了这张卡的 cubin。
# TORCH_CUDA_ARCH_LIST 让 setup.py 走 "works without GPUs" 那条分支，所以建镜像不需要 GPU。
# 断言不是形式主义：pip wheel 装出来**文件名一样**，差别只在里面没有这张卡的 cubin，运行时静默
# 回落 Triton（在 sm_120 上是 1.16× 而不是 1.776×）。
RUN cd /sgl-workspace \
 && git clone --filter=blob:none "${SAGE_REPO}" SageAttention \
 && cd SageAttention \
 && git checkout "${SAGE_REF}" \
 && TORCH_CUDA_ARCH_LIST="${SAGE_ARCH}" MAX_JOBS="${JOBS}" EXT_PARALLEL="${JOBS}" \
    NVCC_APPEND_FLAGS='--threads 2' pip install --no-build-isolation . \
 && rm -rf build \
 && cd /tmp \
 && sagedir=$(python3 -c 'import sageattention,os;print(os.path.dirname(sageattention.__file__))') \
 && /usr/local/cuda/bin/cuobjdump --list-elf "$sagedir"/${SAGE_EXT}*.so | grep -q "${SAGE_CUBIN}" \
 && echo "SAGE OK       ${SAGE_REF} 带 ${SAGE_CUBIN} cubin"

ENV SGLANG_USE_RUNAI_MODEL_STREAMER=0

LABEL h3.base="${BASE}" \
      h3.patches="${PATCHES}" \
      h3.sage.ref="${SAGE_REF}" \
      h3.sage.arch="${SAGE_ARCH}" \
      h3.readme="https://github.com/whn09/minimax_h3_h200"

CMD ["sleep", "infinity"]
