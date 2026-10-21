# lmsysorg/sglang:dev + SageAttention built from source for sm_90 -- nothing else. sglang source is
# untouched (no patches), so the sage arm differs from the stock arms in exactly one package.
# Same SageAttention commit as the earlier H200/g7e rounds (d1a57a5). The cuobjdump check matters:
# a wheel without an sm_90a cubin has the same file name and silently falls back to Triton.
# The c++17 -> c++20 sed: this image's torch headers `#error` below C++20, and setup.py pins c++17.
#   docker build -f Dockerfile.sage -t h3-sglang-sage:dev .
ARG BASE=lmsysorg/sglang:dev
FROM ${BASE}
ARG SAGE_REPO=https://github.com/thu-ml/SageAttention
ARG SAGE_REF=d1a57a5
ARG JOBS=16
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
RUN cd /sgl-workspace \
 && git clone --filter=blob:none "${SAGE_REPO}" SageAttention \
 && cd SageAttention && git checkout "${SAGE_REF}" \
 && sed -i "s/c++17/c++20/g" setup.py \
 && TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS="${JOBS}" EXT_PARALLEL="${JOBS}" \
    NVCC_APPEND_FLAGS='--threads 2' pip install --no-build-isolation . \
 && rm -rf build && cd /tmp \
 && sagedir=$(python3 -c 'import sageattention,os;print(os.path.dirname(sageattention.__file__))') \
 && /usr/local/cuda/bin/cuobjdump --list-elf "$sagedir"/_qattn_sm90*.so | grep -q sm_90a \
 && echo "SAGE OK ${SAGE_REF} sm_90a"
