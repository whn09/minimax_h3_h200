#!/usr/bin/env python3
"""diffusers-native MiniMax-H3 T2VA on N CUDA GPUs: 1344x768, 124 frames, 50 steps, per-stage timing.

    TP=1 U=8 torchrun --nproc_per_node=8 h3_diffusers_bench.py      # Ulysses 8 (diffusers' _cp_plan)
    TP=8 U=1 ...                                                     # TP 8   (PR #14609 _tp_plan)
    TP=2 U=4 ...                                                     # TP2 x U4 (PR #14609 + #14725)

The denoiser is sharded on every rank; the Qwen3-VL text encoder lives on rank 0 only and its output state
is broadcast (62 GB bf16 next to a full 62 GB DiT shard would not fit a 141 GB H200 twice); the video and
audio VAEs run once, natively (no tiling parallelism), on DECODE_RANK. Each stage is timed with a CUDA sync
and a barrier on both sides, so the numbers add up to the request.
"""
import json, os, time

import numpy as np
import torch
import torch.distributed as dist

from diffusers import ContextParallelConfig, MiniMaxH3Transformer3DModel, ModularPipeline, ParallelConfig
from diffusers.utils.export_utils import encode_video

try:
    from diffusers import TensorParallelConfig
except ImportError:
    TensorParallelConfig = None

CKPT = os.environ.get("CKPT", "/root/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-H3/snapshots/"
                      "42ed227ee7df40d41602854ae760620d6eb651fe")
TP, U = int(os.environ.get("TP", "1")), int(os.environ.get("U", "8"))
N, WARM = int(os.environ.get("N", "2")), int(os.environ.get("WARM", "1"))
DECODE_RANK = int(os.environ.get("DECODE_RANK", "1"))
ATTN = os.environ.get("ATTN", "")
OUT = os.environ.get("OUT", "/out")
PROMPT = ("Night-vision bedroom footage: while the owner sleeps, three cats burst in playing tiny brass "
          "instruments at full volume, freeze, then march out as if nothing happened.")
H, W, F, STEPS, SEED = 768, 1344, 124, 50, 1101

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
dev = torch.device("cuda", torch.cuda.current_device())


def log(*a):
    if rank == 0:
        print(*a, flush=True)


def sync():
    torch.cuda.synchronize()
    dist.barrier()


def workflow(keep=None, drop=()):
    wf = ModularPipeline.from_pretrained(CKPT).blocks.get_workflow("t2va")
    for name in list(wf.sub_blocks):
        if (keep is not None and name not in keep) or name in drop:
            wf.sub_blocks.pop(name)
    return wf


t0 = time.time()
cp = ContextParallelConfig(ulysses_degree=U, ulysses_anything=True) if U > 1 else None
if TP > 1:
    pc = ParallelConfig(tensor_parallel_config=TensorParallelConfig(tp_degree=TP), context_parallel_config=cp)
    transformer = MiniMaxH3Transformer3DModel.from_pretrained(
        CKPT, subfolder="transformer", dtype=torch.bfloat16, parallel_config=pc)
else:
    transformer = MiniMaxH3Transformer3DModel.from_pretrained(CKPT, subfolder="transformer", dtype=torch.bfloat16)
    transformer.to(dev)
    if cp is not None:
        transformer.enable_parallelism(config=ParallelConfig(context_parallel_config=cp))
if ATTN:
    transformer.set_attention_backend(ATTN)
denoise_pipe = workflow(drop=("text_encoder", "decode.video", "decode.audio")).init_pipeline(CKPT)
denoise_pipe.update_components(transformer=transformer)
denoise_pipe.load_components(dtype=torch.bfloat16)
cond_pipe = vdec = adec = None
if rank == 0:
    cond_pipe = workflow().sub_blocks.pop("text_encoder").init_pipeline(CKPT)
    cond_pipe.load_components(dtype=torch.bfloat16)
    cond_pipe.text_encoder.to(dev)
if rank == DECODE_RANK:
    vdec = workflow(keep=("decode.video",)).init_pipeline(CKPT)
    vdec.load_components(dtype=torch.bfloat16)
    vdec.vae.to(dev)
    adec = workflow(keep=("decode.audio",)).init_pipeline(CKPT)
    adec.load_components(dtype=torch.bfloat16)
    adec.audio_vae.to(dev)
sync()
log(f"[load] {time.time() - t0:.1f}s TP={TP} U={U} world={world} attn={ATTN or 'default'} "
    f"mem/rank0={torch.cuda.memory_allocated() / 2**30:.1f}GiB")


def one(i, kind):
    t = {}
    sync(); t0 = time.time()
    if rank == 0:
        state = cond_pipe(prompt=PROMPT)
        for k, v in list(state.values.items()):
            if isinstance(v, torch.Tensor):
                state.set(k, v.cpu())
        box = [state]
    else:
        box = [None]
    sync(); t["text_encoder"] = time.time() - t0
    t0 = time.time()
    dist.broadcast_object_list(box, src=0)
    state = box[0]
    for k, v in list(state.values.items()):
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            state.set(k, v.to(dev))
    sync(); t["broadcast"] = time.time() - t0
    t0 = time.time()
    res = denoise_pipe(state=state, height=H, width=W, num_frames=F, num_inference_steps=STEPS,
                       generator=torch.Generator().manual_seed(SEED), output=["latents", "audio_latents"])
    sync(); t["denoise"] = time.time() - t0
    if rank == DECODE_RANK:
        t0 = time.time()
        videos = vdec(latents=res["latents"].to(dev), output_type="np", output=["videos"])["videos"]
        torch.cuda.synchronize(); t["video_vae"] = time.time() - t0
        t0 = time.time()
        a = adec(audio_latents=res["audio_latents"].to(dev), output=["audio", "sampling_rate"])
        torch.cuda.synchronize(); t["audio_vae"] = time.time() - t0
        t0 = time.time()
        frames = np.asarray(videos[0])
        encode_video(frames, fps=24, output_path=f"{OUT}/diffusers_tp{TP}_u{U}_{kind}{i}.mp4",
                     audio=a["audio"].squeeze(0).float().cpu(), audio_sample_rate=a["sampling_rate"])
        t["mux"] = time.time() - t0
        t["frames"] = list(frames.shape)
    box = [t]
    dist.broadcast_object_list(box, src=DECODE_RANK)
    t.update({k: v for k, v in box[0].items() if k not in t})
    sync()
    t["total"] = sum(v for k, v in t.items() if isinstance(v, float))
    t["s_per_step"] = t["denoise"] / STEPS
    log(f"{kind}{i} " + " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in t.items()))
    if rank == 0:
        with open(f"{OUT}/diffusers_tp{TP}_u{U}.jsonl", "a") as f:
            f.write(json.dumps(dict(kind=kind, i=i, tp=TP, u=U, attn=ATTN or "default",
                                    peak_gib_rank0=torch.cuda.max_memory_allocated() / 2**30, **t)) + "\n")


for i in range(WARM):
    one(i, "warmup")
for i in range(N):
    one(i, "run")
dist.destroy_process_group()
