# MiniMax-H3 on p5e.48xlarge (8xH200) — measured, 2026-08-12/13

Box: `ec2-35-163-211-46.us-west-2`, `lmsysorg/sglang:dev` @ `c7c03ec53b`, weights at
`/opt/dlami/nvme/h3` (both the FL2VA and Ref2VA partitions, 196 GiB) mounted as
`/models/MiniMax-H3` (the registry matches on the `--model-path` *basename*, so a local dir must be
named `MiniMax-H3`). All three patches in `patches/` applied inside the container in order
(cpu-offload → short-edge → target-width-height); 480p opt-in via
`SGLANG_MINIMAX_H3_EXTRA_SHORT_EDGES=480`.

Layout: `patches/` the deliverables, `h3gen.py` / `h3req.py` the submitters, `runs/` raw artifacts
(mp4 + request/status json + `frame_*.png`), `videos_named/` the same videos under readable
names, `logs/` server logs. The 4xH100 work it is compared against lives in
`../h3_h100_baseline/`, and the cookbook snapshot those reference numbers come from is
`../h3_h100_baseline/sglang/cookbook_snapshot.html`.

Unless noted, numbers are client-side wall clock from POST to `status: completed`, t2va, 16:9, seed
1101, `flow_shift` 12.0 / `audio_flow_shift` 3.0, warmup already covering the served resolution. The
three-task section reports server-side `inference_time_s` instead, because it compares three tasks
and needs client queueing and 0.5 s poll granularity out of the way; for one request the two differ
by **1.0–1.6 s** (e.g. mode-3 t2va at 16 steps: infer 17.54 s, wall 19.11 s).
Repeats of an identical config are **byte-identical** (md5 match), so single runs are sufficient.

## Patch validation

`short_edge: 480` + `aspect_ratio: "16:9"` → ffprobe reports **864x480, 124 frames**. 768p is
unchanged at 1344x768x124. The 768p 50-step baseline measured **74.28 s** against the cookbook's
published 74.38 s for the same 4xH200 Ulysses4 recipe, so this box reproduces the reference table.

## All three tasks work, and per-step cost differs 3.2x

Measured on a 4-GPU replica (`--ulysses-degree 4`), 864x480, 10 s clip (243 frames), reporting
server-side `inference_time_s`. Every output was ffprobed: `864,480,243` + `10.125000` + `aac`.

| steps | t2va | fl2va | ref2va |
|---|---|---|---|
| 8 | — | **9.79** | **31.15** (repeats 31.14 / 31.16) |
| 16 | **17.54** | 18.35 | 59.56 (repeats 59.07 / 59.10) |
| 32 | — | 36.19 | 114.83 |
| fit (s) | `wall = 2.05 + 1.02×steps` (four points, see the 4-GPU 10 s section) | `0.87 + 1.102×steps` | `3.52 + 3.482×steps` |

Three conclusions:

- **fl2va costs only ~8% more than t2va** (1.102 vs 1.02 s per step). Supplying a first frame (and
  optionally a last frame) is nearly free, so the two can be capacity-planned together.
- **ref2va costs 3.16x more per step** (3.482 vs 1.102 s), with all three points on the fit.
  **The slope difference proves this is not a one-off "encode the reference video" cost** — a one-off
  cost would only raise the 3.52 s intercept.
- **The cost tracks output length, not the mere presence of a reference.** At the same 16 steps with a
  **5 s** reference video (the output follows it to 124 frames, ffprobe `864,480,124` /
  `5.175000`): **22.02 s**, versus 59.56 s for the 10 s reference. At the 5 s length t2va/16 steps is
  ~6.9 s of inference (from `1.54 + 0.400×steps`), a ratio of 3.2x; at 10 s, 17.54 → 59.56 is 3.4x.
  **The multiple holds at both clip lengths**, which further rules out a one-off cost.

ref2va's `duration_seconds` is **derived from the reference material** and cannot be sent in the
request (it is rejected), so "control the ref2va output length" means "supply a reference of a
different length". `h3gen.py` already handles it that way.

## Mode 3: two co-resident replicas do not slow each other down

`./serve.sh both` (fl2va on GPUs 0-3 / :30010, ref2va on GPUs 4-7 / :30030):

| | time to ready | per-GPU memory | one concurrent request each |
|---|---|---|---|
| fl2va replica (4 GPU) | 90 s | 103,041–103,161 MiB | t2va 16 steps **19.11 s** (wall) |
| ref2va replica (4 GPU) | 90 s | 104,045 MiB | ref2va 8 steps **32.25 s** (wall) |

The replicas start sequentially, ~180 s in total. While both requests run, all 8 GPUs report 100%
`utilization.gpu`. **The key control: ref2va at 8 steps takes 32.17 s with the box to itself and
32.25 s co-resident (+0.25%)** — free, because the device sets are disjoint. This is what makes
mixing tasks on one box usable for capacity planning (see `DEPLOYMENT_GUIDE.md` section 3).

## Width/height as parameters: the boundary matrix

Eleven cases, all measured (`negtests.sh` / `negtests2.sh` in `/opt/dlami/nvme/out` on the box):

| case | `target` | outcome |
|---|---|---|
| both groups | `width+height+short_edge+aspect_ratio` | rejected: `target accepts either width+height or short_edge+aspect_ratio, not both` |
| width only | `{"width":800}` | rejected: `target.height must be an integer` |
| not a multiple of 32 | `800x481` | rejected: `target.height must be a positive multiple of 32, got 481` (**not rounded**) |
| over the area cap | `1376x768` | rejected: `target.width*height must be at most 1032192 px, got 1056768 for 1376x768` (**not downscaled**) |
| short edge not opted in | `928x512` | rejected: `min(width, height) is the short edge and must be one of [480, 768] for minimax_h3, got 512 from 928x512` |
| width also unaligned | `912x512` | rejected: hits the 32-alignment check first (`got 912`) — the two checks' order |
| ratio 4.07, outside 1:4–4:1 | `1952x480` | rejected: `adapt_shape_v1 ratio must be within the inclusive range 1:4 to 4:1, got 1952:480` |
| portrait | `480x800` | **completed** (the short edge is 480 — it is `min()`, not `height`) |
| 21:9-ish | `1120x480` | **completed** |
| positive control | `800x480` (5:3, **not** one of the six released ratios) | **completed**, ffprobe `800,480,124` + aac |
| regression control | `short_edge:480 + aspect_ratio:"640:480"` | still rejected with the original message: `must be 'auto' or one of ['21:9','16:9','4:3','1:1','3:4','9:16']` |

That last row is the important regression evidence: **the old ratio path is untouched**, and
`640:480` is still refused despite being 4:3. Note also that `2464x480` trips the area cap before the
ratio cap (both exist, area just comes first), so testing the ratio bound requires a shape whose
area is legal (`1952x480`).

`fl2va` takes the `exact` wire form too: `{"width":800,"height":480}` plus a first frame →
`800,480,124`.

## 4 GPUs, `--num-gpus 4 --ulysses-degree 4`

5 s video. Fit: `wall = 1.54 + 0.400 x steps`.

| short_edge | steps | wall (s) |
|---|---|---|
| 768 | 50 | 74.28 |
| 480 | 50 | 22.10 |
| 480 | 30 | 13.56 |
| 480 | 25 | 11.56 |
| 480 | 20 | **9.55** |
| 480 | 15 | 7.55 |
| 480 | 10 | 5.54 |
| 480 | 8  | 4.54 |

10 s video (243 frames after 17n+5 alignment). Fit: `wall = 2.05 + 1.02 x steps` — 2.55x the
per-step cost of 5 s for 1.96x the frames, i.e. superlinear in frames.

| steps | 50 | 25 | 20 | 12 |
|---|---|---|---|---|
| wall (s) | 53.18 | 27.11 | 22.09 | 14.06 |

## 8 GPUs, `--num-gpus 8 --ulysses-degree 8`

5 s video. Fit: `wall = 1.24 + 0.216 x steps` (480p), `1.36 + 0.746 x steps` (768p).

| short_edge | steps | wall (s) |
|---|---|---|
| 768 | 50 | 38.66 |
| 768 | 15 | 12.55 |
| 768 | 12 | **10.05** |
| 480 | 50 | 12.05 |
| 480 | 45 | 11.05 |
| 480 | 40 | **10.05** |
| 480 | 30 | 7.54 |
| 480 | 25 | 6.54 |
| 480 | 20 | 5.54 |

10 s video: 25 steps 15.06 s, 20 steps 12.06 s, 10 steps 6.54 s → `wall = 0.86 + 0.568 x steps`,
so a 10 s clip inside a 10 s budget is ~16 steps.

Scaling 4 -> 8 GPUs is near-linear: 1.92x at 768p/50, 1.83x at 480p/50 (1.90x on denoise alone
after removing the ~1.3 s fixed overhead). Cost of that latency: 96.4 GPU-s per video at
Ulysses8 vs 88.4 at Ulysses4, i.e. **+9% GPU-seconds for 1.83x lower latency**. Two 4-GPU
replicas remain the better choice if the goal is throughput rather than latency.

## Are 8 GPUs required?

Not required, but clearly better under a 10 s latency target: for the same ~10 s, 8 GPUs buy
twice the steps.

| | 4 GPUs / 20 steps | 8 GPUs / 40 steps |
|---|---|---|
| wall | 9.55 s | 10.05 s |
| SSIM vs 50 steps | 0.8691 | **0.9682** |

## GPU utilization (480p / 40 steps / 8 GPUs, 200 ms sampling)

| metric | value |
|---|---|
| SM util mean | **85.5%** |
| SM util p50 / max | 100% / 100% |
| power | **~609 W of 700 W** |
| memory-bandwidth util | only **~20%** |
| per-GPU memory | ~103 GiB of 139.8 GiB |

Compute-bound and well fed: bandwidth at 20% with power near TDP says the limit is FLOPs, not
memory. The ~15% idle gap is the ~1.3 s fixed encode/decode/mux overhead (the fit's intercept);
denoise itself is essentially saturated, which is why 4 -> 8 scales near-linearly.

## Concurrency: in-flight is always 1, strict FIFO

N threads fired simultaneously with distinct seeds, 480p / 40 steps / 8 GPUs:

| N | wall_all (s) | per-request max (s) | throughput (vid/min) | failures |
|---|---|---|---|---|
| 1 | 9.83 | 9.83 | 6.1 | 0 |
| 2 | 19.41 | 19.41 | 6.2 | 0 |
| 4 | 38.86 | 38.86 | 6.2 | 0 |
| 8 | 77.24 | 77.24 | 6.2 | 0 |

Throughput is flat at 6.2 vid/min while per-request latency grows linearly — fully serialized,
zero batching benefit. Zero failures, so this is queueing, not rejection.

This is architectural, not a misconfiguration: `scheduler.py:967 _dynamic_batching_enabled()`
asks `pipeline_config.supports_dynamic_batching()`; `base.py:405` returns True only for `T2I` /
`T2V`; `minimax_h3.py:48` declares `task_type = ModelTaskType.TI2V` with **no override**, so it
is always False. Logs show `stop_reason=dynamic_disabled`, `merged_rate=0.0%`. Passing
`--batching-max-size 4` is accepted (it appears in `server_args`) and changes nothing.

Capacity must therefore be planned in replicas: one p5e.48xlarge as a single 8-GPU replica is
**~6.2 vid/min**, with tail latency = 10 s x queue depth.

### Two 4-GPU replicas — two launch traps

`--base-gpu-id 4` **does not work**. It shows up in `server_args`, but replica 1's ranks still
land on GPUs 0-3 and collide with replica 0; both then hit CUDA OOM (each rank wants ~85 GiB of
139.8 GiB, so two per GPU cannot fit). Isolate with `CUDA_VISIBLE_DEVICES=4,5,6,7` instead.

A server also binds `127.0.0.1:<port+1>` alongside `0.0.0.0:<port>`, so replicas spaced one port
apart (30010 / 30011) fail with `[Errno 98] address already in use` *after* fully loading the
weights. Space them by at least 2, and give each its own `--master-port` / `--scheduler-port`.

`launch_replicas.sh <gpus_per_replica>` encodes both, plus a third trap: `pkill -f sglang` inside
the container matches the `docker exec` shell's own command line and kills the launcher (exit
137); the pattern must be `[s]glang`.

## Replica sweep: the whole latency/throughput curve (480p, 40 steps)

All four shapes use all 8 GPUs; only the split differs. Concurrency = number of replicas.

| shape | Ulysses | per-request (s) | aggregate (vid/min) | GPU-s per video | speedup vs 1 GPU |
|---|---|---|---|---|---|
| 1 x 8 GPU | 8 | **10.05** | 6.2 | 96.4 | 6.11x |
| 2 x 4 GPU | 4 | 18.17 | 6.69 | 72.7 | 3.38x |
| 4 x 2 GPU | 2 | 34.37 | 7.01 | 68.7 | 1.79x |
| 8 x 1 GPU | 1 | 61.38 | **7.69** | 61.4 | 1.00x |

Two readings:

- **Latency and throughput trade almost 6:1.** Going from 8 single-GPU replicas to one 8-GPU
  replica costs only 24% of aggregate throughput (7.69 -> 6.2 vid/min) and buys **6.11x** lower
  latency. Ulysses on NVLink is efficient: 76% parallel efficiency at 8 GPUs, 84% at 4, 89% at 2.
- **8 x 1 GPU is perfectly parallel:** 8 concurrent requests finish in 62.39 s versus 61.38 s for
  one — the replicas do not interfere.

## Ulysses collapses without NVLink / P2P

Relevant because the customer may move to RTX PRO 6000 (g7e) boxes, which have no NVLink and
where NCCL typically cannot use PCIe P2P either. Emulated with `NCCL_P2P_DISABLE=1` (forces
staging through host memory) on one 8-GPU replica:

| shape, 480p / 40 steps | NVLink | `NCCL_P2P_DISABLE=1` | penalty |
|---|---|---|---|
| TP1 x Ulysses8 | 10.05 s | **151.97 s** (repeat 151.94) | 15.1x |
| TP8 x Ulysses1 | 13.09 s | **248.66 s** (repeat 248.69) | **19.0x** |

**Both parallel modes collapse, and TP collapses harder.** Ulysses exchanges activations twice
per attention (all-to-all); TP all-reduces twice per *layer*, so it is the more
interconnect-hungry of the two. The repeats agree to 0.01%, so this is a stable property of the
topology and not a warmup artifact.

The consequence for g7e is decisive: **neither the Ulysses lever nor the TP lever is available
without P2P.** The only viable shape is one replica per GPU, where the encoder also cannot fold,
so per-GPU memory becomes the binding constraint and CPU offload is the only remaining tool.
(Side finding: disabling P2P also drops per-GPU memory from 100.6 to 90.0 GiB, so ~10.6 GiB of
the footprint is NCCL communication buffers.)

## Where the memory actually goes (per-component, measured from the loader log)

The loader prints every component's resident size. Its "GB" are really GiB — confirmed below by
two independent cross-checks. At `--num-gpus 8 --ulysses-degree 8 --encoder-parallel auto`:

| component | size / GPU | sharded by |
|---|---|---|
| `transformer` (DiT) | **61.73** | `--tp-size` only |
| `video_vae` | 9.7 | nothing |
| `text_encoder` | **8.23** (of 47.97) | encoder folding |
| `audio_vae` | 0.56 | nothing |
| sum of weights | 80.22 | |
| measured high-water | **95.9 GiB** | +15.7 GiB activations / NCCL / allocator |

**The replicated 61.73 GB DiT is the dominant term, not the text encoder.** An earlier draft of
this document called the encoder "the biggest memory lever" on the strength of its 63 GB
checkpoint directory; that was wrong twice over. The directory holds all 64 Qwen3-VL layers plus
`lm_head`, while H3 trims to 50 and replaces the final norm with `nn.Identity`, so only 47.97 GB
is ever loaded — and of that, only 8.23 GB per GPU survives folding.

### The encoder is already distributed across all 8 GPUs

Answering "can the encoder be spread over 4/8 GPUs instead of deployed standalone" — it already
is, by default. `--encoder-parallel` has **four** modes, not the three the cookbook's picker
shows: `auto | fold | dp | replicate` (`server_args.py:1598`). Measured at 8 GPUs, 480p/40:

| `--encoder-parallel` | `text_encoder` size | per-GPU high-water | latency |
|---|---|---|---|
| `replicate` | 47.97 GB | 135.6 GiB | 10.09 s |
| `fold` (explicit) | **8.23 GB** | **95.9 GiB** | **10.08 s** |
| `auto` (default here) | 8.23 GB | 95.9 GiB | 10.08 s |

**Folding is free: 39.7 GiB/GPU for zero latency.** The saving is exactly the difference in
encoder size (47.97 - 8.23 = 39.74), which is the first cross-check that the loader's units are
GiB. The second: only the transformer *layers* fold, so the expected folded size is
`50 layers / 8 + embed_tokens + vision tower` = `45.4/8 + 1.45 + 1.1` = **8.23 GiB**, matching
the log to the digit. (Layer arithmetic: hidden 5120, intermediate 25600, 64 q-heads / 8 kv-heads
of dim 128 → 487.6 M params = 0.975 GB per layer.)

Two caveats that matter for the g7e plan:

- **The fold decision requires P2P.** `encoders/base.py:104 group_has_measured_topology()` calls
  `torch.cuda.can_device_access_peer()` for every peer and, if any fails, `auto` deliberately
  stays replicated — the docstring's reasoning is that on host-routed topologies "a rule that
  barely paid on NVLink can invert". Explicit `--encoder-parallel fold` overrides this
  ("topology is the caller's call"), so on a no-NVLink box the flag must be passed by hand.
- **Folding needs ranks to fold over.** `server_args.py:669` requires
  `replica_size > tp_size` with `replica_size = num_gpus // dp_size`, so a 1-GPU replica can never
  fold. That is why the 8 x 1 shape below pays the full 47.97 GB.
- `dp` mode additionally needs `batch_size > 1`, and H3 is hard-capped at `batching_max_size=1`,
  so `prefer_dp` is never satisfied for H3 — `dp` is dead code on this model.

## `--tp-size` is the memory lever, and it is cheap

This is the significant new finding. Ulysses is *sequence* parallel — weights replicated, only
activations split — so it does nothing for the 61.73 GB DiT. `--tp-size` shards the DiT itself,
and on NVLink it costs far less latency than expected (8 GPUs, 480p / 40 steps):

| shape | DiT / GPU | per-GPU high-water | latency | vs best |
|---|---|---|---|---|
| TP1 x Ulysses8 | 61.73 | 95.9 GiB | **10.08 s** | — |
| TP2 x Ulysses4 | 30.86 | 63.9 GiB | 11.08 s | +10.2% |
| TP4 x Ulysses2 | 15.43 | 47.5 GiB | 11.59 s | +15.0% |
| TP8 x Ulysses1 | **7.72** | **39.0 GiB** | 13.09 s | +30.0% |

**2.5x less memory for +30% latency**, and the TP2 row alone gives -32 GiB for +10%. This
corroborates the cookbook's claim that TP2 is worth "about 30 GB lower peak memory per GPU" and
extends it: the curve keeps paying out to TP8. The practical consequence is new capability —
at TP4 or TP8 the 480p config fits an **80 GB** card (A100-80G / H100-80G), which the stock
Ulysses=8 shape at 95.9 GiB does not.

`video_vae` (9.7 GB) is not sharded by TP, so ~17.5 GiB of the TP8 footprint is
VAE + encoder + overhead and the curve flattens there.

### Replica shapes: per-GPU memory rises as the degree shrinks

480p / 40 steps, `nvidia-smi` high-water mark, one replica per row group:

| shape | Ulysses | per-GPU memory | encoder folded? |
|---|---|---|---|
| 1 x 8 GPU | 8 | 100.6 GiB | yes |
| 2 x 4 GPU | 4 | 100.7 GiB | yes |
| 4 x 2 GPU | 2 | 113.9 GiB | yes |
| 8 x 1 GPU | 1 | **132.2 GiB** | **no** (`replica_size > tp_size` fails) |
| 8 x 1 GPU + encoder/VAE CPU offload | 1 | **79.4 GiB** | n/a |

The 113.9 -> 132.2 GiB jump is mostly the encoder un-folding, not activations: weights alone at
1 GPU are `47.97 + 61.73 + 9.7 + 0.56` = 119.96 GiB, leaving 12.2 GiB of overhead, versus
15.7 GiB at 8 ranks (more NCCL buffers). Both budgets close to within a GiB, which is the third
consistency check on the component table.

Every unoffloaded shape exceeds a 96 GB card (~95.6 GiB usable). With
`--text-encoder-cpu-offload --vae-cpu-offload` the single-GPU footprint drops by **52.8 GiB
(-40%)** for **+7.9% latency** (66.22 s vs 61.38 s) — those components run once per request, not
once per step, so offloading them is cheap.

**Verified, not extrapolated:** holding 45,268 MiB of ballast on GPU 0 to make it behave like a
96 GB card, the offloaded single-GPU server loads, warms up and generates with zero OOM at the
same 66.24 s. Peak with ballast was 127,137 MiB, i.e. the server itself stayed at 79.4 GiB.

So a no-NVLink 96 GB box is viable at **~0.91 vid/min per GPU** (8 GPUs -> ~7.25 vid/min,
vs 7.69 on H200), but each request takes ~66 s and cannot be accelerated by adding GPUs.

### CPU offload needs a one-line fix (`patches/minimax-h3-cpu-offload-inplace.patch`)

Out of the box, any `*-cpu-offload` flag makes H3 fail during warmup:

```
RuntimeError: Inplace update to inference tensor outside InferenceMode is not allowed.
  ... minimax_h3/stages/decoding.py, line 92, in _reverse_normalize_latents_
```

`_reverse_normalize_latents_` does `latents.mul_(std).add_(mean)`. The latents come from
`denoise_loop.py:33 @torch.inference_mode()`, so they are inference tensors; the offload manager
runs the stage under `torch.inference_mode(False)` (`layerwise_offload.py:389`), where mutating an
inference tensor is illegal. Making the denormalization out-of-place fixes it and is a no-op in
the non-offload path.

### Moving components to *other GPUs* is closed; spreading them over all GPUs is not

Two different questions, with two different answers.

**Dedicating a GPU to the encoder/VAEs is closed.** `minimax_h3_pipeline.py:94
validate_disagg_role()` raises for any role other than `MONOLITHIC`, so `--disagg-role
encoder|denoiser|decoder` with `--encoder-urls` / `--denoiser-urls` / `--decoder-urls` is closed
for H3, and `--srt-encoder-url` (a separate text-encoder server) is wired only for GLM-Image
(`glm_image.py`, `vl_encoder_loader.py`). See `SRT_ENCODER_PR_ASSESSMENT.md` for what enabling
either would take.

**Spreading them over the GPUs you already have is open and already on** — that is encoder
folding, above, and it is where most of the win from a dedicated encoder GPU would have come
from anyway. Combined with `--tp-size` for the DiT, the in-server levers cover the memory problem
on any P2P-capable box without touching disaggregation at all. CPU offload remains the tool for
the no-P2P case.

## What 40 steps costs (SSIM vs the 50-step reference, same topology)

Run-to-run within one topology is byte-identical (SSIM inf), so these are clean signal.

| steps | SSIM (All) vs 50 |
|---|---|
| 45 | 0.9746 |
| 40 | 0.9682 |
| 30 | 0.9267 |
| 25 | 0.8719 |
| 20 | 0.8691 |

Control: the *same* 50-step request run at Ulysses4 vs Ulysses8 scores **0.9598** against itself.
Changing the GPU count perturbs the sample more than dropping 50 -> 40 steps does. 30 steps and
below is a visibly different sample, though still coherent; at 20 steps the cats lose detail.
Extracted mid-frames: `frame_*.png`.

### No topology degrades quality — with a proper noise floor

Every shape in the TP/encoder sweep produces a different bitstream (md5 differs), which is
expected: changing the parallel decomposition changes floating-point reduction order. To show
that none of them *degrades* output, the right control is two runs whose math is identical and
whose reduction order is not — `TP8 x Ulysses1` with and without P2P, which differ only in the
NCCL algorithm NCCL picks:

| pair (480p / 40 steps) | SSIM (All) |
|---|---|
| **control:** TP8 vs TP8 `NCCL_P2P_DISABLE=1` | **0.9444** |
| `fold` vs `replicate` | 0.9376 |
| `fold` vs TP4 x Ulysses2 | 0.9348 |
| `fold` vs TP2 x Ulysses4 | 0.9199 |
| `fold` vs TP8 x Ulysses1 | 0.9054 |

All four sit at or just under the 0.9444 reduction-order floor, i.e. within the same band that a
pure NCCL-algorithm change produces on its own. The trajectory moves, the quality does not.
`ssim_pairs.sh` reproduces the table from `videos_topo/`.

Worth noting for anyone reading the cookbook: it flags only `dp` as "not bitwise-identical", but
`fold` is not bitwise-identical either (0.9376 vs `replicate`). Sharding the encoder's linear
layers reorders reductions just as much. That is a documentation gap, not a bug.

## Cache-DiT is a no-op on H3

`SGLANG_CACHE_DIT_ENABLED=true FN=1 BN=0 WARMUP=4 RDT=0.12 MC=2` (the cookbook's own manual
recipe) and an aggressive `RDT=0.50 MC=6` both register
(`Cache-DiT] Collected Context Config: DBCache_F1B0_W4I1M0MC2_R0.12_N49`) and then skip **zero**
blocks: 12.09 s vs 12.05 s, and the output mp4 is **byte-identical** to the run without it. Not a
usable lever here. `quality: "high"` is not an alternative — its gate
(`release_metadata.py::_MINIMAX_H3_QUALITY_WORKLOAD`) pins width 1344 / height 768 / 50 steps, so
it rejects both a non-768 resolution and any step change.

## Recommendation

The customer has confirmed that "10 s" means clip *duration*, so item 1 is the main line; items 2–3
are kept for the 10-s-latency reading.

1. **The customer's reading (10 s clip): 8 GPUs, Ulysses=8, 864x480, 16 steps → 10.58 s wall
   (9.18 s inference).** 243 frames @ 24 fps = 10.125 s. Step count is a parameter; 8–32 steps are
   all measured (see the three-task section).
2. For 10 s of *latency* with a 5 s clip: 8 GPUs / 40 steps → 10.05 s, a fidelity loss smaller than
   the topology-change control.
3. Must have 768p: 12 steps → 10.05 s. Holds together (H3 is guidance-distilled) but prompt
   adherence degrades — the 12-step frame shows one cat where the prompt asks for three.
4. **Plan ref2va separately**: 3.16x more per step, so a 10 s clip at 16 steps is 60.24 s wall.
   Either cut steps (8 steps → 32.2 s), give it more GPUs, or accept lower QPS.
5. Memory-constrained variant: add `--tp-size 2 --ulysses-degree 4` for 11.08 s at 63.9 GiB/GPU,
   which still clears the 10 s bar only if the customer accepts ~11 s. On an 80 GB card use
   `--tp-size 4 --ulysses-degree 2` (47.5 GiB, 11.59 s).

Fleet size for 1 QPS (scaled in replicas, since H3 does not batch) is in `DEPLOYMENT_GUIDE.md`
section 3: **10 boxes** for t2va at 2x4 GPUs, **30 boxes** for ref2va at 2x4 GPUs.

Always pass `--warmup-resolutions` for every resolution served; the cookbook measures ~10 s of
first-request cost otherwise, and the builder takes raw `WxH` so `864x480` is accepted even
unpatched. `serve.sh` now defaults to both shapes (`WARMUP="1344x768 864x480"`) — see the open
question below on whether the list is honoured.

### Open observation: the warmup list may be ignored (image `c7c03ec53b`)

One 8-GPU run recorded `"warmup_resolutions": ["864x480"]` in `server_args`, yet the scheduler's only
warmup request was `server warmup req (1344x768x124f, 2/50 steps)`, 7.65 s — the shape asked for was
not the shape warmed, and 50 steps is the release default.

Suspected cause, from reading only: the list does reach request construction
(`server_warmup.py:137` passes it, `build_warmup_reqs()` calls `parse_size`), but `width` is not a
declared `Req` field and `Req.__getattr__`/`__setattr__`
(`runtime/pipelines_core/schedule_batch.py:251`/`269`) delegate to `sampling_params`, whose release
defaults are 1344x768 / 50 steps. **Not confirmed by experiment.**

Consequences, already in the guides: don't treat warmup success as evidence the short-edge patch is
live (use `git diff --stat`), and don't assume a listed shape is warm — check with
`grep -o 'warmup req ([^)]*)'` and budget ~10 s extra for any served shape that is missing.

`warmtest.sh` would settle it (launch on 4 free GPUs at each `WARMUP` value, print the warmup
requests actually run, time cold vs warm at both shapes). Not yet run.

The platform-specific commands, trap list and Fabric Manager recovery recipe are collected in
`DEPLOYMENT_GUIDE.md` / `DEPLOYMENT_GUIDE_zh.md` — that is the customer-facing document.

## Videos for inspection

`videos_named/` holds every run under a self-describing name
(`{gpus}_{WxH}_{clip length}_{steps}_{measured wall}`). All 29 raw files are also in `runs/`
under their run ids; the `cd_*` / `cda_*` Cache-DiT files are omitted from
`videos_named/` because they are byte-identical to their `u8_*` counterparts. Each mp4 carries the
generated AAC soundtrack, so audio quality is judgeable from the same file. `runs/frame_*.png` are
mid-clip (frame 62) stills for the side-by-side that mattered: 480p/40 steps vs 768p/12 steps at
the same 10 s budget.

Samples for the three tasks and the new features are in `videos_named/` too: `mode3_t2va.mp4` /
`mode3_ref2va.mp4` (the two generated simultaneously while co-resident) and `geturl.mp4` (produced
from a single GET URL via `h3get.py`).

Both readings of "10 s" are covered by the tables above: for a 10 s *clip* (the customer's reading)
see the three-task section and the 10 s rows, for a 10 s *latency* see the 5 s-clip tables.
