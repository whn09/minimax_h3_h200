# Turbo LoRA (8-step distilled weights) on H200 — measured 2026-08-21, `p5en.48xlarge`

`RESULTS_QUANT.md` only changed *how* the model runs (FP8 / SageAttention / Cache-DiT); the weights
stayed official. This round changes *the weights*: swap in
[`larryvrh/MiniMax-H3-Turbo-Lora`](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)
(`minimax_h3_turbo_v4_step600_ema.safetensors`, strength 1.0; the model card says 4–8 steps work
and 8 is best) and drop from 20 steps to 8. Driver `h200_turbo.sh`, merge script
`lora_merge_transformer.py`.

## Conclusion

**8-step turbo + Cache-DiT `RDT=0.24` is both the cheapest and the lowest-latency tier on this
box**: **2.47× (480p) / 2.79× (768p)** faster than same-geometry stock 20 steps, **59%/64%** cheaper
per video-second. On 8-GPU Ulysses: 480p **2.010 s**, 768p **5.136 s** — the lowest latency measured
anywhere in this H3 project.

| fl2va, 5.175 s clip | 480p 1 GPU | 480p 8 GPU | 768p 1 GPU | 768p 8 GPU |
|---|---|---|---|---|
| stock 20 steps (FP8[+sage], denominator) | 29.015 s | — | 97.474 s | — |
| turbo 8 steps | 12.694 s | 2.263 s | 39.758 s | 5.792 s |
| **turbo 8 steps + `RDT=0.24`** | **11.766 s** | **2.010 s** | **34.953 s** | **5.136 s** |
| vs stock 20 steps | 2.47× | — | 2.79× | — |
| $/video-second (spot) | **$0.002124** | $0.002902 | **$0.006308** | $0.007416 |
| saving vs stock 20 steps | 59% | — | 64% | — |

480p runs plain FP8 (no sage), 768p adds sage — same as the stock round; turbo does not change that.

**The cost is quality: turbo 8 steps scores SSIM Y 0.905–0.918 against stock 20 steps, below the
0.942–0.951 band that FP8 quantization alone costs.** That is inherent to distilled weights, not a
misconfiguration. Visually (frames 10/60/110 side by side, plus a 768p center crop) it is
**indistinguishable**: same cat, same curtain pattern and shadows, same window frame, no artifacts,
no motion collapse. Motion energy 0.254 vs reference 0.239 (480p) and 0.336 vs 0.286 (768p) — very
slightly higher, not collapsed.

**Reports that "this distilled LoRA looks terrible" do not reproduce on H200.** If you see that, check
two things first: whether the step count was actually lowered to 8 (running distilled weights at 20
steps overshoots), and how the LoRA is loaded (see "Offline merge is mandatory").

## One step shorter than the g7e chain: H200 needs no offline quantization

On g7e the chain is merge-to-bf16 → `nvfp4_quantize_transformer.py` → `--transformer-weights-path
<file>`, because NVFP4 checkpoints are an offline format. H200's quantization tier is
`--quantization fp8`, which is **online** (applied in `process_weights_after_loading` after load),
so `--transformer-weights-path` can point straight at the **merged bf16 directory**:

```bash
mkdir -p /opt/dlami/nvme/out/lora && curl -sL --retry 5 \
  -o /opt/dlami/nvme/out/lora/minimax_h3_turbo_v4_step600_ema.safetensors \
  https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/resolve/main/minimax_h3_turbo_v4_step600_ema.safetensors
docker cp lora_merge_transformer.py h3:/tmp/
docker exec -e SRC=/models/MiniMax-H3/FL2VA/transformer \
  -e LORA=/out/lora/minimax_h3_turbo_v4_step600_ema.safetensors \
  -e DST=/out/turbo_v4_600_bf16 h3 python3 /tmp/lora_merge_transformer.py
#  -> lora modules: 259  strength=1.0 ... merged 259/259 modules, max |delta|/|W| = 0.0036
#  -> 62 GiB, ~4 min (pure CPU; do not run alongside timed requests)
```

Two source facts behind this: `resolve_transformer_safetensors_to_load` takes the single-file branch
only when `os.path.isfile(...) and endswith(".safetensors")`, otherwise it walks the directory via
`_list_safetensors_files`; and in `_resolve_quant_config` an explicit `--quantization` has top
priority, with `fp8` constructing `quant_cls()` with no args = online quantization. So "override the
weights" and "quantize online" do not conflict.

Serving (480p delivery tier / 768p adds sage):

```bash
IMAGE=h3-h200:local NAME=h3 GPUS=1 EXTRA="--quantization fp8 \
  --layerwise-offload-components text_encoder \
  --transformer-weights-path /out/turbo_v4_600_bf16" \
  ENVX="SGLANG_CACHE_DIT_ENABLED=1 SGLANG_CACHE_DIT_RDT=0.24 SGLANG_CACHE_DIT_SECONDARY_RDT=0.24" \
  WARMUP="864x480" ./serve.sh start
```

## Offline merge is mandatory — `--lora-path` will not work

`runtime/layers/lora/linear.py` does an in-place add on an `[out, in]` weight, but FP8 stores the
weight **transposed** (`21504` vs `5376`) — the shapes do not match and it raises. So the merge has
to happen **before** quantization, on bf16. Merge semantics (copied from sglang's own
implementation): `W_eff = W + strength * (B @ A)`, **no** alpha/rank scaling, delta accumulated in
fp32 then cast back to bf16. All 259 modules / 518 tensors must hit; the script `exit 1`s otherwise —
silently missing a few modules yields "looks like stock but worse", which is very hard to diagnose.

## Step curve

1 GPU, **plain-FP8 arm** (numerator and denominator in the same arm), `inference_time_s` / SSIM Y /
motion energy:

| Steps | 480p | 768p |
|---|---|---|
| stock 20 (denominator) | 29.015 s / 1.000 / 0.2362 | 103.154 s / 1.000 / 0.2748 |
| turbo 4 | 7.608 s / 0.8898 / 0.3025 | 21.201 s / 0.8774 / 0.3263 |
| turbo 6 | 10.086 s / 0.9062 / 0.3059 | 31.449 s / 0.8830 / 0.2985 |
| **turbo 8** | **12.694 s / 0.9180 / 0.2540** | **41.781 s / 0.9097 / 0.3268** |

(Against the FP8+sage arm's own denominator, 8 steps is 480p 0.9159 / 0.2571 and 768p 0.9047 /
0.3356 — the two denominators differ by 0.002–0.005, i.e. noise.)

**8 steps is the only one of the three worth using.** 4 and 6 steps drop to SSIM 0.877–0.906 with
motion energy 20–38% above reference (overshoot); the 5 s / 21 s saved is not worth it. The model
card's "8 is best" holds on H200.

sage behaves the same on turbo as on stock: worth +4.9% at 768p (41.781 → 39.758) and **negative at
480p** (12.694 → 12.939, 1.9% slower).

## Cache-DiT on top of turbo: `RDT=0.24`

1 GPU, turbo-8 base, RDT sweep:

| RDT | 480p | 768p |
|---|---|---|
| off (turbo base) | 12.939 s / 0.9159 / 0.2571 | 39.758 s / 0.9047 / 0.3356 |
| 0.16 | 13.049 s / byte-identical to base | 39.622 s / byte-identical to base |
| **0.24** | **11.766 s / 0.9146 / 0.2527** | **34.953 s / 0.9057 / 0.3233** |
| 0.32 | 10.470 s / 0.8941 / 0.3110 | 34.941 s / byte-identical to 0.24 |

Three readings:

1. **`RDT=0.16` never fires at 8 steps.** Not "small gain" — the mp4's md5 is **byte-identical to
   Cache-DiT off** at both resolutions, and the 13.049/39.622 s difference is pure run-to-run noise.
   The stock round's 480p recommendation of 0.16 **must be changed** for turbo: fewer steps → larger
   per-step residual change → the threshold has to move up. The knee for turbo-8 is **0.24**, the
   same knee as on g7e.
2. **`RDT=0.24` is nearly free on top of turbo**: 480p SSIM 0.9159 → 0.9146, motion 0.2571 → 0.2527;
   768p 0.9047 → 0.9057 (up slightly), 0.3356 → 0.3233 (closer to the reference's 0.2856). The cache
   pulls turbo's overshot motion back a little.
3. **`RDT=0.32` splits by resolution**: at 768p it is **byte-identical to 0.24** (0.24 already skips
   every skippable step), so 768p should just use 0.24; at 480p it really does skip more — 11% faster
   but SSIM drops to 0.8941 and motion rises 21%, so **do not use it**.

## Scaling out: turbo does not dilute Ulysses

Turbo thins each step, so communication should take a larger share. Measured, it barely does
(1 → 8 GPU efficiency; stock 20-step equivalent in parentheses):

| Tier | 1 GPU | 2 GPU | 4 GPU | 8 GPU | 1→8 eff. |
|---|---|---|---|---|---|
| 480p turbo 8 | 12.939 s | 7.698 s | 4.124 s | 2.263 s | 71% (stock 81%) |
| 480p turbo + `RDT=0.24` | 11.766 s | 6.996 s | 3.667 s | **2.010 s** | 73% (stock 75%) |
| 768p turbo 8 | 39.758 s | 22.115 s | 11.273 s | 5.792 s | 86% (stock 88%) |
| 768p turbo + `RDT=0.24` | 34.953 s | 19.546 s | 9.874 s | **5.136 s** | 85% (stock 88%) |

768p loses 2–3 points, 480p loses 2–10 (short sequence, so fixed communication cost is a larger
share to begin with). Scale-out premium ($/video-second, spot): 480p turbo+R24 $0.002124 →
$0.002902 (+37%), 768p $0.006308 → $0.007416 (+18%).

Peak memory per GPU is 47.9–49.2 GiB (1 GPU) / 52.2–52.5 GiB (8 GPU), same as the stock FP8 tier —
turbo changes the weights, not the memory layout.

## ref2va

The same LoRA (the repo only ships one, t2v-named) merged into the `Ref2VA/transformer` partition:
**259/259 hit, same max |delta|/|W| = 0.0036**. 1 GPU / 8 GPU:

| ref2va (`REF_SHORT_EDGE=1024`) | 480p 1 GPU | 480p 8 GPU | 768p 1 GPU | 768p 8 GPU |
|---|---|---|---|---|
| stock 20 steps (FP8+sage) | 33.850 s | — | 101.277 s | — |
| turbo 8 steps | 14.973 s | 2.493 s | 41.239 s | 5.960 s |
| **turbo 8 steps + `RDT=0.24`** | **13.430 s** | **2.262 s** | **36.353 s** | **5.290 s** |
| vs stock 20 steps | 2.52× | — | 2.79× | — |
| $/video-second (spot) | **$0.002424** | $0.003266 | **$0.006561** | $0.007638 |

The speedups match fl2va (2.52×/2.79× vs 2.47×/2.79×), and scale-out efficiency (74–86%) is the same
class.

**SSIM and motion energy are not valid quality criteria for ref2va.** Measured SSIM Y is only 0.389
(480p) / 0.635 (768p) with motion energy 1.27 vs the reference's 0.18 (480p) — but that is not a
quality defect: frame by frame, both stock and turbo contain a **real camera pan**, turbo's just runs
faster, so frame-aligned SSIM collapses. ref2va gets a reference image but no first frame, so the
camera trajectory is free to differ. Visually (480p 3 rows × 4 frames, 768p center-crop triptych):
same cat, same window, same curtain detail and wall grain; turbo matches stock, and `RDT=0.24` is
indistinguishable from turbo base.

## Versus the g7e turbo tier

Each platform's delivery config (H200 = FP8[+sage], g7e = NVFP4+sage), both turbo 8 steps +
`RDT=0.24`:

| | 480p 1 GPU | 768p 1 GPU | Lowest latency |
|---|---|---|---|
| H200 latency | 11.766 s | 34.953 s | 8 GPU 2.010 s / 5.136 s |
| g7e latency | 11.882 s | 36.400 s | 2 GPU 8.983 s / 24.020 s |
| **H200 faster** | 1.01× | 1.04× | **4.5× / 4.7×** |
| H200 $/video-s (spot) | $0.002124 | $0.006308 | |
| g7e $/video-s (3-yr SP) | $0.001142 | $0.003497 | |
| **g7e cheaper** | **1.86×** | **1.80×** | |

**On the turbo tier H200's single-GPU latency edge essentially disappears** (1.09–1.17× on stock,
only 1.01–1.04× here): turbo removes more than half of the DiT work, the remaining fixed cost (text
encode + VAE decode + muxing) is comparable on both boxes, and once DiT's share of the denominator
drops, the raw compute gap stops showing. **The more aggressive the acceleration, the smaller the
cross-platform latency gap and the larger the unit-cost gap.** So "pick g7e for unit cost" holds
*more* strongly on turbo than on stock (1.80–1.86× vs 1.61–1.73×); what H200 still buys is the
8-GPU latency ceiling (2.0 s / 5.1 s), which g7e cannot reach.

## 1 QPS fleet (turbo 8 steps + `RDT=0.24`, 1 replica per GPU)

| Tier | Per-GPU throughput | GPUs needed | `p5en` boxes | spot $/h | CB $/h |
|---|---|---|---|---|---|
| 768p fl2va | 1 clip / 34.953 s | 35 | 4.4 | **$118** | $209 |
| 768p ref2va | 1 clip / 36.353 s | 37 | 4.6 | $124 | $221 |
| 480p fl2va | 1 clip / 11.766 s | 12 | 1.5 | **$40** | $72 |
| 480p ref2va | 1 clip / 13.430 s | 14 | 1.8 | $47 | $84 |

That is **64% / 59%** below the stock round's equivalents ($330 for 768p / $98 for 480p at FP8+sage,
20 steps).

## Reproducing

```bash
scp h200_turbo.sh lora_merge_transformer.py H200:~/h3run/
# Prerequisite (once, CPU): download the LoRA and merge into each partition — see the script header
setsid nohup ./h200_turbo.sh > /opt/dlami/nvme/out/h200_turbo.log 2>&1 < /dev/null &
PHASES=tc ./h200_turbo.sh   # one phase only: t0 reference / to step curve / tc RDT / tg scale-out / tr ref2va
```

Same protocol as `h200_grid.sh`: `short_edge` 480/768 + `aspect 16:9`, `duration 5.0` (5.175 s clip,
124 frames), `flow_shift 12.0`, `audio_flow_shift 3.0`, fixed seed (fl2va 6201 / ref2va 8201), image
`nightly-dev-20260818-c0b6474b`. Times are `inference_time_s` (whole request).

**Every denominator was re-measured in this round rather than quoted from `RESULTS_QUANT.md`**:
same-config latency drifts 5.4% across restarts, so numerator and denominator must share one boot.
This box was rebooted mid-round by a DLAMI auto-upgrade (see trap 8 in `DEPLOYMENT_GUIDE.md`); the
pre- and post-reboot reference clips are **md5-identical** and latency drifted 0.02%–2.1%.

## Traps

1. **Pass `TAGSUF=` when sweeping RDT** — the tag has no RDT field, so without it the mp4s silently
   overwrite each other.
2. **`serve_grid_*.log` filenames have no `TAGSUF`**, so a later phase reusing the same arm and GPU
   count overwrites the earlier phase's server log. Rename between phases if you need the evidence.
3. **The merge script must be copied into the container** (`docker cp ... h3:/tmp/`), and `serve.sh`
   recreates the container for every arm — which empties `/tmp`. Do the merges before the grid, or
   re-copy each time.
4. **`--lora-path` does not work under any quantization tier** (shape mismatch, see above).
