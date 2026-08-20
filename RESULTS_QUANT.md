# H200 quantization and acceleration tiers, measured (2026-08-21, `p5en.48xlarge`)

This round measures **how cheaply H3 runs on H200**: BF16 → FP8 → +SageAttention → +Cache-DiT, four
arms × GPU counts × four geometries, with every GPU count carrying its own BF16 denominator.
`RESULTS.md` answered "how do we get latency down to 10 s" on an older sglang (`c7c03ec53b`) in
BF16; this one answers "what does a second of finished video cost", on
`nightly-dev-20260818-c0b6474b`.

**It also overturns one of `RESULTS.md`'s negative results**: Cache-DiT really did skip zero blocks
on the old image, but it works on `c0b6474b` and is worth 1.94–2.40×. See "Cache-DiT" below.

Driver: `h200_grid.sh`. Delivery image: `Dockerfile` + `build_image.sh`. Bare metal:
`h200_bringup.sh`.

## Conclusions

**Delivery configuration: single-GPU FP8 (w8a8)** — 1.13–1.19× over BF16, memory 78.8 → 48.1 GB
(−39%). At 768p, adding SageAttention is worth a further **+5.5–5.8%**; at **480p do not add it**
(1.6% slower than plain FP8). If quality can be traded for cost, Cache-DiT on top is worth another
**1.67–2.40×** (`RDT=0.16` at 480p, `RDT=0.24` at 768p — see "Picking the Cache-DiT RDT"; the table
below uses 0.24 throughout so it lines up with the scaling table).

| Case (fl2va, 5.175 s clip) | 480p/20 | 480p/30 | 768p/20 | 768p/30 |
|---|---|---|---|---|
| **BF16 (denominator)** | | | | |
| 1 GPU | 33.663 s | 49.515 s | 117.099 s | 175.779 s |
| 8 GPU Ulysses=8 | 5.324 s | 7.423 s | 16.446 s | 24.570 s |
| **FP8 (delivered)** | | | | |
| 1 GPU | **28.454 s** | **41.506 s** | 103.581 s | 155.087 s |
| 8 GPU Ulysses=8 | **4.367 s** | **6.079 s** | 14.648 s | 21.836 s |
| vs BF16 | 1.18× | 1.19× | 1.13× | 1.13× |
| **FP8 + SageAttention** | | | | |
| 1 GPU | 28.909 s | 42.143 s | **97.918 s** | **146.295 s** |
| 8 GPU Ulysses=8 | 4.462 s | 6.186 s | **13.863 s** | **20.414 s** |
| vs plain FP8 | 0.98× | 0.99× | **1.06×** | **1.06×** |
| **plus Cache-DiT `RDT=0.24` (lossy, optional)** | | | | |
| 1 GPU | 14.737 s | 18.790 s | 50.473 s | 61.087 s |
| 8 GPU Ulysses=8 | 2.458 s | 2.890 s | 7.274 s | 8.679 s |
| vs same-shape FP8+sage | 1.96× | 2.24× | 1.94× | 2.40× |
| vs BF16 | 2.28× | 2.64× | 2.32× | 2.88× |

Request shape: `short_edge` 480/768 + `aspect 16:9`, `duration 5.0` (5.175 s clip, 124 frames,
24 fps), `flow_shift 12.0`, `audio_flow_shift 3.0`, fixed seed 6201, fl2va (the keyframe is the
first frame of a clip we generated ourselves). The time is `inference_time_s`, i.e. the whole
request (text encode + sampling + VAE decode + mux), all measured on the **same**
`p5en.48xlarge`. The first three blocks are **lossless** (bit-reproducible at a fixed GPU count and
seed), so they compare directly.

### Three findings

1. **FP8 is worth more on H200 than on g7e.** The same lever is worth 1.073× on g7e (RTX PRO 6000
   Blackwell) and 1.13–1.19× here. 480p (1.18–1.19×) gains more than 768p (1.13×), because 768p
   spends a larger share of the step in attention, and **weight quantization does not touch
   attention**.

2. **SageAttention does not port to sm_90.** It is worth 1.26–1.29× on g7e (sm_120); on H200 it is a
   **regression** at 480p (28.909 vs 28.454, 1.6% slower) and only +5.5–5.8% at 768p. This is not a
   build or configuration problem: the image asserts at build time that `_qattn_sm90*.so` contains
   an `sm_90a` cubin (see `Dockerfile`), and the server log reads back
   `Using sage_attn attention backend`. What differs is **the baseline it replaces**: with sage off,
   the DiT on H200 runs `fa` (FlashAttention, confirmed in the log), whereas on g7e FA's fast path
   is constrained on sm_120, so sage there displaces a weaker baseline. Same kernel, different
   opponent, different win. (A harder attribution needs a backend ablation on both boxes; not done
   this round.) Conclusion: **no sage at 480p**, sage at 768p (+5.8% for free, with no extra quality
   cost — see "Quality").

3. **Cache-DiT ports completely, and it is where the cost savings are.** More steps, more value
   (2.24–2.40× at 30 steps, 1.94–1.96× at 20) — it skips the steps whose residual barely changes,
   and there are more of those when there are more steps. It is also the only accelerator that does
   not dilute multi-GPU scaling (7.04× at 768p vs FP8+sage's 7.06×).

**Memory: Ulysses shards the sequence, not the weights, so adding GPUs does not save memory.**
Single-GPU peak BF16 78,765 MiB → FP8 48,091 MiB (+sage 47,969, +Cache-DiT 48,487). At 8 GPUs the
**per-GPU** peak is instead ~82–83 GiB (BF16) / ~52.0–52.5 GiB (FP8 tier), i.e. 4–5 GB *higher* than
single-GPU (communication and activation buffers). The memory lever is `--tp-size` (see
`RESULTS.md`).

## Scaling (Ulysses 1 / 2 / 4 / 8)

Ulysses efficiency on H200 is high — much better than g7e (74–75% at 768p on 2 GPUs). 1 → 8:

| Case | 1 → 8 GPU speedup | efficiency |
|---|---|---|
| 768p/30 FP8+sage | 7.17× | 90% |
| 768p/30 BF16 | 7.15× | 89% |
| 768p/20 FP8+sage | 7.06× | 88% |
| 768p/30 +Cache-DiT | 7.04× | 88% |
| 480p/30 FP8+sage | 6.81× | 85% |
| 480p/20 FP8+sage | 6.48× | 81% |
| 480p/20 +Cache-DiT | 6.00× | 75% |

Two regularities: **higher resolution scales better** (longer sequence, smaller communication
share), and **the more aggressive the accelerator, the worse the scaling** (the numerator shrinks
while the communication cost does not).

Full curve (`GPUSETS="1 2 4 8"`, BF16 and +Cache-DiT arms each with their own 1-GPU denominator;
each cell is `latency / speedup / efficiency / $ per second of video (spot)`):

| Case | 1 GPU | 2 GPU | 4 GPU | 8 GPU |
|---|---|---|---|---|
| **BF16** | | | | |
| 480p/20 | 33.663 s — $0.006076 | 18.978 s 1.77×/89% $0.006850 | 9.915 s 3.40×/85% $0.007158 | 5.324 s 6.32×/79% $0.007687 |
| 480p/30 | 49.515 s — $0.008937 | 27.624 s 1.79×/90% $0.009971 | 14.181 s 3.49×/87% $0.010238 | 7.423 s 6.67×/83% $0.010718 |
| 768p/20 | 117.099 s — $0.021134 | 62.746 s 1.87×/93% $0.022649 | 31.920 s 3.67×/92% $0.023044 | 16.446 s 7.12×/89% $0.023746 |
| 768p/30 | 175.779 s — $0.031725 | 93.856 s 1.87×/94% $0.033879 | 47.638 s 3.69×/92% $0.034391 | 24.570 s 7.15×/89% $0.035476 |
| **FP8+sage+Cache-DiT `RDT=0.24`** | | | | |
| 480p/20 | 14.737 s — $0.002660 | 8.641 s 1.71×/85% $0.003119 | 4.477 s 3.29×/82% $0.003232 | 2.458 s 6.00×/75% $0.003549 |
| 480p/30 | 18.790 s — $0.003391 | 10.793 s 1.74×/87% $0.003896 | 5.464 s 3.44×/86% $0.003945 | 2.890 s 6.50×/81% $0.004173 |
| 768p/20 | 50.473 s — $0.009109 | 28.150 s 1.79×/90% $0.010161 | 14.082 s 3.58×/90% $0.010166 | 7.274 s 6.94×/87% $0.010503 |
| 768p/30 | 61.087 s — $0.011025 | 34.057 s 1.79×/90% $0.012293 | 16.992 s 3.60×/90% $0.012267 | 8.679 s 7.04×/88% $0.012531 |

**At 768p, 4 GPUs are free: 3.6–3.7× the latency for the same $/video-second as 2 GPUs**
($0.010166 vs $0.010161; BF16 $0.023044 vs $0.022649, a 0.05%–1.7% spread), and only 11–12% more
than 1 GPU. At 480p, 4 GPUs cost 16–19% more. Only the step from 4 to 8 costs real money
(+3% to +10%). **If you want latency, take 4 GPUs — it is the best-value point on this box.**

## Cost

Two bases, both measured prices:

- **`p5en.48xlarge` spot (us-east-2b) $26.8991/h = $3.36239 / GPU-hour = $0.000933997 / GPU-second**
  — the box this table ran on.
- **The customer's `p5e` Capacity Block $47.76/h = $5.97 / GPU-hour** — 1.78× more expensive.

Clip length 5.175 s. **Per second of finished video** (fl2va):

| Case | 480p/20 | 480p/30 | 768p/20 | 768p/30 |
|---|---|---|---|---|
| **spot $26.90/h** | | | | |
| FP8 1 GPU | **$0.005135** | **$0.007491** | $0.018695 | $0.027990 |
| FP8+sage 1 GPU | $0.005218 | $0.007606 | **$0.017672** | **$0.026404** |
| FP8+sage 8 GPU | $0.006443 (+22%) | $0.008932 (+17%) | $0.020016 (+13%) | $0.029475 (+12%) |
| +Cache-DiT 1 GPU | $0.002660 | $0.003391 | $0.009109 | $0.011025 |
| **Capacity Block $47.76/h** | | | | |
| FP8 1 GPU | $0.009118 | $0.013301 | $0.033193 | $0.049698 |
| FP8+sage 1 GPU | $0.009264 | $0.013505 | $0.031378 | $0.046880 |
| +Cache-DiT 1 GPU | $0.004722 | $0.006021 | $0.016174 | $0.019575 |

**Per clip** (spot, 1 GPU): FP8 480p/20 $0.0266, 480p/30 $0.0388, 768p/20 $0.0967, 768p/30 $0.1449;
with Cache-DiT $0.0138 / $0.0176 / $0.0471 / $0.0571.

The multi-GPU premium is exactly `1/efficiency − 1`, which on H200 is only **+12% to +22%** (on g7e
it is +33% to +63%), so "adding GPUs buys latency, not unit cost" is a much weaker statement here —
8 GPUs take 768p from 98 s to 13.9 s for 13% more per video-second.

A fleet for 1 QPS (one request per second), one replica per GPU, fl2va, FP8+sage:

| Case | per-GPU throughput | GPUs needed | `p5en` boxes | spot $/h | CB $/h |
|---|---|---|---|---|---|
| 768p/30 | 1 / 146.3 s | 147 | 18.4 | $494 | $878 |
| 768p/20 | 1 / 97.9 s | 98 | 12.2 | **$330** | $585 |
| 480p/30 | 1 / 42.1 s | 43 | 5.4 | $145 | $257 |
| 480p/20 | 1 / 28.9 s | 29 | 3.6 | **$98** | $173 |
| 480p/20 +Cache-DiT | 1 / 14.7 s | 15 | 1.9 | **$50** | $90 |

**480p is 3.4× cheaper than 768p**, so the resolution tier still matters more than any knob. Use DP
replicas (one per GPU) rather than Ulysses to build throughput: sglang serialises concurrent
requests to the same replica.

### Versus g7e

Same sglang, same prompt/seed/geometry, each tier in its own delivery configuration (H200 =
FP8[+sage], g7e = NVFP4+sage):

| | 480p/20 | 480p/30 | 768p/20 | 768p/30 |
|---|---|---|---|---|
| H200 1-GPU latency | 28.454 s | 41.506 s | 97.918 s | 146.295 s |
| g7e 1-GPU latency | 31.125 s | 45.145 s | 114.419 s | 170.278 s |
| **H200 faster by** | 1.09× | 1.09× | **1.17×** | **1.16×** |
| H200 $/video-second (spot) | $0.005135 | $0.007491 | $0.017672 | $0.026404 |
| g7e $/video-second (3-yr SP) | $0.002990 | $0.004337 | $0.010992 | $0.016359 |
| **g7e cheaper by** | 1.72× | 1.73× | **1.61×** | **1.61×** |

**H200 is only 1.09–1.17× faster per GPU but 1.88× more expensive per GPU, so g7e is 1.61–1.73×
cheaper per second of video.** The price basis does not favour g7e: its 3-year Savings Plan rate
(`g7e.48xlarge` $14.31835/h) is 4% *above* its current us-east-2 spot price ($14.8815/h).

What H200 does buy is a **latency ceiling**: g7e has only 2 GPUs at 74–75% Ulysses efficiency, H200
has 8 at 88–90%, so 768p/30 comes down to 20.4 s (8.7 s with Cache-DiT) where g7e's two-GPU best is
115.4 s (52.0 s with Cache-DiT). **Choose g7e for unit cost, H200 for the latency ceiling.** The g7e
line lives at [whn09/minimax_h3_g7e](https://github.com/whn09/minimax_h3_g7e).

## Quality

Reference = the **BF16 single-GPU** clip at the same geometry (single-GPU runs are bit-reproducible
at a fixed seed, so the run-to-run floor is SSIM 1.0). Besides SSIM, read **inter-frame motion
energy**: motion collapse pushes SSIM *up*, so SSIM alone will score a ruined clip as a good one.
Tool: `quality_pair_local.sh`.

| Arm (1 GPU) | 480p/20 | 480p/30 | 768p/20 | 768p/30 |
|---|---|---|---|---|
| BF16 8 GPU (reduction-order floor) | 0.9691 | 0.9711 | 0.9699 | 0.9712 |
| FP8 | 0.9514 | 0.9470 | 0.9429 | 0.9418 |
| FP8+sage | 0.9480 | 0.9467 | 0.9393 | 0.9405 |
| +Cache-DiT `RDT=0.24` | 0.9309 | 0.9361 | 0.9411 | 0.9390 |

(SSIM Y. Motion energy: reference 0.2172 / 0.2142 / 0.2615 / 0.2521; the arms sit between 0.21 and
0.30.)

Four things to read out of it:

1. **The multi-GPU floor is 0.969–0.971, not 1.0.** Ulysses changes reduction order, so the same
   seed is not bit-identical. Nothing below 0.969 can be attributed to "multi-GPU".
2. **FP8's 0.942–0.951 is a real quantization difference** (below the multi-GPU floor), but the
   composition and motion are the same and motion energy is only 3–9% higher. That is the normal
   cost of w8a8.
3. **sage costs no extra quality**: FP8+sage differs from plain FP8 by 0.0001–0.0036 across the four
   cases, i.e. noise. So the +5.8% at 768p is free.
4. **Cache-DiT is nearly lossless at 768p (0.939–0.941, on par with FP8); 480p/20 is the one cell to
   watch**: SSIM drops to 0.9309 and motion energy rises 25% (0.2726 vs 0.2172) — at 20 steps
   `RDT=0.24` skips too much. The next section sweeps the RDT.

## Picking the Cache-DiT RDT

Single GPU, FP8+sage+Cache-DiT, `RDT` swept 0.12 / 0.16 / 0.20 / 0.24 (`SECONDARY_RDT` moves with
it). Each cell is `latency / SSIM(Y) / motion energy`; the reference is still the same-geometry BF16
single-GPU clip (reference motion energy 0.2172 / 0.2142 / 0.2615 / 0.2521):

| RDT | 480p/20 | 480p/30 | 768p/20 | 768p/30 |
|---|---|---|---|---|
| off (plain FP8+sage) | 28.909 s / 0.9480 / 0.2386 | 42.143 s / 0.9467 / 0.2378 | 97.918 s / 0.9393 / 0.2856 | 146.295 s / 0.9405 / 0.2957 |
| 0.12 | 21.187 s / 0.9454 / 0.2393 | 22.691 s / 0.9425 / 0.2289 | 69.395 s / 0.9394 / 0.2823 | 80.031 s / 0.9436 / 0.2841 |
| **0.16** | **17.275 s / 0.9432 / 0.2401** | **20.059 s / 0.9411 / 0.2152** | 59.890 s / 0.9393 / 0.2765 | 65.823 s / 0.9410 / 0.2669 |
| 0.20 | 15.996 s / 0.9402 / 0.2501 | 18.758 s / 0.9361 / 0.2100 | 55.168 s / 0.9415 / 0.2801 | 61.050 s / 0.9390 / 0.2632 |
| 0.24 | 14.737 s / 0.9309 / 0.2726 | 18.790 s / 0.9361 / 0.2100 | **50.473 s / 0.9411 / 0.2784** | **61.087 s / 0.9390 / 0.2632** |

Three readings:

1. **At 768p the RDT does not cost quality anywhere in 0.12–0.24**: all four rows sit at SSIM
   0.9390–0.9415, the same as plain FP8+sage's 0.9393/0.9405, and motion energy does not move. So
   768p takes **0.24**, the fastest row.
2. **At 30 steps, RDT 0.24 and 0.20 are the same output** — not close, **byte-identical by md5** (at
   both 480p/30 and 768p/30), and the 0.03–0.04 s latency difference is noise. With more steps, 0.20
   already skips every step whose residual is small enough; raising the threshold finds no new ones.
3. **Only 480p/20 needs to come down**: at 0.24 it is SSIM 0.9309 with motion energy 0.2726, the one
   cell in the table outside the FP8 quantization band (0.942–0.951). Dropping to **0.16** buys back
   SSIM 0.9432 and puts motion energy at 0.2401, the same as with Cache-DiT off (0.2386), for
   17.275 s instead of 14.737 s. Going further to 0.12 buys almost nothing (0.9454) and costs
   21.187 s.

**Recommendation: `RDT=0.16` at 480p, `RDT=0.24` at 768p.** Cost at that setting (1 GPU, spot):

| | 480p/20 | 480p/30 | 768p/20 | 768p/30 |
|---|---|---|---|---|
| RDT | 0.16 | 0.16 | 0.24 | 0.24 |
| latency | 17.275 s | 20.059 s | 50.473 s | 61.087 s |
| vs FP8+sage | 1.67× | 2.10× | 1.94× | 2.40× |
| vs BF16 | 1.95× | 2.47× | 2.32× | 2.88× |
| $/video-second | **$0.003118** | **$0.003621** | **$0.009109** | **$0.011025** |

The Cache-DiT rows in the scaling table above use `RDT=0.24`; switching 480p to 0.16 scales those
two rows' $/video-second by 17.275/14.737 = 1.17× and 20.059/18.790 = 1.07×.

## Cache-DiT (correcting `RESULTS.md`)

`RESULTS.md` states that Cache-DiT is a no-op on H3 — registers, skips zero blocks, output
byte-identical. That was measured on `c7c03ec53b` and **was correct there** (H3's residual read back
as 0, so the threshold never fired). On `c0b6474b` it works.

It is **env-driven**, not `--cache-dit-config` (that is the diffusers backend):

```bash
ENVX="SGLANG_CACHE_DIT_ENABLED=1 SGLANG_CACHE_DIT_RDT=0.24 SGLANG_CACHE_DIT_SECONDARY_RDT=0.24"  # 768p
ENVX="SGLANG_CACHE_DIT_ENABLED=1 SGLANG_CACHE_DIT_RDT=0.16 SGLANG_CACHE_DIT_SECONDARY_RDT=0.16"  # 480p
```

Two traps:

- **The request must not carry a `quality` field** — it switches this generic Cache-DiT path off.
- **The "enabled" log line is emitted during warmup** (`cache-dit enabled on transformer (steps=19,
  Fn=1, Bn=0, rdt=0.240)`), in the same second `/health` goes green. Reading the log immediately
  after starting the server misses it and looks like "it did not attach". `h200_grid.sh` reads it
  back after the cases have run.

## Deployment (delta over `DEPLOYMENT_GUIDE.md`)

This round's image is **baked by `Dockerfile`** instead of being patched inside the container by
`serve.sh` at run time:

```bash
setsid nohup ./h200_bringup.sh > ~/bringup.log 2>&1 < /dev/null &   # ~40 min, mostly 269 GiB of weights
./build_image.sh                                                    # -> h3-h200:local, ~3 min
```

Five differences from `DEPLOYMENT_GUIDE.md`:

- **The base is pinned by digest** (`lmsysorg/sglang@sha256:51e576f0…` =
  `nightly-dev-20260818-c0b6474b`), **the same digest as the g7e line** — a cross-platform
  cost comparison needs the same sglang on both sides.
- **The AMI is the Base OSS-driver DLAMI, not the PyTorch DLAMI**, so there is no
  `/opt/pytorch/bin/activate`. `h200_bringup.sh` creates its own venv, and **disables
  `unattended-upgrades` first** (the DLAMI otherwise upgrades packages and reboots ~20 minutes after
  boot; upgrading docker kills in-flight timed requests, which looks like a spot reclaim but is not).
- **`huggingface_hub` 1.28.0 has no `cli` / `hf_transfer` extra**:
  `pip install "huggingface_hub[hf_transfer,cli]"` only warns and then installs a hub with no
  acceleration. `hf_transfer` must be installed as its own package.
- **sm_90's SageAttention is a separate extension**: unlike sm_120 (which reuses the sm89 sources and
  only adds a cubin), sm_90 builds `_qattn_sm90*.so`, so the build-time assertion checks a different
  filename. `cuobjdump --list-elf | grep sm_90a` must hit — a pip wheel installs a file with the
  identical name that simply lacks that cubin and silently falls back to Triton at run time.
  `TORCH_CUDA_ARCH_LIST=9.0` cross-compiles, so building the image needs **no GPU**.
- **The two NVFP4 patches are not applied**: sm_90 has no FP4 tensor cores, so the quantization tier
  here is FP8 via sglang's own `--quantization fp8`, with no offline quantized checkpoint.

Starting the server (single-GPU 480p delivery / 8-GPU 768p with sage):

```bash
IMAGE=h3-h200:local NAME=h3 GPUS=1 ULYSSES=1 \
  EXTRA="--quantization fp8 --layerwise-offload-components text_encoder" \
  WARMUP="864x480" ./serve.sh start

IMAGE=h3-h200:local NAME=h3 GPUS=8 \
  EXTRA="--quantization fp8 --attention-backend sage_attn \
    --component-attention-backends text_encoder=torch_sdpa,audio_vae=torch_sdpa,video_vae=torch_sdpa" \
  WARMUP="1344x768" ./serve.sh start
```

⚠️ **`--attention-backend` is global** (upstream issue
[#35743](https://github.com/sgl-project/sglang/issues/35743)), so turning sage on requires
`--component-attention-backends` to pull `text_encoder` / `audio_vae` / `video_vae` back onto
`torch_sdpa`; without it the scheduler dies.

ref2va additionally needs `ENVX="SGLANG_MINIMAX_H3_REF_IMAGE_SHORT_EDGE=1024"`: the default 2048
takes 480p/20 from ~36 s to ~65 s, and the reference short edge is purely an input-side choice.

The whole table is `./h200_grid.sh` (it starts and stops the server per arm, and every GPU count
carries its own BF16 denominator).

## Traps

1. **`p5en` is far easier to get than `p5e`.** Both are 8×H200 141 GB and equivalent for
   single-box H3, but p5e is only offered in us-east-2 (3 AZs) / us-west-2c / eu-north-1a, while
   p5en adds us-east-1b/d, us-west-2a/d, ap-south-1a/b, ap-northeast-1a. Hunting p5e alone burned
   13.5 hours; adding p5en landed one in 62 seconds.
2. **Disable `unattended-upgrades` first** (see "Deployment").
3. **`inference_time_s` sits at a version-dependent nesting depth in `status.json`** — do not
   hard-code the path (the scripts search for the key recursively).
4. **Pass `TAGSUF=` when sweeping RDT**: the tag does not contain the RDT, so without it each
   sweep point silently overwrites the previous mp4 and status.json and only the last one survives.
5. **Spot instances get reclaimed — pull results while the run is going** (`pull_results_loop.sh`).
