# MiniMax-H3 on H200 (p5e.48xlarge) — the customer deployment

The customer runs H3 on `p5e.48xlarge` (8xH200) via SGLang and insists on **480P**. They have since
confirmed the requirements: "10 s" means a **10-second clip** (243 frames @ 24 fps = 10.125 s), not
10 s of latency; **step count and width/height must both be request parameters**, with
**resolution-ratio and width/height as two mutually exclusive parameter groups**; and **all three
tasks (t2va / fl2va / ref2va) are required**. All of that is implemented and validated on hardware.

**To deploy, go straight to section 0 "Quick start" of `DEPLOYMENT_GUIDE.md`**: download weights →
start the server → send a request → collect the video.

There are **three measurement rounds** in this repo. `RESULTS.md` is the BF16 latency round (image
`c7c03ec53b`) — how to get a request under 10 s. **`RESULTS_QUANT.md` is the cost round** (image
`nightly-dev-20260818-c0b6474b`) — FP8 / SageAttention / Cache-DiT and dollars per second of finished
video. **`RESULTS_TURBO.md` changes the weights themselves** (Turbo LoRA distillation, 20 steps → 8):
the cheapest and lowest-latency tier (2.47–2.79×; 8 GPU 480p 2.010 s / 768p 5.136 s), at the cost of
dropping from the SSIM-0.94 band to the 0.91 band. Where rounds disagree (Cache-DiT, the 480p `RDT`),
the newer round wins.

## Deliverables

1. **`patches/` — three patches, in a fixed order**

   | order | patch | what it does |
   |---|---|---|
   | 1 | `minimax-h3-cpu-offload-inplace.patch` | One-line fix. Unpatched, any `*-cpu-offload` flag makes H3 die during warmup at `decoding.py:92` (`Inplace update to inference tensor outside InferenceMode`). This is the prerequisite for fitting H3 on a 96 GB card (RTX PRO 6000 / g7e) |
   | 2 | `minimax-h3-short-edge.patch` | Lets SGLang accept a non-768 short edge. Three hunks, opt-in behind `SGLANG_MINIMAX_H3_EXTRA_SHORT_EDGES`; unset, the released behaviour and both error strings are byte-for-byte unchanged |
   | 3 | `minimax-h3-target-width-height.patch` | Accepts `target.width` / `target.height` as a **second geometry group** and enforces that the two groups are mutually exclusive — exactly the customer's "pick one of the two". Released code accepts only 6 aspect literals and rejects even `"640:480"`, despite it being 4:3 |

   **#3 is diffed against a tree that already has #2 applied** (both edit
   `request_validation.py::_validate_target`), so the order cannot change. All three apply cleanly to
   `lmsysorg/sglang:dev` @ `c7c03ec53b`, and `serve.sh` applies them in order, idempotently.

   Validated on hardware: `short_edge: 480` + `16:9` really serves **864x480**, and
   `{"width": 800, "height": 480}` — a ratio that is not one of the released six — really serves
   `800x480`.

2. **`DEPLOYMENT_GUIDE.md`** (and `DEPLOYMENT_GUIDE_zh.md`) — **start here for the customer.**
   Section 0 is the quick start (weights download, the three deployment modes, request parameters,
   where videos land, the API shape); then the recommended commands for H200 and for g7e, a
   pick-a-shape-by-objective table, the `--tp-size` memory lever, encoder folding, the
   `target.width/height` semantics, the per-task cost differences and 1 QPS fleet sizing, the shared
   trap list, the Fabric Manager recovery recipe, and the dead ends.

3. **`RESULTS.md`** (and `RESULTS_zh.md`) — the measured latency/quality tables at 4 and 8 GPUs, the
   full replica sweep (1x8 / 2x4 / 4x2 / 8x1), the TP/encoder-parallel sweep, what Ulysses and TP
   cost without NVLink, the verified 96 GB verdict, the per-task step sweeps, the 11 width/height
   boundary cases, and the two negative results (Cache-DiT is a no-op **on image `c7c03ec53b`** —
   superseded, see below; `quality: "high"` is gated on 1344x768 *and* 50 steps).

4. **`RESULTS_QUANT.md`** (and `RESULTS_QUANT_zh.md`) — **the cost round**, measured on a newer
   sglang (`nightly-dev-20260818-c0b6474b`) on `p5en.48xlarge`: BF16 → FP8 → +SageAttention →
   +Cache-DiT × 1/2/4/8 GPUs × four geometries, the Ulysses 1→8 curve at 88–90% efficiency, dollars
   per second of finished video on two price bases, the 1 QPS fleet, the SSIM/motion quality table,
   and the g7e comparison. Headline: **single-GPU FP8 is the delivery config** (1.13–1.19× over
   BF16, 78.8 → 48.1 GB), sage is worth +5.8% at 768p but is a **regression at 480p**, and
   **Cache-DiT works on this image** (1.94–2.40×), which reverses `RESULTS.md`'s negative result.
   Its `Dockerfile` bakes the delivery image instead of patching the container at run time.

5. **`RESULTS_TURBO.md`** (and `RESULTS_TURBO_zh.md`) — **the Turbo LoRA round** (20 steps → 8-step
   distilled weights): the step curve (4/6/8), the Cache-DiT `RDT` sweep on top of turbo, the
   1/2/4/8-GPU curve, both fl2va and ref2va, quality (SSIM + motion energy + visual), and the
   comparison against g7e's turbo tier. Conclusion: **8 steps + `RDT=0.24`** is **2.47×/2.79×**
   faster than stock 20 steps and **59%/64%** cheaper per video-second, reaching **2.010 s** (480p) /
   **5.136 s** (768p) on 8 GPUs; `RDT=0.16` never fires at 8 steps, which **reverses the 0.16 that
   `RESULTS_QUANT.md` recommends for 480p**. **Reports that "this distilled LoRA looks terrible" do not
   reproduce here** (visually indistinguishable from stock 20 steps).

6. **Scripts**: `serve.sh` (start/stop/status, three deployment modes), `fill_ref2va.sh` (fills in
   the Ref2VA weights for 73 GiB less disk), `h3gen.py` (submitter covering all three tasks),
   `h3get.py` (a sidecar that turns one GET URL into an mp4), `h3req.py` (the original polling
   submitter), `h200_grid.sh` / `h200_turbo.sh` (the drivers behind the last two rounds),
   `lora_merge_transformer.py` (merges a LoRA into the bf16 transformer offline).

Headline: **8 GPUs, `--ulysses-degree 8`, 864x480 → 10.05 s for a 5 s clip at 40 steps, 10.58 s for
a 10 s clip at 16 steps.** 4 → 8 GPUs scales near-linearly (1.9x) for +9% GPU-seconds, which is what
buys back the step budget (40 vs 20 steps, SSIM 0.9682 vs 0.8691).

Two memory levers, in order of preference:

- **On a P2P-capable box, use `--tp-size`.** It shards the 61.73 GB DiT, which Ulysses does not
  touch: 95.9 → 39.0 GiB per GPU for +30% latency, or -32 GiB for +10% at `--tp-size 2`. At TP4
  or above the 480p config fits an 80 GB card.
- **On a no-NVLink box, neither Ulysses nor TP is usable** (15x and 19x slower respectively with
  P2P disabled). Run one replica per GPU with `--num-gpus 1 --ulysses-degree 1
  --text-encoder-cpu-offload --vae-cpu-offload` (79.4 GiB per GPU, ~66 s per video, verified to fit
  under a 96 GB cap).

## Serving

`serve.sh` wraps "create a long-lived container → apply the three patches in order, idempotently →
start the server detached → wait for `/health`", and covers the three deployment modes the customer
asked for:

```bash
./serve.sh                                   # mode 1: 8-GPU fl2va, serves t2va + fl2va (:30010)
GPUS=4 ./serve.sh                            #   4 GPUs (the cookbook's 4xH200 recipe)
CUDA_VISIBLE_DEVICES=0,1,2,3 ./serve.sh      #   exactly those 4; the count is inferred

VARIANT=ref2va ./serve.sh                    # mode 2: ref2va (:30030)
VARIANT=ref2va CUDA_VISIBLE_DEVICES=4,5,6,7 ./serve.sh

./serve.sh both                              # mode 3: two replicas, all three tasks (4 + 4)
GPUS_A=2 GPUS_B=6 ./serve.sh both            #   uneven: ref2va costs 3.3x per step

DRYRUN=1 ./serve.sh both                     # print the resolved placement, touch no GPU
TP=2 ULYSSES=4 ./serve.sh                    # shard the DiT: 63.9 GiB/GPU instead of 95.9
SHORT_EDGES= ./serve.sh                      # leave 480p off (patch inert), verify released behaviour
./serve.sh stop | logs | status               # acts on all replicas unless VARIANT is set
```

**Why mode 3 needs two processes**: `--model-variant` selects which DiT is loaded, and the task →
partition map is a hard gate — `fl2va` serves `t2va` + `fl2va`, `ref2va` serves only `ref2va`, so one
process can never cover all three. The two replicas are isolated with `CUDA_VISIBLE_DEVICES`
(`--base-gpu-id` is silently ignored), and co-residency is measured to be free (ref2va 32.25 s
alongside fl2va vs 32.17 s alone).

If you would rather drive the server yourself:

```bash
SGLANG_MINIMAX_H3_EXTRA_SHORT_EDGES=480 sglang serve \
  --model-path /models/MiniMax-H3 --model-variant fl2va \
  --num-gpus 8 --ulysses-degree 8 --performance-mode speed \
  --warmup-resolutions 1344x768 864x480 \
  --output-path /out/videos \
  --host 0.0.0.0 --port 30010
```

Three things to get right:

- **`--warmup-resolutions` must list every resolution served**; without it the first request at a
  cold shape pays about 10 s. It is `nargs="+"`, so one server can warm both — which is what
  `serve.sh` does by default (`WARMUP="1344x768 864x480"`, ~7.65 s extra at startup; narrow it to
  `864x480` on a 96 GB card). It takes raw `WxH` via `parse_size` and bypasses the canonical
  short-edge validator, so `864x480` is *accepted* even on an unpatched server — but acceptance is
  not warming: on image `c7c03ec53b` a server configured with `["864x480"]` logged its only warmup
  request as `1344x768x124f`. Always confirm with
  `grep -o 'warmup req ([^)]*)'` on the server log (guide §1.1).
- **Omitting `--output-path` is a trap**: the server writes to a relative `outputs/` inside the
  container, which is usually not mounted, and the videos die with the container. `serve.sh` passes
  it, so videos land on the host at `/opt/dlami/nvme/out/videos/<id>.mp4`.
- **A local weights directory must be *named* `MiniMax-H3`.** `registry.py:1199`
  `get_non_diffusers_pipeline_name()` matches the **basename** of `--model-path`, not `--model-id`.
  A different name reads the root `model_index.json` and dies with `module diffusers has no attribute
  MiniMaxH3ModularPipeline`. (`serve.sh` mounts the weights as `/models/MiniMax-H3`, so the host name
  is free.)

Patch idempotency uses stamp files in `/sgl-workspace/.h3-patches/`, not `git apply -R --check` —
the latter is a broken test here because patch #3 rewrites patch #2's hunk context. The log prints
`APPLIED` / `ALREADY`; if a newer image no longer takes a patch it fails hard with
`DOES_NOT_APPLY ... image moved off c7c03ec53b` rather than serving half-patched.

## No-NVLink boxes (RTX PRO 6000 / g7e)

Do not use Ulysses: with P2P disabled, 8-GPU Ulysses=8 goes from 10.05 s to 151.97 s (15x slower).
The right shape is **one replica per GPU with encoder/VAEs offloaded to CPU** (which requires
`minimax-h3-cpu-offload-inplace.patch`, or warmup crashes):

```bash
OFFLOAD=1 GPUS=1 ULYSSES=1 ./serve.sh        # or by hand:
CUDA_VISIBLE_DEVICES=<one gpu> SGLANG_MINIMAX_H3_EXTRA_SHORT_EDGES=480 sglang serve \
  --model-path /models/MiniMax-H3 --model-variant fl2va \
  --num-gpus 1 --ulysses-degree 1 --performance-mode speed \
  --text-encoder-cpu-offload --vae-cpu-offload \
  --warmup-resolutions 864x480 --host 0.0.0.0 --port <port>
```

79.4 GiB per GPU (132.2 without offload), verified under a 96 GB cap with a ballast allocation;
~66 s per request, ~0.91 videos/min per GPU. **Adding GPUs raises throughput but cannot cut that
66 s.** `launch_replicas.sh <gpus_per_replica>` does the fan-out and `conc_multi.py <N> <steps>
<ports>` is the concurrency probe.

## Submitting requests

`h3gen.py` is the main submitter: all three tasks, both geometry groups, any step count and clip
length.

```bash
python3 h3gen.py --width 864 --height 480 --steps 16 --duration 10        # t2va, 10 s clip
python3 h3gen.py --short-edge 480 --aspect 21:9 --steps 20                # the other group
python3 h3gen.py --task fl2va --image assets/first.png --inline --steps 16  # first frame
python3 h3gen.py --task ref2va --ref-video assets/ref.mp4 --inline --steps 8 --port 30030
python3 h3gen.py --host <box> --task fl2va --image assets/first.png --inline   # from a laptop
```

It prints the wire form it actually used (`wire=ratio|literal|exact`), so "I asked for 864x480 and
the server quietly gave me something else" cannot happen. The two geometry groups are mutually
exclusive, as the customer required.

**`--inline` is how the bytes get there.** The server resolves the condition `uri` itself, so a bare
path is read *on the server*; `--inline` instead sends the file as a `data:` URI in the request body,
and an `http(s)://` URI makes the server fetch it. Sample materials are in `assets/`. That HTTP fetch
skips SGLang's SSRF policy on purpose (`material_io.py:719`), so put an allowlist in front of it if
untrusted callers can reach the API — see §0.3 of the guide.

For one URL that returns a video (demos, debugging) use `h3get.py`. sglang's API is an asynchronous
three-step (`POST /v1/videos` → poll → `GET /{id}/content`) and **has no GET generate route at all**;
this sidecar walks those three steps for you:

```bash
python3 h3get.py --ref2va-port 30030 &
curl "http://127.0.0.1:8080/gen?prompt=three+cats&width=864&height=480&steps=16&duration=10" -o v.mp4
```

It holds the connection for 10–60 s, so **do not use it in production** and do not expose it
publicly.

`h3req.py` is the original polling submitter (t2va only, still works — the model card's own scripts
do a single status GET and download a truncated file):

```bash
python3 h3req.py <short_edge> <steps> <duration_s> <out-prefix> [aspect_ratio]
python3 h3req.py 480 40 5 u8_480p_40
```

## Contents

| path | what |
|---|---|
| `DEPLOYMENT_GUIDE.md` / `DEPLOYMENT_GUIDE_zh.md` | **deployment best practice (EN/ZH) — start here** |
| `RESULTS.md` / `RESULTS_zh.md` | measured results, BF16 round on `c7c03ec53b` (EN/ZH) |
| `RESULTS_QUANT.md` / `RESULTS_QUANT_zh.md` | measured results, quantization/cost round on `c0b6474b` (EN/ZH) |
| `RESULTS_TURBO.md` / `RESULTS_TURBO_zh.md` | measured results, Turbo LoRA (8-step) round on `c0b6474b` (EN/ZH) |
| `Dockerfile` / `build_image.sh` / `.dockerignore` | bakes the delivery image (sm_90 SageAttention + patches) |
| `h200_bringup.sh` | bare-metal prep: disable auto-upgrades, venv, weights, image pull |
| `h200_grid.sh` | the 4-arm x GPU-count x geometry driver behind `RESULTS_QUANT.md` |
| `h200_turbo.sh` | the 5-phase driver behind `RESULTS_TURBO.md` (reference / step curve / RDT / scale-out / ref2va) |
| `lora_merge_transformer.py` | merges a LoRA into the bf16 transformer offline (259/259 modules must hit) |
| `quality_pair.sh` / `quality_pair_local.sh` | SSIM + inter-frame motion energy against a reference clip |
| `pull_results_loop.sh` | keeps pulling artifacts off a spot box while it runs |
| `patches/` | the three delivered patches + `make_patch.sh` for re-diffing |
| `serve.sh` | start/stop/status, three deployment modes (validated on hardware) |
| `fill_ref2va.sh` | fills in Ref2VA: downloads only the transformer, hardlinks the other 16 files from FL2VA, saving 73 GiB |
| `h3gen.py` | general submitter: three tasks / both geometry groups / any steps and duration |
| `assets/` | sample condition materials: `first.png`, `last.png`, `ref.mp4` (10.125 s), `ref5s.mp4` (5.04 s), `refaudio.wav` |
| `h3get.py` | sidecar turning one GET URL into an mp4 (demo use) |
| `h3req.py` | the original t2va polling submitter |
| `launch_replicas.sh` / `launch_mixed.sh` / `serve_topo.sh` | multi-replica and topology-sweep launchers |
| `conc_multi.py` | concurrency probe |
| `ssim_pairs.sh` | scores topology outputs against each other |
| `SRT_ENCODER_PR_ASSESSMENT.md` | feasibility of wiring `--srt-encoder-url` upstream |
| `runs/` | raw artifacts (mp4 + request/status json + `frame_*.png`) |
| `videos_named/` | the same videos under readable names |
| `logs/` | server logs |

The 4xH100 baseline this is compared against is `../h3_h100_baseline/`; the Trainium port is
`../h3_plugin_src/`.

## Operational note

None of the three patches is upstream. They must be re-applied after an SGLang image upgrade
(`patches/make_patch.sh` re-diffs them), or MiniMax/SGLang should be pushed to make non-768 short
edges and `target.width/height` first-class options. Also, H3-Base is released at 768px, so non-768
resolutions are out-of-distribution: **there is no official reference output to compare against**,
and the customer needs to judge quality on their own prompts.
