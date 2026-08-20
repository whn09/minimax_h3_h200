# MiniMax-H3 deployment guide: H200 and g7e (RTX PRO 6000)

Customer-facing deployment recommendations. Every number was measured on `p5e.48xlarge` (8xH200),
timed client-side (POST to `status: completed`). Full experimental record in `RESULTS.md`, patches
in `patches/`.

Two baseline conditions, do not mix them: the **topology and memory tables use 864x480 / 124 frames
(5 s clip) / 40 steps / t2va / seed 1101**; **section 3 (the three tasks and capacity planning) uses
864x480 / 243 frames (10 s clip) / 16 steps**, because the customer confirmed "10 seconds" means
**clip duration**, not time-to-first-video.

**All three tasks (t2va / fl2va / ref2va) are validated on real hardware**, and one box can serve
all three at once — see 0.2.

Headline: **H200 has two cheap parallel levers (Ulysses and TP); g7e has neither.** The optimal
shape differs completely between the two platforms — do not port the H200 command to g7e.

---

## 0. Quick start (the current recommended flow)

Four steps: download weights → start the server → send a request → collect the video. All measured
on `p5e.48xlarge`.

### 0.1 Download the weights

`serve.sh` does **not** download the model. One command gets both weight partitions:

```bash
source /opt/pytorch/bin/activate
pip install -U "huggingface_hub[cli]"
hf download MiniMaxAI/MiniMax-H3 \
  --include "FL2VA/*" --include "Ref2VA/*" \
  --local-dir /opt/dlami/nvme/h3     # 162 files, 269 GiB
```

**Repeat `--include` per pattern** — each one takes a single value, so `--include "A" "B"` treats `B`
as a filename and fails with `Error: File not found in repository ... /Ref2VA/%2A`. Add `--dry-run`
to check a variant before transferring; the correct command lists 162 files, 81 per partition.

That is everything the server ever opens: **`FL2VA/` serves t2va and fl2va, `Ref2VA/` serves
ref2va**. Do not clone the whole repo (**464 GiB**) — the remainder is a flat diffusers layout that
sglang never reads. Serving only t2va/fl2va, drop the second `--include` and it is **134 GiB**.

Optional, saves **73 GiB**: **every one of Ref2VA's 16 non-transformer LFS files has the same oid as
FL2VA's** (compared oid-by-oid through the HF tree API, not "the sizes look equal"), so they can be
hardlinked instead of downloaded twice:

```bash
D=/opt/dlami/nvme/h3
hf download MiniMaxAI/MiniMax-H3 --include "FL2VA/*" --local-dir $D
hf download MiniMaxAI/MiniMax-H3 --include "Ref2VA/*" \
  --exclude "Ref2VA/transformer/*.safetensors" --local-dir $D   # config/index only, ~29 MB
bash fill_ref2va.sh $D        # downloads Ref2VA/transformer (62 GiB) + hardlinks the other 16
```

**196 GiB** instead of 269 GiB. Forgetting the third command is what fails at load time with
`ValueError: no safetensors files found in .../Ref2VA/transformer`.

Either way the directory name matters, not the download route: `serve.sh` defaults to
`/opt/dlami/nvme/h3` and mounts it as `/models/MiniMax-H3` (see note 3 in 0.2).

### 0.2 The three deployment modes

`serve.sh` is these three and nothing else worth remembering:

```bash
# mode 1: fl2va (serves t2va + fl2va), port 30010
./serve.sh                                   # all 8 GPUs -- the recommended H200 config
GPUS=4 ./serve.sh                            # 4 GPUs, sglang picks which
CUDA_VISIBLE_DEVICES=0,1,2,3 ./serve.sh      # exactly those 4 (GPU count is inferred)

# mode 2: ref2va (serves ref2va only), port 30030
VARIANT=ref2va ./serve.sh
VARIANT=ref2va CUDA_VISIBLE_DEVICES=4,5,6,7 ./serve.sh

# mode 3: both at once (all three tasks), isolated from each other by CUDA_VISIBLE_DEVICES
./serve.sh both                              # 4 + 4
GPUS_A=2 GPUS_B=6 ./serve.sh both            # uneven: ref2va costs 3.3x per step, give it more

# prefix any of these with DRYRUN=1 to print the resolved placement without touching a GPU
DRYRUN=1 ./serve.sh both
./serve.sh status | logs | stop               # acts on all replicas unless VARIANT is set
```

**Why two processes are unavoidable: `--model-variant` selects which DiT is loaded, and the
task → partition map is a hard gate, not a preference.**

| `--model-variant` | tasks served | default port |
|---|---|---|
| `fl2va` (default) | `t2va`, `fl2va` | 30010 |
| `ref2va` | `ref2va` | 30030 |

Asking an fl2va server for ref2va is refused explicitly:

```
task 'ref2va' is not served by MiniMax H3 partition 'fl2va'; supported tasks: ['t2va', 'fl2va']
```

So **the client must route by task**: t2va/fl2va → 30010, ref2va → 30030.

Mode 3 is measured: each replica is ready in 90 s (started sequentially, ~180 s total), 103 GiB per
GPU for fl2va and 104 GiB for ref2va, all 8 GPUs at 100% when one request runs against each, and
**ref2va takes the same time as when it owns the box (32.25 s vs 32.17 s) — co-residency is free**,
because the two device sets are disjoint.

Three things to watch:

- **`--base-gpu-id` does not work** (it is silently ignored); isolation has to go through
  `CUDA_VISIBLE_DEVICES`.
- **The two replicas need separate `--master-port` / `--scheduler-port`.** `serve.sh` presets them
  per variant (30100/5700 and 30120/5720), so there is nothing to fill in.
- **A local weights directory must be *named* `MiniMax-H3`**: `registry.py:1199` resolves the
  pipeline class from the basename of `--model-path`. `serve.sh` mounts `$WEIGHTS` as
  `/models/MiniMax-H3` inside the container, so the host name is free — this only matters if you
  write the docker command yourself. (`serve.sh` defaults to `/opt/dlami/nvme/h3`, named for the
  model because that one directory holds **both** the FL2VA and Ref2VA partitions. It still falls
  back to the older `h3-fl2va` name if that is what a box already has.)

### 0.3 Sending requests: everything is a parameter

`h3gen.py` covers all three tasks, any geometry, any step count, any clip length:

```bash
# t2va: 10 s clip, 16 steps, 864x480
python3 h3gen.py --width 864 --height 480 --steps 16 --duration 10

# the other geometry group (mutually exclusive with width/height)
python3 h3gen.py --short-edge 480 --aspect 21:9 --steps 20 --duration 10

# fl2va: first frame (last frame optional)
python3 h3gen.py --task fl2va --image assets/first.png --inline --steps 16 --duration 10

# ref2va: reference video (its audio track comes along) -- note the port
python3 h3gen.py --task ref2va --ref-video assets/ref.mp4 --inline --steps 8 --port 30030
```

**The two geometry groups are mutually exclusive**, matching the customer's "pick one of the two
groups" requirement, and that is how the server validates them too (see 1.6). `--duration` is bound
by the model's own **4–15 s** range; a 10 s clip is really **243 frames @ 24 fps = 10.125 s**.

**How a condition file reaches the server.** The `uri` string in each condition is resolved *by the
server* (`minimax_h3_localize_material_uri`, `.../minimax_h3/material_io.py:761`), which accepts
four useful schemes:

| `uri` | server behaviour | use it when |
|---|---|---|
| `data:image/png;base64,…`, `base64://…` | decodes the payload out of the request body | **the normal case** — `h3gen.py --inline` builds this |
| `http://…`, `https://…` | the **server** fetches the URL | the material already lives in object storage / a CDN |
| `/path/to/x.png`, `file:///…` | read in place, no copy | client and server share a filesystem |
| `tar+offset://`, `tar+b64header://` | member of a local tar | batch pipelines |

`s3://` raises `NotImplementedError` unless an artifact resolver is configured. Two properties are
worth knowing before this is exposed to real callers:

- **The HTTP fetch is deliberately unguarded.** The comment at `material_io.py:719` states it skips
  the shared SSRF policy and has no cumulative deadline, so the server will fetch whatever URL a
  caller sends, including link-local metadata addresses. **Put your own allowlist in front of it**
  if untrusted clients can reach the API.
- **Both base64 alphabets work** — standard `+/` and URL-safe `-_` (`_BASE64_ALPHABET`, line 33) —
  and whitespace plus percent-escapes are tolerated, so ordinary `base64.b64encode` output is fine.

Sample materials are in `assets/` (`first.png`, `last.png`, `ref.mp4` 10.125 s, `ref5s.mp4` 5.04 s,
`refaudio.wav`). They are just fixtures cut by `mkmat.sh` out of an earlier generated clip.

### 0.4 Where the videos land

`serve.sh` passes `--output-path /out/videos`, and `/out` is the mounted `$OUTDIR`, so they appear
**on the host at `/opt/dlami/nvme/out/videos/<video_id>.mp4`**.

**Omitting that flag is a trap**: the server then writes to a relative `outputs/` inside the
container, which nothing mounts, and the videos die with the container. To disable saving entirely
and fetch only over HTTP, use `OUTPATH= ./serve.sh`.

The status JSON also carries `file_paths`, so a caller on the same host **does not need a second
HTTP transfer**:

```json
{"status": "completed", "file_paths": ["/out/videos/<id>.mp4"], "inference_time_s": 4.26}
```

### 0.5 The API is an asynchronous three-step (and how to make it one GET URL)

This is the only way sglang's video API works — the POST handler in `video_api.py` ends with
`# Enqueue the job asynchronously and return immediately`:

```
POST /v1/videos              -> {"id": ..., "status": "queued"}   returns before any compute
GET  /v1/videos/{id}         -> "status": "completed"             poll
GET  /v1/videos/{id}/content -> mp4 bytes
```

**There is no GET generate route and no parameter that makes POST block for the result** (checked
across all of `runtime/entrypoints/`). If you want one URL that returns a video — for demos and
debugging — use `h3get.py` in this directory. It is a sidecar, it does not modify sglang, and it
walks those three steps for you and returns the mp4 bytes:

```bash
python3 h3get.py --ref2va-port 30030 &        # listens on 8080
curl "http://127.0.0.1:8080/gen?prompt=three+cats+playing+brass+instruments\
&width=864&height=480&steps=16&duration=10" -o v.mp4     # measured 10.07 s
```

Paste it in a browser and it plays. It routes on `task=` to the right replica and its parameter
names match `h3gen.py`
(`prompt/task/width/height/short_edge/aspect/steps/duration/seed/image/ref_video/...`); add
`&json=1` for the job metadata instead. **It holds the connection for 10–60 s, so do not use it in
production** — production should POST and poll. Do not expose it publicly either: the paths in the
URL are read by the server.

---

## 1. H200 (p5e.48xlarge, NVLink)

### 1.1 First: the patch must be applied inside the container

**`SGLANG_MINIMAX_H3_EXTRA_SHORT_EDGES=480` does nothing on a stock image.** Only patched code
reads that variable (`minimax_h3/resolved_plan.py`); unpatched, nothing reads it and a
`short_edge: 480` request is still rejected by the released validator. Setting the env var without
applying the patch is the single most likely way to "follow the guide and have it not work".

Good news: **no image rebuild is needed.** In `lmsysorg/sglang:dev`, sglang is an *editable*
install (`Editable project location: /sgl-workspace/sglang/python`), so `git apply` against
`/sgl-workspace/sglang` takes effect the next time the server process starts.

The easiest route is the wrapper in this directory, which mounts the patches, applies them
idempotently, starts the server detached and waits for `/health` (the three deployment modes are in
0.2):

```bash
cd h3_h200_baseline
./serve.sh                      # <- the recommended H200 config; see below
GPUS=4 ./serve.sh               # the cookbook's 4xH200 recipe
VARIANT=ref2va ./serve.sh       # ref2va only, on port 30030
./serve.sh both                 # all three tasks: two replicas, 4 + 4
TP=2 ULYSSES=4 ./serve.sh       # shard the DiT: 63.9 GiB/GPU instead of 95.9
SHORT_EDGES= ./serve.sh         # released 768-only policy, patch stays inert
./serve.sh status | logs | stop
```

**Bare `./serve.sh`, with no arguments and no env vars, is the recommended configuration**:
8 GPUs, TP=1, Ulysses=8, `encoder-parallel auto`, 480p enabled, and `WARMUP="1344x768 864x480"` —
the measured 10.05 s / 6.2 vid/min / 95.9 GiB-per-GPU shape. It creates a long-lived
`sleep infinity` container and patches the source *inside* it, so restarting the server with
different flags neither re-pulls the image nor re-applies the patch. `stop` kills only the server
process and keeps the container, so the patched source survives.

**The warmup default covers both shapes the customer may ask for** — `1344x768` and `864x480` — since
a cold shape's first request pays ~10 s extra and the extra warmup costs ~7.65 s once per server
lifetime. Narrow it with `WARMUP="864x480" ./serve.sh`, and **do narrow it on a 96 GB card (g7e)**:
the 79.4 GiB fit in 2.2 was measured at 480p only and `1344x768` is 2.49x the area. `serve.sh` warns
if you leave both on with `OFFLOAD=1`.

**Confirm the list was honoured — do not assume it.** On image `c7c03ec53b` a server recording
`warmup_resolutions=["864x480"]` warmed `1344x768x124f` instead (likely cause and status in
`RESULTS.md`). Check with:

```bash
docker exec h3 bash -lc "tr '\r' '\n' < /out/serve_fl2va.log | grep -o 'warmup req ([^)]*)'"
```

Any served shape missing from that output should be budgeted at ~10 s extra on its first request.

**All three patches in `patches/` are applied, and the order is fixed:**

| order | patch | what it does | cost of skipping it |
|---|---|---|---|
| 1 | `minimax-h3-cpu-offload-inplace.patch` | one line, allows CPU offload | `OFFLOAD=1` dies during warmup (2.3) |
| 2 | `minimax-h3-short-edge.patch` | enables non-768 short edges (480p) | `SGLANG_..._EXTRA_SHORT_EDGES` does nothing |
| 3 | `minimax-h3-target-width-height.patch` | accepts `target.width/height` (1.6) | only the 6 released aspect strings work |

**#3 is diffed against a tree that already has #2 applied**, because both edit
`request_validation.py::_validate_target`. The order is therefore not alphabetical luck, and
`serve.sh` lists them explicitly; to regenerate, use `patches/make_patch.sh` (it reverse-applies #3,
commits temporarily, then re-applies, so the diff stays clean).

Applying all three unconditionally is safe: #1 is a no-op without an offload flag, #2 is inert
without the env var, and #3 is inert unless you send width/height.

**Idempotency uses stamp files, not `git apply -R --check`.** The latter is broken as an
"already applied?" test here: patch #3 rewrites #2's hunk context, so on a tree with all three
applied, the reverse check fails for #2 and the script would wrongly report `DOES_NOT_APPLY`. What
it does instead is try `git apply --check` first (apply and drop a stamp if it succeeds), and if
that fails, look for a stamp in `/sgl-workspace/.h3-patches/`: present means `ALREADY`, absent means
genuinely inapplicable and it exits. Stamps live and die with the patched source (both are in the
container filesystem), so "stamp present but source unpatched" cannot happen. Verified: three
`APPLIED` lines on a fresh container, three `ALREADY` lines on the next run.

Then submit the recommended request, also with no arguments:

```bash
python3 h3req.py                # 864x480 / 40 steps / 5 s clip -> measured 10.09 s
python3 h3req.py 768 12 5 my768 # override when needed: [short_edge [steps [duration [prefix]]]]
```

### 1.2 Recommended: latency-first (10 s target)

If you would rather drive Docker yourself, mount the patch read-only and apply it in the same
`bash -lc` that starts the server:

```bash
docker run -d --name h3 --gpus all --ipc=host --network host --shm-size 32g \
  -v /opt/dlami/nvme/h3:/models/MiniMax-H3:ro \
  -v $PWD/patches/minimax-h3-short-edge.patch:/patch.patch:ro \
  -v /opt/dlami/nvme/out:/out \
  lmsysorg/sglang:dev bash -lc '
    cd /sgl-workspace/sglang && git apply -p1 /patch.patch &&
    SGLANG_MINIMAX_H3_EXTRA_SHORT_EDGES=480 sglang serve \
      --model-path /models/MiniMax-H3 --model-variant fl2va \
      --num-gpus 8 --ulysses-degree 8 \
      --performance-mode speed \
      --warmup-resolutions 1344x768 864x480 \
      --host 0.0.0.0 --port 30010 > /out/serve.log 2>&1'
```

**10.05 s per request, 6.2 videos/min, 95.9 GiB per GPU.**

`git apply` is **not idempotent** — running it twice fails. This one-shot form is therefore only
suitable for a container you throw away; for anything you restart repeatedly, use `serve.sh`,
which distinguishes "not yet applied" from "already applied" with `--check` plus stamp files (1.1).

Verify the patch is actually live — do **not** use "warmup succeeded" as evidence. Warmup builds its
requests from raw `WxH` through `parse_size` rather than through the canonical short-edge validator,
and (see 1.1) it does not reliably warm the shapes you list at all, so it tells you nothing either
way. The source tree is the evidence:

```bash
docker exec h3 git -C /sgl-workspace/sglang diff --stat
docker exec h3 git -C /sgl-workspace/sglang log --oneline -1   # expect c7c03ec53b
```

**The patch is diffed against image commit `c7c03ec53b`.** If you pull a newer `:dev` it may not
apply; `serve.sh` fails loudly with `DOES_NOT_APPLY ... image moved off c7c03ec53b` rather than
serving half-patched, and the `&&` in the command above short-circuits the same way. Re-diff before
trusting it on a newer image.

### 1.3 Pick a shape by objective

| objective | shape | latency | throughput | per-GPU mem |
|---|---|---|---|---|
| **lowest latency** | 1 replica x 8 GPU, Ulysses=8 | **10.05 s** | 6.2 vid/min | 95.9 GiB |
| **highest throughput** | 8 replicas x 1 GPU | 61.38 s | **7.69 vid/min** | 132.2 GiB |
| balanced | 2 replicas x 4 GPU, Ulysses=4 | 18.17 s | 6.69 vid/min | 100.7 GiB |
| **least memory** | 1 replica x 8 GPU, TP=8 | 13.09 s | 4.8 vid/min | **39.0 GiB** |

Latency and throughput trade roughly 6:1: going from 8 single-GPU replicas to one 8-GPU replica
costs only 24% of throughput and buys **6.11x** lower latency. Ulysses is efficient on NVLink
(76% parallel efficiency at 8 GPUs, 84% at 4, 89% at 2).

### 1.4 `--tp-size` is the memory lever, and it is cheap (new finding)

Ulysses is *sequence* parallel: weights replicated, only activations split, so it does nothing for
the 61.73 GB DiT. `--tp-size` shards the DiT itself and costs far less latency than expected:

| shape | DiT / GPU | per-GPU peak | latency | vs best |
|---|---|---|---|---|
| TP1 x Ulysses8 | 61.73 | 95.9 GiB | **10.08 s** | — |
| TP2 x Ulysses4 | 30.86 | 63.9 GiB | 11.08 s | +10.2% |
| TP4 x Ulysses2 | 15.43 | 47.5 GiB | 11.59 s | +15.0% |
| TP8 x Ulysses1 | 7.72 | **39.0 GiB** | 13.09 s | +30.0% |

**2.5x less memory for +30% latency**; TP2 alone saves 32 GiB for +10%. The practical consequence
is new capability: at TP4/TP8 the 480p config fits an **80 GB** card (A100-80G / H100-80G), which
the default Ulysses=8 shape at 95.9 GiB does not. Add `--tp-size N` and set
`--ulysses-degree` to `8/N`:

```bash
--num-gpus 8 --tp-size 2 --ulysses-degree 4     # 63.9 GiB, 11.08 s
```

### 1.5 The encoder is already distributed — nothing to do

The cookbook's picker lists 3 modes; there are actually **4**: `auto | fold | dp | replicate`
(`server_args.py:1598`). On multi-GPU, `auto` already folds `text_encoder` across the Ulysses
ranks:

| `--encoder-parallel` | text_encoder | per-GPU peak | latency |
|---|---|---|---|
| `replicate` | 47.97 GB | 135.6 GiB | 10.09 s |
| `fold` / `auto` (default) | **8.23 GB** | **95.9 GiB** | **10.08 s** |

**Folding is free: 39.7 GiB/GPU at zero latency cost.** So the idea of dedicating one GPU to the
encoder to save memory has already had most of its benefit taken by the default behaviour — on
H200 no action is needed.

Two traps:
- **`dp` mode is dead code for H3**: it requires `batch_size > 1`, and H3 is hard-capped at
  `batching_max_size=1`.
- **A 1-GPU replica never folds**: `server_args.py:669` requires `replica_size > tp_size`, so the
  8 x 1 shape carries the full 47.97 GB — the main source of its 132.2 GiB.

### 1.6 Width and height as parameters: `target.width/height` (new patch)

**The released wire contract has no `width`/`height`** — only `{short_edge, aspect_ratio,
duration_seconds}` — and there are two independent filters:
`configs/sample/minimax_h3.py::_validate` **projects** `target` onto those three keys (extra
width/height are dropped silently, with no error), and `request_validation.py::_validate_target`
accepts only those three. Worse, **`aspect_ratio` is a string membership test** against exactly
`21:9 / 16:9 / 4:3 / 1:1 / 3:4 / 9:16`, so `"640:480"` is rejected **even though it is 4:3**. (One
exception: `fl2va` is not bound by that whitelist and takes any `"W:H"`.)

`patches/minimax-h3-target-width-height.patch` adds the second parameter group, and **matches the
customer's requirement that the two groups are mutually exclusive**. The design principle is refuse
rather than silently alter:

- both groups given → refused: `target accepts either width+height or short_edge+aspect_ratio, not both`
- only one of the pair → refused (the missing one reports `must be an integer`)
- not a multiple of 32 → refused, **not rounded**: `must be a positive multiple of 32, got 481`
- above the `768*1344 = 1032192` px area cap → refused, **not downscaled** (the ratio path does downscale)
- `min(w,h)` must be in the allowed short-edge list (`SGLANG_..._EXTRA_SHORT_EDGES` + 768)
- ratio outside 1:4–4:1 → rejected cleanly by the resolver, not a worker crash

The 11 boundary cases that were measured are in `RESULTS.md`. Positive example:
`target: {"width": 800, "height": 480}` (5:3, **not** one of the six released ratios) produces an mp4
that ffprobes as `800,480,124` + aac. Regression evidence: `640:480` is still rejected with the
original message, so the ratio path was not disturbed.

`h3gen.py` picks the most portable wire form automatically and prints which one it used (`wire=`), so
"I asked for 864x480 and the server quietly gave me something else" cannot happen:

| `wire` | wire form | requires |
|---|---|---|
| `ratio` | `short_edge` + one of the 6 released ratios | nothing; works on a stock image |
| `literal` | `short_edge` + an arbitrary `"W:H"` | `fl2va` only, also unpatched |
| `exact` | `target.width` + `target.height` | the patch in this section |

The selection compares **canvases, not ratios**, and the difference matters: `864x480` is ratio
**1.8, not 16:9's 1.7778** (it is `round32(480 × 16/9)`), so reducing the fraction never yields
`"16:9"`. The right question is "does 16:9 at short edge 480 land exactly on 864x480?".

---

## 2. g7e (RTX PRO 6000, 96 GB, no NVLink)

### 2.1 Neither parallel lever is available — this is a hard result

Emulating no-P2P with `NCCL_P2P_DISABLE=1` (forces staging through host memory):

| shape | NVLink | no P2P | penalty |
|---|---|---|---|
| TP1 x Ulysses8 | 10.05 s | **151.97 s** (repeat 151.94) | 15.1x |
| TP8 x Ulysses1 | 13.09 s | **248.66 s** (repeat 248.69) | **19.0x** |

Repeats agree to 0.01%, so this is a stable property of the topology, not warmup noise. **TP is
worse than Ulysses**: Ulysses exchanges activations twice per attention, TP all-reduces twice per
*layer*.

So on g7e the only viable shape is **one replica per GPU**, and per-GPU memory is the sole
constraint.

### 2.2 Recommended: one replica per GPU + CPU offload

This shape needs **two of the three** patches — the short-edge one for 480p and the cpu-offload one
without which every `*-cpu-offload` flag dies during warmup (section 2.3). The width/height one is
optional here (inert unless you send `target.width`). `serve.sh` applies all three, so the
single-GPU shape is:

```bash
OFFLOAD=1 GPUS=1 ULYSSES=1 ./serve.sh
```

For all 8 GPUs as 8 independent replicas, `launch_replicas.sh 1` does the fan-out; the underlying
per-replica command, inside a container where the patches are already applied, is:

```bash
# one per GPU, n = 0..7; space ports by at least 2
CUDA_VISIBLE_DEVICES=$n SGLANG_MINIMAX_H3_EXTRA_SHORT_EDGES=480 sglang serve \
  --model-path /models/MiniMax-H3 --model-variant fl2va \
  --num-gpus 1 --ulysses-degree 1 \
  --text-encoder-cpu-offload --vae-cpu-offload \
  --performance-mode speed \
  --warmup-resolutions 864x480 \
  --host 0.0.0.0 --port $((30010 + 2*n))
```

Note `--encoder-parallel` is absent on purpose: a 1-GPU replica has nothing to fold over, so the
flag would have no effect (section 1.5). The warmup list is deliberately **480p only** here, unlike
the H200 default: the 79.4 GiB fit below was measured at 480p and `1344x768` is 2.49x the area.

**79.4 GiB per GPU, 66.22 s per request, ~7.25 videos/min across 8 GPUs.**

Without offload it is 132.2 GiB, which **does not fit a 96 GB card** (~95.6 GiB usable). Offload
drops it by **52.8 GiB (-40%)** for **+7.9%** latency (66.22 s vs 61.38 s) — the encoder and VAEs
run once per request, not once per step.

**Verified, not extrapolated:** holding 45,268 MiB of ballast on GPU 0 to make it behave like a
96 GB card, the offloaded single-GPU server loads, warms up and generates with zero OOM at the
same 66.24 s.

### 2.3 Required patch for g7e

`patches/minimax-h3-cpu-offload-inplace.patch` (one line). Without it, any `*-cpu-offload` flag
fails during warmup:

```
RuntimeError: Inplace update to inference tensor outside InferenceMode is not allowed.
  ... minimax_h3/stages/decoding.py, line 92, in _reverse_normalize_latents_
```

`_reverse_normalize_latents_` does `latents.mul_(std).add_(mean)`; the latents come from
`denoise_loop.py:33 @torch.inference_mode()`, and the offload manager runs the stage under
`torch.inference_mode(False)` (`layerwise_offload.py:389`), where mutating an inference tensor is
illegal. Out-of-place fixes it and is a no-op in the non-offload path.

### 2.4 What to expect on g7e

| | H200 8 GPU | g7e 8 GPU |
|---|---|---|
| per-request latency | **10.05 s** | 66.22 s |
| aggregate throughput | 6.2 vid/min | ~7.25 vid/min |
| can adding GPUs cut latency? | yes (near-linear) | **no** |

Throughput is fine; **per-request latency is pinned at ~66 s and cannot be improved by adding
GPUs.** If the customer's 10 s target is firm, g7e cannot meet it except at very low step counts
(see the step tables in `RESULTS.md`). Set this expectation early.

---

## 3. Cost differences between the three tasks, and sizing for 1 QPS

### 3.1 ref2va costs 3.2x more per step (10 s clip, 864x480, 4 GPUs)

Per-step cost from multi-point sweeps (server-side `inference_time_s`; client wall adds 1.0–1.6 s):

| task | per step | fit and measured points |
|---|---|---|
| t2va | **1.02 s/step** | `2.05 + 1.02×steps` (four points at 12/20/25/50 steps, wall) |
| fl2va | **1.10 s/step** | `0.87 + 1.102×steps` (8/16/32 steps → 9.79 / 18.35 / 36.19 s) |
| ref2va | **3.48 s/step** | `3.52 + 3.482×steps` (8/16/32 steps → 31.15 / 59.56 / 114.83 s) |

**fl2va costs only ~8% more than t2va**, so the two can be capacity-planned together. **ref2va costs
3.16x more per step**, with all three points on the line (the 8-step point repeats at 31.14 / 31.16 s,
the 16-step at 59.07 / 59.10 s). **The slope difference proves this is not a one-off "encode the
reference video" cost but a per-step one** — a one-off cost would only raise the 3.52 s intercept.
The same multiple holds at a 5 s clip (ref2va with a 5 s reference at 16 steps = 22.02 s, ~3.2x t2va
at the same length). Do not extrapolate ref2va capacity from t2va numbers.

Also, **ref2va derives its output length from the reference material** — passing
`duration_seconds` in the request is rejected, so a shorter clip means a shorter reference.

### 3.2 How many boxes for 1 QPS

H3 **never batches** (`stop_reason=dynamic_disabled`, see trap 5 in section 4), so

```
QPS = replicas / per-request latency
```

and concurrency only queues. At a 10 s clip / 16 steps / 864x480:

| task | shape | per request | QPS per p5e | boxes for 1 QPS |
|---|---|---|---|---|
| t2va / fl2va | 2 replicas × 4 GPU | 19.11 s | 0.105 | **10** |
| t2va / fl2va | 1 replica × 8 GPU | 10.58 s | 0.095 | 11 |
| ref2va | 2 replicas × 4 GPU | 60.24 s | 0.033 | **30** |

Two points worth making to the customer:

- **4-GPU replicas have slightly higher throughput than 8-GPU ones** (0.105 vs 0.095 QPS/box),
  because Ulysses is 76% efficient at 8 GPUs. Use 8 GPUs for latency and 4 for throughput; at a
  1 QPS target, 4-GPU replicas need fewer boxes.
- **Step count is the only large lever**: at the same 1 QPS, dropping t2va from 16 to 8 steps takes
  latency from 19.11 s to ~10.6 s and halves the fleet. For quality vs steps see the SSIM table in
  `RESULTS.md` (40 steps 0.9682 / 20 steps 0.8691).

If ref2va is a small share of traffic, **mixing tasks on one box** (mode 3, unevenly split) is far
cheaper than sizing the whole fleet for the most expensive task: at, say, 10% ref2va traffic,
`GPUS_A=4 GPUS_B=4` serves both streams from one box with no separate ref2va fleet.

## 4. Traps common to both platforms

1. **Always pass `--warmup-resolutions`, covering every resolution served** — otherwise the first
   request at a cold shape pays ~10 s extra — **and then check the log that it was honoured**:
   `grep -o 'warmup req ([^)]*)'` on the server log. On image `c7c03ec53b` a server configured with
   `["864x480"]` warmed `1344x768x124f` instead (1.1). The flag takes raw `WxH` via `parse_size`,
   which is why `864x480` is accepted even unpatched, but acceptance is not proof of warming.
2. **`--base-gpu-id` does not work for multiple replicas.** It appears in `server_args`, but
   replica 1's ranks still land on GPUs 0-3, collide with replica 0, and both OOM. Isolate with
   `CUDA_VISIBLE_DEVICES`.
3. **Space replica ports by at least 2.** A server binds `127.0.0.1:<port+1>` alongside
   `0.0.0.0:<port>`, so spacing of 1 fails with `[Errno 98] address already in use` *after* the
   weights have loaded. Also give each its own `--master-port` / `--scheduler-port`.
4. **`pkill -f sglang` inside the container kills the launcher** (it matches the `docker exec`
   shell's own command line, exit 137). Use the pattern `[s]glang`.
5. **Dynamic batching is permanently off for H3**, and `--batching-max-size 4` is accepted but
   does nothing: `base.py:405` returns True only for `T2I`/`T2V`, while `minimax_h3.py:48`
   declares `TI2V` with no override. Logs show `stop_reason=dynamic_disabled`. **Plan capacity in
   replicas** — concurrent requests queue, they never merge.
6. **Cache-DiT is image-dependent.** On `c7c03ec53b` the cookbook's own manual recipe registers and
   then skips **zero** blocks (output mp4 byte-identical); on
   `nightly-dev-20260818-c0b6474b` it works and is worth **1.94–2.40×** (`RESULTS_QUANT.md`).
   Always read back the `cache-dit enabled on transformer (... rdt=...)` line — it is printed
   during warmup, not at request time.
7. **On NVSwitch boxes, check Fabric Manager after any host reboot** (next section) or CUDA will
   not initialize at all.

## 5. Fabric Manager recovery (p5e and other NVSwitch boxes)

After a host reboot, if the server dies with `Error 802: system not yet initialized` /
`cudaGetDeviceCount()` failing, check `systemctl status nvidia-fabricmanager`. The usual cause is
a version mismatch against the driver:

```
fabric manager NVIDIA GPU driver interface version 610.57.04
  don't match with driver version 595.71.05
```

**FM must match the driver exactly.** Fix (substitute the version `nvidia-smi` reports):

```bash
# note: nvidia-fabricmanager-595 is a virtual package and will not install; pin the full version
sudo apt-get install -y --allow-downgrades --allow-change-held-packages \
  nvidia-fabricmanager=595.71.05-1ubuntu1
sudo nvidia-smi -r                              # reset all GPUs and NVSwitches
sudo systemctl restart nvidia-fabricmanager     # FM must be restarted after the reset
nvidia-smi -q | grep -A2 "Fabric"               # want State: Completed / Status: Success
```

**Order matters**: installing the package without the reset leaves `Fabric State: In Progress` and
CUDA still broken. Verify with:

```bash
python3 -c "import torch; print(torch.cuda.device_count(), torch.cuda.can_device_access_peer(0,1))"
# expect: 8 True
```

`can_device_access_peer` matters especially for H3 — it is exactly what `auto`'s encoder-folding
decision tests. If P2P is down, `auto` silently falls back to `replicate` and wastes 39.7 GiB per
GPU. In that situation pass `--encoder-parallel fold` explicitly to force it.

## 6. Dead ends — do not spend time here

All confirmed by measurement or by reading the code:

- **Splitting encoder/VAEs onto dedicated GPUs.** `minimax_h3_pipeline.py:94
  validate_disagg_role()` raises for any non-`MONOLITHIC` role, closing `--disagg-role` and
  `--encoder-urls`. `--srt-encoder-url` is wired only for GLM-Image. Enabling either needs
  upstream changes — see `SRT_ENCODER_PR_ASSESSMENT.md`. And most of the benefit is already
  captured by default encoder folding.
- **Using concurrency for throughput.** See trap 5; H3 does not batch.
- **`quality: "high"`.** Its gate
  (`release_metadata.py::_MINIMAX_H3_QUALITY_WORKLOAD`) pins 1344x768 / 50 steps and rejects any
  change of resolution or step count.
- **`--vae-config.parallel-decode-mode spatial` / `spatial_shard`.** H3 rejects both.
- **`--use-fsdp-inference`.** Shards only the DiT, which the TP lever above already does.

## 7. Verification commands

```bash
# after startup, confirm component sizes and the folding decision
grep -E "Loaded (text_encoder|transformer|video_vae|audio_vae):" serve.log
grep -i "encoder parallel folding" serve.log

# per-GPU high-water mark
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

# submit one 480p / 40-step request and time it
python3 h3req.py 480 40 5 check
```

`serve_topo.sh <gpus> <tp> <ulysses> <encoder_parallel> <logname>` does launch + wait-for-ready +
print per-GPU memory + print component sizes in one command; every topology number here came
from it.
