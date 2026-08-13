# MiniMax-H3 deployment guide: H200 and g7e (RTX PRO 6000)

Customer-facing deployment recommendations. Every number was measured on `p5e.48xlarge` (8xH200)
at a fixed **864x480 / 124 frames / 40 steps / t2va / seed 1101**, timed client-side (POST to
`status: completed`). Full experimental record in `RESULTS.md`, patches in `patches/`.

Headline: **H200 has two cheap parallel levers (Ulysses and TP); g7e has neither.** The optimal
shape differs completely between the two platforms — do not port the H200 command to g7e.

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

The easiest route is the wrapper in this directory, which mounts the patch, applies it
idempotently, starts the server detached and waits for `/health`:

```bash
cd h3_h200_baseline
./serve.sh                      # <- the recommended H200 config; see below
GPUS=4 ./serve.sh               # the cookbook's 4xH200 recipe
TP=2 ULYSSES=4 ./serve.sh       # shard the DiT: 63.9 GiB/GPU instead of 95.9
SHORT_EDGES= ./serve.sh         # released 768-only policy, patch stays inert
./serve.sh status | logs | stop
```

**Bare `./serve.sh`, with no arguments and no env vars, is the recommended configuration**:
8 GPUs, TP=1, Ulysses=8, `encoder-parallel auto`, 480p enabled and warmed, nothing else warmed —
the measured 10.05 s / 6.2 vid/min / 95.9 GiB-per-GPU shape. It creates a long-lived
`sleep infinity` container and patches the source *inside* it, so restarting the server with
different flags neither re-pulls the image nor re-applies the patch. `stop` kills only the server
process and keeps the container, so the patched source survives.

**Both** patches in `patches/` are applied: the short-edge one enables 480p, and the cpu-offload
one is required whenever `OFFLOAD=1` (section 2.3). Applying both unconditionally is safe — the
offload patch is a no-op when no offload flag is passed.

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
  -v /opt/dlami/nvme/h3-fl2va:/models/MiniMax-H3:ro \
  -v $PWD/patches/minimax-h3-short-edge.patch:/patch.patch:ro \
  -v /opt/dlami/nvme/out:/out \
  lmsysorg/sglang:dev bash -lc '
    cd /sgl-workspace/sglang && git apply -p1 /patch.patch &&
    SGLANG_MINIMAX_H3_EXTRA_SHORT_EDGES=480 sglang serve \
      --model-path /models/MiniMax-H3 --model-variant fl2va \
      --num-gpus 8 --ulysses-degree 8 \
      --performance-mode speed \
      --warmup-resolutions 864x480 \
      --host 0.0.0.0 --port 30010 > /out/serve.log 2>&1'
```

**10.05 s per request, 6.2 videos/min, 95.9 GiB per GPU.**

`git apply` is **not idempotent** — running it twice fails. This one-shot form is therefore only
suitable for a container you throw away; for anything you restart repeatedly, use `serve.sh`,
which distinguishes "not yet applied" from "already applied" with `--check` and `-R --check`.

Verify the patch is actually live — do **not** use "warmup succeeded" as evidence, because
`--warmup-resolutions` takes raw `WxH` and bypasses the short-edge validator even unpatched:

```bash
docker exec h3 git -C /sgl-workspace/sglang diff --stat
docker exec h3 git -C /sgl-workspace/sglang log --oneline -1   # expect c7c03ec53b
```

**The patch is diffed against image commit `c7c03ec53b`.** If you pull a newer `:dev` it may not
apply; `serve.sh` fails loudly with `PATCH_DOES_NOT_APPLY -- image moved off c7c03ec53b` rather
than serving half-patched, and the `&&` in the command above short-circuits the same way. Re-diff
before trusting it on a newer image.

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

This shape needs **both** patches — the short-edge one for 480p and the cpu-offload one without
which every `*-cpu-offload` flag dies during warmup (section 2.3). `serve.sh` applies both, so the
single-GPU shape is:

```bash
OFFLOAD=1 GPUS=1 ULYSSES=1 ./serve.sh
```

For all 8 GPUs as 8 independent replicas, `launch_replicas.sh 1` does the fan-out; the underlying
per-replica command, inside a container where both patches are already applied, is:

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
flag would have no effect (section 1.5).

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

## 3. Traps common to both platforms

1. **Always pass `--warmup-resolutions`, covering every resolution served.** Otherwise the first
   request pays ~10 s extra. It takes raw `WxH`, so `864x480` works even unpatched.
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
6. **Cache-DiT is a no-op on H3.** The cookbook's own manual recipe registers and then skips
   **zero** blocks; the output mp4 is byte-identical.
7. **On NVSwitch boxes, check Fabric Manager after any host reboot** (next section) or CUDA will
   not initialize at all.

## 4. Fabric Manager recovery (p5e and other NVSwitch boxes)

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

## 5. Dead ends — do not spend time here

All confirmed by measurement or by reading the code:

- **Splitting encoder/VAEs onto dedicated GPUs.** `minimax_h3_pipeline.py:94
  validate_disagg_role()` raises for any non-`MONOLITHIC` role, closing `--disagg-role` and
  `--encoder-urls`. `--srt-encoder-url` is wired only for GLM-Image. Enabling either needs
  upstream changes — see `SRT_ENCODER_PR_ASSESSMENT.md`. And most of the benefit is already
  captured by default encoder folding.
- **Using concurrency for throughput.** See trap 5; H3 does not batch.
- **Cache-DiT / `quality: "high"`.** The former is a no-op; the latter's gate
  (`release_metadata.py::_MINIMAX_H3_QUALITY_WORKLOAD`) pins 1344x768 / 50 steps and rejects any
  change of resolution or step count.
- **`--vae-config.parallel-decode-mode spatial` / `spatial_shard`.** H3 rejects both.
- **`--use-fsdp-inference`.** Shards only the DiT, which the TP lever above already does.

## 6. Verification commands

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
