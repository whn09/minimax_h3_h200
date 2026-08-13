# MiniMax-H3 on H200 (p5e.48xlarge) — the customer deployment

The customer runs H3 on `p5e.48xlarge` (8xH200) via SGLang, insists on **480P**, and wants
"10 s" (whether that means 10 s of latency or a 10 s clip was still being confirmed with them —
both readings are measured here, so no re-run is needed either way).

Two deliverables:

1. **`patches/minimax-h3-short-edge.patch`** — lets SGLang accept a non-768 short edge.
   Three hunks, opt-in behind `SGLANG_MINIMAX_H3_EXTRA_SHORT_EDGES`; unset, the released
   behaviour and both error strings are byte-for-byte unchanged. Applies cleanly to
   `lmsysorg/sglang:dev` @ `c7c03ec53b`:

   ```
   git apply -p1 minimax-h3-short-edge.patch   # from /sgl-workspace/sglang
   ```

   Validated on hardware: `short_edge: 480` + `16:9` really serves **864x480 x 124 frames**.

2. **`patches/minimax-h3-cpu-offload-inplace.patch`** — one-line fix: unpatched, any
   `*-cpu-offload` flag makes H3 die during warmup at `decoding.py:92`
   (`Inplace update to inference tensor outside InferenceMode`). Needed to enable CPU offload,
   which is the only way to fit H3 on a 96 GB card (RTX PRO 6000 / g7e).

3. **`DEPLOYMENT_GUIDE.md`** (and `DEPLOYMENT_GUIDE_zh.md`) — **start here for the customer.** The
   two-platform best practice: recommended commands for H200 and for g7e, a pick-a-shape-by-
   objective table, the `--tp-size` memory lever, encoder folding, the shared trap list, the
   Fabric Manager recovery recipe, and the dead ends.

4. **`RESULTS.md`** (and `RESULTS_zh.md`) — the measured latency/quality tables at 4 and 8 GPUs,
   the full replica sweep (1x8 / 2x4 / 4x2 / 8x1), the TP/encoder-parallel sweep, what Ulysses and
   TP cost without NVLink, the verified 96 GB verdict, the two negative results (Cache-DiT is a
   no-op on H3; `quality: "high"` is gated on 1344x768 *and* 50 steps), and the recommendation.

`launch_replicas.sh <gpus_per_replica>` splits the 8 GPUs into replicas (and encodes three launch
traps); `serve_topo.sh <gpus> <tp> <ulysses> <encoder_parallel> <logname>` launches one topology
and reports per-GPU memory plus component sizes; `conc_multi.py <N> <steps> <ports>` is the
concurrency probe; `ssim_pairs.sh` scores the topology outputs against each other.

Two memory levers, in order of preference:

- **On a P2P-capable box, use `--tp-size`.** It shards the 61.73 GB DiT, which Ulysses does not
  touch: 95.9 -> 39.0 GiB per GPU for +30% latency, or -32 GiB for +10% at `--tp-size 2`. At TP4
  or above the 480p config fits an 80 GB card.
- **On a no-NVLink box, neither Ulysses nor TP is usable** (15x and 19x slower respectively with
  P2P disabled). Run one replica per GPU with `--num-gpus 1 --ulysses-degree 1
  --text-encoder-cpu-offload --vae-cpu-offload` (79.4 GiB per GPU, ~66 s per video, verified to fit
  under a 96 GB cap).

Headline: **8 GPUs, `--ulysses-degree 8`, 864x480, 40 steps -> 10.05 s.** 4 -> 8 GPUs scales
near-linearly (1.9x) for +9% GPU-seconds, which is what buys back the step budget.

Serve it with (weights dir must be *named* `MiniMax-H3` if local — the registry matches the
`--model-path` basename, not `--model-id`):

```
SGLANG_MINIMAX_H3_EXTRA_SHORT_EDGES=480 sglang serve \
  --model-path MiniMaxAI/MiniMax-H3 --model-variant fl2va \
  --num-gpus 8 --ulysses-degree 8 --performance-mode speed \
  --warmup-resolutions 1344x768 864x480 \
  --host 0.0.0.0 --port 30010
```

`--warmup-resolutions` must list every resolution served; without it the first request at a cold
shape pays about 10 s. It takes raw `WxH` and bypasses the canonical short-edge validator, so
`864x480` warms up even on an unpatched server.

Contents: `h3req.py` (submitter that *polls* to completion — the model card's own scripts do a
single status GET and download a truncated file), `runs/` raw artifacts, `videos_named/` the same
videos under readable names, `logs/` server logs.

The 4xH100 baseline this is compared against is `../h3_h100_baseline/`; the Trainium port is
`../h3_plugin_src/`.
