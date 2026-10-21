# Stock SGLang 768p steady-state baseline (8×H200) — for the Trainium2 comparison

Reference config, no patches / quantization / cache: `lmsysorg/sglang:dev`, 8 GPUs, Ulysses=8,
`--model-variant fl2va --performance-mode speed`. Request: t2va, 1344×768, **124 frames** (5.17 s @
24 fps), **50 steps**, flow_shift 12 / audio_flow_shift 3.

| file | what |
|---|---|
| `serve_sglang.sh` | the reference `docker run ... sglang serve` (only the HF cache path is adapted) |
| `bench_sglang.sh` | (1) reference `bench_serving` cmd with N vbench prompts, (2) reference curl payload N times |
| `req_768p.py` | the reference curl payload, polled to completion, N back-to-back runs |
| `serve_vomni.sh` / `req_vomni.sh` | vllm-omni server (8-GPU) and the reference `/v1/videos/sync` request ×N |
| `bench_matrix.sh` / `req_matrix.py` / `summarize_matrix.py` | sglang arms × tasks with per-stage breakdown |
| `bench_vomni_versions.sh` / `req_vomni_async.py` | vllm-omni images side by side; per-component `stage_durations` |
| `Dockerfile.vomni029` | `v0.29.0rc1` image made startable (matching vllm 0.29.0) |
| `Dockerfile.nccl` | swap only the NCCL wheel in a vllm-omni image (NCCL A/B) |
| `h3_diffusers_bench.py` / `run_diffusers.sh` | diffusers-native H3 under torchrun, TP×Ulysses, per-stage timing |
| `results_20261021_034117/` | the run below (one mp4 kept; all 5 fixed-prompt mp4s are md5-identical) |

## Result (2026-10-21, p5en.48xlarge, driver 595.91.07, sglang `c1c287a42`, image `sha256:b0473834…`)

All numbers after warmup (server startup warmup of 2 steps + one 50-step warmup request).

| measurement | n | mean | min | max | stdev |
|---|---|---|---|---|---|
| `bench_serving` latency (vbench prompts) | 5 | **42.08 s** | 42.08 | 42.08 | <0.01 |
| fixed prompt (seed 1101) client wall | 5 | **41.81 s** | 41.80 | 41.82 | 0.008 |
| fixed prompt server `inference_time_s` | 5 | 41.67 s | 41.62 | 41.73 | 0.043 |
| DenoisingStage (all 11 requests) | 11 | 39.63 s → **0.793 s/step** | 39.03 | 39.98 | |
| DecodingStage | 11 | 1.03 s | 0.92 | 1.10 | |

- Peak memory 82,460 MB per GPU (of 143,771).
- Output: 1344×768, 124 video frames, 163 audio frames; the fixed prompt is bit-identical across runs.
- Denoise drifts slowly upward over back-to-back requests (39.03 → 39.98 s, +2.4% over ~7 min), so
  quote the mean, not the first run.

Reproduce: `./serve_sglang.sh` → wait for `/health` 200 (~3 min) → `N=5 ./bench_sglang.sh`.

## vllm-omni on the same box (`vllm/vllm-omni:minimax-h3`, `sha256:e930db8e…`, vllm-omni `310b4b477`)

The reference 4-GPU command with every 4 raised to 8 (`--num-gpus 8 --usp 8 --ring 1
--vae-patch-parallel-size 8 --vae-parallel-mode tile --vae-use-tiling`); regional `torch.compile` of
the DiT blocks and FlashAttention are the image defaults. Request = the reference `/v1/videos/sync`
curl at 1344×768, 5 s, 50 steps, seed 1101 (its own prompt, not the sglang one). Output checked:
H.264 1344×768 124 frames + AAC 32 kHz stereo 163 frames, decodes clean.

| config | n | client wall mean | min–max | stdev |
|---|---|---|---|---|
| U8 + VPP8 tile (reference flags) | 5 | **49.38 s** | 49.04–49.59 | 0.196 |
| + `--text-encoder-tp-size 8` + pipeline profiler | 2 | **49.09 s** | 48.97–49.22 | 0.123 |

The first request is 82–83 s (regional compile) and is excluded. Breakdown from
`--enable-diffusion-pipeline-profiler` (`vomni_u8_tetp8_prof/breakdown.txt`):

| component | time |
|---|---|
| text encode (Qwen3-VL, TP8) | 0.06 s |
| DiT denoise, 50 steps | 43.35 s (0.867 s/step) |
| VAE decode (tile-parallel 8) | 1.17 s |
| engine overhead | 1.67 s |
| MP4 encode in the HTTP layer | 2.88 s |
| **client total** | **49.2 s** |

vs sglang (41.81 s): the DiT is 9% slower per step (0.867 vs 0.793 s) and vllm-omni pays 2.9 s
of MP4 encoding after the engine returns. Text-encoder TP does not change latency, but flattens
memory: rank 0 drops from 121 GiB (encoder resident there) to 80 GiB, same as every other rank.
Note the two runs are bit-stable within a config but TE-TP8 changes the output md5 (likely the sharded encoder's
reductions rounding differently), and the first (compile) request's output differs from the steady-state ones.

Reproduce: `./serve_vomni.sh` (or `EXTRA="--text-encoder-tp-size 8 --enable-diffusion-pipeline-profiler"
./serve_vomni.sh`) → `/health` 200 (~90 s) → `N=5 WARM=1 ./req_vomni.sh`.

## T2VA per-component breakdown + vllm-omni versions (2026-10-21)

Same box, same 8-GPU shape, T2VA only, all after warmup. sglang rows: `bench_matrix.sh` with
`ARMS="exact lossless"` (stage times from `SGLANG_DIFFUSION_STAGE_LOGGING=1` +
`SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=1`, raw rows in `matrix_lossless/*/requests.jsonl`). vllm-omni
rows: the reference flags scaled to 8 GPUs + `--enable-diffusion-pipeline-profiler`; breakdown is the
job's `stage_durations` from the async API (`req_vomni_async.py`; nightly no longer logs the
profiler lines), end-to-end is the sync client wall (`req_vomni.sh`, includes MP4 encoding).

| stack / version | E2E | text enc | DiT, 50 steps | s/step | VAE decode | MP4 encode |
|---|---|---|---|---|---|---|
| sglang `dev` (`c1c287a42`), `quality: exact` | **41.38 s** (inference_time_s) | ~0.002 s* | 40.32 s | 0.806 | 1.04 s | in decode |
| sglang `dev`, `quality: lossless` + `--warmup-resolutions 1344x768` | **41.11 s** | ~0.002 s* | 40.15 s | 0.803 | 0.93 s | in decode |
| vllm-omni nightly (`0.32.0rc2.dev5+g7b316e3ae`, vllm 0.32.0) | **44.22 s** | 0.043 s | 42.39 s | 0.848 | 0.81 s | 0.37 s |
| vllm-omni latest (`v0.31.0rc1`, vllm 0.31.0) | **44.01 s** | 0.043 s | 42.41 s | 0.848 | 0.83 s | not logged |
| vllm-omni `v0.29.0rc1` + vllm 0.29.0 (the Trn2 major version; `Dockerfile.vomni029`) | **42.96 s** | 0.044 s | 41.51 s | 0.830 | 0.79 s | 0.38 s |
| vllm-omni `minimax-h3` (`0.1.dev2381+g310b4b477`, 2026-08-02, vllm 0.26.0) | 49.38 s | 0.06 s (TP8) | 43.35 s | 0.867 | 1.17 s | 2.88 s |

\* sglang caches the prompt embedding: the first request with a prompt costs 0.08–0.14 s, repeats
of the same prompt 1.7 ms. Either way it is <0.4% of the request.

- The DiT is ≥96% of every request on both stacks; encoder + decoder is ~1 s.
- `lossless` vs `exact` is worth 0.65% (mostly VAE decode 1.04 → 0.93 s); both are bit-stable
  run to run (one md5 per cell).
- The new vllm-omni is 10.4% faster than the `minimax-h3` image, but 2.5 s of that is MP4
  encoding (2.88 → 0.37 s); its DiT is only 2% faster. It still trails sglang by 7.6%, almost all
  in the DiT (0.848 vs 0.803 s/step).
- Across vllm-omni releases the steady state is close: 0.29 / 0.31 / nightly = 42.96 / 44.01 /
  44.22 s. The 0.29 lead is **real, not restart drift**: alternating 0.29 and 0.31 over two more
  restarts each (`ab_029_031.sh`, `vomni_*_ab{1,2}/`) gives 0.29 = 42.84 / 42.90 / 42.96 s
  (DiT 41.29–41.51 s) vs 0.31 = 43.79 / 43.96 / 44.01 s (DiT 42.41–42.43 s), ranges disjoint —
  0.31 is ~1.0 s (2.4%) slower, all in the DiT. Output is the same trajectory (0.29 vs nightly SSIM
  0.951, closer than 0.31 vs nightly 0.929; identical composition), so 0.29 is not skipping work.
  The H3 block code (incl. the fused modulation kernels) is identical; torch / triton / cuDNN / FA3
  are identical; NCCL is **ruled out**: swapping only the NCCL wheel (`Dockerfile.nccl`; the library torch loads
  at runtime is the pip `nvidia-nccl-cu13` one — `torch.cuda.nccl.version()` reports the compile-time
  2.29.7 everywhere) gives 0.31 + NCCL 2.29.7 = DiT 41.94 s and 0.29 + NCCL 2.30.7 = DiT 40.95 s
  (`vomni_*nccl*/`): both ~1% faster than their baselines (an environment drift, run 30 min later),
  and the 0.29–0.31 gap stays 0.99 s. The gap follows the vllm-omni/vllm version, i.e. code changed
  between 0.29 and 0.31 (attention backends, `attention/layer.py`, `hooks/sequence_parallel.py`,
  `packed_sequence.py` all differ) — not bisected further. All vllm-omni versions
  are 4–7% behind sglang's 41.11–41.38 s.
- The stock `vllm/vllm-omni:v0.29.0rc1` image does not start: it ships vllm 0.28.0 while its
  vllm-omni CLI imports `vllm.entrypoints.launchers` (vllm 0.29), so `vllm serve` dies at import
  (`vomni_v0_29_0rc1_stock_broken/`). `Dockerfile.vomni029` adds only vllm 0.29.0 (torch 2.13.0
  unchanged) plus `FLASHINFER_DISABLE_VERSION_CHECK=1`, because vllm 0.29.0 lifts flashinfer-python
  to 0.6.18 and there is no matching flashinfer-cubin; H3's DiT attention is FlashAttention. PyPI has
  no vllm-omni 0.29.0 final.
- Two nightly bring-up traps (handled in the scripts): pass the `FL2VA/` subdirectory, not the
  repo id — the official repo root carries `modular_model_index.json`, which nightly takes as a
  FastH3 release and then dies looking for `fastvideo_inference.json`; and t2va requests must now
  carry `aspect_ratio=16:9`.

Reproduce: sglang `ARMS="exact lossless" PARTS=fl2va ./bench_matrix.sh`; vllm-omni
`TAGS="nightly latest h3-vomni:0.29.0rc1-vllm0.29.0" ./bench_vomni_versions.sh`.

## diffusers native (2026-10-21, same box, T2VA only)

`h3_diffusers_bench.py` under `torchrun --nproc_per_node=8` (`run_diffusers.sh`), in `lmsysorg/sglang:dev`
(torch 2.14.1+cu130, transformers 5.19) with diffusers installed `--no-deps -e` from PR #14609 head
`bc35ec2d5` with PR #14725 (`ce1581f0f`, TP+CP in one `ParallelConfig`) merged in — the one source
conflict in `modeling_utils.py` keeps both sides (`diffusers_out/combo_conflict_resolution.patch`). U8 alone
is diffusers main's `_cp_plan` path; TP8 is #14609's `_tp_plan`; TP2×U4 needs both PRs. Text encoder on
rank 0 only (output state broadcast), video + audio VAE run once, natively, on rank 1 (no tile
parallelism), default attention backend (SDPA). Same prompt/seed as the sglang run; 1 warmup + 2 runs.

| config | E2E | text enc | DiT, 50 steps | s/step | video VAE | audio VAE | MP4 mux | peak rank 0 |
|---|---|---|---|---|---|---|---|---|
| Ulysses 8 (main `_cp_plan`) | **63.08 s** | 0.059 s | 50.60 s | 1.012 | 7.80 s | 0.064 s | 4.56 s | 126.0 GiB |
| TP2 × U4 (#14609 + #14725) | 65.20 s | 0.057 s | 52.76 s | 1.055 | 7.81 s | 0.063 s | 4.50 s | 94.8 GiB |
| TP8 (#14609) | 74.02 s | 0.056 s | 61.52 s | 1.230 | 7.83 s | 0.064 s | 4.54 s | 73.1 GiB |

- diffusers' best (U8, 63.08 s) is 1.53× sglang (41.11 s): the DiT is 1.26× slower per step (1.012 vs
  0.803 s) and the native VAE decode + mux cost 12.4 s that sglang/vllm-omni spend ~1 s on.
- On 8×H200 with NVLink, **TP does not help latency**: each TP step adds per-block all-reduces, so U8 >
  TP2×U4 > TP8. What #14609's TP buys is memory: rank-0 peak 126 → 95 → 73 GiB (the 62 GB DiT is
  replicated under pure Ulysses). That is the lever on memory-bound hardware (Trn2 cores), not here.
- `_flash_3_hub` could not be tried: the hub's `kernels-community/flash-attn3` has no torch 2.14 build
  (`diffusers_out/tp1_u8_fa3hub_failed.log`).

<details><summary>中文</summary>

# 官方 SGLang 768p 稳态基线（8×H200）——用于和 Trainium2 对比

参考配置，无 patch / 量化 / cache：`lmsysorg/sglang:dev`，8 卡 Ulysses=8，
`--model-variant fl2va --performance-mode speed`。请求：t2va，1344×768，**124 帧**（24fps 5.17 s），
**50 步**，flow_shift 12 / audio_flow_shift 3。

| 文件 | 作用 |
|---|---|
| `serve_sglang.sh` | 参考的 `docker run ... sglang serve`（只改了 HF cache 路径） |
| `bench_sglang.sh` | (1) 参考 `bench_serving` 命令跑 N 个 vbench prompt；(2) 参考 curl 请求跑 N 次 |
| `req_768p.py` | 参考 curl 请求体，轮询到完成，连续跑 N 次 |
| `serve_vomni.sh` / `req_vomni.sh` | vllm-omni 8 卡 server 与参考 `/v1/videos/sync` 请求 ×N |
| `bench_matrix.sh` / `req_matrix.py` / `summarize_matrix.py` | sglang 各档 × 场景，带分阶段耗时 |
| `bench_vomni_versions.sh` / `req_vomni_async.py` | vllm-omni 各镜像对比；分组件 `stage_durations` |
| `Dockerfile.vomni029` | 让 `v0.29.0rc1` 镜像能起（补对应的 vllm 0.29.0） |
| `Dockerfile.nccl` | 只换 vllm-omni 镜像里的 NCCL wheel（NCCL A/B） |
| `h3_diffusers_bench.py` / `run_diffusers.sh` | diffusers 原生 H3（torchrun，TP×Ulysses，分阶段计时） |
| `results_20261021_034117/` | 下面这轮的原始结果（只留一份 mp4；5 次固定 prompt 输出 md5 完全相同） |

## 结果（2026-10-21，p5en.48xlarge，驱动 595.91.07，sglang `c1c287a42`，镜像 `sha256:b0473834…`）

全部是 warmup 之后的数（server 启动时自带 2 步 warmup + 一个 50 步 warmup 请求）。

| 测量 | 次数 | 均值 | 最小 | 最大 | 标准差 |
|---|---|---|---|---|---|
| `bench_serving` 延迟（vbench prompt） | 5 | **42.08 s** | 42.08 | 42.08 | <0.01 |
| 固定 prompt（seed 1101）客户端墙钟 | 5 | **41.81 s** | 41.80 | 41.82 | 0.008 |
| 固定 prompt 服务端 `inference_time_s` | 5 | 41.67 s | 41.62 | 41.73 | 0.043 |
| DenoisingStage（全部 11 个请求） | 11 | 39.63 s → **0.793 s/step** | 39.03 | 39.98 | |
| DecodingStage | 11 | 1.03 s | 0.92 | 1.10 | |

- 每卡峰值显存 82,460 MB（共 143,771）。
- 输出 1344×768、124 视频帧、163 音频帧；固定 prompt 多次运行逐位相同。
- 连续请求下 denoise 缓慢变慢（39.03 → 39.98 s，约 7 分钟内 +2.4%），报数用均值而不是第一次。

复现：`./serve_sglang.sh` → 等 `/health` 返回 200（约 3 分钟）→ `N=5 ./bench_sglang.sh`。

## 同一台机器上的 vllm-omni（`vllm/vllm-omni:minimax-h3`，`sha256:e930db8e…`，vllm-omni `310b4b477`）

参考的 4 卡命令把 4 全换成 8（`--num-gpus 8 --usp 8 --ring 1 --vae-patch-parallel-size 8
--vae-parallel-mode tile --vae-use-tiling`）；DiT block 的 regional `torch.compile` 和 FlashAttention
是镜像默认。请求 = 参考的 `/v1/videos/sync` curl，1344×768、5 s、50 步、seed 1101（用它自己的 prompt，
不是 sglang 那个）。输出已校验：H.264 1344×768 124 帧 + AAC 32 kHz 双声道 163 帧，能完整解码。

| 配置 | 次数 | 客户端墙钟均值 | 最小–最大 | 标准差 |
|---|---|---|---|---|
| U8 + VPP8 tile（参考 flag） | 5 | **49.38 s** | 49.04–49.59 | 0.196 |
| + `--text-encoder-tp-size 8` + pipeline profiler | 2 | **49.09 s** | 48.97–49.22 | 0.123 |

第一个请求 82–83 s（regional compile），不计入。`--enable-diffusion-pipeline-profiler` 的拆分
（`vomni_u8_tetp8_prof/breakdown.txt`）：

| 组件 | 耗时 |
|---|---|
| 文本编码（Qwen3-VL，TP8） | 0.06 s |
| DiT 去噪 50 步 | 43.35 s（0.867 s/step） |
| VAE 解码（tile 并行 8） | 1.17 s |
| 引擎开销 | 1.67 s |
| HTTP 层 MP4 编码 | 2.88 s |
| **客户端合计** | **49.2 s** |

对比 sglang（41.81 s）：DiT 每步慢 9%（0.867 vs 0.793 s），且 vllm-omni 在引擎返回后还要付 2.9 s
MP4 编码。text encoder TP 不改变延迟，但把显存摊平：rank 0 从 121 GiB（encoder 常驻在它上面）降到
80 GiB，与其他 rank 相同。同配置内输出逐位稳定，但 TE-TP8 会改变 md5（推测是 encoder 分片后归约的舍入不同），
第一个（compile）请求的输出也与稳态不同。

复现：`./serve_vomni.sh`（或 `EXTRA="--text-encoder-tp-size 8 --enable-diffusion-pipeline-profiler"
./serve_vomni.sh`）→ `/health` 200（约 90 s）→ `N=5 WARM=1 ./req_vomni.sh`。

## T2VA 分组件耗时 + vllm-omni 各版本（2026-10-21）

同一台机器、同样 8 卡、只测 T2VA，全部是 warmup 之后。sglang 行：`bench_matrix.sh`
`ARMS="exact lossless"`（阶段时间来自 `SGLANG_DIFFUSION_STAGE_LOGGING=1` +
`SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=1`，原始数据在 `matrix_lossless/*/requests.jsonl`）。vllm-omni
行：参考 flag 扩到 8 卡 + `--enable-diffusion-pipeline-profiler`；拆分取异步接口返回的
`stage_durations`（`req_vomni_async.py`；nightly 不再把 profiler 打进日志），端到端是同步接口的客户端
墙钟（`req_vomni.sh`，含 MP4 编码）。

| 栈 / 版本 | 端到端 | text enc | DiT 50 步 | s/step | VAE decode | MP4 编码 |
|---|---|---|---|---|---|---|
| sglang `dev`（`c1c287a42`），`quality: exact` | **41.38 s**（inference_time_s） | ~0.002 s* | 40.32 s | 0.806 | 1.04 s | 含在 decode |
| sglang `dev`，`quality: lossless` + `--warmup-resolutions 1344x768` | **41.11 s** | ~0.002 s* | 40.15 s | 0.803 | 0.93 s | 含在 decode |
| vllm-omni nightly（`0.32.0rc2.dev5+g7b316e3ae`，vllm 0.32.0） | **44.22 s** | 0.043 s | 42.39 s | 0.848 | 0.81 s | 0.37 s |
| vllm-omni latest（`v0.31.0rc1`，vllm 0.31.0） | **44.01 s** | 0.043 s | 42.41 s | 0.848 | 0.83 s | 未打日志 |
| vllm-omni `v0.29.0rc1` + vllm 0.29.0（与 Trn2 同大版本；`Dockerfile.vomni029`） | **42.96 s** | 0.044 s | 41.51 s | 0.830 | 0.79 s | 0.38 s |
| vllm-omni `minimax-h3`（`0.1.dev2381+g310b4b477`，2026-08-02，vllm 0.26.0） | 49.38 s | 0.06 s（TP8） | 43.35 s | 0.867 | 1.17 s | 2.88 s |

\* sglang 会缓存 prompt embedding：某个 prompt 第一次请求 0.08–0.14 s，重复请求 1.7 ms。怎么算都不到
请求的 0.4%。

- 两个栈 DiT 都占每个请求的 96% 以上；encoder + decoder 合计约 1 s。
- `lossless` 比 `exact` 快 0.65%（主要是 VAE decode 1.04 → 0.93 s）；两档都逐次逐位稳定（每格一个 md5）。
- 新版 vllm-omni 比 `minimax-h3` 镜像快 10.4%，但其中 2.5 s 是 MP4 编码（2.88 → 0.37 s），DiT 只快 2%。
  仍比 sglang 慢 7.6%，几乎全在 DiT（0.848 vs 0.803 s/step）。
- vllm-omni 各版本稳态接近：0.29 / 0.31 / nightly = 42.96 / 44.01 / 44.22 s。0.29 领先**是真的，不是重启漂移**：
  0.29 与 0.31 各再交替重启两次（`ab_029_031.sh`，`vomni_*_ab{1,2}/`），0.29 = 42.84 / 42.90 / 42.96 s
  （DiT 41.29–41.51 s），0.31 = 43.79 / 43.96 / 44.01 s（DiT 42.41–42.43 s），区间不重叠 —— 0.31 稳定慢约 1.0 s
  （2.4%），全在 DiT。输出是同一条轨迹（0.29 vs nightly SSIM 0.951，比 0.31 vs nightly 的 0.929 还近；构图一致），
  0.29 没有少算。H3 block 代码（含融合 modulation kernel）相同；torch / triton / cuDNN / FA3 相同；唯一与 DiT 相关的
  NCCL **已排除**：只换 NCCL wheel（`Dockerfile.nccl`；torch 运行时加载的是 pip 的 `nvidia-nccl-cu13`，
  `torch.cuda.nccl.version()` 处处报编译时的 2.29.7，不能用来判断），0.31 + NCCL 2.29.7 = DiT 41.94 s，
  0.29 + NCCL 2.30.7 = DiT 40.95 s（`vomni_*nccl*/`）：两者都比各自基线快约 1%（晚 30 分钟的环境漂移），而 0.29 与 0.31
  的差距仍是 0.99 s。差距跟着 vllm-omni/vllm 版本走，即 0.29 → 0.31 之间的代码改动（attention 后端、
  `attention/layer.py`、`hooks/sequence_parallel.py`、`packed_sequence.py` 都有改）—— 未进一步二分。
  所有 vllm-omni 版本都比 sglang 的 41.11–41.38 s 慢 4–7%。
- 原版 `vllm/vllm-omni:v0.29.0rc1` 镜像起不来：里面是 vllm 0.28.0，而 vllm-omni CLI 要 import
  `vllm.entrypoints.launchers`（vllm 0.29 才有），`vllm serve` 在 import 时就死（`vomni_v0_29_0rc1_stock_broken/`）。
  `Dockerfile.vomni029` 只加装 vllm 0.29.0（torch 2.13.0 不动）并设 `FLASHINFER_DISABLE_VERSION_CHECK=1` ——
  vllm 0.29.0 把 flashinfer-python 拉到 0.6.18 而没有对应的 flashinfer-cubin；H3 的 DiT attention 走
  FlashAttention。PyPI 上没有 vllm-omni 0.29.0 正式版。
- nightly 起服务的两个坑（脚本里已处理）：要传 `FL2VA/` 子目录而不是 repo id —— 官方仓库根目录有
  `modular_model_index.json`，nightly 会把它当成 FastH3 发布，然后因找不到 `fastvideo_inference.json`
  崩掉；以及 t2va 请求现在必须带 `aspect_ratio=16:9`。

复现：sglang `ARMS="exact lossless" PARTS=fl2va ./bench_matrix.sh`；vllm-omni
`TAGS="nightly latest h3-vomni:0.29.0rc1-vllm0.29.0" ./bench_vomni_versions.sh`。

## diffusers 原生（2026-10-21，同一台机器，只测 T2VA）

`h3_diffusers_bench.py` 用 `torchrun --nproc_per_node=8` 跑（`run_diffusers.sh`），环境是 `lmsysorg/sglang:dev`
（torch 2.14.1+cu130、transformers 5.19），diffusers 以 `--no-deps -e` 装自 PR #14609 head `bc35ec2d5` 合入
PR #14725（`ce1581f0f`，TP+CP 同一个 `ParallelConfig`）—— `modeling_utils.py` 唯一的源码冲突两边都保留
（`diffusers_out/combo_conflict_resolution.patch`）。纯 U8 走的是 diffusers main 的 `_cp_plan`；TP8 是 #14609 的
`_tp_plan`；TP2×U4 两个 PR 都要。text encoder 只在 rank 0（输出 state 广播），视频 + 音频 VAE 在 rank 1 原生跑一次
（不做 tile 并行），attention 用默认后端（SDPA）。prompt/seed 与 sglang 相同；1 次 warmup + 2 次测量。

| 配置 | 端到端 | text enc | DiT 50 步 | s/step | 视频 VAE | 音频 VAE | MP4 封装 | rank 0 峰值 |
|---|---|---|---|---|---|---|---|---|
| Ulysses 8（main `_cp_plan`） | **63.08 s** | 0.059 s | 50.60 s | 1.012 | 7.80 s | 0.064 s | 4.56 s | 126.0 GiB |
| TP2 × U4（#14609 + #14725） | 65.20 s | 0.057 s | 52.76 s | 1.055 | 7.81 s | 0.063 s | 4.50 s | 94.8 GiB |
| TP8（#14609） | 74.02 s | 0.056 s | 61.52 s | 1.230 | 7.83 s | 0.064 s | 4.54 s | 73.1 GiB |

- diffusers 最好档（U8，63.08 s）是 sglang（41.11 s）的 1.53×：DiT 每步慢 1.26×（1.012 vs 0.803 s），且原生 VAE 解码 +
  封装要 12.4 s，而 sglang/vllm-omni 约 1 s。
- 在带 NVLink 的 8×H200 上，**TP 不帮延迟**：TP 每个 block 多出 all-reduce，所以 U8 > TP2×U4 > TP8。#14609 的 TP 买的是
  显存：rank 0 峰值 126 → 95 → 73 GiB（纯 Ulysses 下 62 GB 的 DiT 每卡一份）。这是显存受限硬件（Trn2 核）上的杠杆，不是这里。
- `_flash_3_hub` 没法试：hub 上 `kernels-community/flash-attn3` 没有 torch 2.14 的构建（`diffusers_out/tp1_u8_fa3hub_failed.log`）。

</details>
