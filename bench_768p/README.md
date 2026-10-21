# Stock SGLang 768p steady-state baseline (8×H200) — for the Trainium2 comparison

Reference config, no patches / quantization / cache: `lmsysorg/sglang:dev`, 8 GPUs, Ulysses=8,
`--model-variant fl2va --performance-mode speed`. Request: t2va, 1344×768, **124 frames** (5.17 s @
24 fps), **50 steps**, flow_shift 12 / audio_flow_shift 3.

| file | what |
|---|---|
| `serve_stock.sh` | the reference `docker run ... sglang serve` (only the HF cache path is adapted) |
| `bench_stock.sh` | (1) reference `bench_serving` cmd with N vbench prompts, (2) reference curl payload N times |
| `req_768p.py` | the reference curl payload, polled to completion, N back-to-back runs |
| `serve_vomni.sh` / `req_vomni.sh` | vllm-omni server (8-GPU) and the reference `/v1/videos/sync` request ×N |
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

Reproduce: `./serve_stock.sh` → wait for `/health` 200 (~3 min) → `N=5 ./bench_stock.sh`.

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

<details><summary>中文</summary>

# 官方 SGLang 768p 稳态基线（8×H200）——用于和 Trainium2 对比

参考配置，无 patch / 量化 / cache：`lmsysorg/sglang:dev`，8 卡 Ulysses=8，
`--model-variant fl2va --performance-mode speed`。请求：t2va，1344×768，**124 帧**（24fps 5.17 s），
**50 步**，flow_shift 12 / audio_flow_shift 3。

| 文件 | 作用 |
|---|---|
| `serve_stock.sh` | 参考的 `docker run ... sglang serve`（只改了 HF cache 路径） |
| `bench_stock.sh` | (1) 参考 `bench_serving` 命令跑 N 个 vbench prompt；(2) 参考 curl 请求跑 N 次 |
| `req_768p.py` | 参考 curl 请求体，轮询到完成，连续跑 N 次 |
| `serve_vomni.sh` / `req_vomni.sh` | vllm-omni 8 卡 server 与参考 `/v1/videos/sync` 请求 ×N |
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

复现：`./serve_stock.sh` → 等 `/health` 返回 200（约 3 分钟）→ `N=5 ./bench_stock.sh`。

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

</details>
