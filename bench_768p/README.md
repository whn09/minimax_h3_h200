# Stock SGLang 768p steady-state baseline (8×H200) — for the Trainium2 comparison

Reference config, no patches / quantization / cache: `lmsysorg/sglang:dev`, 8 GPUs, Ulysses=8,
`--model-variant fl2va --performance-mode speed`. Request: t2va, 1344×768, **124 frames** (5.17 s @
24 fps), **50 steps**, flow_shift 12 / audio_flow_shift 3.

| file | what |
|---|---|
| `serve_stock.sh` | the reference `docker run ... sglang serve` (only the HF cache path is adapted) |
| `bench_stock.sh` | (1) reference `bench_serving` cmd with N vbench prompts, (2) reference curl payload N times |
| `req_768p.py` | the reference curl payload, polled to completion, N back-to-back runs |
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

</details>
