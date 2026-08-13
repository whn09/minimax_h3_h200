# MiniMax-H3 在 H200（p5e.48xlarge）上的部署 —— 客户交付件

客户在 `p5e.48xlarge`（8×H200）上用 SGLang 跑 H3，**坚持要 480P**，并且希望"10 秒出视频"。
（"10 秒"到底指出片延迟还是视频时长尚待客户确认——两种读法本文都测了，确认后不需要重跑。）

两个交付件：

1. **`patches/minimax-h3-short-edge.patch`** —— 让 SGLang 接受非 768 的短边。
   三个 hunk，通过环境变量 `SGLANG_MINIMAX_H3_EXTRA_SHORT_EDGES` opt-in；**不设这个变量时，
   已发布的行为和两条报错文案逐字节不变**。可干净打到 `lmsysorg/sglang:dev` @ `c7c03ec53b`：

   ```
   cd /sgl-workspace/sglang && git apply -p1 minimax-h3-short-edge.patch
   ```

   已在真机验证：`short_edge: 480` + `16:9` 实际出的是 **864×480 × 124 帧**（ffprobe 读出）。

2. **`patches/minimax-h3-cpu-offload-inplace.patch`** —— 一行修复：原版只要加任何
   `*-cpu-offload`，H3 就会在预热时崩在 `decoding.py:92`
   （`Inplace update to inference tensor outside InferenceMode`）。修了才能开 CPU offload，
   而 offload 是让 H3 装进 96 GB 卡（RTX PRO 6000 / g7e）的唯一办法。

3. **`DEPLOYMENT_GUIDE_zh.md`** —— **给客户看的话从这里开始。** 两个平台的部署最佳实践：
   H200 和 g7e 各自的推荐命令、按目标选形态的对照表、`--tp-size` 这个显存杠杆、encoder 折叠、
   两平台共同的坑清单、Fabric Manager 恢复步骤，以及不要浪费时间的方向。

4. **`RESULTS_zh.md`** —— 4 卡 / 8 卡的延迟与画质实测表，副本切分全曲线
   （1×8 / 2×4 / 4×2 / 8×1），TP / encoder-parallel 扫描，无 NVLink 时 Ulysses 和 TP 的代价，
   96 GB 卡可行性验证，两个负面结论（Cache-DiT 在 H3 上是 no-op；`quality: "high"` 同时钉死
   1344×768 **和** 50 步），GPU 利用率、并发能力，以及推荐配置。

结论一句话：**8 卡、`--ulysses-degree 8`、864×480、40 步 → 10.05 秒。**
4→8 卡近线性加速（1.9×），代价只有 +9% GPU-秒，正是这部分加速买回了步数预算
（40 步 vs 20 步，SSIM 0.9682 vs 0.8691）。

显存不够时按这个顺序选杠杆：

- **有 P2P 的机器用 `--tp-size`。** 它切的是 Ulysses 碰不到的 61.73 GB DiT：每卡
  95.9 → 39.0 GiB，延迟 +30%；只用 `--tp-size 2` 就是省 32 GiB 付 10%。TP4 以上 480p 能装进
  80 GB 卡。
- **无 NVLink 的机器上 Ulysses 和 TP 都不能用**（关掉 P2P 分别慢 15× 和 19×）。只能一卡一副本 +
  `--num-gpus 1 --ulysses-degree 1 --text-encoder-cpu-offload --vae-cpu-offload`
  （每卡 79.4 GiB、约 66 s 一个视频、已验证能压在 96 GB 以内）。

## 起服务

```bash
SGLANG_MINIMAX_H3_EXTRA_SHORT_EDGES=480 sglang serve \
  --model-path MiniMaxAI/MiniMax-H3 --model-variant fl2va \
  --num-gpus 8 --ulysses-degree 8 --performance-mode speed \
  --warmup-resolutions 1344x768 864x480 \
  --host 0.0.0.0 --port 30010
```

两个必须注意的点：

- **`--warmup-resolutions` 要列出所有会被请求的分辨率。** 否则某个分辨率的第一个请求要额外付
  约 10 秒。它吃的是原始 `WxH`、走 `parse_size`，绕过了 H3 的短边校验器，所以 `864x480`
  即使在**没打 patch** 的服务上也能预热。它是 `nargs="+"`，一个服务可以同时预热两种分辨率。
- **本地权重目录必须叫 `MiniMax-H3`。** `registry.py:1199` 的
  `get_non_diffusers_pipeline_name()` 匹配的是 `--model-path` 的 **basename**，不看
  `--model-id`。目录名不对会去读根 `model_index.json`，然后报
  `module diffusers has no attribute MiniMaxH3ModularPipeline`。

`serve.sh` 把上面这套（建常驻容器 → 幂等打 patch → 后台起服务 → 等 health）封好了：

```bash
./serve.sh                # 8 卡 Ulysses=8，开 480p，两种分辨率都预热
GPUS=4 ./serve.sh         # cookbook 的 4×H200 配方
SHORT_EDGES= ./serve.sh   # 不启用 480p（patch 保持惰性），验证发布行为
./serve.sh stop | logs | status
```

`stop` 只杀 server、保留容器，所以打过 patch 的源码在换参数重启时不用重新打。
patch 用 `git apply --check` 双向探测，会打印 `PATCH_APPLIED` / `PATCH_ALREADY_APPLIED`，
镜像升级后如果不再适用会硬失败并提示 `PATCH_DOES_NOT_APPLY`（提醒重新 diff）。

## 无 NVLink 的机器（RTX PRO 6000 / g7e）

不要用 Ulysses：关掉 P2P 后 8 卡 Ulysses=8 从 10.05 s 变成 151.97 s（慢 15×）。正确形态是
**一卡一副本 + 把 encoder/VAE 卸到 CPU**（必须先打 `minimax-h3-cpu-offload-inplace.patch`，
否则预热就崩）：

```bash
CUDA_VISIBLE_DEVICES=<单张卡> SGLANG_MINIMAX_H3_EXTRA_SHORT_EDGES=480 sglang serve \
  --model-path MiniMaxAI/MiniMax-H3 --model-variant fl2va \
  --num-gpus 1 --ulysses-degree 1 --performance-mode speed \
  --text-encoder-cpu-offload --vae-cpu-offload \
  --warmup-resolutions 864x480 --host 0.0.0.0 --port <端口>
```

每卡显存 79.4 GiB（不开 offload 是 132.2 GiB），已用"压舱物"实测能装进 96 GB；
单请求约 66 s，每卡约 0.91 视频/分钟。**加卡只能提吞吐，不能降这个 66 s。**
多副本用 `launch_replicas.sh <每副本卡数>`，并发压测用
`conc_multi.py <并发数> <步数> <端口列表>`。

## 提交请求

`h3req.py` 是一个**轮询到完成**的提交脚本（模型卡自带的脚本只 GET 一次 status 就下载，
会拿到被截断的文件）：

```bash
python3 h3req.py <short_edge> <steps> <duration_s> <输出前缀> [aspect_ratio]
# 例：480p / 40 步 / 5 秒片长
python3 h3req.py 480 40 5 u8_480p_40
```

## 目录内容

| 路径 | 内容 |
|---|---|
| `patches/` | 交付的 patch |
| `serve.sh` | 起/停/看日志脚本（已在真机验证） |
| `h3req.py` | 轮询式提交脚本 |
| `RESULTS_zh.md` / `RESULTS.md` | 实测结果（中/英） |
| `runs/` | 原始产物（mp4 + request/status json + `frame_*.png`） |
| `videos_named/` | 同样的视频，改成可读文件名 `{卡数}_{WxH}_{片长}_{步数}_{实测耗时}` |
| `logs/` | 服务端日志 |

对照组：4×H100 的基线在 `../h3_h100_baseline/`；Trainium 移植在 `../h3_plugin_src/`。

## 运维提醒

这个 patch **没有进上游**。SGLang 镜像升级后需要重新打（或者让 MiniMax/SGLang 把非 768
短边做成正式选项）。另外，H3-Base 官方是按 768px 发布的，非 768 分辨率属于
out-of-distribution：**没有官方参考输出可比**，画质需要客户自己在自己的 prompt 上确认。
