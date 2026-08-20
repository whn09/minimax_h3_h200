# MiniMax-H3 在 H200（p5e.48xlarge）上的部署 —— 客户交付件

客户在 `p5e.48xlarge`（8×H200）上用 SGLang 跑 H3，**坚持要 480P**。客户已确认："10 秒"指的是
**视频时长**（243 帧 @ 24 fps = 10.125 s），不是出片延迟；**步数、长宽都要作为参数**，而且
**「分辨率比例」与「长宽」这两组参数二选一**；**三种任务（t2va / fl2va / ref2va）全都要**。
这些都已经落地并在真机验证。

**要开始部署，直接看 `DEPLOYMENT_GUIDE_zh.md` 的第零章「快速开始」**：下权重 → 起服务 →
发请求 → 取视频，四步。

这个库里有**两轮实测**，跑在两个 sglang 镜像上。`RESULTS_zh.md` 是 BF16 延迟那一轮
（镜像 `c7c03ec53b`），回答「怎么把一条请求压到 10 s 内」；**`RESULTS_QUANT_zh.md` 是性价比那一轮**
（镜像 `nightly-dev-20260818-c0b6474b`），量 FP8 / SageAttention / Cache-DiT 与每成片秒的钱。
两轮结论冲突的地方（Cache-DiT）以新的那轮为准。

## 交付件

1. **`patches/`（三个 patch，顺序固定）**

   | 顺序 | patch | 作用 |
   |---|---|---|
   | 1 | `minimax-h3-cpu-offload-inplace.patch` | 一行修复。原版只要加任何 `*-cpu-offload`，H3 就在预热时崩在 `decoding.py:92`（`Inplace update to inference tensor outside InferenceMode`）。这是让 H3 装进 96 GB 卡（RTX PRO 6000 / g7e）的前提 |
   | 2 | `minimax-h3-short-edge.patch` | 让 SGLang 接受非 768 短边。三个 hunk，靠环境变量 `SGLANG_MINIMAX_H3_EXTRA_SHORT_EDGES` opt-in；**不设这个变量时，已发布的行为和两条报错文案逐字节不变** |
   | 3 | `minimax-h3-target-width-height.patch` | 收 `target.width` / `target.height` 作为**第二组几何参数**，并强制两组互斥（正是客户要的"二选一"）。原版只认 6 个 aspect 字面量，连 `"640:480"` 都会被拒——尽管它就是 4:3 |

   **第 3 个是对着"已打完第 2 个"的树 diff 的**（两者都改
   `request_validation.py::_validate_target`），所以顺序不能变。全部可干净打到
   `lmsysorg/sglang:dev` @ `c7c03ec53b`，`serve.sh` 会按序幂等地打好。

2. **`DEPLOYMENT_GUIDE_zh.md`** —— **给客户看的话从这里开始。** 第零章是快速开始（权重下载、
   三种部署模式、请求参数、视频落盘位置、API 形态）；后面是 H200 与 g7e 各自的推荐命令、
   按目标选形态的对照表、`--tp-size` 这个显存杠杆、encoder 折叠、`target.width/height` 的语义、
   三种任务的成本差异与 1 QPS 容量规划、两平台共同的坑、Fabric Manager 恢复步骤，以及不要
   浪费时间的方向。

3. **`RESULTS_zh.md`** —— 全部实测记录：4/8 卡延迟与画质、副本切分全曲线（1×8 / 2×4 / 4×2 /
   8×1）、TP 与 encoder-parallel 扫描、无 NVLink 时 Ulysses 与 TP 的代价、96 GB 卡可行性验证、
   三种任务的步数扫描、长宽参数的 11 个边界用例、两个负面结论（Cache-DiT 在**镜像 `c7c03ec53b`
   上**是 no-op —— 已被下面那一轮推翻；`quality: "high"` 同时钉死 1344×768 **和** 50 步）。

4. **`RESULTS_QUANT_zh.md`** —— **性价比那一轮**，用更新的 sglang
   （`nightly-dev-20260818-c0b6474b`）在 `p5en.48xlarge` 上量：BF16 → FP8 → +SageAttention →
   +Cache-DiT × 1/2/4/8 卡 × 四个几何、Ulysses 1→8 的 88–90% 效率曲线、两个价格口径下的每成片秒
   成本、1 QPS 机队、SSIM/运动能量画质表，以及和 g7e 的对比。结论：**交付方案是单卡 FP8**
   （对 BF16 快 1.13–1.19×，显存 78.8 → 48.1 GB），sage 在 768p 值 +5.8% 但在 **480p 是负收益**，
   而 **Cache-DiT 在这个镜像上是有效的**（1.94–2.40×），推翻了 `RESULTS_zh.md` 里的负面结论。
   配套的 `Dockerfile` 把交付镜像烤出来，不再在容器里做运行时改动。

5. **脚本**：`serve.sh`（起/停/状态，三种部署模式）、`fill_ref2va.sh`（省 73 GiB 补齐 Ref2VA
   权重）、`h3gen.py`（三种任务的通用提交脚本）、`h3get.py`（一条 GET URL 直接出 mp4 的
   sidecar）、`h3req.py`（最早的轮询式提交脚本）。

结论一句话：**8 卡、`--ulysses-degree 8`、864×480 → 5 秒片长 40 步 10.05 秒；
10 秒片长 16 步 10.58 秒。** 4→8 卡近线性加速（1.9×），代价只有 +9% GPU-秒，正是这部分加速
买回了步数预算（40 步 vs 20 步，SSIM 0.9682 vs 0.8691）。

显存不够时按这个顺序选杠杆：

- **有 P2P 的机器用 `--tp-size`。** 它切的是 Ulysses 碰不到的 61.73 GB DiT：每卡
  95.9 → 39.0 GiB，延迟 +30%；只用 `--tp-size 2` 就是省 32 GiB 付 10%。TP4 以上 480p 能装进
  80 GB 卡。
- **无 NVLink 的机器上 Ulysses 和 TP 都不能用**（关掉 P2P 分别慢 15× 和 19×）。只能一卡一副本 +
  `--num-gpus 1 --ulysses-degree 1 --text-encoder-cpu-offload --vae-cpu-offload`
  （每卡 79.4 GiB、约 66 s 一个视频、已验证能压在 96 GB 以内）。

## 起服务

`serve.sh` 把「建常驻容器 → 按序幂等打三个 patch → 后台起服务 → 等 `/health`」封好了，
并且覆盖客户要的**三种部署模式**：

```bash
./serve.sh                                   # 模式 1：8 卡 fl2va，服务 t2va + fl2va（:30010）
GPUS=4 ./serve.sh                            #   只用 4 卡（cookbook 的 4×H200 配方）
CUDA_VISIBLE_DEVICES=0,1,2,3 ./serve.sh      #   指定就是这 4 卡，卡数自动推出来

VARIANT=ref2va ./serve.sh                    # 模式 2：ref2va（:30030）
VARIANT=ref2va CUDA_VISIBLE_DEVICES=4,5,6,7 ./serve.sh

./serve.sh both                              # 模式 3：两个副本，三种任务全覆盖（4 + 4）
GPUS_A=2 GPUS_B=6 ./serve.sh both            #   不均分：ref2va 每步贵 3.3 倍

DRYRUN=1 ./serve.sh both                     # 只打印解析后的放置，不动 GPU
TP=2 ULYSSES=4 ./serve.sh                    # 切 DiT：每卡 63.9 GiB 而不是 95.9
SHORT_EDGES= ./serve.sh                      # 不启用 480p（patch 保持惰性），验证发布行为
./serve.sh stop | logs | status               # 不带 VARIANT 时作用于全部副本
```

**为什么模式 3 需要两个进程**：`--model-variant` 决定加载哪一份 DiT，而任务到分区的映射是
硬闸门——`fl2va` 服务 `t2va` + `fl2va`，`ref2va` 只服务 `ref2va`，一个进程永远覆盖不了三种。
两副本靠 `CUDA_VISIBLE_DEVICES` 隔离（`--base-gpu-id` 会被静默忽略），实测同机共驻不互相
拖慢（ref2va 32.25 s vs 独占 32.17 s）。

要自己写 docker 命令的话：

```bash
SGLANG_MINIMAX_H3_EXTRA_SHORT_EDGES=480 sglang serve \
  --model-path /models/MiniMax-H3 --model-variant fl2va \
  --num-gpus 8 --ulysses-degree 8 --performance-mode speed \
  --warmup-resolutions 1344x768 864x480 \
  --output-path /out/videos \
  --host 0.0.0.0 --port 30010
```

三个必须注意的点：

- **`--warmup-resolutions` 要列出所有会被请求的分辨率。** 否则某个分辨率的第一个请求要额外付
  约 10 秒。它是 `nargs="+"`，一个服务可以同时预热两种分辨率——`serve.sh` 默认就是两种
  （`WARMUP="1344x768 864x480"`，启动多付约 7.65 秒；96 GB 卡上收窄成 `864x480`）。它吃的是原始
  `WxH`、走 `parse_size`，绕过了 H3 的短边校验器，所以 `864x480` 即使在**没打 patch** 的服务上也
  **能被接受**——但"被接受"不等于"真预热了"：镜像 `c7c03ec53b` 上配了 `["864x480"]` 的服务，
  唯一那次预热请求打的是 `1344x768x124f`。务必用服务日志上的
  `grep -o 'warmup req ([^)]*)'` 复核（详见指南 1.1）。
- **`--output-path` 不传是个坑。** 服务端默认写容器内的相对路径 `outputs/`，那个目录一般没被
  挂载，视频会跟容器一起消失。`serve.sh` 已经传了，落在宿主机
  `/opt/dlami/nvme/out/videos/<id>.mp4`。
- **本地权重目录必须叫 `MiniMax-H3`。** `registry.py:1199` 的
  `get_non_diffusers_pipeline_name()` 匹配的是 `--model-path` 的 **basename**，不看
  `--model-id`。目录名不对会去读根 `model_index.json`，然后报
  `module diffusers has no attribute MiniMaxH3ModularPipeline`。（`serve.sh` 把权重挂成容器里的
  `/models/MiniMax-H3`，所以宿主机上叫什么都行。）

patch 的幂等性用 stamp 文件（`/sgl-workspace/.h3-patches/`）而不是 `git apply -R --check`——
后者在这里是坏的判据，因为第 3 个 patch 改了第 2 个的 hunk 上下文。日志里会打印
`APPLIED` / `ALREADY`；镜像升级后不再适用会硬失败并提示 `DOES_NOT_APPLY ... image moved off
c7c03ec53b`（提醒重新 diff），不会带着半个 patch 起服务。

## 无 NVLink 的机器（RTX PRO 6000 / g7e）

不要用 Ulysses：关掉 P2P 后 8 卡 Ulysses=8 从 10.05 s 变成 151.97 s（慢 15×）。正确形态是
**一卡一副本 + 把 encoder/VAE 卸到 CPU**（必须先打 `minimax-h3-cpu-offload-inplace.patch`，
否则预热就崩）：

```bash
OFFLOAD=1 GPUS=1 ULYSSES=1 ./serve.sh        # 或者手写：
CUDA_VISIBLE_DEVICES=<单张卡> SGLANG_MINIMAX_H3_EXTRA_SHORT_EDGES=480 sglang serve \
  --model-path /models/MiniMax-H3 --model-variant fl2va \
  --num-gpus 1 --ulysses-degree 1 --performance-mode speed \
  --text-encoder-cpu-offload --vae-cpu-offload \
  --warmup-resolutions 864x480 --host 0.0.0.0 --port <端口>
```

每卡显存 79.4 GiB（不开 offload 是 132.2 GiB），已用"压舱物"实测能装进 96 GB；
单请求约 66 s，每卡约 0.91 视频/分钟。**加卡只能提吞吐，不能降这个 66 s。**
多副本用 `launch_replicas.sh <每副本卡数>`，并发压测用
`conc_multi.py <并发数> <步数> <端口列表>`。

## 提交请求

`h3gen.py` 是现在的主力脚本，覆盖三种任务、两组几何参数、任意步数与片长：

```bash
python3 h3gen.py --width 864 --height 480 --steps 16 --duration 10        # t2va，10 秒片长
python3 h3gen.py --short-edge 480 --aspect 21:9 --steps 20                # 另一组几何参数
python3 h3gen.py --task fl2va --image assets/first.png --inline --steps 16   # 首帧（可选末帧）
python3 h3gen.py --task ref2va --ref-video assets/ref.mp4 --inline --steps 8 --port 30030
python3 h3gen.py --host <机器> --task fl2va --image assets/first.png --inline   # 从笔记本发
```

它会打印实际用的线上表达形式（`wire=ratio|literal|exact`），所以不会发生"我要 864x480、
服务器悄悄给了别的"。**两组几何参数互斥**，与客户要求一致。

**素材的字节是靠 `--inline` 过去的。** condition 里的 `uri` 是服务端自己解析的，所以裸路径是
**在服务端**读；`--inline` 会把文件作为 `data:` URI 放进请求体，而 `http(s)://` 则是让服务端自己去拉。
示例素材在 `assets/`。注意那条 HTTP 拉取是故意跳过 SGLang 的 SSRF 防护的
（`material_io.py:719`），不可信调用方能访问时必须自己加白名单——详见指南 0.3。

想"一条 URL 直接出视频"（演示/调试）用 `h3get.py`——sglang 的接口是异步三段式
（`POST /v1/videos` → 轮询 → `GET /{id}/content`），**没有任何 GET 生成接口**，这个 sidecar
替你把三步走完：

```bash
python3 h3get.py --ref2va-port 30030 &
curl "http://127.0.0.1:8080/gen?prompt=three+cats&width=864&height=480&steps=16&duration=10" -o v.mp4
```

它会占住连接 10~60 秒，**生产不要用**，也别暴露到公网。

`h3req.py` 是最早的轮询式提交脚本，只做 t2va，仍然可用：

```bash
python3 h3req.py <short_edge> <steps> <duration_s> <输出前缀> [aspect_ratio]
python3 h3req.py 480 40 5 u8_480p_40
```

## 目录内容

| 路径 | 内容 |
|---|---|
| `DEPLOYMENT_GUIDE_zh.md` / `DEPLOYMENT_GUIDE.md` | **部署最佳实践（中/英），从这里开始** |
| `RESULTS_zh.md` / `RESULTS.md` | 实测结果，`c7c03ec53b` 上的 BF16 那一轮（中/英） |
| `RESULTS_QUANT_zh.md` / `RESULTS_QUANT.md` | 实测结果，`c0b6474b` 上的量化/性价比那一轮（中/英） |
| `Dockerfile` / `build_image.sh` / `.dockerignore` | 烤交付镜像（sm_90 SageAttention + 补丁） |
| `h200_bringup.sh` | 裸机准备：关自动升级、建 venv、下权重、拉镜像 |
| `h200_grid.sh` | `RESULTS_QUANT_zh.md` 背后的四臂 × 卡数 × 几何驱动脚本 |
| `quality_pair.sh` / `quality_pair_local.sh` | 对参考片算 SSIM + 帧间运动能量 |
| `pull_results_loop.sh` | spot 机器边跑边把产物拉回本地 |
| `patches/` | 三个交付 patch + `make_patch.sh`（重新 diff 用） |
| `serve.sh` | 起/停/状态，三种部署模式（已在真机验证） |
| `fill_ref2va.sh` | 补齐 Ref2VA 权重：只下 transformer，其余 16 个文件硬链 FL2VA，省 73 GiB |
| `h3gen.py` | 通用提交脚本：三种任务 / 两组几何参数 / 任意步数片长 |
| `assets/` | 示例素材：`first.png`、`last.png`、`ref.mp4`（10.125 s）、`ref5s.mp4`（5.04 s）、`refaudio.wav` |
| `h3get.py` | 一条 GET URL 直接返回 mp4 的 sidecar（演示用） |
| `h3req.py` | 早期的 t2va 轮询式提交脚本 |
| `launch_replicas.sh` / `launch_mixed.sh` / `serve_topo.sh` | 多副本与拓扑扫描脚本 |
| `conc_multi.py` | 并发压测 |
| `SRT_ENCODER_PR_ASSESSMENT.md` | `--srt-encoder-url` 上游改造的可行性评估 |
| `runs/` | 原始产物（mp4 + request/status json + `frame_*.png`） |
| `videos_named/` | 同样的视频，改成可读文件名 `{卡数}_{WxH}_{片长}_{步数}_{实测耗时}` |
| `logs/` | 服务端日志 |

对照组：4×H100 的基线在 `../h3_h100_baseline/`；Trainium 移植在 `../h3_plugin_src/`。

## 运维提醒

这三个 patch **都没有进上游**。SGLang 镜像升级后需要重新打（`patches/make_patch.sh` 可以
重新 diff），或者推动 MiniMax/SGLang 把非 768 短边和 `target.width/height` 做成正式选项。
另外，H3-Base 官方是按 768px 发布的，非 768 分辨率属于 out-of-distribution：
**没有官方参考输出可比**，画质需要客户自己在自己的 prompt 上确认。
