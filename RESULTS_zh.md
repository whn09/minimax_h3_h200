# MiniMax-H3 在 p5e.48xlarge（8×H200）上的实测 —— 2026-08-12/13

机器：`ec2-35-163-211-46.us-west-2`，镜像 `lmsysorg/sglang:dev` @ `c7c03ec53b`，权重在
`/opt/dlami/nvme/h3`（FL2VA + Ref2VA 两个分区，共 196 GiB），挂成 `/models/MiniMax-H3`
（registry 匹配 `--model-path` 的 **basename**，所以本地目录必须叫 `MiniMax-H3`）。容器内按序打了
`patches/` 下的三个 patch（cpu-offload → short-edge → target-width-height），480p 通过
`SGLANG_MINIMAX_H3_EXTRA_SHORT_EDGES=480` 开启。

除特别注明外，数字都是**客户端 wall clock**（POST 到 `status: completed`），t2va，16:9，seed 1101，
`flow_shift` 12.0 / `audio_flow_shift` 3.0，服务已预热对应分辨率。第二章（三种任务）报的是
**服务端 `inference_time_s`**，因为那批是三个任务横比、要去掉客户端排队与轮询粒度；同一请求两者
差 **1.0~1.6 s**（例：mode3 的 t2va 16 步 infer 17.54 s / wall 19.11 s）。
**同配置重跑逐字节相同**（md5 一致），所以单次 run 即可，不需要取平均。

## patch 验证

`short_edge: 480` + `aspect_ratio: "16:9"` → ffprobe 读出 **864×480、124 帧**；768p 不受影响，
仍是 1344×768×124。768p / 50 步基线实测 **74.28 s**，官方 cookbook 同配方（4×H200 Ulysses4）
公布 74.38 s —— 机器和口径都可信。

## 三种任务：全部跑通，且每步成本差 3.2 倍

以下都在 4 卡副本（`--ulysses-degree 4`）、864×480、10 秒片长（243 帧）上测，报的是服务端
`inference_time_s`。三种任务的输出都 ffprobe 过：`864,480,243` + `10.125000` + `aac`。

| 步数 | t2va | fl2va | ref2va |
|---|---|---|---|
| 8 | — | **9.79** | **31.15**（复测 31.14 / 31.16） |
| 16 | **17.54** | 18.35 | 59.56（复测 59.07 / 59.10） |
| 32 | — | 36.19 | 114.83 |
| 拟合（s） | `wall = 2.05 + 1.02×steps`（见下面 4 卡 10 秒片长那节，4 个点） | `0.87 + 1.102×steps` | `3.52 + 3.482×steps` |

三条结论：

- **fl2va 只比 t2va 贵约 8%**（每步 1.102 vs 1.02 s）。给首帧（可选末帧）几乎是免费的，
  容量规划可以和 t2va 合并。
- **ref2va 每步贵 3.16 倍**（3.482 vs 1.102 s/step），三个点都落在拟合线上。
  **斜率差说明这不是"编码参考视频"的一次性开销**——一次性开销只会抬高 3.52 s 的截距。
- **代价随输出长度走，不随"有参考"这件事走。** 同样 16 步，换成 **5 秒**的参考视频
  （输出也跟着变成 124 帧，ffprobe `864,480,124` / `5.175000`）：**22.02 s**，
  而 10 秒参考是 59.56 s。5 秒档下 t2va 16 步约 6.9 s infer（由 `1.54+0.400×steps` 反推），
  比值 3.2×；10 秒档 17.54 → 59.56 也是 3.4×。**两个片长上倍数一致**，进一步排除一次性开销。

ref2va 的 `duration_seconds` 是**从参考素材推出来的**，请求里不能传（传了会被拒），所以
"控制 ref2va 的输出长度"等于"换一段不同长度的参考"。`h3gen.py` 已按这个语义处理。

### 素材传输：inline base64 已验证，而且首帧确实被用上了

之前所有 fl2va/ref2va 实测传的都是**服务端能看见**的路径，等于没测过"远端客户端"这条路。
在 4 卡 fl2va 副本上做了配对实验，故意用一个**没有挂进容器**的宿主机路径：

| 请求 | 结果 |
|---|---|
| 裸路径，服务端看不到 | **HTTP 500** —— `FileNotFoundError: MiniMax H3 material source does not exist or is not a file`（`material_io.py:209`，经 `:798`） |
| 同一路径加 `--inline`（344,381 B → 请求体里 459,176 个 base64 字符） | **5.15 s 完成**，864x480x124 h264 + aac |

注意失败是 500 而不是 4xx——素材路径不可达会表现成服务端错误。

而且 inline 的字节是**真被用上了**，不只是被接受：输出的第 0 帧与条件图的 SSIM 是 **0.775**；
对照组（同 prompt、同 seed、同 8 步，**不给图**）只有 **0.306**。（0.775 而不是接近 1.0 是正常的：
首帧要过 VAE，而且这是 8 步的样本。`first.png` 与 `last.png` 之间是 0.917，说明这个固定机位场景
本身两帧就很像，那个数不能当地板用。）

## 模式 3：两个副本同机共驻，互不拖慢

`./serve.sh both`（fl2va 在 GPU 0-3 / :30010，ref2va 在 GPU 4-7 / :30030）：

| | 就绪时间 | 每卡显存 | 同时各发一个请求 |
|---|---|---|---|
| fl2va 副本（4 卡） | 90 s | 103,041–103,161 MiB | t2va 16 步 **19.11 s**（wall） |
| ref2va 副本（4 卡） | 90 s | 104,045 MiB | ref2va 8 步 **32.25 s**（wall） |

两副本串行启动，合计约 180 s。并发时 8 张卡 `utilization.gpu` 全部 100%。
**关键对照：ref2va 8 步独占整机时是 32.17 s，共驻时 32.25 s（+0.25%）** —— 因为两组卡不重叠，
共驻是免费的。这条决定了"按任务混合部署"在容量规划上是可用的（见
`DEPLOYMENT_GUIDE_zh.md` 第三章）。

## 长宽作为参数：边界矩阵（`minimax-h3-target-width-height.patch`）

11 个用例，全部实测（脚本 `negtests.sh` / `negtests2.sh` 在机器上的 `/opt/dlami/nvme/out`）：

| 用例 | `target` | 结果 |
|---|---|---|
| 两组都给 | `width+height+short_edge+aspect_ratio` | 拒：`target accepts either width+height or short_edge+aspect_ratio, not both` |
| 只给 width | `{"width":800}` | 拒：`target.height must be an integer` |
| 不是 32 的倍数 | `800x481` | 拒：`target.height must be a positive multiple of 32, got 481`（**不四舍五入**） |
| 超面积上限 | `1376x768` | 拒：`target.width*height must be at most 1032192 px, got 1056768 for 1376x768`（**不缩放**） |
| 短边未 opt-in | `928x512` | 拒：`min(width, height) is the short edge and must be one of [480, 768] for minimax_h3, got 512 from 928x512` |
| 宽也不是 32 的倍数 | `912x512` | 拒：先撞 32 对齐（`got 912`）——两道校验的先后顺序 |
| 比例 4.07 超出 1:4~4:1 | `1952x480` | 拒：`adapt_shape_v1 ratio must be within the inclusive range 1:4 to 4:1, got 1952:480` |
| 竖屏 | `480x800` | **通过**（短边是 480，min() 而不是 height） |
| 21:9 近似 | `1120x480` | **通过** |
| 正例 | `800x480`（5:3，**不在**发布的 6 个比例里） | **通过**，ffprobe `800,480,124` + aac |
| 回归对照 | `short_edge:480 + aspect_ratio:"640:480"` | 仍按原文案拒：`must be 'auto' or one of ['21:9','16:9','4:3','1:1','3:4','9:16']` |

最后一行是关键的回归证据：**ratio 那条老路一字未动**，`640:480` 仍然被拒——尽管它就是 4:3。
另外 `2464x480` 会先撞面积上限而不是比例上限（两个上限都在，只是面积先到），所以要测比例上限
必须挑一个面积合法的形状（`1952x480`）。

`fl2va` 也走得通 `exact` 这条线：`{"width":800,"height":480}` + 首帧 → `800,480,124`。

## 4 卡（`--num-gpus 4 --ulysses-degree 4`）

5 秒片长（124 帧）。拟合：`wall = 1.54 + 0.400 × steps`。

| short_edge | 步数 | 耗时 (s) |
|---|---|---|
| 768 | 50 | 74.28 |
| 480 | 50 | 22.10 |
| 480 | 30 | 13.56 |
| 480 | 25 | 11.56 |
| 480 | 20 | **9.55** |
| 480 | 15 | 7.55 |
| 480 | 10 | 5.54 |
| 480 | 8  | 4.54 |

10 秒片长（17n+5 对齐后 243 帧）。拟合：`wall = 2.05 + 1.02 × steps` —— 帧数 1.96×，
每步成本 2.55×，**对帧数是超线性的**。

| 步数 | 50 | 25 | 20 | 12 |
|---|---|---|---|---|
| 耗时 (s) | 53.18 | 27.11 | 22.09 | 14.06 |

## 8 卡（`--num-gpus 8 --ulysses-degree 8`）

5 秒片长。拟合：480p `wall = 1.24 + 0.216 × steps`；768p `1.36 + 0.746 × steps`。

| short_edge | 步数 | 耗时 (s) |
|---|---|---|
| 768 | 50 | 38.66 |
| 768 | 15 | 12.55 |
| 768 | 12 | **10.05** |
| 480 | 50 | 12.05 |
| 480 | 45 | 11.05 |
| 480 | 40 | **10.05** |
| 480 | 30 | 7.54 |
| 480 | 25 | 6.54 |
| 480 | 20 | 5.54 |

10 秒片长：25 步 15.06 s、20 步 12.06 s、10 步 6.54 s → `wall = 0.86 + 0.568 × steps`，
即"10 秒预算出 10 秒片长" ≈ 16 步。

## 4 卡 → 8 卡：近线性，且 8 卡是划算的

| 配置 | 4 卡 | 8 卡 | 加速 |
|---|---|---|---|
| 768p / 50 步 | 74.28 s | 38.66 s | **1.92×** |
| 480p / 50 步 | 22.10 s | 12.05 s | **1.83×** |
| 480p denoise（去掉 ~1.3 s 固定开销） | — | — | **1.90×** |

代价：**每个视频 96.4 GPU-秒（8 卡）vs 88.4 GPU-秒（4 卡），只贵 9%**，却换来 1.83× 的延迟。
所以：

- **要延迟 → 上 8 卡**（同样 10 秒预算里，8 卡给 40 步，4 卡只给 20 步）；
- **要吞吐 → 两个 4 卡副本**更好（见下面并发一节）。

## 8 卡是必须的吗？

不是必须，但**在 10 秒延迟目标下 8 卡明显更好**：

| | 4 卡 / 20 步 | 8 卡 / 40 步 |
|---|---|---|
| 耗时 | 9.55 s | 10.05 s |
| 与 50 步的 SSIM | 0.8691 | **0.9682** |

即同样约 10 秒，8 卡能多跑一倍步数，画质从"明显是另一个样本"回到"和 50 步几乎一致"。
只有当目标是**吞吐**而不是延迟时，才应该切成多副本——极端是 8×1 卡，吞吐 7.69 视频/分钟
（比 1×8 卡高 24%），但单请求要 61.38 s。完整曲线见下面"副本切分全曲线"。

## 40 步的代价（同拓扑下对 50 步参考的 SSIM）

同拓扑重跑是逐字节相同（SSIM inf），所以下表是干净信号：

| 步数 | SSIM (All) vs 50 步 |
|---|---|
| 45 | 0.9746 |
| 40 | 0.9682 |
| 30 | 0.9267 |
| 25 | 0.8719 |
| 20 | 0.8691 |

**对照基准：**同一个 50 步请求，在 4 卡和 8 卡上跑，两者互相的 SSIM 只有 **0.9598** ——
**换卡数对样本的扰动比 50→40 步更大**。这是解读上表的关键：40 步（0.9682）的差异
比"换个拓扑"还小。30 步及以下是肉眼可辨的另一个样本，但仍然连贯；20 步猫的细节开始丢。
抽帧对比见 `runs/frame_*.png`（第 62 帧）：480p/40 步三只猫和铜管乐器都在；
768p/12 步画面干净但只有一只猫，prompt 遵循度掉了。

### 没有哪个拓扑会掉质量——前提是先量出噪声地板

TP / encoder 这轮扫出来的每个形态，输出 mp4 的 md5 都不一样，这是预期的：换并行拆分方式就会换
浮点归约顺序。要证明它们不是**掉质量**，正确的对照是"数学完全相同、只有归约顺序不同"的两次跑——
即 `TP8 × Ulysses1` 开/关 P2P，两者唯一的差别是 NCCL 选了不同算法：

| 配对（480p / 40 步） | SSIM (All) |
|---|---|
| **对照：** TP8 vs TP8 `NCCL_P2P_DISABLE=1` | **0.9444** |
| `fold` vs `replicate` | 0.9376 |
| `fold` vs TP4 × Ulysses2 | 0.9348 |
| `fold` vs TP2 × Ulysses4 | 0.9199 |
| `fold` vs TP8 × Ulysses1 | 0.9054 |

四个配对都落在 0.9444 这条归约顺序地板上或略低，也就是说都在"纯粹换 NCCL 算法"本身就能造成的
波动区间里。轨迹变了，质量没变。`ssim_pairs.sh` 可以从 `videos_topo/` 复现这张表。

顺带一个值得记的点：cookbook 只标注了 `dp` 模式"不是逐位一致"，但 `fold` 其实也不是逐位一致
（对 `replicate` 是 0.9376）。切 encoder 的线性层一样会重排归约顺序。这是文档的疏漏，不是 bug。

## GPU 利用率（480p / 40 步 / 8 卡，200 ms 采样）

| 指标 | 值 |
|---|---|
| SM 利用率 均值 | **85.5%** |
| SM 利用率 p50 / max | 100% / 100% |
| 功耗 | **~609 W / 700 W** |
| 显存带宽利用率 | 仅 **~20%** |
| 每卡显存占用 | ~103 GiB / 139.8 GiB |

结论：**compute-bound，卡是喂饱的。**带宽只用到 20%、功耗接近 TDP 上限，说明瓶颈在算力
而不是访存；剩下那 ~15% 的空档就是 encode/decode/mux 的 ~1.3 s 固定开销
（拟合截距），denoise 段本身基本满载。这也解释了为什么 4→8 卡能近线性扩展。

## 并发能力：在飞请求恒为 1，严格 FIFO

用 N 个线程同时打（各自不同 seed），480p / 40 步 / 8 卡：

| 并发 N | 总墙钟 (s) | 单请求 max (s) | 吞吐 (视频/分钟) | 失败 |
|---|---|---|---|---|
| 1 | 9.83 | 9.83 | 6.1 | 0 |
| 2 | 19.41 | 19.41 | 6.2 | 0 |
| 4 | 38.86 | 38.86 | 6.2 | 0 |
| 8 | 77.24 | 77.24 | 6.2 | 0 |

**吞吐恒定在 6.2 视频/分钟，单请求延迟随并发线性增长**——完全串行，没有任何批处理收益。
0 失败，所以是排队而不是拒绝。

这**不是配置问题，是架构决定的**：`scheduler.py:967 _dynamic_batching_enabled()` 去问
`pipeline_config.supports_dynamic_batching()`，`base.py:405` 只对 `T2I` / `T2V` 返回 True，
而 `minimax_h3.py:48` 声明的是 `task_type = ModelTaskType.TI2V` 且**没有 override**，
所以恒为 False。日志里能看到 `stop_reason=dynamic_disabled`、`merged_rate=0.0%`。
给 `--batching-max-size 4` 会被接受但完全无效（server_args 里能看到，行为不变）。

**所以容量规划只能按副本数横向扩**，不能按"一台机器扛 N 并发"估。给客户的口径：
一台 p5e.48xlarge（8 卡单副本）≈ **6.2 视频/分钟**，P99 延迟 = 10 s × 队列深度。

### 多副本（2×4 卡）

`--base-gpu-id 4` **是个坑**：它会出现在 `server_args` 里，但**不生效**——第二个副本的 rank
仍然落在 GPU 0-3，和第一个副本撞车，两边一起 CUDA OOM
（每个 rank 要 ~85 GiB / 139.8 GiB，一张卡放不下两个）。正确做法是用
`CUDA_VISIBLE_DEVICES=4,5,6,7` 隔离，再配不同的 `--port` / `--master-port` / `--scheduler-port`。

第二个坑：服务除了 `0.0.0.0:<port>` 还会绑 `127.0.0.1:<port+1>`，所以两个副本端口只差 1
（30010 / 30011）会在**权重全部加载完之后**才报 `[Errno 98] address already in use`。
端口至少间隔 2。

第三个坑：容器内 `pkill -f sglang` 会匹配到 `docker exec` 自己的命令行，把 launcher 一起杀掉
（退出码 137，配合 `set -e` 直接静默中止）。pattern 要写成 `[s]glang`。
以上三个都封进了 `launch_replicas.sh <每副本卡数>`。

## 副本切分全曲线（480p / 40 步）

四种形态都用满 8 张卡，只是切法不同。并发数 = 副本数。

| 形态 | Ulysses | 单请求 (s) | 总吞吐 (视频/分钟) | GPU-秒/视频 | 相对单卡加速 |
|---|---|---|---|---|---|
| 1 × 8 卡 | 8 | **10.05** | 6.2 | 96.4 | 6.11× |
| 2 × 4 卡 | 4 | 18.17 | 6.69 | 72.7 | 3.38× |
| 4 × 2 卡 | 2 | 34.37 | 7.01 | 68.7 | 1.79× |
| 8 × 1 卡 | 1 | 61.38 | **7.69** | 61.4 | 1.00× |

两个读法：

- **延迟和吞吐的兑换比接近 6:1。** 从"8 个单卡副本"换成"1 个 8 卡副本"，总吞吐只掉 24%
  （7.69 → 6.2 视频/分钟），但单请求延迟降 **6.11×**。说明 NVLink 上的 Ulysses 效率很高：
  8 卡并行效率 76%、4 卡 84%、2 卡 89%。
- **8×1 卡是完全并行的**：8 个并发 62.39 s，单个请求 61.38 s，副本之间基本不互相干扰。

## 没有 NVLink / P2P 时 Ulysses 直接崩

这条对客户后面可能换的 RTX PRO 6000（g7e）很关键——那类卡没有 NVLink，NCCL 通常也用不了
PCIe P2P。用 `NCCL_P2P_DISABLE=1`（强制走 host 内存中转）在单个 8 卡副本上模拟：

| 形态，480p / 40 步 | 有 NVLink | `NCCL_P2P_DISABLE=1` | 惩罚 |
|---|---|---|---|
| TP1 × Ulysses8 | 10.05 s | **151.97 s**（重跑 151.94） | 15.1× |
| TP8 × Ulysses1 | 13.09 s | **248.66 s**（重跑 248.69） | **19.0×** |

**两个并行旋钮都崩，而且 TP 崩得更厉害。** Ulysses 每次 attention 交换两次激活（all-to-all），
TP 每**层** all-reduce 两次，通信更密。两次重跑都吻合到 0.01%，说明这是拓扑的稳定性质，不是
warmup 抖动。

对 g7e 的结论是决定性的：**没有 P2P 时 Ulysses 和 TP 两条路都不可用。** 唯一可行的形态是
一卡一副本，而这种形态下 encoder 也无法折叠，于是每卡显存成为硬约束，CPU offload 是唯一剩下的
工具。（附带发现：关掉 P2P 后每卡显存从 100.6 GiB 降到 90.0 GiB，说明其中约 10.6 GiB 是 NCCL
通信 buffer。）

## 显存到底花在哪（按组件，从 loader 日志实测）

loader 会打印每个组件的驻留大小，它写的 "GB" 其实是 GiB —— 下面有两处独立交叉验证。在
`--num-gpus 8 --ulysses-degree 8 --encoder-parallel auto` 下：

| 组件 | 每卡大小 | 被谁切 |
|---|---|---|
| `transformer`（DiT） | **61.73** | 只有 `--tp-size` |
| `video_vae` | 9.7 | 不切 |
| `text_encoder` | **8.23**（全量 47.97） | encoder 折叠 |
| `audio_vae` | 0.56 | 不切 |
| 权重合计 | 80.22 | |
| 实测高水位 | **95.9 GiB** | 多出 15.7 GiB 是激活 / NCCL / allocator |

**占主导的是每卡整份的 61.73 GB DiT，不是 text encoder。** 本文档早先的版本根据 63 GB 的
checkpoint 目录把 encoder 称作"最大的显存杠杆"，那个判断错了两次：目录里装的是完整 64 层
Qwen3-VL 外加 `lm_head`，而 H3 只截取 50 层并把最后的 norm 换成 `nn.Identity`，真正加载的只有
47.97 GB；这 47.97 GB 里每卡又只剩 8.23 GB。

### Encoder 默认就已经分布到全部 8 张卡上了

回答"能不能把 encoder 分布式部署到 4/8 卡而不是独立部署"——它**默认就是**。`--encoder-parallel`
有 **4 个**模式而不是 cookbook 选择器展示的 3 个：`auto | fold | dp | replicate`
（`server_args.py:1598`）。8 卡 480p/40 步实测：

| `--encoder-parallel` | `text_encoder` 大小 | 每卡高水位 | 延迟 |
|---|---|---|---|
| `replicate` | 47.97 GB | 135.6 GiB | 10.09 s |
| `fold`（显式） | **8.23 GB** | **95.9 GiB** | **10.08 s** |
| `auto`（此处默认） | 8.23 GB | 95.9 GiB | 10.08 s |

**折叠是免费的：省 39.7 GiB/卡，延迟一分不涨。** 省下来的量正好等于 encoder 大小之差
（47.97 − 8.23 = 39.74），这是"loader 的单位是 GiB"的第一处交叉验证。第二处：只有 transformer
**层**参与折叠，所以折叠后应该是 `50 层 / 8 + embed_tokens + vision tower` =
`45.4/8 + 1.45 + 1.1` = **8.23 GiB**，和日志一位不差。（单层账：hidden 5120、intermediate 25600、
64 个 q head / 8 个 kv head、head_dim 128 → 487.6 M 参数 = 0.975 GB/层。）

两个和 g7e 方案直接相关的注意点：

- **折叠决策依赖 P2P。** `encoders/base.py:104 group_has_measured_topology()` 会对每个 peer 调
  `torch.cuda.can_device_access_peer()`，只要有一个失败，`auto` 就故意保持 `replicate`——注释里的
  理由是在走 host 中转的拓扑上"在 NVLink 上勉强划算的规则可能会反过来"。显式
  `--encoder-parallel fold` 可以覆盖这个判断（"拓扑是调用方自己的判断"），所以无 NVLink 的机器上
  这个 flag 必须手动传。
- **折叠需要有 rank 可折。** `server_args.py:669` 要求 `replica_size > tp_size`，其中
  `replica_size = num_gpus // dp_size`，所以 1 卡副本永远无法折叠——这正是下面 8 × 1 形态要背满
  47.97 GB 的原因。
- `dp` 模式还额外要求 `batch_size > 1`，而 H3 硬性 `batching_max_size=1`，所以 `prefer_dp` 对 H3
  永不成立，`dp` 在这个模型上是死代码。

## `--tp-size` 才是显存杠杆，而且很便宜

这是本轮最重要的新发现。Ulysses 是**序列并行**——权重每卡整份，只切激活——所以它对 61.73 GB 的
DiT 毫无帮助。真正切 DiT 的是 `--tp-size`，而它在 NVLink 上的延迟代价远低于预期
（8 卡，480p / 40 步）：

| 形态 | DiT / 卡 | 每卡高水位 | 延迟 | 相对最优 |
|---|---|---|---|---|
| TP1 × Ulysses8 | 61.73 | 95.9 GiB | **10.08 s** | — |
| TP2 × Ulysses4 | 30.86 | 63.9 GiB | 11.08 s | +10.2% |
| TP4 × Ulysses2 | 15.43 | 47.5 GiB | 11.59 s | +15.0% |
| TP8 × Ulysses1 | **7.72** | **39.0 GiB** | 13.09 s | +30.0% |

**2.5 倍显存换 30% 延迟**，光 TP2 一档就是省 32 GiB 只多付 10%。这印证了 cookbook 关于 TP2 值
"每卡低约 30 GB 峰值显存"的说法，并且往后延伸：曲线一直到 TP8 都还在给回报。实际意义是多了一种
新能力——TP4 或 TP8 下 480p 配置能装进 **80 GB** 卡（A100-80G / H100-80G），而默认 Ulysses=8 的
95.9 GiB 装不进。

`video_vae`（9.7 GB）不被 TP 切，所以 TP8 那 39.0 GiB 里约 17.5 GiB 是 VAE + encoder + 开销，
曲线到这里就平了。

### 副本形态：度数越小，每卡显存越大

480p / 40 步，`nvidia-smi` 高水位：

| 形态 | Ulysses | 每卡显存 | encoder 折叠？ |
|---|---|---|---|
| 1 × 8 卡 | 8 | 100.6 GiB | 是 |
| 2 × 4 卡 | 4 | 100.7 GiB | 是 |
| 4 × 2 卡 | 2 | 113.9 GiB | 是 |
| 8 × 1 卡 | 1 | **132.2 GiB** | **否**（`replica_size > tp_size` 不成立） |
| 8 × 1 卡 + text encoder/VAE 卸到 CPU | 1 | **79.4 GiB** | 不适用 |

113.9 → 132.2 GiB 这一跳主要是 encoder 不再折叠，不是激活变多：单卡时权重本身就是
`47.97 + 61.73 + 9.7 + 0.56` = 119.96 GiB，剩 12.2 GiB 是开销，而 8 rank 时是 15.7 GiB
（NCCL buffer 更多）。两笔账都能对到 1 GiB 以内，这是组件表的第三处一致性检验。

**不做卸载的所有形态都超过 96 GB 卡**（可用约 95.6 GiB）。加上
`--text-encoder-cpu-offload --vae-cpu-offload` 之后，单卡占用降 **52.8 GiB（−40%）**，
延迟只涨 **7.9%**（66.22 s vs 61.38 s）——因为这些组件每请求只跑一次，不在每步循环里。

**这是实测不是推算**：在 GPU 0 上先占住 45,268 MiB"压舱物"，把它变成一张 96 GB 卡的行为，
再起开了 offload 的单卡服务——加载、预热、生成全部通过，**0 次 OOM**，耗时同样 66.24 s
（含压舱物峰值 127,137 MiB，即服务本身 79.4 GiB）。

所以无 NVLink 的 96 GB 机器**可行**：每卡约 **0.91 视频/分钟**（8 卡 ≈ 7.25 视频/分钟，
H200 是 7.69），但**单请求就是 ~66 s，且加卡不能降延迟**。

### CPU offload 需要一行修复（`patches/minimax-h3-cpu-offload-inplace.patch`）

原版只要加任何 `*-cpu-offload`，H3 在预热阶段就崩：

```
RuntimeError: Inplace update to inference tensor outside InferenceMode is not allowed.
  ... minimax_h3/stages/decoding.py, line 92, in _reverse_normalize_latents_
```

`_reverse_normalize_latents_` 里是 `latents.mul_(std).add_(mean)` 原地操作。latents 来自
`denoise_loop.py:33 @torch.inference_mode()`，所以是 inference tensor；而 offload 管理器会用
`torch.inference_mode(False)` 包住 stage 执行（`layerwise_offload.py:389`），在那里原地改
inference tensor 是非法的。改成非原地即可，对不开 offload 的路径无影响。

### "挪到别的卡上"是关的，"摊到所有卡上"不是

这是两个不同的问题，答案也不同。

**给 encoder/VAE 独占一张卡是关着的。** `minimax_h3_pipeline.py:94 validate_disagg_role()` 对任何
非 `MONOLITHIC` 角色直接 raise，所以 `--disagg-role encoder|denoiser|decoder` 配
`--encoder-urls` / `--denoiser-urls` / `--decoder-urls` 这条路对 H3 是关着的；
`--srt-encoder-url`（独立 text encoder 服务）也只接了 GLM-Image（`glm_image.py`、
`vl_encoder_loader.py`）。要打开任何一条需要改上游，评估见 `SRT_ENCODER_PR_ASSESSMENT.md`。

**但"摊到已有的卡上"是开着的，而且默认就在生效**——就是上面的 encoder 折叠，而独占一张卡本来
想要的收益大部分也就在这里。再配合 `--tp-size` 切 DiT，在任何有 P2P 的机器上，服务内部的这两个
旋钮就已经覆盖了显存问题，完全不需要碰 disaggregation。CPU offload 留给没有 P2P 的场景。

## 两个负面结论

**Cache-DiT 在 H3 上是 no-op —— 仅限镜像 `c7c03ec53b`（已被更新的镜像推翻）。**

> 在 `nightly-dev-20260818-c0b6474b` 上 Cache-DiT 正常工作，值 **1.94–2.40×**，见
> `RESULTS_QUANT_zh.md`。下面这段对 `c7c03ec53b` 仍然成立，但结论不要往后带。

cookbook 自己的手动配方
`SGLANG_CACHE_DIT_ENABLED=true FN=1 BN=0 WARMUP=4 RDT=0.12 MC=2`，以及更激进的
`RDT=0.50 MC=6`，都只是注册成功
（`Cache-DiT] Collected Context Config: DBCache_F1B0_W4I1M0MC2_R0.12_N49`）然后**跳过 0 个
block**：12.09 s vs 12.05 s，输出 mp4 与不开时**逐字节相同**。在**这个镜像上**不是可用旋钮。

**`quality: "high"` 也不能用。** 它的门控
（`release_metadata.py::_MINIMAX_H3_QUALITY_WORKLOAD`）同时钉死 width 1344 / height 768 /
50 步（shift 还用 `math.isclose` 比），所以**既拒绝非 768 分辨率，也拒绝任何步数改动**——
和两个降延迟手段都不能叠加。

## 推荐配置

客户已确认"10 秒"指的是**视频时长**，所以主线是下面第 1 条；第 2~3 条保留给"10 秒延迟"这个读法。

1. **客户口径（10 秒片长）：8 卡、Ulysses=8、864×480、16 步 → 10.58 s wall（infer 9.18 s）。**
   243 帧 @ 24 fps = 10.125 s。步数是参数，8~32 步都实测过（见上面三种任务那节）。
2. 若要 10 秒**延迟**、5 秒片长：8 卡 / 40 步 → 10.05 s，画质损失比"换拓扑"这个对照还小。
3. 必须 768p：12 步 → 10.05 s。画面能立住（H3 是 guidance-distilled 的），但 prompt 遵循度掉。
4. **ref2va 单独规划**：每步贵 3.16 倍，10 秒片长 16 步要 60.24 s wall。要么降步数（8 步 32.2 s），
   要么多给卡，要么接受更低的 QPS。
5. 显存吃紧的变体：加 `--tp-size 2 --ulysses-degree 4` → 11.08 s / 每卡 63.9 GiB（客户要能接受
   ~11 s）；80 GB 卡上用 `--tp-size 4 --ulysses-degree 2` → 47.5 GiB / 11.59 s。

1 QPS 的机器数（按副本横向扩，H3 不合批）见 `DEPLOYMENT_GUIDE_zh.md` 第三章：
t2va 2×4 卡 **10 台**、ref2va 2×4 卡 **30 台**。

无论哪种，**都要给 `--warmup-resolutions` 列全所有会服务的分辨率**（cookbook 自己也测到
不预热的首个请求要多付约 10 s；builder 吃原始 `WxH`，所以 `864x480` 即使不打 patch 也能被接受）。
`serve.sh` 现在默认两个形状都预热（`WARMUP="1344x768 864x480"`）——但这个列表到底有没有被采纳，
见下面这条待定观察。

### 待定观察：预热列表可能被忽略（镜像 `c7c03ec53b`）

某次 8 卡运行的 `server_args` 记着 `"warmup_resolutions": ["864x480"]`，但调度器唯一那条预热请求是
`server warmup req (1344x768x124f, 2/50 steps)`，7.65 s —— 要求预热的形状不是实际预热的形状，
而 50 步是发布默认值。

只靠读码得到的推测：列表确实到了构造请求这一步（`server_warmup.py:137` 传了，
`build_warmup_reqs()` 做了 `parse_size`），但 `width` 不是 `Req` 的声明字段，
`Req.__getattr__`/`__setattr__`（`runtime/pipelines_core/schedule_batch.py:251`/`269`）会转发到
`sampling_params`，那里的发布默认值正是 1344x768 / 50 步。**未经实验确认。**

两条已经写进指南的后果：别把预热成功当作短边 patch 生效的证据（要看 `git diff --stat`）；
别假定列了的形状真的热了——用 `grep -o 'warmup req ([^)]*)'` 核，缺的那些按第一个请求多付约 10 s 算。

`warmtest.sh` 能给出结论（用 4 张空卡分别按两个 `WARMUP` 值起服务，打印实际跑了哪些预热请求，
两个形状各测冷/热）。**还没跑。**

两个平台各自的推荐命令、踩坑清单和 Fabric Manager 恢复步骤都收在
`DEPLOYMENT_GUIDE_zh.md` —— **那份才是给客户看的文档。**

## 可供检查的视频

`videos_named/` 下每个文件名自解释：`{卡数}_{WxH}_{片长}_{步数}_{实测耗时}`。
全部原始文件也在 `runs/` 里（按 run id）；`cd_*` / `cda_*` 这些 Cache-DiT 的文件没有放进
`videos_named/`，因为它们与对应的 `u8_*` 逐字节相同。每个 mp4 都带生成的 AAC 音轨，
所以音频质量也能从同一个文件里判断。`runs/frame_*.png` 是片中（第 62 帧）抽帧，
用于同样 10 秒预算下 480p/40 步 vs 768p/12 步的并排对比。

三种任务与新功能的样片也在 `videos_named/`：`mode3_t2va.mp4` / `mode3_ref2va.mp4`（模式 3
共驻时同时生成的两条），`geturl.mp4`（一条 GET URL 出的片，`h3get.py`）。

"10 秒"两种读法上面的表都覆盖了：要 10 秒**片长**（客户的口径）看"三种任务"那节和 10 秒片长
那几行，要 10 秒**延迟**看 5 秒片长的表。
