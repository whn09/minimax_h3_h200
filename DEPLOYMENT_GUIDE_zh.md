# MiniMax-H3 部署最佳实践：H200 与 g7e（RTX PRO 6000）

面向客户的部署建议。所有数字都是在 `p5e.48xlarge`（8xH200）上实测的，测试条件统一为
**864x480 / 124 帧 / 40 步 / t2va / seed 1101**，口径是客户端 wall clock（POST 到
`status: completed`）。完整实验记录见 `RESULTS_zh.md`，补丁在 `patches/`。

结论先行：**H200 上有两个便宜的并行旋钮（Ulysses 与 TP），g7e 上两个都不能用**，所以两个平台
的最佳形态完全不同，不要把 H200 的命令直接搬到 g7e。

---

## 一、H200（p5e.48xlarge，有 NVLink）

### 1.1 前提：patch 必须打进容器里

**`SGLANG_MINIMAX_H3_EXTRA_SHORT_EDGES=480` 在原版镜像上是完全无效的。** 这个变量只被 patch 新增
的代码读（`minimax_h3/resolved_plan.py`）；没打 patch 的镜像里根本没有代码读它，`short_edge: 480`
的请求照样被原来的校验器拒掉。**"设了环境变量却没打 patch"是照着文档做却跑不起来的最常见原因。**

好消息：**不用重新 build 镜像。** `lmsysorg/sglang:dev` 里 sglang 是 **editable** 安装
（`Editable project location: /sgl-workspace/sglang/python`），所以 `git apply` 改
`/sgl-workspace/sglang` 的源码后，下次启动 server 进程就直接生效。

最省事的是本目录里的封装脚本，它负责挂载 patch、幂等地 apply、后台起服务、等 `/health`：

```bash
cd h3_h200_baseline
./serve.sh                      # <- 默认就是 H200 最优配置，见下
GPUS=4 ./serve.sh               # cookbook 的 4×H200 配方
TP=2 ULYSSES=4 ./serve.sh       # 切 DiT：每卡 63.9 GiB 而不是 95.9
SHORT_EDGES= ./serve.sh         # 保持发布的 768-only 策略，patch 保持惰性
./serve.sh status | logs | stop
```

**`./serve.sh` 不带任何参数就是推荐配置**：8 卡、TP=1、Ulysses=8、`encoder-parallel auto`、
480p 打开并预热、不预热其他分辨率 —— 也就是实测 10.05 s / 6.2 视频/分 / 每卡 95.9 GiB 那个形态。
它建的是常驻的 `sleep infinity` 容器、patch 打在容器里，所以之后换参数重启既不会重新 pull 镜像
也不会重复打 patch；`stop` 只杀 server 进程、保留容器，打好的源码不会丢。

`patches/` 下的**两个 patch 都会被应用**：short-edge 那个开 480p，cpu-offload 那个在
`OFFLOAD=1` 时是必须的（见 2.3）。无条件都打是安全的——不传 offload flag 时后者是 no-op。

起完服务发一个推荐请求（同样不需要参数）：

```bash
python3 h3req.py                # 864x480 / 40 步 / 5 秒片长 → 实测 10.09 s
python3 h3req.py 768 12 5 my768 # 需要时再覆盖：[短边 [步数 [片长 [输出前缀]]]]
```

### 1.2 推荐命令：延迟优先（10 秒目标）

如果你想自己写 docker 命令，就把 patch 只读挂进去，并在同一个 `bash -lc` 里先 apply 再起服务：

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

**10.05 s / 请求，6.2 视频/分钟，每卡 95.9 GiB。**

注意 `git apply` **不是幂等的**，跑第二次会失败。所以这种一次性写法只适合用完就删的容器；要反复
重启就用 `serve.sh`（它用 `--check` 和 `-R --check` 区分"还没打"和"已经打过"）。

验证 patch 真的生效了——**不要用"预热成功"当证据**，因为 `--warmup-resolutions` 吃原始 `WxH`、
绕过短边校验器，不打 patch 也能预热：

```bash
docker exec h3 git -C /sgl-workspace/sglang diff --stat
docker exec h3 git -C /sgl-workspace/sglang log --oneline -1   # 期望 c7c03ec53b
```

**patch 是对着镜像 commit `c7c03ec53b` diff 的。** 如果 pull 到更新的 `:dev` 可能打不上；
`serve.sh` 会明确报 `DOES_NOT_APPLY ... image moved off c7c03ec53b` 并退出，不会带着半个 patch
起服务，上面那条命令里的 `&&` 也会同样短路。真遇到就需要重新 diff 一版。

### 1.3 按目标选形态

| 目标 | 形态 | 延迟 | 吞吐 | 每卡显存 |
|---|---|---|---|---|
| **最低延迟** | 1 副本 x 8 卡，Ulysses=8 | **10.05 s** | 6.2 视频/分 | 95.9 GiB |
| **最高吞吐** | 8 副本 x 1 卡 | 61.38 s | **7.69 视频/分** | 132.2 GiB |
| 折中 | 2 副本 x 4 卡，Ulysses=4 | 18.17 s | 6.69 视频/分 | 100.7 GiB |
| **显存最省** | 1 副本 x 8 卡，TP=8 | 13.09 s | 4.8 视频/分 | **39.0 GiB** |

延迟和吞吐大约按 6:1 交换：从 8 个单卡副本换成 1 个 8 卡副本，只损失 24% 吞吐，换来
**6.11 倍**更低的延迟。Ulysses 在 NVLink 上效率很高（8 卡 76%、4 卡 84%、2 卡 89%）。

### 1.4 `--tp-size` 是显存旋钮，而且很便宜（本轮新发现）

Ulysses 是**序列并行**：权重每卡一份，只切激活，所以它对 61.73 GB 的 DiT 权重毫无帮助。
真正切 DiT 的是 `--tp-size`，而它在 NVLink 上的延迟代价比预期小得多：

| 形态 | DiT / 卡 | 每卡峰值 | 延迟 | 相对最优 |
|---|---|---|---|---|
| TP1 x Ulysses8 | 61.73 | 95.9 GiB | **10.08 s** | — |
| TP2 x Ulysses4 | 30.86 | 63.9 GiB | 11.08 s | +10.2% |
| TP4 x Ulysses2 | 15.43 | 47.5 GiB | 11.59 s | +15.0% |
| TP8 x Ulysses1 | 7.72 | **39.0 GiB** | 13.09 s | +30.0% |

**2.5 倍显存换 30% 延迟**，只要 TP2 就能省 32 GiB 而只多付 10%。实际意义是多了一种新能力：
TP4/TP8 下 480p 能装进 **80 GB** 卡（A100-80G / H100-80G），而默认的 Ulysses=8（95.9 GiB）装不进。

命令上就是加 `--tp-size N`，并把 `--ulysses-degree` 改成 `8/N`：

```bash
--num-gpus 8 --tp-size 2 --ulysses-degree 4     # 63.9 GiB，11.08 s
```

### 1.5 Encoder 已经是分布式的，不用管

cookbook 的选择器只列了 3 个模式，实际有 **4 个**：`auto | fold | dp | replicate`
（`server_args.py:1598`）。多卡下 `auto` 默认就会把 text_encoder 折叠到所有 Ulysses rank 上：

| `--encoder-parallel` | text_encoder | 每卡峰值 | 延迟 |
|---|---|---|---|
| `replicate` | 47.97 GB | 135.6 GiB | 10.09 s |
| `fold` / `auto`（默认） | **8.23 GB** | **95.9 GiB** | **10.08 s** |

**折叠是免费的：省 39.7 GiB/卡，延迟一分不涨。** 所以"把 encoder 单独部署到一张卡上省显存"这个
想法，收益其实已经被默认行为拿掉了大部分——H200 上不需要做任何事。

两个坑：
- **`dp` 模式对 H3 是死代码**：它要求 `batch_size > 1`，而 H3 硬性 `batching_max_size=1`。
- **1 卡副本永远不会折叠**：`server_args.py:669` 要求 `replica_size > tp_size`，所以 8x1 形态
  要背满 47.97 GB，这正是它 132.2 GiB 的主要来源。

---

## 二、g7e（RTX PRO 6000，96 GB，无 NVLink）

### 2.1 两个并行旋钮都不能用——这是硬结论

用 `NCCL_P2P_DISABLE=1` 模拟无 P2P（走 host 内存中转）实测：

| 形态 | 有 NVLink | 无 P2P | 惩罚 |
|---|---|---|---|
| TP1 x Ulysses8 | 10.05 s | **151.97 s**（复测 151.94） | 15.1x |
| TP8 x Ulysses1 | 13.09 s | **248.66 s**（复测 248.69） | **19.0x** |

复测都吻合到 0.01%，是拓扑的稳定性质，不是 warmup 抖动。**TP 比 Ulysses 还更糟**：Ulysses 每次
attention 交换 2 次激活，TP 每**层** all-reduce 2 次，通信更密。

所以 g7e 上唯一可行的形态是**一张卡一个副本**，per-GPU 显存成为唯一约束。

### 2.2 推荐命令：每卡一个副本 + CPU offload

这个形态**两个 patch 都需要**：short-edge 那个开 480p，cpu-offload 那个不打的话任何
`*-cpu-offload` 都会在 warmup 直接崩（见 2.3）。`serve.sh` 两个都会打，所以单卡形态就是：

```bash
OFFLOAD=1 GPUS=1 ULYSSES=1 ./serve.sh
```

要把 8 张卡起成 8 个独立副本用 `launch_replicas.sh 1`；在两个 patch 都已打好的容器里，
每个副本底层的命令是：

```bash
# 对每张卡 n = 0..7 各起一个，端口至少间隔 2
CUDA_VISIBLE_DEVICES=$n SGLANG_MINIMAX_H3_EXTRA_SHORT_EDGES=480 sglang serve \
  --model-path /models/MiniMax-H3 --model-variant fl2va \
  --num-gpus 1 --ulysses-degree 1 \
  --text-encoder-cpu-offload --vae-cpu-offload \
  --performance-mode speed \
  --warmup-resolutions 864x480 \
  --host 0.0.0.0 --port $((30010 + 2*n))
```

这里故意没有 `--encoder-parallel`：1 卡副本没有可折叠的 rank，传了也没有任何效果（见 1.5）。

**每卡 79.4 GiB，66.22 s / 请求，8 卡合计约 7.25 视频/分钟。**

不开 offload 是 132.2 GiB，**装不进 96 GB 卡**（可用约 95.6 GiB）。开了之后降 **52.8 GiB
(-40%)**，只多付 **7.9%** 延迟（66.22 s vs 61.38 s）——因为 encoder 和 VAE 每个请求只跑一次，
不是每步跑一次，卸载它们很便宜。

**已实测不是外推**：在 GPU 0 上占住 45,268 MiB 压舱物、把它伪装成 96 GB 卡，offload 后的单卡
服务加载、warmup、生成全程零 OOM，延迟同样是 66.24 s。

### 2.3 g7e 必须打的补丁

`patches/minimax-h3-cpu-offload-inplace.patch`（一行）。不打的话任何 `*-cpu-offload` 都会在
warmup 阶段直接失败：

```
RuntimeError: Inplace update to inference tensor outside InferenceMode is not allowed.
  ... minimax_h3/stages/decoding.py, line 92, in _reverse_normalize_latents_
```

原因：`_reverse_normalize_latents_` 做 `latents.mul_(std).add_(mean)`，而 latents 来自
`denoise_loop.py:33 @torch.inference_mode()`，是 inference tensor；offload manager 在
`torch.inference_mode(False)` 下跑这个 stage（`layerwise_offload.py:389`），此时改 inference
tensor 是非法的。改成 out-of-place 即可，非 offload 路径上是 no-op。

### 2.4 g7e 的现实预期

| | H200 8 卡 | g7e 8 卡 |
|---|---|---|
| 单请求延迟 | **10.05 s** | 66.22 s |
| 总吞吐 | 6.2 视频/分 | ~7.25 视频/分 |
| 能否加卡降延迟 | 能（近线性） | **不能** |

吞吐上 g7e 不吃亏，但**单请求延迟固定在 ~66 s 且加卡无法改善**。如果客户的 10 秒目标是硬指标，
g7e 达不到，除非降到很少的步数（见 `RESULTS_zh.md` 的步数表）。这是需要提前跟客户讲清楚的点。

---

## 三、两个平台共同的坑

1. **`--warmup-resolutions` 必须传，且要覆盖所有会用到的分辨率**。不传的话首个请求要多付约
   10 秒。它吃原始 `WxH`，所以 `864x480` 即使不打补丁也认。
2. **`--base-gpu-id` 对多副本无效**。它出现在 `server_args` 里，但副本 1 的 rank 仍然落在
   GPU 0-3 上和副本 0 撞车，然后双双 OOM。用 `CUDA_VISIBLE_DEVICES` 隔离。
3. **副本端口至少间隔 2**。服务除了 `0.0.0.0:<port>` 还会绑 `127.0.0.1:<port+1>`，间隔 1 会在
   权重**加载完之后**才报 `[Errno 98] address already in use`。另外各给一个
   `--master-port` / `--scheduler-port`。
4. **容器内 `pkill -f sglang` 会杀掉自己**（匹配到 `docker exec` 那个 shell 的命令行，exit 137）。
   模式要写成 `[s]glang`。
5. **动态 batching 对 H3 永远关闭**，`--batching-max-size 4` 传了也不生效：
   `base.py:405` 只对 `T2I`/`T2V` 返回 True，而 `minimax_h3.py:48` 声明 `TI2V` 且没有 override。
   日志里是 `stop_reason=dynamic_disabled`。**容量必须按副本规划**，并发只会排队不会合批。
6. **Cache-DiT 在 H3 上是 no-op**。cookbook 自己的手工配方会注册成功然后跳过 **0** 个 block，
   输出 mp4 逐字节相同。
7. **NVSwitch 机型重启后要检查 Fabric Manager**（见下节），否则 CUDA 直接起不来。

## 四、Fabric Manager 恢复步骤（p5e 等 NVSwitch 机型）

主机重启后如果服务报 `Error 802: system not yet initialized` / `cudaGetDeviceCount()` 失败，
先看 `systemctl status nvidia-fabricmanager`。典型原因是 FM 版本和驱动不一致：

```
fabric manager NVIDIA GPU driver interface version 610.57.04
  don't match with driver version 595.71.05
```

**FM 版本必须和驱动完全一致。** 修复（把 595.71.05 换成 `nvidia-smi` 报的版本）：

```bash
# 注意：nvidia-fabricmanager-595 是虚拟包，装不上，必须写全版本号
sudo apt-get install -y --allow-downgrades --allow-change-held-packages \
  nvidia-fabricmanager=595.71.05-1ubuntu1
sudo nvidia-smi -r                              # 复位所有 GPU 与 NVSwitch
sudo systemctl restart nvidia-fabricmanager     # 复位后必须重启 FM
nvidia-smi -q | grep -A2 "Fabric"               # 要看到 State: Completed / Status: Success
```

**顺序很关键**：只装包不复位的话 `Fabric State` 会卡在 `In Progress`，CUDA 仍然不可用。验证：

```bash
python3 -c "import torch; print(torch.cuda.device_count(), torch.cuda.can_device_access_peer(0,1))"
# 期望 8 True
```

`can_device_access_peer` 这一项对 H3 尤其重要——`auto` 的 encoder 折叠决策就是靠它，
P2P 不通的话 `auto` 会退回 `replicate`，白多吃 39.7 GiB/卡。这种情况下要显式传
`--encoder-parallel fold` 强制折叠（`encoders/base.py` 的注释说"拓扑是调用方自己的判断"）。

## 五、不要浪费时间的方向

以下都实测/读码确认过是死路，客户如果问可以直接答：

- **把 encoder/VAE 拆到独立的卡**：`minimax_h3_pipeline.py:94 validate_disagg_role()` 对任何非
  `MONOLITHIC` 的 role 直接 raise，`--disagg-role` / `--encoder-urls` 全部关闭。
  `--srt-encoder-url` 只给 GLM-Image 接了线。要打开需要改上游，评估见
  `SRT_ENCODER_PR_ASSESSMENT.md`。而且真正的收益已经被默认的 encoder 折叠拿走了。
- **靠并发提吞吐**：见上面第 5 条，H3 不合批。
- **Cache-DiT / `quality: "high"`**：前者 no-op；后者的 gate
  （`release_metadata.py::_MINIMAX_H3_QUALITY_WORKLOAD`）把 1344x768/50 步写死，改分辨率或步数
  都会被拒。
- **`--vae-config.parallel-decode-mode spatial` / `spatial_shard`**：H3 明确拒绝。
- **`--use-fsdp-inference`**：只切 DiT，且和上面的 TP 旋钮目标重叠，没有额外好处。

## 六、验收命令

```bash
# 起服务后确认组件驻留大小与折叠决策
grep -E "Loaded (text_encoder|transformer|video_vae|audio_vae):" serve.log
grep -i "encoder parallel folding" serve.log

# 确认每卡峰值
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

# 发一个 480p / 40 步请求并计时
python3 h3req.py 480 40 5 check
```

`serve_topo.sh <卡数> <tp> <ulysses> <encoder_parallel> <日志名>` 可以一条命令跑完
「起服务 + 等就绪 + 打印每卡显存 + 打印组件大小」，本轮所有拓扑数字都是它测的。
