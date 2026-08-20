# MiniMax-H3 部署最佳实践：H200 与 g7e（RTX PRO 6000）

面向客户的部署建议。所有数字都是在 `p5e.48xlarge`（8xH200）上实测的，口径是客户端 wall clock
（POST 到 `status: completed`）。完整实验记录见 `RESULTS_zh.md`，补丁在 `patches/`。

两套基准条件，别混用：**拓扑/显存那些表用 864x480 / 124 帧（5 秒片长）/ 40 步 / t2va /
seed 1101**；**三种任务与容量规划（第三章）用 864x480 / 243 帧（10 秒片长）/ 16 步**，
因为客户确认"10 秒"指的是**视频时长**而不是出片延迟。

**三种任务（t2va / fl2va / ref2va）全部已在真机跑通**，一台机器就能同时服务，见 0.2。

结论先行：**H200 上有两个便宜的并行旋钮（Ulysses 与 TP），g7e 上两个都不能用**，所以两个平台
的最佳形态完全不同，不要把 H200 的命令直接搬到 g7e。

---

## 零、快速开始（当前推荐流程）

四步：下权重 → 起服务 → 发请求 → 取视频。全部在 `p5e.48xlarge` 上实测过。

### 0.1 下载权重

`serve.sh` **不会**自动下模型。一条命令把两个权重分区都下下来：

```bash
source /opt/pytorch/bin/activate
pip install -U "huggingface_hub[cli]"
hf download MiniMaxAI/MiniMax-H3 \
  --include "FL2VA/*" --include "Ref2VA/*" \
  --local-dir /opt/dlami/nvme/h3     # 162 个文件，269 GiB
```

**`--include` 每个 pattern 要写一次** —— 一个 `--include` 只吃一个值，写成 `--include "A" "B"` 会把
`B` 当文件名，报 `Error: File not found in repository ... /Ref2VA/%2A`。改过这条命令就先加
`--dry-run` 核一遍：正确的那条列出 162 个文件，每个分区 81 个。

这就是服务端会读的全部内容：**`FL2VA/` 服务 t2va 和 fl2va，`Ref2VA/` 服务 ref2va**。
别整仓拉（**464 GiB**）——剩下的是一套 sglang 从来不读的 diffusers 扁平布局。
只服务 t2va/fl2va 的话去掉第二个 `--include`，是 **134 GiB**。

可选，省 **73 GiB**：**Ref2VA 除 transformer 外的 16 个 LFS 文件与 FL2VA 逐个 oid 相同**
（用 HF tree API 对过 oid，不是"看文件大小一样就当一样"），所以那 16 个可以硬链而不必下两遍：

```bash
D=/opt/dlami/nvme/h3
hf download MiniMaxAI/MiniMax-H3 --include "FL2VA/*" --local-dir $D
hf download MiniMaxAI/MiniMax-H3 --include "Ref2VA/*" \
  --exclude "Ref2VA/transformer/*.safetensors" --local-dir $D   # 只要 config/index，约 29 MB
bash fill_ref2va.sh $D        # 自动下 Ref2VA/transformer（62 GiB）+ 硬链其余 16 个
```

**196 GiB** 而不是 269 GiB。漏掉第三条命令就会在加载时死在
`ValueError: no safetensors files found in .../Ref2VA/transformer`。

两条路都一样：真正要紧的是**目录名**而不是下载方式 —— `serve.sh` 默认用
`/opt/dlami/nvme/h3`，并把它挂成 `/models/MiniMax-H3`（见 0.2 第三条注）。

### 0.2 三种部署模式

`serve.sh` 就是这三种，其它都不用记：

```bash
# 模式 1：fl2va（服务 t2va + fl2va），端口 30010
./serve.sh                                   # 全部 8 卡 —— H200 推荐配置
GPUS=4 ./serve.sh                            # 只用 4 卡
CUDA_VISIBLE_DEVICES=0,1,2,3 ./serve.sh      # 指定就是这 4 卡（卡数自动推出来）

# 模式 2：ref2va（只服务 ref2va），端口 30030
VARIANT=ref2va ./serve.sh
VARIANT=ref2va CUDA_VISIBLE_DEVICES=4,5,6,7 ./serve.sh

# 模式 3：两个一起上（三种任务全覆盖），靠 CUDA_VISIBLE_DEVICES 互相隔离
./serve.sh both                              # 4 + 4
GPUS_A=2 GPUS_B=6 ./serve.sh both            # 不均分：ref2va 每步贵 3.3 倍，可以多给它卡

# 任何一条前面加 DRYRUN=1 只打印解析结果，不动 GPU
DRYRUN=1 ./serve.sh both
./serve.sh status | logs | stop               # 不带 VARIANT 时作用于全部副本
```

**为什么必须两个进程：`--model-variant` 决定加载哪一份 DiT，而任务到分区的映射是硬闸门。**

| `--model-variant` | 能服务的任务 | 默认端口 |
|---|---|---|
| `fl2va`（默认） | `t2va`、`fl2va` | 30010 |
| `ref2va` | `ref2va` | 30030 |

拿 ref2va 去问 fl2va 服务会被明确拒掉：

```
task 'ref2va' is not served by MiniMax H3 partition 'fl2va'; supported tasks: ['t2va', 'fl2va']
```

所以**客户端要按任务路由**：t2va/fl2va → 30010，ref2va → 30030。

模式 3 已实测：两副本各 90 秒就绪（串行起，共约 180 秒），每卡 103 GiB（fl2va）/ 104 GiB
（ref2va），同时各发一个请求时 8 张卡全部 100% 利用率，且 **ref2va 的耗时与它独占时一致
（32.25 s vs 32.17 s）——同机共驻不互相拖慢**，因为两组卡不重叠。

三个注意点：

- **`--base-gpu-id` 不管用**（会被静默忽略），隔离只能靠 `CUDA_VISIBLE_DEVICES`。
- **两个副本的 `--master-port` / `--scheduler-port` 必须分开**，`serve.sh` 已按 variant 预置
  （30100/5700 与 30120/5720），不用手填。
- **本地权重目录名必须以 `MiniMax-H3` 结尾**：`registry.py:1199` 拿 `--model-path` 的 basename
  找 pipeline 类。`serve.sh` 把 `$WEIGHTS` 挂成容器里的 `/models/MiniMax-H3`，所以宿主机上叫
  什么都行——自己写 docker 命令时才要注意。（`serve.sh` 默认用 `/opt/dlami/nvme/h3`，按模型命名
  而不是按分区命名，因为**这一个目录里 FL2VA 和 Ref2VA 两个分区都有**；老机器上如果还叫
  `h3-fl2va`，脚本会自动回退到那个名字。）

### 0.3 发请求：所有参数都是参数

`h3gen.py` 覆盖三种任务、任意几何、任意步数、任意片长：

```bash
# t2va：10 秒片长、16 步、864x480
python3 h3gen.py --width 864 --height 480 --steps 16 --duration 10

# 另一组几何参数（与 width/height 二选一）
python3 h3gen.py --short-edge 480 --aspect 21:9 --steps 20 --duration 10

# fl2va：首帧（可选末帧）
python3 h3gen.py --task fl2va --image assets/first.png --inline --steps 16 --duration 10

# ref2va：参考视频（连带它的音轨）→ 注意端口
python3 h3gen.py --task ref2va --ref-video assets/ref.mp4 --inline --steps 8 --port 30030
```

**两组几何参数是互斥的**，与客户"这两组参数二选一"的要求一致，服务端也是这么校验的
（见 1.6）。`--duration` 的合法范围是模型自己的 **4~15 秒**；10 秒片长实际出的是 **243 帧
@ 24 fps = 10.125 s**。

**素材是怎么传到服务端的。** 每个 condition 里的 `uri` 字符串是**服务端自己解析**的
（`minimax_h3_localize_material_uri`，`.../minimax_h3/material_io.py:761`），支持四种有用的形式：

| `uri` | 服务端行为 | 什么时候用 |
|---|---|---|
| `data:image/png;base64,…`、`base64://…` | 从请求体里解出 payload | **常规做法** —— `h3gen.py --inline` 生成的就是这个 |
| `http://…`、`https://…` | **服务端自己**去拉这个 URL | 素材已经在对象存储 / CDN 上 |
| `/path/to/x.png`、`file:///…` | 原地读，不拷贝 | 客户端和服务端共享文件系统 |
| `tar+offset://`、`tar+b64header://` | 本地 tar 里的成员 | 批处理流水线 |

`s3://` 会直接 `NotImplementedError`（除非配了 artifact resolver）。在把这个接口暴露给真实调用方
之前，有两点必须知道：

- **HTTP 那条路是故意不做防护的。** `material_io.py:719` 的注释写明它跳过了共享的 SSRF 策略、也
  没有累计超时，所以服务端会去拉调用方给的任何 URL，包括 link-local 的元数据地址。
  **如果不可信客户端能访问这个 API，必须自己在前面加白名单。**
- **两套 base64 字母表都能用** —— 标准的 `+/` 和 URL-safe 的 `-_`（`_BASE64_ALPHABET`，第 33 行），
  空白字符和百分号转义也都容忍，所以普通 `base64.b64encode` 的输出直接可用。

示例素材在 `assets/`（`first.png`、`last.png`、`ref.mp4` 10.125 s、`ref5s.mp4` 5.04 s、
`refaudio.wav`），都只是 `mkmat.sh` 从之前生成的一条片子里切出来的测试素材。

### 0.4 视频落在哪里

`serve.sh` 会传 `--output-path /out/videos`，而 `/out` 就是挂进去的 `$OUTDIR`，所以
**宿主机上在 `/opt/dlami/nvme/out/videos/<video_id>.mp4`**。

**不传这个参数是个坑**：服务端默认写容器内的相对路径 `outputs/`，那个目录没被挂载，
视频跟容器一起消失。想彻底不落盘就 `OUTPATH= ./serve.sh`，只走 HTTP 取。

另外，状态 JSON 里带 `file_paths`，所以同机调用方**不必再走一次 HTTP** 下载视频：

```json
{"status": "completed", "file_paths": ["/out/videos/<id>.mp4"], "inference_time_s": 4.26}
```

### 0.5 API 是异步的三段式（以及怎么变成一条 GET URL）

sglang 的视频接口只能这么用——`video_api.py` 的 POST handler 结尾就写着
`# Enqueue the job asynchronously and return immediately`：

```
POST /v1/videos              -> {"id": ..., "status": "queued"}   立即返回，还没开始算
GET  /v1/videos/{id}         -> "status": "completed"             轮询
GET  /v1/videos/{id}/content -> mp4 字节
```

**没有任何 GET 生成接口，也没有让 POST 阻塞等结果的参数**（整个 `runtime/entrypoints/` 查过）。
需要"一条 URL 直接出视频"（演示、调试）的话，用本目录的 `h3get.py`——它是个 sidecar，
不改 sglang，内部替你走完那三步，直接把 mp4 字节返回：

```bash
python3 h3get.py --ref2va-port 30030 &        # 监听 8080
curl "http://127.0.0.1:8080/gen?prompt=three+cats+playing+brass+instruments\
&width=864&height=480&steps=16&duration=10" -o v.mp4     # 实测 10.07 s
```

浏览器直接粘贴就能播。它按 `task=` 自动分流到两个副本，参数名和 `h3gen.py` 一致
（`prompt/task/width/height/short_edge/aspect/steps/duration/seed/image/ref_video/...`），
加 `&json=1` 返回 job 元数据。**它会占住连接 10~60 秒，生产别用**，生产走 POST + 轮询。
也别把它暴露到公网——URL 里的路径是服务端读的。

---

## 一、H200（p5e.48xlarge，有 NVLink）

### 1.1 前提：patch 必须打进容器里

**`SGLANG_MINIMAX_H3_EXTRA_SHORT_EDGES=480` 在原版镜像上是完全无效的。** 这个变量只被 patch 新增
的代码读（`minimax_h3/resolved_plan.py`）；没打 patch 的镜像里根本没有代码读它，`short_edge: 480`
的请求照样被原来的校验器拒掉。**"设了环境变量却没打 patch"是照着文档做却跑不起来的最常见原因。**

好消息：**不用重新 build 镜像。** `lmsysorg/sglang:dev` 里 sglang 是 **editable** 安装
（`Editable project location: /sgl-workspace/sglang/python`），所以 `git apply` 改
`/sgl-workspace/sglang` 的源码后，下次启动 server 进程就直接生效。

最省事的是本目录里的封装脚本，它负责挂载 patch、幂等地 apply、后台起服务、等 `/health`
（三种部署模式见 0.2）：

```bash
cd h3_h200_baseline
./serve.sh                      # <- 默认就是 H200 最优配置，见下
GPUS=4 ./serve.sh               # cookbook 的 4×H200 配方
TP=2 ULYSSES=4 ./serve.sh       # 切 DiT：每卡 63.9 GiB 而不是 95.9
SHORT_EDGES= ./serve.sh         # 保持发布的 768-only 策略，patch 保持惰性
./serve.sh status | logs | stop
```

**`./serve.sh` 不带任何参数就是推荐配置**：8 卡、TP=1、Ulysses=8、`encoder-parallel auto`、
480p 打开、`WARMUP="1344x768 864x480"` —— 也就是实测 10.05 s / 6.2 视频/分 / 每卡 95.9 GiB
那个形态。它建的是常驻的 `sleep infinity` 容器、patch 打在容器里，所以之后换参数重启既不会重新
pull 镜像也不会重复打 patch；`stop` 只杀 server 进程、保留容器，打好的源码不会丢。

**预热默认值覆盖客户可能用的两种分辨率** —— `1344x768` 和 `864x480`：没预热的分辨率第一个请求要
多付约 10 秒，而多出来的那次预热只花约 7.65 秒、整个服务生命周期只付一次。要收窄用
`WARMUP="864x480" ./serve.sh`；**96 GB 卡（g7e）上必须收窄**——2.2 节那个 79.4 GiB 只在 480p 下量过，
而 `1344x768` 面积是它的 2.49 倍。`OFFLOAD=1` 还留着 768 时 `serve.sh` 会警告。

**要去日志确认这个列表被采纳了，别默认它生效。** 镜像 `c7c03ec53b` 上，记着
`warmup_resolutions=["864x480"]` 的服务实际预热的是 `1344x768x124f`（原因推测与状态见
`RESULTS_zh.md`）。自查：

```bash
docker exec h3 bash -lc "tr '\r' '\n' < /out/serve_fl2va.log | grep -o 'warmup req ([^)]*)'"
```

要服务的形状没出现在输出里，就按"它的第一个请求慢约 10 秒"来算。

`patches/` 下的**三个 patch 全部会被应用，而且顺序是固定的**：

| 顺序 | patch | 作用 | 不打的后果 |
|---|---|---|---|
| 1 | `minimax-h3-cpu-offload-inplace.patch` | 一行，允许 CPU offload | `OFFLOAD=1` 在预热时崩（见 2.3） |
| 2 | `minimax-h3-short-edge.patch` | 开非 768 短边（480p） | `SGLANG_..._EXTRA_SHORT_EDGES` 完全无效 |
| 3 | `minimax-h3-target-width-height.patch` | 收 `target.width/height`（见 1.6） | 只能用 6 个发布的 aspect 字符串 |

**第 3 个是对着"已打完第 2 个"的树 diff 出来的**，因为两者都改
`request_validation.py::_validate_target`。所以顺序不是字母序的巧合，`serve.sh` 里是显式列表；
要重新生成用 `patches/make_patch.sh`（它会先反打第 3 个、临时提交、再正打，保证 diff 干净）。

不传 offload flag 时第 1 个是 no-op，不设环境变量时第 2 个惰性，不发 width/height 时第 3 个惰性
——所以无条件全打是安全的。

**幂等性用 stamp 文件而不是 `git apply -R --check`。** 后者作为"是否已打过"的判据在这里是坏的：
第 3 个 patch 改了第 2 个的 hunk 上下文，于是在一棵已经打好三个 patch 的树上，
反打检查会对第 2 个失败，脚本就会误报 `DOES_NOT_APPLY`。现在的做法是先试 `git apply --check`
（能打就打并落 stamp），打不上再看 `/sgl-workspace/.h3-patches/` 里有没有 stamp：有就是
`ALREADY`，没有才是真的不适用并退出。stamp 和被改的源码同生共死（都在容器文件系统里），
所以不会出现"stamp 在但源码没打"。已验证：全新容器三行 `APPLIED`，再跑一次三行 `ALREADY`。

起完服务发一个推荐请求（同样不需要参数）：

```bash
python3 h3req.py                # 864x480 / 40 步 / 5 秒片长 → 实测 10.09 s
python3 h3req.py 768 12 5 my768 # 需要时再覆盖：[短边 [步数 [片长 [输出前缀]]]]
```

### 1.2 推荐命令：延迟优先（10 秒目标）

如果你想自己写 docker 命令，就把 patch 只读挂进去，并在同一个 `bash -lc` 里先 apply 再起服务：

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

**10.05 s / 请求，6.2 视频/分钟，每卡 95.9 GiB。**

注意 `git apply` **不是幂等的**，跑第二次会失败。所以这种一次性写法只适合用完就删的容器；要反复
重启就用 `serve.sh`（它用 `--check` + stamp 文件区分"还没打"和"已经打过"，见 1.1）。

验证 patch 真的生效了——**不要用"预热成功"当证据**。预热是拿原始 `WxH` 走 `parse_size` 造请求，
根本不经过 H3 的短边校验器；而且（见 1.1）它连"列了就一定预热"都不保证，所以它两个方向都说明
不了问题。真正的证据是源码树：

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

### 1.6 长宽作为参数：`target.width/height`（本轮新增 patch）

**发布版的线上契约里没有 `width`/`height`**，只有 `{short_edge, aspect_ratio, duration_seconds}`，
而且有两道各自独立的过滤：`configs/sample/minimax_h3.py::_validate` 会把 `target` **投影**到那
三个键（多传的 width/height 被静默丢掉，不报错），`request_validation.py::_validate_target`
也只认那三个。更麻烦的是 **`aspect_ratio` 是字符串成员检查**，只认
`21:9 / 16:9 / 4:3 / 1:1 / 3:4 / 9:16` 这 6 个字面量——所以 `"640:480"` 会被拒，**尽管它就是 4:3**。
（例外：`fl2va` 不受这个白名单约束，任意 `"W:H"` 都收。）

`patches/minimax-h3-target-width-height.patch` 补上了第二组参数，并且**与客户的要求一致：
两组互斥**。设计上是"宁可拒绝也不悄悄改数"：

- 两组都给 → 拒绝：`target accepts either width+height or short_edge+aspect_ratio, not both`
- 只给一个 → 拒绝（缺的那个报 `must be an integer`）
- 不是 32 的倍数 → 拒绝，**不四舍五入**：`must be a positive multiple of 32, got 481`
- 超过面积上限 `768*1344 = 1032192` px → 拒绝，**不缩放**（ratio 那条路是会缩放的）
- `min(w,h)` 必须在允许的短边列表里（`SGLANG_..._EXTRA_SHORT_EDGES` + 768）
- 比例超出 1:4~4:1 → 由 resolver 干净地拒掉，不是 worker 崩溃

实测的 11 个边界用例见 `RESULTS_zh.md`。正例：`target: {"width": 800, "height": 480}`
（5:3，**不是**发布的 6 个比例之一）出来的 mp4 ffprobe 是 `800,480,124` + aac。回归证据：
`640:480` 仍然被原文案拒掉，说明 ratio 那条路没被动过。

`h3gen.py` 会自动选最可移植的表达方式，并把选了哪种打印出来（`wire=`），所以不会发生
"我要 864x480，服务器悄悄给了别的"：

| `wire` | 线上形式 | 需要什么 |
|---|---|---|
| `ratio` | `short_edge` + 6 个发布比例之一 | 什么都不用打，原版镜像就行 |
| `literal` | `short_edge` + 任意 `"W:H"` | 仅 `fl2va`，也不用打 patch |
| `exact` | `target.width` + `target.height` | 需要本节这个 patch |

选择依据是**画布比较而不是比例比较**，这个区别很重要：`864x480` 的比例是 **1.8 而不是
16:9 的 1.7778**（它是 `round32(480 × 16/9)`），所以约分永远得不到 `"16:9"`；正确的问法是
"16:9 在短边 480 上落到的画布是不是正好 864x480"。

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

这个形态需要**三个里的两个**：short-edge 那个开 480p，cpu-offload 那个不打的话任何
`*-cpu-offload` 都会在 warmup 直接崩（见 2.3）。width/height 那个在这里可选（不发
`target.width` 就是惰性的）。`serve.sh` 三个都会打，所以单卡形态就是：

```bash
OFFLOAD=1 GPUS=1 ULYSSES=1 ./serve.sh
```

要把 8 张卡起成 8 个独立副本用 `launch_replicas.sh 1`；在 patch 都已打好的容器里，
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
预热列表这里**故意只留 480p**，和 H200 的默认值不同：下面那个 79.4 GiB 是在 480p 下量的，
而 `1344x768` 面积是它的 2.49 倍。

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

## 三、三种任务的成本差异与 1 QPS 容量规划

### 3.1 ref2va 每步贵 3.2 倍（10 秒片长，864x480，4 卡）

每步成本由多点扫描拟合得到（服务端 `inference_time_s`；客户端 wall 再加 1.0~1.6 s）：

| 任务 | 每步 | 拟合与实测点 |
|---|---|---|
| t2va | **1.02 s/step** | `2.05 + 1.02×steps`（12/20/25/50 步四点，wall） |
| fl2va | **1.10 s/step** | `0.87 + 1.102×steps`（8/16/32 步 → 9.79 / 18.35 / 36.19 s） |
| ref2va | **3.48 s/step** | `3.52 + 3.482×steps`（8/16/32 步 → 31.15 / 59.56 / 114.83 s） |

**fl2va 只比 t2va 贵约 8%**，容量上可以和 t2va 合并规划。**ref2va 每步贵 3.16 倍**，三个点都落在
拟合线上（8 步复测 31.14 / 31.16 s，16 步复测 59.07 / 59.10 s）。**斜率差说明这不是"编码参考
视频"的一次性开销，而是每步都要付的代价**——一次性开销只会抬高 3.52 s 的截距。同样的倍数在
5 秒片长上也成立（ref2va 5 秒参考 / 16 步 = 22.02 s，对 t2va 同档约 3.2×）。做容量规划时不能用
t2va 的数字推 ref2va。

另外，**ref2va 的输出长度是从参考素材推出来的**，请求里传 `duration_seconds` 会被拒——
要短片就换一段短的参考。

### 3.2 1 QPS 需要多少台

H3 **永不合批**（`stop_reason=dynamic_disabled`，见第四章第 5 条），所以

```
QPS = 副本数 / 单请求延迟
```

并发只会排队。按 10 秒片长 / 16 步 / 864x480 算：

| 任务 | 形态 | 单请求 | 每台 p5e QPS | 1 QPS 需要 |
|---|---|---|---|---|
| t2va / fl2va | 2 副本 × 4 卡 | 19.11 s | 0.105 | **10 台** |
| t2va / fl2va | 1 副本 × 8 卡 | 10.58 s | 0.095 | 11 台 |
| ref2va | 2 副本 × 4 卡 | 60.24 s | 0.033 | **30 台** |

两点值得跟客户说清：

- **4 卡副本的吞吐比 8 卡副本略高**（0.105 vs 0.095 QPS/台），因为 Ulysses 8 卡的并行效率是
  76%。要低延迟就 8 卡，要吞吐就 4 卡，1 QPS 这个目标下 4 卡更省机器。
- **步数是唯一的大杠杆**：同样 1 QPS，t2va 从 16 步降到 8 步，延迟 19.11 → ~10.6 s，机器数直接
  减半。画质与步数的关系见 `RESULTS_zh.md` 的 SSIM 表（40 步 0.9682 / 20 步 0.8691）。

如果 ref2va 的实际占比不高，**按任务混合部署**（模式 3，不均分）比全部按最贵的任务堆机器便宜
得多：比如 ref2va 只占 10% 的流量，`GPUS_A=4 GPUS_B=4` 一台就能同时承担两种流量，
不需要为 ref2va 单独开一批机器。

## 四、两个平台共同的坑

1. **`--warmup-resolutions` 必须传，且要覆盖所有会用到的分辨率**（没预热的分辨率首个请求要多付约
   10 秒），**传完还要去日志里确认它被采纳了**：对服务日志 `grep -o 'warmup req ([^)]*)'`。在镜像
   `c7c03ec53b` 上，配了 `["864x480"]` 的服务实际预热的是 `1344x768x124f`（见 1.1）。这个 flag 走
   `parse_size` 吃原始 `WxH`，所以 `864x480` 不打补丁也认——但"参数被接受"不等于"真的预热了"。
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
6. **Cache-DiT 的效果和镜像版本有关**。在 `c7c03ec53b` 上 cookbook 自己的手工配方会注册成功然后
   跳过 **0** 个 block，输出 mp4 逐字节相同；在 `nightly-dev-20260818-c0b6474b` 上它正常工作，
   值 **1.94–2.40×**（见 `RESULTS_QUANT_zh.md`）。一定要回读
   `cache-dit enabled on transformer (... rdt=...)` 那行日志 —— 它是 warmup 时打的，不是请求时打的。
7. **NVSwitch 机型重启后要检查 Fabric Manager**（见下节），否则 CUDA 直接起不来。

## 五、Fabric Manager 恢复步骤（p5e 等 NVSwitch 机型）

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

## 六、不要浪费时间的方向

以下都实测/读码确认过是死路，客户如果问可以直接答：

- **把 encoder/VAE 拆到独立的卡**：`minimax_h3_pipeline.py:94 validate_disagg_role()` 对任何非
  `MONOLITHIC` 的 role 直接 raise，`--disagg-role` / `--encoder-urls` 全部关闭。
  `--srt-encoder-url` 只给 GLM-Image 接了线。要打开需要改上游，评估见
  `SRT_ENCODER_PR_ASSESSMENT.md`。而且真正的收益已经被默认的 encoder 折叠拿走了。
- **靠并发提吞吐**：见上面第 5 条，H3 不合批。
- **`quality: "high"`**：它的 gate
  （`release_metadata.py::_MINIMAX_H3_QUALITY_WORKLOAD`）把 1344x768/50 步写死，改分辨率或步数
  都会被拒。
- **`--vae-config.parallel-decode-mode spatial` / `spatial_shard`**：H3 明确拒绝。
- **`--use-fsdp-inference`**：只切 DiT，且和上面的 TP 旋钮目标重叠，没有额外好处。

## 七、验收命令

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
