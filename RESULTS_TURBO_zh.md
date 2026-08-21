# H200 上的 Turbo LoRA 蒸馏权重（8 步）实测（2026-08-21，`p5en.48xlarge`）

`RESULTS_QUANT_zh.md` 那一轮只动**执行方式**（FP8 / SageAttention / Cache-DiT），权重还是官方那份。
这一轮动的是**权重本身**：换成 [`larryvrh/MiniMax-H3-Turbo-Lora`](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)
的 `minimax_h3_turbo_v4_step600_ema.safetensors`（strength 1.0，模型卡说 4–8 步可用、8 步最好），
把 20 步降到 8 步。驱动脚本 `h200_turbo.sh`，合并脚本 `lora_merge_transformer.py`。

## 结论

**8 步 turbo + Cache-DiT `RDT=0.24` 是这台机器上最省的一档，也是延迟最低的一档**，对同几何的
stock 20 步快 **2.47×（480p）/ 2.79×（768p）**，每成片秒省 **59%/64%**。8 卡 Ulysses 下 480p
**2.010 s**、768p **5.136 s** —— 这是整个 H3 项目里量到的最低延迟。

| fl2va，5.175 s 成片 | 480p 1 卡 | 480p 8 卡 | 768p 1 卡 | 768p 8 卡 |
|---|---|---|---|---|
| stock 20 步（FP8[+sage]，分母） | 29.015 s | — | 97.474 s | — |
| turbo 8 步 | 12.694 s | 2.263 s | 39.758 s | 5.792 s |
| **turbo 8 步 + `RDT=0.24`** | **11.766 s** | **2.010 s** | **34.953 s** | **5.136 s** |
| 对 stock 20 步 | 2.47× | — | 2.79× | — |
| $/成片秒（spot） | **$0.002124** | $0.002902 | **$0.006308** | $0.007416 |
| 对 stock 20 步省 | 59% | — | 64% | — |

480p 用纯 FP8（不加 sage），768p 加 sage —— 和 stock 那轮的结论一致，turbo 没有改变这一条。

**代价是画质：turbo 8 步 SSIM Y 0.905–0.918（对 stock 20 步），低于 FP8 量化本身的 0.942–0.951。**
这是蒸馏权重的固有代价，不是配置错误。目视（frame 10/60/110 三帧对照 + 768p 中心裁切）**看不出
差别**：同一只猫、同样的窗帘花纹与阴影、同样的窗框，没有伪影、没有运动塌缩。运动能量
0.254 vs 参考 0.239（480p）、0.336 vs 0.286（768p），略高而非塌缩。

**「这份蒸馏 LoRA 画质很差」的说法在 H200 上复现不出来。** 出现这种情况先核对两件事：步数是不是
真的降到 8（用 20 步跑蒸馏权重会过冲）、以及 LoRA 是怎么加载的（见下面「必须离线合并」）。

## 和 g7e 那条线只差一步：H200 不需要离线量化

g7e 的链路是「合并成 bf16 → 用 `nvfp4_quantize_transformer.py` 量成 NVFP4 文件 → `--transformer-weights-path <文件>`」。
H200 的量化档是 `--quantization fp8`，是**在线**量化（load 之后在 `process_weights_after_loading` 里量），
所以直接把 `--transformer-weights-path` 指到**合并后的 bf16 目录**就行：

```bash
mkdir -p /opt/dlami/nvme/out/lora && curl -sL --retry 5 \
  -o /opt/dlami/nvme/out/lora/minimax_h3_turbo_v4_step600_ema.safetensors \
  https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/resolve/main/minimax_h3_turbo_v4_step600_ema.safetensors
docker cp lora_merge_transformer.py h3:/tmp/
docker exec -e SRC=/models/MiniMax-H3/FL2VA/transformer \
  -e LORA=/out/lora/minimax_h3_turbo_v4_step600_ema.safetensors \
  -e DST=/out/turbo_v4_600_bf16 h3 python3 /tmp/lora_merge_transformer.py
#  -> lora modules: 259  strength=1.0 ... merged 259/259 modules, 最大 |delta|/|W| = 0.0036
#  -> 62 GiB，约 4 分钟（纯 CPU，别和计时请求并跑）
```

两条源码依据：`resolve_transformer_safetensors_to_load` 只在
`os.path.isfile(...) and endswith(".safetensors")` 时走单文件分支，否则对目录走
`_list_safetensors_files`；`_resolve_quant_config` 里显式 `--quantization` 优先级最高，
`fp8` 会走 `quant_cls()` 无参构造 = 在线量化。所以「换权重」和「在线 fp8」两件事不打架。

起服务（480p 交付档 / 768p 加 sage）：

```bash
IMAGE=h3-h200:local NAME=h3 GPUS=1 EXTRA="--quantization fp8 \
  --layerwise-offload-components text_encoder \
  --transformer-weights-path /out/turbo_v4_600_bf16" \
  ENVX="SGLANG_CACHE_DIT_ENABLED=1 SGLANG_CACHE_DIT_RDT=0.24 SGLANG_CACHE_DIT_SECONDARY_RDT=0.24" \
  WARMUP="864x480" ./serve.sh start
```

## 必须离线合并，不能用 `--lora-path`

`runtime/layers/lora/linear.py` 的运行时合并是对 `[out, in]` 形状做 in-place add，而 FP8 把权重
**转置**存（`21504` vs `5376`）—— 形状对不上，直接抛。所以只能在**量化之前**、在 bf16 上合并。
合并语义（照抄 sglang 那份实现）：`W_eff = W + strength * (B @ A)`，**没有** alpha/rank 缩放，
delta 用 fp32 算完再 cast 回 bf16。259 个模块 / 518 个张量必须全命中，脚本不命中就 `exit 1` ——
静默漏几个模块的后果是「像 stock 但更差」，很难查。

## 步数曲线

1 卡，**纯 FP8 臂**（分子分母同臂），`inference_time_s` / SSIM Y / 运动能量：

| 步数 | 480p | 768p |
|---|---|---|
| stock 20 步（分母） | 29.015 s / 1.000 / 0.2362 | 103.154 s / 1.000 / 0.2748 |
| turbo 4 步 | 7.608 s / 0.8898 / 0.3025 | 21.201 s / 0.8774 / 0.3263 |
| turbo 6 步 | 10.086 s / 0.9062 / 0.3059 | 31.449 s / 0.8830 / 0.2985 |
| **turbo 8 步** | **12.694 s / 0.9180 / 0.2540** | **41.781 s / 0.9097 / 0.3268** |

（换成 FP8+sage 臂自己的分母，8 步是 480p 0.9159 / 0.2571、768p 0.9047 / 0.3356 —— 两个分母
差 0.002–0.005，在噪声里。）

**8 步是三档里唯一该用的。** 4 步和 6 步的 SSIM 掉到 0.877–0.906 且运动能量涨 20–38%（冲过头），
省下来的 5 s / 21 s 不值。模型卡说的「8 步最好」在 H200 上成立。

sage 在 turbo 上和在 stock 上表现一致：768p 值 +4.9%（41.781 → 39.758），**480p 是负收益**
（12.694 → 12.939，慢 1.9%）。

## turbo 之上再叠 Cache-DiT：`RDT=0.24`

1 卡，turbo 8 步 base，扫 RDT：

| RDT | 480p | 768p |
|---|---|---|
| 关（turbo base） | 12.939 s / 0.9159 / 0.2571 | 39.758 s / 0.9047 / 0.3356 |
| 0.16 | 13.049 s / 逐字节同 base | 39.622 s / 逐字节同 base |
| **0.24** | **11.766 s / 0.9146 / 0.2527** | **34.953 s / 0.9057 / 0.3233** |
| 0.32 | 10.470 s / 0.8941 / 0.3110 | 34.941 s / 逐字节同 0.24 |

三条读法：

1. **`RDT=0.16` 在 8 步上一次都不触发。** 不是「收益小」，是 mp4 的 **md5 和不开 Cache-DiT 逐字节
   相同**（480p 和 768p 都是），13.049/39.622 s 的差异纯粹是 run-to-run 噪声。stock 那轮给 480p
   推荐的 0.16 在 turbo 上**必须改掉**：步数少 → 每步 residual 变化大 → 阈值要往上走。
   turbo 8 步的膝点是 **0.24**，和 g7e 上同一个膝点。
2. **`RDT=0.24` 在 turbo 之上几乎是白送的**：480p SSIM 0.9159 → 0.9146、运动能量 0.2571 → 0.2527；
   768p 0.9047 → 0.9057（略升）、0.3356 → 0.3233（更贴近参考的 0.2856）。cache 把 turbo 冲过头
   的运动往回收了一点。
3. **`RDT=0.32` 分两种情况**：768p 上它和 0.24 **逐字节相同**（0.24 已经把能跳的步全跳完了），
   所以 768p 用 0.24 就够，不必再抬；480p 上它真的多跳了步，快 11% 但 SSIM 掉到 0.8941、
   运动能量涨 21%，**不要用**。

## 加卡：turbo 没有稀释 Ulysses 收益

turbo 把每步压薄了，通信占比理应升高。实测**几乎没被稀释**（1 → 8 卡效率，括号是 stock 20 步同档）：

| 档 | 1 卡 | 2 卡 | 4 卡 | 8 卡 | 1→8 效率 |
|---|---|---|---|---|---|
| 480p turbo 8 步 | 12.939 s | 7.698 s | 4.124 s | 2.263 s | 71%（stock 81%） |
| 480p turbo + `RDT=0.24` | 11.766 s | 6.996 s | 3.667 s | **2.010 s** | 73%（stock 75%） |
| 768p turbo 8 步 | 39.758 s | 22.115 s | 11.273 s | 5.792 s | 86%（stock 88%） |
| 768p turbo + `RDT=0.24` | 34.953 s | 19.546 s | 9.874 s | **5.136 s** | 85%（stock 88%） |

768p 掉 2–3 个点，480p 掉 2–10 个点（480p 序列短，固定的通信开销占比本来就高）。加卡溢价
（$/成片秒，spot）：480p turbo+R24 $0.002124 → $0.002902（+37%）、768p $0.006308 → $0.007416（+18%）。

每卡显存峰值 47.9–49.2 GiB（1 卡）/ 52.2–52.5 GiB（8 卡），和 stock FP8 档一样 —— turbo 换的是
权重不是显存布局。

## ref2va

同一份 LoRA（这个库里只有一份 t2v 命名的权重）合进 `Ref2VA/transformer` 分区：**259/259 命中，
最大 |delta|/|W| 同样 0.0036**。1 卡 / 8 卡：

| ref2va（`REF_SHORT_EDGE=1024`） | 480p 1 卡 | 480p 8 卡 | 768p 1 卡 | 768p 8 卡 |
|---|---|---|---|---|
| stock 20 步（FP8+sage） | 33.850 s | — | 101.277 s | — |
| turbo 8 步 | 14.973 s | 2.493 s | 41.239 s | 5.960 s |
| **turbo 8 步 + `RDT=0.24`** | **13.430 s** | **2.262 s** | **36.353 s** | **5.290 s** |
| 对 stock 20 步 | 2.52× | — | 2.79× | — |
| $/成片秒（spot） | **$0.002424** | $0.003266 | **$0.006561** | $0.007638 |

倍数和 fl2va 基本一样（2.52×/2.79× vs 2.47×/2.79×）。加卡效率 74–86%，也和 fl2va 同级。

**ref2va 的 SSIM 和运动能量不能当画质判据。** 实测 SSIM Y 只有 0.389（480p）/ 0.635（768p）、
运动能量 1.27 vs 参考 0.18（480p）—— 但这不是画质问题：逐帧看，stock 和 turbo **都有真实的镜头
平移**，只是 turbo 的平移更快，所以按帧对齐的 SSIM 直接塌掉。ref2va 只给参考图不给首帧，
镜头轨迹本来就允许不同。目视（480p 三行 × 四帧对照、768p 中心裁切三联）：同一只猫、同一个窗户、
同样的窗帘细节与墙面颗粒，turbo 与 stock 同级，`RDT=0.24` 与 turbo base 看不出差别。

## 和 g7e 的 turbo 档比

各自的交付配置（H200 = FP8[+sage]，g7e = NVFP4+sage），都是 turbo 8 步 + `RDT=0.24`：

| | 480p 1 卡 | 768p 1 卡 | 最低延迟 |
|---|---|---|---|
| H200 延迟 | 11.766 s | 34.953 s | 8 卡 2.010 s / 5.136 s |
| g7e 延迟 | 11.882 s | 36.400 s | 2 卡 8.983 s / 24.020 s |
| **H200 快** | 1.01× | 1.04× | **4.5× / 4.7×** |
| H200 $/成片秒（spot） | $0.002124 | $0.006308 | |
| g7e $/成片秒（3 年 SP） | $0.001142 | $0.003497 | |
| **g7e 便宜** | **1.86×** | **1.80×** | |

**在 turbo 档上 H200 的单卡延迟优势基本消失**（stock 是 1.09–1.17×，这里只有 1.01–1.04×）：
turbo 把 DiT 那部分压掉了一多半，剩下的固定开销（文本编码 + VAE 解码 + 封装）在两台上差不多，
分母里 DiT 占比一降，卡的算力差就体现不出来。**加速手段越猛，平台间的延迟差越小、单价差越大。**
所以 turbo 档下「要单价选 g7e」这条比 stock 档更成立（1.80–1.86× vs 1.61–1.73×），
H200 买的只剩 8 卡那个延迟天花板（2.0 s / 5.1 s，g7e 到不了）。

## 1 QPS 机队（turbo 8 步 + `RDT=0.24`，1 卡 1 副本）

| 档位 | 单卡产出 | 需要卡数 | `p5en` 台数 | spot $/h | CB $/h |
|---|---|---|---|---|---|
| 768p fl2va | 1 条 / 34.953 s | 35 | 4.4 | **$118** | $209 |
| 768p ref2va | 1 条 / 36.353 s | 37 | 4.6 | $124 | $221 |
| 480p fl2va | 1 条 / 11.766 s | 12 | 1.5 | **$40** | $72 |
| 480p ref2va | 1 条 / 13.430 s | 14 | 1.8 | $47 | $84 |

对 stock 那轮的同口径（768p $330 / 480p $98，FP8+sage 20 步）省 **64% / 59%**。

## 复现

```bash
scp h200_turbo.sh lora_merge_transformer.py H200:~/h3run/
# 前置（一次，CPU）：下 LoRA + 两个分区各合一次，见脚本头部
setsid nohup ./h200_turbo.sh > /opt/dlami/nvme/out/h200_turbo.log 2>&1 < /dev/null &
PHASES=tc ./h200_turbo.sh          # 只补一个 phase：t0 参考 / to 步数曲线 / tc RDT / tg 加卡 / tr ref2va
```

口径同 `h200_grid.sh`：`short_edge` 480/768 + `aspect 16:9`、`duration 5.0`（5.175 s 成片、124 帧）、
`flow_shift 12.0`、`audio_flow_shift 3.0`、固定 seed（fl2va 6201 / ref2va 8201）、
镜像 `nightly-dev-20260818-c0b6474b`。时间是 `inference_time_s`（整条请求）。

**所有分母都在这一轮里重跑，没有引用 `RESULTS_QUANT_zh.md` 的数**：跨重启同配置实测漂 5.4%，
倍数和 SSIM 的分子分母必须同一次开机。这台机器在这一轮中间被 DLAMI 的自动升级重启过一次
（见 `DEPLOYMENT_GUIDE_zh.md` 的坑 8），重启前后的参考片 md5 **完全相同**，延迟漂 0.02%–2.1%。

## 坑

1. **扫 RDT 时必须传 `TAGSUF=`**（tag 里没有 RDT），不传就静默覆盖同名 mp4。
2. **`serve_grid_*.log` 的文件名里没有 `TAGSUF`**，同 arm 同卡数的后一个 phase 会覆盖前一个的服务
   端日志。要留证据就在 phase 之间把日志改名。
3. **合并脚本要放进容器**（`docker cp ... h3:/tmp/`），而 `serve.sh` 每换一个臂都会重建容器 ——
   容器一重建 `/tmp` 就空了。合并要么在跑网格之前做完，要么每次重新 cp。
4. **`--lora-path` 在量化档下用不了**（形状不匹配，见上）。
