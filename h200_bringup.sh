#!/usr/bin/env bash
# 把一台全新的 p5e/p5en.48xlarge（8×H200 141 GB）带到「serve.sh 能起服务」。
# **必须 detached 跑** —— 权重 269 GiB + 镜像 ~48 GiB，远超任何交互式 SSH 的空闲超时：
#
#   setsid nohup ./h200_bringup.sh > ~/bringup.log 2>&1 < /dev/null &
#   PARTS="FL2VA" ./h200_bringup.sh      # 只要 t2va/fl2va 那一半（135 GiB）
#
# 两个分区默认都下：ref2va 在 `Ref2VA/`，t2va 和 fl2va 在 `FL2VA/`，一个进程只能加载一个。
# FL2VA 先下，所以它下完就能起 fl2va 服务，不用等 Ref2VA。
#
# 和 g7e 版（`g7e_bringup.sh`）的两处差异，都是 AMI 决定的：
#   1. **这个 AMI 是 Base OSS-driver DLAMI，不是 PyTorch DLAMI** —— 没有 `/opt/pytorch/bin/activate`
#      （2026-08-13 那份指南里的第一步在这台上直接找不到文件）。所以自己建 venv。
#   2. **`huggingface_hub` 1.28.0 没有 `cli` / `hf_transfer` extra** —— `pip install
#      "huggingface_hub[hf_transfer,cli]"` 只会 WARNING 然后装个没有加速的 hub。`hf_transfer`
#      要当独立包装；装上之后 `HF_HUB_ENABLE_HF_TRANSFER=1` 才真的生效。
#
# 权重目录必须**叫** `MiniMax-H3`：sglang 从 `--model-path` 的 basename 反解 pipeline 类。
set -euo pipefail

VENV=${VENV:-$HOME/hf}
DEST=${DEST:-/opt/dlami/nvme/h3}
IMAGE=${IMAGE:-lmsysorg/sglang@sha256:51e576f02368480c055c7aadb67590d82b172e2392123ce4cf4cc8251b2d8caf}
PARTS=${PARTS:-"FL2VA Ref2VA"}

sudo mkdir -p /opt/dlami/nvme/out
sudo chown -R "$(id -u):$(id -g)" /opt/dlami/nvme
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv ffmpeg

# **先关掉 unattended-upgrades。** DLAMI 会在开机 20 分钟左右自己升包并重启：升 docker 时
# docker.sock 权限翻掉 → 正在计时的 arm 报 "permission denied while trying to connect to the
# docker API"，1 分钟后真重启。spot 请求还是 fulfilled、实例还是 running，**看起来像被回收但不是**。
sudo systemctl disable --now unattended-upgrades apt-daily.timer apt-daily-upgrade.timer 2>/dev/null || true
sudo systemctl mask unattended-upgrades 2>/dev/null || true
printf 'APT::Periodic::Unattended-Upgrade "0";\nAPT::Periodic::Update-Package-Lists "0";\n' \
  | sudo tee /etc/apt/apt.conf.d/99-no-auto-upgrade >/dev/null
sudo rm -f /var/run/reboot-required /var/run/reboot-required.pkgs

[ -x "$VENV/bin/hf" ] || { rm -rf "$VENV"; python3 -m venv "$VENV"; \
  "$VENV/bin/pip" -q install -U pip huggingface_hub hf_transfer; }

# 拉镜像和下权重并行 —— 除了网卡它们不抢任何东西。base 按 digest 钉死，不是移动标签。
docker pull "$IMAGE" > /tmp/pull.log 2>&1 &
PULL=$!

export HF_HUB_ENABLE_HF_TRANSFER=1 HF_XET_HIGH_PERFORMANCE=1
export HF_HOME=${HF_HOME:-/opt/dlami/nvme/out/hf}    # 别让 269 GiB 的临时块落到 484 GiB 的根盘上
for part in $PARTS; do
  "$VENV/bin/hf" download MiniMaxAI/MiniMax-H3 --include "$part/*" --local-dir "$DEST"
  echo "${part}_DONE $(date -u +%FT%TZ) size=$(du -sh "$DEST" | cut -f1)"
done

wait $PULL && echo "IMAGE_DONE $(docker images --format '{{.ID}} {{.Repository}}:{{.Tag}}' | head -1)"
echo "BRINGUP_DONE $(date -u +%FT%TZ)  下一步：cd scripts && ./build_image.sh"
