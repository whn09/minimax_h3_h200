#!/usr/bin/env bash
# fill_ref2va.sh [checkpoint_root]   (default /opt/dlami/nvme/h3, or h3-fl2va if that is what exists)
#
# The MiniMax-H3 repo ships two weight partitions, FL2VA/ and Ref2VA/, and `--model-variant
# ref2va` needs the second one. A `snapshot_download` that was scoped to FL2VA leaves Ref2VA/ as
# metadata only (~29 MB of config/index/tokenizer), and the server then dies with
#
#   ValueError: no safetensors files found in <root>/Ref2VA/transformer
#   RuntimeError: Failed to load customized transformer; native fallback is disabled ...
#
# Only Ref2VA/transformer is genuinely different -- that is the ref DiT, 13 shards / 66 GB, and it
# must be downloaded. Everything else Ref2VA needs is *bit-identical* to FL2VA: all 16 remaining
# large files (14 text_encoder shards, video_vae/source/model.safetensors, audio_vae/model.
# safetensors) have matching LFS oids on the Hub, verified with the tree API, not assumed from
# equal file sizes. So they are hardlinked, which costs no disk and no download -- 74 GB of each
# saved. Hardlinks are safe here because the checkpoint is mounted read-only into the container.
set -euo pipefail
ROOT=${1:-/opt/dlami/nvme/h3}
[ -d "$ROOT" ] || [ -n "${1:-}" ] || ROOT=/opt/dlami/nvme/h3-fl2va
SRC=$ROOT/FL2VA
DST=$ROOT/Ref2VA

[ -d "$SRC" ] && [ -d "$DST" ] || { echo "need $SRC and $DST"; exit 1; }

# 1. the ref DiT: the one thing that really is different
if ! ls "$DST"/transformer/*.safetensors >/dev/null 2>&1; then
  echo "== downloading Ref2VA/transformer (66 GB)"
  python3 - "$ROOT" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download("MiniMaxAI/MiniMax-H3", allow_patterns=["Ref2VA/transformer/*"],
                  local_dir=sys.argv[1], max_workers=16)
PY
else
  echo "== Ref2VA/transformer already present ($(du -sh "$DST"/transformer | cut -f1))"
fi

# 2. everything else: hardlink from FL2VA, never overwriting what Ref2VA already ships
for comp in text_encoder video_vae audio_vae processor tokenizer; do
  [ -d "$SRC/$comp" ] || continue
  n=0
  while IFS= read -r rel; do
    [ -e "$DST/$comp/$rel" ] && continue
    mkdir -p "$(dirname "$DST/$comp/$rel")"
    ln "$SRC/$comp/$rel" "$DST/$comp/$rel"
    n=$((n + 1))
  done < <(cd "$SRC/$comp" && find . -type f -printf '%P\n')
  echo "== $comp: linked $n file(s)"
done

echo "== result"
du -sh "$DST"/* | sort -h
echo "== sanity: every component has weights"
for comp in transformer text_encoder; do
  c=$(ls "$DST/$comp"/*.safetensors 2>/dev/null | wc -l)
  echo "$comp: $c safetensors shard(s)"
done
ls -l "$DST"/video_vae/source/model.safetensors "$DST"/audio_vae/model.safetensors
