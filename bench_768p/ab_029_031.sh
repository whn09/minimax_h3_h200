#!/usr/bin/env bash
# A/B across restarts: 0.29 and 0.31 alternated twice, 3 sync + 2 async each.
set -euo pipefail
cd "$(dirname "$0")"
for r in 1 2; do
  for t in h3-vomni:0.29.0rc1-vllm0.29.0 latest; do
    SUF=_ab$r TAGS=$t N=3 NA=2 ./bench_vomni_versions.sh
  done
done
