#!/usr/bin/env python3
"""Summarize a bench_matrix.sh output dir into a markdown table: one row per (task, arm), measured
runs only (warmups excluded), mean per-stage time.

Usage: summarize_matrix.py MATRIX_DIR [MATRIX_DIR ...]

Columns: TextEncoding (Qwen3-VL), VisualEncoding (keyframe / reference image+video through the VAE
encoder), AudioEncoding (reference soundtrack), Denoising (the DiT, 50 steps; s/step = /50),
Decoding (video + audio VAE decode and mux), other = inference_time_s - sum of logged stages.
"""
import glob, json, os, statistics, sys

ARM_ORDER = ["exact", "lossless", "compile", "fp8", "fp8sage", "fp8sagecache"]
TASK_ORDER = ["t2va", "fl2va", "ref2va_image", "ref2va_video"]
COLS = ["TextEncoding", "VisualEncoding", "AudioEncoding", "Denoising", "Decoding"]

rows = []
for d in sys.argv[1:]:
    for f in glob.glob(os.path.join(d, "*_*", "requests.jsonl")):
        arm = os.path.basename(os.path.dirname(f)).split("_", 1)[1]
        for line in open(f):
            r = json.loads(line)
            if r["kind"] == "run":
                r["arm"] = arm
                rows.append(r)

groups = {}
for r in rows:
    groups.setdefault((r["task"], r["arm"]), []).append(r)

mean = statistics.mean
print("| task | arm | n | inference_time_s | stdev | text enc | visual enc | audio enc | DiT denoise | s/step "
      "| VAE decode | other | peak MB/GPU | md5s |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for (task, arm) in sorted(groups, key=lambda k: (TASK_ORDER.index(k[0]), ARM_ORDER.index(k[1]))):
    g = groups[(task, arm)]
    inf = [r["inference_time_s"] for r in g]
    st = {c: mean(r["stages"].get(c, 0.0) for r in g) for c in COLS}
    other = mean(r["inference_time_s"] - sum(r["stages"].values()) for r in g)
    print(f"| {task} | {arm} | {len(g)} | **{mean(inf):.2f}** | {statistics.pstdev(inf):.3f} | "
          + " | ".join(f"{st[c]:.2f}" for c in COLS[:4])
          + f" | {st['Denoising'] / 50:.3f} | {st['Decoding']:.2f} | {other:.2f} | "
          f"{max(r['peak_memory_mb'] or 0 for r in g):,.0f} | {len({r['md5'] for r in g})} |")
