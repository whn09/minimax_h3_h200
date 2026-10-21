#!/usr/bin/env python3
"""Send the reference 768p / 124-frame / 50-step t2va request N times back to back, poll each to
completion, and report per-request wall clock plus the server's own inference_time_s.

Usage: req_768p.py [N [out-dir]]      (default N=5, out-dir=./out_768p)

The payload is the reference curl request verbatim (fixed prompt, seed 1101, quality=exact).
Same prompt + seed every time, so the spread across runs is pure run-to-run noise.
"""
import json, os, statistics, sys, time, urllib.request

n   = int(sys.argv[1]) if len(sys.argv) > 1 else 5
out = sys.argv[2] if len(sys.argv) > 2 else "out_768p"
PORT = 30010
payload = {
    "model": "MiniMaxAI/MiniMax-H3",
    "prompt": "Night-vision bedroom footage: while the owner sleeps, three cats burst in playing tiny "
              "brass instruments at full volume, freeze, then march out as if nothing happened.",
    "seconds": 5,
    "task": "t2va",
    "conditions": [],
    "target": {"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5},
    "quality": "exact",
    "num_outputs_per_prompt": 1,
    "num_inference_steps": 50,
    "flow_shift": 12,
    "audio_flow_shift": 3,
    "seed": 1101,
}

def call(url, data=None):
    req = urllib.request.Request(url, data=json.dumps(data).encode() if data else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())

os.makedirs(out, exist_ok=True)
walls, infs = [], []
for i in range(n):
    t0 = time.time()
    vid = call(f"http://127.0.0.1:{PORT}/v1/videos", payload)["id"]
    while True:  # poll -- a single status GET downloads a truncated file
        st = call(f"http://127.0.0.1:{PORT}/v1/videos/{vid}")
        if st.get("status") in ("completed", "failed"):
            break
        time.sleep(0.2)
    wall = time.time() - t0
    json.dump(st, open(f"{out}/run{i}_status.json", "w"), indent=2)
    if st["status"] == "failed":
        sys.exit(f"run {i} FAILED: {json.dumps(st)[:2000]}")
    urllib.request.urlretrieve(f"http://127.0.0.1:{PORT}/v1/videos/{vid}/content", f"{out}/run{i}.mp4")
    inf = st.get("inference_time_s")
    walls.append(wall); infs.append(inf)
    print(f"run {i}: wall={wall:.2f}s inference_time_s={inf} id={vid}", flush=True)

def stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs: return "n/a"
    return (f"mean={statistics.mean(xs):.2f}s median={statistics.median(xs):.2f}s "
            f"min={min(xs):.2f}s max={max(xs):.2f}s stdev={statistics.pstdev(xs):.3f}s")
print("WALL     ", stats(walls))
print("INFERENCE", stats(infs))
