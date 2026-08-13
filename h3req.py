#!/usr/bin/env python3
"""Submit one H3 t2va request to the local SGLang server, poll to completion, report wall clock.

Usage: h3req.py [short_edge [steps [duration_seconds [out-prefix [aspect_ratio]]]]]

Every argument is optional and defaults to the recommended H200 request -- 864x480 (short edge
480), 40 steps, a 5 s clip -- so bare `h3req.py` reproduces the headline 10.05 s measurement
against a server started by bare `serve.sh`.

Polls -- the model card's own scripts do a single status GET and download a truncated file.
"""
import json, sys, time, urllib.request

a = sys.argv[1:]
short_edge = int(a[0]) if len(a) > 0 else 480
steps      = int(a[1]) if len(a) > 1 else 40
dur        = float(a[2]) if len(a) > 2 else 5.0
out        = a[3] if len(a) > 3 else f"h3_{short_edge}p_{steps}steps_{int(dur)}s"
ar         = a[4] if len(a) > 4 else "16:9"
PORT = 30010
payload = {
    "model": "MiniMax-H3",
    "prompt": "At night, while their owner sleeps in a bedroom, three cats march in loudly playing tiny brass instruments, then abruptly file out.",
    "seconds": int(dur),
    "task": "t2va",
    "conditions": [],
    "target": {"short_edge": short_edge, "aspect_ratio": ar, "duration_seconds": dur},
    "num_outputs_per_prompt": 1,
    "num_inference_steps": steps,
    "flow_shift": 12.0,
    "audio_flow_shift": 3.0,
    "seed": 1101,
}

def call(url, data=None):
    req = urllib.request.Request(url, data=json.dumps(data).encode() if data else None,
                                headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        print("HTTP", e.code, e.read().decode()[:2000]); sys.exit(1)

open(out + "_request.json", "w").write(json.dumps(payload, indent=2))
t0 = time.time()
resp = call(f"http://127.0.0.1:{PORT}/v1/videos", payload)
vid = resp["id"]
print(f"job {vid} submitted  short_edge={short_edge} steps={steps} dur={dur}", flush=True)
while True:
    st = call(f"http://127.0.0.1:{PORT}/v1/videos/{vid}")
    if st.get("status") == "completed":
        break
    if st.get("status") == "failed":
        json.dump(st, open(out + "_failed.json", "w"), indent=2)
        print("FAILED:", json.dumps(st)[:2000]); sys.exit(1)
    time.sleep(0.5)
wall = time.time() - t0
json.dump(st, open(out + "_status.json", "w"), indent=2)
urllib.request.urlretrieve(f"http://127.0.0.1:{PORT}/v1/videos/{vid}/content", out + ".mp4")
import os
print(f"RESULT short_edge={short_edge} steps={steps} dur={dur} wall={wall:.2f}s "
      f"mp4={os.path.getsize(out+'.mp4')/1e6:.2f}MB id={vid}", flush=True)
