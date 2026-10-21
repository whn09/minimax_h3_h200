#!/usr/bin/env python3
"""One cell of the 768p / 124-frame / 50-step matrix: WARM excluded warmups + N measured requests of
one task against a running sglang server, with a per-stage latency breakdown for every request.

Usage: req_matrix.py TASK OUT_DIR [--n 3] [--warm 1] [--quality exact|lossless|none] [--tag TAG]

TASK is one of
  t2va          the reference curl request (fixed prompt, seed 1101)
  fl2va         first-frame keyframe = assets/first.png, seed 2101         (FL2VA partition)
  ref2va_image  image reference      = assets/first.png, seed 3101         (Ref2VA partition)
  ref2va_video  video reference      = assets/ref5s.mp4 (+ its soundtrack), seed 4101 (Ref2VA)
All are 1344x768 (short edge 768, 16:9), 5 s target -> 124 frames, 50 steps, shifts 12 / 3.

--quality none omits the field: required for the generic Cache-DiT path, because any explicit
quality (even "exact") takes H3 off it.

Breakdown: the server must run with SGLANG_DIFFUSION_STAGE_LOGGING=1 and
SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=1; each request's "[XxxStage] finished in N seconds" lines are
read back with `docker logs --since <submit time>`. One request is in flight at a time, so the
window holds exactly that request's stages.
"""
import argparse, hashlib, json, os, re, statistics, subprocess, sys, time, urllib.request

ASSET = "file:///data/minimax-h3"
TARGET = {"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0}
TASKS = {
    "t2va": dict(
        task="t2va", seed=1101, conditions=[],
        prompt="Night-vision bedroom footage: while the owner sleeps, three cats burst in playing tiny "
               "brass instruments at full volume, freeze, then march out as if nothing happened."),
    "fl2va": dict(
        task="fl2va", seed=2101,
        conditions=[{"type": "image", "uri": f"{ASSET}/first.png", "role": "keyframe", "frame_index": 0}],
        prompt="A dim bedroom at night. Three kittens keep tumbling and batting at each other on the "
               "patterned rug in front of the radiator, tails flicking, while the person in the bed stays "
               "asleep under the blanket. The bedside lamp keeps its steady warm glow, the tall curtains "
               "stir faintly, night sky outside the windows. Warm low-key lamp light, static cinematic "
               "camera, quiet room tone with soft paw pats and faint fabric rustle."),
    "ref2va_image": dict(
        task="ref2va", seed=3101,
        conditions=[{"type": "image", "uri": f"{ASSET}/first.png", "role": "reference"}],
        prompt="Use <Picture 1> as the visual subject. The same dim night bedroom: the three kittens go on "
               "tussling on the rug by the radiator, the sleeping person does not stir, the bedside lamp "
               "holds its warm glow and the curtains move slightly. Preserve the layout, colours and "
               "low-key lighting of the reference, realistic coherent motion, static cinematic camera, "
               "synchronized quiet ambience."),
    "ref2va_video": dict(
        task="ref2va", seed=4101,
        conditions=[{"type": "video", "uri": f"{ASSET}/ref5s.mp4", "role": "reference"}],
        prompt="Follow the motion and appearance of <Video 1>, changing the setting to a moonlit bedroom "
               "while preserving coherent timing."),
}
STAGE_RE = re.compile(r"\[(\w+?)(?:Stage)?\] finished in ([0-9.]+) seconds")


def payload(name, quality):
    t = TASKS[name]
    p = {"model": "MiniMaxAI/MiniMax-H3", "prompt": t["prompt"], "seconds": 5, "task": t["task"],
         "conditions": t["conditions"], "target": TARGET, "num_outputs_per_prompt": 1,
         "num_inference_steps": 50, "flow_shift": 12.0, "audio_flow_shift": 3.0, "seed": t["seed"]}
    if quality != "none":
        p["quality"] = quality
    return p


def call(url, data=None):
    req = urllib.request.Request(url, data=json.dumps(data).encode() if data else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def stages_since(container, since):
    p = subprocess.run(["docker", "logs", "--since", f"{since:.3f}", container],
                       capture_output=True, text=True)
    out = (p.stdout + p.stderr).replace("\r", "\n")
    st = {}
    for name, sec in STAGE_RE.findall(out):
        name = name.replace("MiniMaxH3", "")
        st[name] = st.get(name, 0.0) + float(sec)  # a stage logged twice (e.g. per partition) sums
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=TASKS)
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--warm", type=int, default=1)
    ap.add_argument("--quality", default="exact", choices=["exact", "lossless", "none"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--port", type=int, default=30010)
    ap.add_argument("--container", default="h3-stock")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    base = f"http://127.0.0.1:{a.port}/v1/videos"
    body = payload(a.task, a.quality)
    tag = a.tag or a.task
    json.dump(body, open(f"{a.out}/{tag}_request.json", "w"), indent=2)
    rows = []
    for i in range(a.warm + a.n):
        kind = "warmup" if i < a.warm else "run"
        since = time.time() - 0.5
        t0 = time.time()
        vid = call(base, body)["id"]
        while True:  # poll -- a single status GET downloads a truncated file
            st = call(f"{base}/{vid}")
            if st.get("status") in ("completed", "failed"):
                break
            time.sleep(0.2)
        wall = time.time() - t0
        if st["status"] == "failed":
            json.dump(st, open(f"{a.out}/{tag}_failed.json", "w"), indent=2)
            sys.exit(f"{tag} {kind}{i} FAILED: {json.dumps(st.get('error'))[:2000]}")
        mp4 = f"{a.out}/{tag}_{kind}{i}.mp4"
        urllib.request.urlretrieve(f"{base}/{vid}/content", mp4)
        time.sleep(1.0)  # let the last stage lines reach the log
        row = dict(tag=tag, task=a.task, kind=kind, i=i, id=vid, wall_s=round(wall, 3),
                   inference_time_s=st.get("inference_time_s"), peak_memory_mb=st.get("peak_memory_mb"),
                   size=st.get("size"), seconds=st.get("seconds"), quality=st.get("quality"),
                   md5=hashlib.md5(open(mp4, "rb").read()).hexdigest(),
                   stages=stages_since(a.container, since))
        rows.append(row)
        with open(f"{a.out}/requests.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        if kind == "run" and i > a.warm:  # one measured mp4 is enough; md5 is in the row
            os.remove(mp4)
        print(f"{tag} {kind}{i}: wall={wall:.2f}s inf={row['inference_time_s']:.2f}s "
              f"size={row['size']} md5={row['md5'][:10]} stages="
              + " ".join(f"{k}={v:.2f}" for k, v in row["stages"].items()), flush=True)
    runs = [r for r in rows if r["kind"] == "run"]
    m = lambda k: statistics.mean(r[k] for r in runs)
    print(f"SUMMARY {tag} n={len(runs)} wall={m('wall_s'):.2f}s inf={m('inference_time_s'):.2f}s "
          f"(min {min(r['inference_time_s'] for r in runs):.2f} max {max(r['inference_time_s'] for r in runs):.2f}) "
          f"peak={max(r['peak_memory_mb'] or 0 for r in runs):.0f}MB "
          f"distinct_md5={len({r['md5'] for r in runs})}", flush=True)


if __name__ == "__main__":
    main()
