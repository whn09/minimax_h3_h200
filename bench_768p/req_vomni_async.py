#!/usr/bin/env python3
"""vllm-omni t2va per-component breakdown: the reference request through the async /v1/videos API,
which (with return_stage_metrics=true and a server started with --enable-diffusion-pipeline-profiler)
returns the pipeline's stage_durations in the job status. Newer images (nightly 0.32) stopped
logging the profiler lines, so this is the way to read them there.

Usage: req_vomni_async.py OUT_DIR [--n 3] [--warm 0] [--port 8000] [--aspect 16:9]
"""
import argparse, json, os, statistics, time, urllib.request, uuid

PROMPT = ("At night, three cats march into a bedroom playing tiny brass instruments, then abruptly "
          "file out, with synchronized room ambience.")


def multipart(fields):
    b = uuid.uuid4().hex
    body = b"".join(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
                    for k, v in fields.items()) + f"--{b}--\r\n".encode()
    return body, f"multipart/form-data; boundary={b}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--warm", type=int, default=0)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--aspect", default="16:9", help="'' to omit (old minimax-h3 image)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    base = f"http://127.0.0.1:{a.port}/v1/videos"
    fields = {"prompt": PROMPT, "width": 1344, "height": 768, "fps": 24, "num_inference_steps": 50,
              "flow_shift": 12, "seed": 1101, "return_stage_metrics": "true",
              "extra_params": json.dumps({"task": "t2va", "duration": 5.0, "audio_flow_shift": 3.0})}
    if a.aspect:
        fields["aspect_ratio"] = a.aspect
    runs = []
    for i in range(a.warm + a.n):
        body, ctype = multipart(fields)
        t0 = time.time()
        req = urllib.request.Request(base, data=body, headers={"Content-Type": ctype})
        vid = json.loads(urllib.request.urlopen(req).read())["id"]
        while True:
            st = json.loads(urllib.request.urlopen(f"{base}/{vid}").read())
            if st["status"] in ("completed", "failed"):
                break
            time.sleep(0.2)
        wall = time.time() - t0
        if st["status"] == "failed":
            raise SystemExit(f"FAILED: {st.get('error')}")
        sd = {k.replace("MiniMaxH3Pipeline.", ""): v for k, v in (st.get("stage_durations") or {}).items()}
        gen = st["metrics"]["stage_metrics"]["0"]["stage_gen_time_ms"] / 1000
        row = dict(kind="warmup" if i < a.warm else "run", i=i, wall_s=wall,
                   inference_time_s=st.get("inference_time_s"), stage_gen_s=gen, stage_durations=sd)
        open(f"{a.out}/async_requests.jsonl", "a").write(json.dumps(row) + "\n")
        print(f"{row['kind']}{i}: wall={wall:.2f}s inference={st.get('inference_time_s'):.2f}s gen={gen:.2f}s "
              + " ".join(f"{k}={v:.3f}" for k, v in sd.items()), flush=True)
        if i >= a.warm:
            runs.append(row)
    keys = sorted({k for r in runs for k in r["stage_durations"]})
    print("MEAN " + " ".join(f"{k}={statistics.mean(r['stage_durations'].get(k, 0) for r in runs):.3f}"
                             for k in keys)
          + f" inference={statistics.mean(r['inference_time_s'] for r in runs):.2f}"
          + f" wall={statistics.mean(r['wall_s'] for r in runs):.2f}")


if __name__ == "__main__":
    main()
