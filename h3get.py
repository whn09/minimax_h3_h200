#!/usr/bin/env python3
"""One-GET-URL front end for MiniMax-H3: paste a URL in a browser, get the mp4 back.

sglang's video API is POST-only (`POST /v1/videos` -> poll `GET /v1/videos/{id}` -> fetch
`GET /v1/videos/{id}/content`), so no single GET URL can generate anything. This is a sidecar that
collapses those three calls into one GET and streams the finished mp4 back inline, so a browser
plays it directly.

Deliberately a sidecar rather than a patch to sglang: it survives image upgrades, and it can route
by task across the two replicas that all three modes require (t2va/fl2va live in the fl2va weight
partition, ref2va in its own).

Start it:
    python3 h3get.py                                    # one replica on :30010, listen on :8080
    python3 h3get.py --ref2va-port 30030                # two replicas: t2va/fl2va + ref2va
    python3 h3get.py --listen 0.0.0.0 --port 8080       # reachable from outside the box

Then just open a URL:
    http://<host>:8080/gen?prompt=three+cats+playing+brass+instruments
    http://<host>:8080/gen?prompt=a+dog+surfing&width=864&height=480&steps=16&duration=10
    http://<host>:8080/gen?prompt=neon+city&short_edge=480&aspect=21:9&steps=20
    http://<host>:8080/gen?task=fl2va&image=/out/first.png&prompt=it+starts+moving
    http://<host>:8080/gen?task=ref2va&ref_video=/out/ref.mp4&prompt=same+scene+at+night

Every geometry/step/duration parameter the customer asked for is a query parameter, and the wire
form is chosen by the same ladder h3gen.py uses -- this imports it rather than restating it, so the
two can never disagree.

Add &json=1 to get the job metadata instead of the video bytes, and &download=1 to force a file
download instead of inline playback.
"""
import argparse, json, sys, time, urllib.error, urllib.parse, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import h3gen

USAGE = """MiniMax-H3 one-URL front end

  /gen?prompt=...            generate and return the mp4

geometry -- pick ONE group (they are mutually exclusive, as on the server):
  width=&height=             exact canvas, each a multiple of 32
  short_edge=&aspect=        e.g. short_edge=480&aspect=16:9, or aspect=auto

other parameters:
  task=t2va|fl2va|ref2va     default t2va
  steps=N                    denoise steps, default 20
  duration=S                 clip seconds, 4..15, default 5
  seed=N                     default 1101
  image=PATH                 fl2va first frame, or ref2va subject image
  last_image=PATH            fl2va last frame
  ref_video=PATH             ref2va reference video
  ref_audio=PATH             ref2va reference audio
  wire=auto|ratio|literal|exact   how width/height is expressed on the wire
  json=1                     return job metadata instead of the video
  download=1                 force download instead of inline playback

Condition values are forwarded as the `uri` the SERVER resolves, so a plain path must exist inside
its container (e.g. /out/first.png). An `http(s)://` URL works too and is fetched by the server --
prefer that here, since a GET query string is the wrong place for a megabyte of base64. To send
local bytes, use `h3gen.py --inline` instead.
"""

# Query parameters that are forwarded as-is to h3gen's geometry/conditions logic. Anything else in
# the query string is rejected rather than ignored, so a typo'd parameter is never silently dropped.
STR_PARAMS = ("prompt", "task", "aspect", "wire", "image", "last_image", "ref_video", "ref_audio")
INT_PARAMS = ("width", "height", "short_edge", "steps", "seed")
FLOAT_PARAMS = ("duration", "flow_shift", "audio_flow_shift")
CONTROL_PARAMS = ("json", "download")


class Bad(Exception):
    """A client error: reported as 400 with the message, not as a traceback."""


def as_query_names(message):
    """Rewrite h3gen's CLI flag names into the query-parameter names a URL user actually typed.

    h3gen is a CLI, so its errors say `--last-image`; a browser user has no such flag. Only the
    spelling changes -- the message itself is reused verbatim so the two front ends never drift.
    """
    for flag in sorted(STR_PARAMS + INT_PARAMS + FLOAT_PARAMS, key=len, reverse=True):
        message = message.replace("--" + flag.replace("_", "-"), flag)
    return message


def build_args(query):
    unknown = set(query) - set(STR_PARAMS + INT_PARAMS + FLOAT_PARAMS + CONTROL_PARAMS)
    if unknown:
        raise Bad(f"unknown parameter(s): {sorted(unknown)}\n\n{USAGE}")
    # The two geometry groups are mutually exclusive, as the customer required and as the patched
    # server enforces. Checked HERE on presence in the query string, because h3gen.geometry() only
    # sees defaults and cannot tell "short_edge was asked for" from "short_edge defaulted to 480" --
    # so without this, width+short_edge would silently win on width and drop short_edge.
    explicit = {"width", "height"} & set(query), {"short_edge", "aspect"} & set(query)
    if all(explicit):
        raise Bad(f"pick ONE geometry group: {sorted(explicit[0])} or {sorted(explicit[1])}, "
                  f"not both\n\n{USAGE}")
    a = SimpleNamespace(
        task="t2va", prompt=h3gen.DEFAULT_PROMPT, width=None, height=None,
        short_edge=480, aspect="16:9", wire="auto", steps=20, duration=5.0, seed=1101,
        flow_shift=12.0, audio_flow_shift=3.0,
        image=None, last_image=None, ref_video=None, ref_audio=None,
    )
    for name in STR_PARAMS:
        if name in query:
            setattr(a, name, query[name][0])
    for name, cast in [(n, int) for n in INT_PARAMS] + [(n, float) for n in FLOAT_PARAMS]:
        if name in query:
            try:
                setattr(a, name, cast(query[name][0]))
            except ValueError:
                raise Bad(f"{name}={query[name][0]!r} is not a valid {cast.__name__}")
    if a.task not in ("t2va", "fl2va", "ref2va"):
        raise Bad(f"task must be t2va, fl2va or ref2va, got {a.task!r}")
    return a


def generate(a, ports, timeout):
    """POST, poll, fetch. Returns (mp4_bytes, status_json, asked_wh, wire)."""
    port = ports.get(a.task) or ports["default"]
    # h3gen.geometry/conditions call sys.exit on bad input; turn that into a 400 instead of killing
    # the sidecar, which would otherwise take down the whole endpoint on one malformed URL.
    try:
        target, pw, ph, wire = h3gen.geometry(a)
        conds = h3gen.conditions(a)
    except SystemExit as exc:
        raise Bad(as_query_names(str(exc)))

    derive_duration = a.task == "ref2va" and (a.ref_video or a.ref_audio) and not a.image
    if not derive_duration:
        if not 4.0 <= a.duration <= 15.0:
            raise Bad(f"duration must be within the model's 4..15 s range, got {a.duration}")
        target["duration_seconds"] = a.duration
    payload = {
        "model": "MiniMax-H3", "prompt": a.prompt, "task": a.task, "conditions": conds,
        "target": target, "num_outputs_per_prompt": 1, "num_inference_steps": a.steps,
        "flow_shift": a.flow_shift, "audio_flow_shift": a.audio_flow_shift, "seed": a.seed,
    }
    if not derive_duration:
        payload["seconds"] = int(a.duration)

    base = f"http://127.0.0.1:{port}/v1/videos"
    try:
        req = urllib.request.Request(base, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            job = json.loads(r.read())
    except urllib.error.HTTPError as exc:
        raise Bad(f"server rejected the request (HTTP {exc.code}): {exc.read().decode()[:1000]}")
    except urllib.error.URLError as exc:
        raise Bad(f"cannot reach the {a.task} replica on port {port}: {exc}")

    vid = job["id"]
    deadline = time.time() + timeout
    while True:
        with urllib.request.urlopen(f"{base}/{vid}") as r:
            st = json.loads(r.read())
        if st.get("status") == "completed":
            break
        if st.get("status") == "failed":
            raise Bad(f"generation failed: {json.dumps(st)[:1000]}")
        if time.time() > deadline:
            raise Bad(f"timed out after {timeout}s waiting for job {vid}")
        time.sleep(0.5)
    with urllib.request.urlopen(f"{base}/{vid}/content") as r:
        return r.read(), st, (pw, ph), wire


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    ports = {}
    timeout_s = 900

    def _send(self, code, body, ctype, extra=None):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ("/", "/help"):
            return self._send(200, USAGE, "text/plain; charset=utf-8")
        if parsed.path != "/gen":
            return self._send(404, f"no such path {parsed.path!r}\n\n{USAGE}",
                              "text/plain; charset=utf-8")
        query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
        t0 = time.time()
        try:
            a = build_args(query)
            mp4, st, (pw, ph), wire = generate(a, self.ports, self.timeout_s)
        except Bad as exc:
            return self._send(400, str(exc) + "\n", "text/plain; charset=utf-8")
        except Exception as exc:  # never let one bad request kill the endpoint
            return self._send(500, f"{type(exc).__name__}: {exc}\n", "text/plain; charset=utf-8")
        wall = time.time() - t0
        if "json" in query:
            st["_sidecar"] = {"wall_seconds": round(wall, 2), "asked": f"{pw}x{ph}",
                              "wire": wire, "task": a.task, "steps": a.steps}
            return self._send(200, json.dumps(st, indent=2), "application/json")
        name = f"h3_{a.task}_{pw}x{ph}_{a.steps}st.mp4"
        disp = "attachment" if "download" in query else "inline"
        self._send(200, mp4, "video/mp4", {
            "Content-Disposition": f'{disp}; filename="{name}"',
            # Surfaced as headers so a browser's network tab shows what was actually generated,
            # including which wire form the geometry was expressed as.
            "X-H3-Wall-Seconds": f"{wall:.2f}", "X-H3-Asked": f"{pw}x{ph}", "X-H3-Wire": wire,
        })

    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (self.log_date_time_string(), fmt % args))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--listen", default="127.0.0.1",
                   help="bind address; use 0.0.0.0 to reach it from outside the box")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--sglang-port", type=int, default=30010,
                   help="replica serving t2va and fl2va")
    p.add_argument("--ref2va-port", type=int,
                   help="replica started with --model-variant ref2va; without this, ref2va "
                        "requests go to --sglang-port and the server will refuse them")
    p.add_argument("--timeout", type=int, default=900, help="seconds to wait for one generation")
    a = p.parse_args()

    Handler.ports = {"default": a.sglang_port, "t2va": a.sglang_port, "fl2va": a.sglang_port,
                     "ref2va": a.ref2va_port or a.sglang_port}
    Handler.timeout_s = a.timeout
    shown = a.listen if a.listen != "0.0.0.0" else "<this-host>"
    print(f"listening on http://{a.listen}:{a.port}  ->  t2va/fl2va :{a.sglang_port}, "
          f"ref2va :{Handler.ports['ref2va']}")
    print(f"try: http://{shown}:{a.port}/gen?prompt=three+cats+playing+brass+instruments"
          f"&width=864&height=480&steps=16&duration=10")
    ThreadingHTTPServer((a.listen, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
