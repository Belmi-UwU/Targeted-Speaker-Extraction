"""Streaming live demo + a browser view of the input and output spectrograms in real time.

Runs live_demo_streaming.py unchanged (same options), taps the audio going into
and coming out of the extractor, and serves a local page that draws two
scrolling spectrograms: microphone input on top, extracted output below.

  .venv/bin/python live_demo_visual.py --model retrained_finetune_research --input-device 3 --output-device 2
  .venv/bin/python live_demo_visual.py --model retrained_finetune_research --input-file ../data/user_recordings/mixture/Mix.wav

Extra option: --port (default 8765). The page opens automatically at
http://127.0.0.1:8765. It is served only on this machine, and nothing is saved.
"""

import base64
import json
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

HERE = Path(__file__).resolve().parent
RATE, N_FFT, HOP, ROWS = 16000, 512, 256, 128
KEEP_SECONDS, MAX_COLUMNS = 30, 400
PAGE = HERE / "live_spectrum.html"


class Tap:
    """Recent input/output audio with absolute sample positions, turned into dB spectrogram columns on request."""

    def __init__(self):
        self.lock = threading.Lock()
        self.audio = {"in": np.zeros(0, dtype="float32"), "out": np.zeros(0, dtype="float32")}
        self.start = {"in": 0, "out": 0}
        self.window = np.hanning(N_FFT + 1)[:-1].astype("float32")
        self.scale = 2.0 / self.window.sum()  # full-scale sine -> 0 dB

    def append(self, kind, samples):
        with self.lock:
            buf = np.concatenate([self.audio[kind], np.asarray(samples, dtype="float32")])
            excess = len(buf) - KEEP_SECONDS * RATE
            if excess > 0:
                buf, self.start[kind] = buf[excess:], self.start[kind] + excess
            self.audio[kind] = buf

    def columns(self, kind, since):
        with self.lock:
            buf, start = self.audio[kind], self.start[kind]
        end = start + len(buf)
        if since < 0 or since < start:  # first request or fell behind: start 5 s back
            since = max(start, end - 5 * RATE)
            since -= (since - start) % HOP
        count = min(MAX_COLUMNS, max(0, (end - since - N_FFT) // HOP + 1))
        if count == 0:
            return "", 0, since
        offset = since - start
        frames = np.stack([buf[offset + i * HOP: offset + i * HOP + N_FFT] for i in range(count)])
        mag = np.abs(np.fft.rfft(frames * self.window, axis=1))[:, :ROWS * 2] * self.scale
        mag = mag.reshape(count, ROWS, 2).mean(axis=2)
        db = 20 * np.log10(mag + 1e-9)
        q = np.clip((db + 120.0) * 255.0 / 120.0, 0, 255).astype("uint8")  # -120..0 dBFS -> 0..255
        return base64.b64encode(q.tobytes()).decode(), count, since + count * HOP

    def level(self, kind, seconds=0.1):
        with self.lock:
            tail = self.audio[kind][-int(seconds * RATE):]
        return float(20 * np.log10(np.sqrt(np.mean(tail.astype("float64") ** 2)) + 1e-9)) if len(tail) else -120.0


TAP = Tap()
STATE = {"adapter": None, "model": None, "started": time.time()}


def status():
    adapter = STATE["adapter"]
    if adapter is None:
        return {"phase": "discovering / choosing a speaker (see the terminal)"}
    compute = np.array(adapter.compute[-200:]) if adapter.compute else np.zeros(0)
    return {"phase": "live", "model": STATE["model"], "speaker": adapter.target_index + 1,
            "speakers": len(adapter.enrollments), "bypass": adapter.bypass,
            "chunk_ms": round(adapter.chunk / RATE * 1000),
            "rtf": round(float(np.median(compute)) * RATE / adapter.chunk, 2) if len(compute) else None,
            "level_in": round(TAP.level("in"), 1), "level_out": round(TAP.level("out"), 1)}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep the terminal for the demo
        pass

    def send(self, body, kind):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            self.send(PAGE.read_text(), "text/html; charset=utf-8")
        elif url.path == "/data":
            query = parse_qs(url.query)
            reply = {"rows": ROWS, "hop_ms": HOP / RATE * 1000, "status": status()}
            for kind in ("in", "out"):
                cols, count, cursor = TAP.columns(kind, int(query.get(kind, ["-1"])[0]))
                reply[kind] = {"cols": cols, "count": count, "next": cursor}
            self.send(json.dumps(reply), "application/json")
        else:
            self.send_response(404)
            self.end_headers()


def main():
    port = 8765
    if "--port" in sys.argv:
        i = sys.argv.index("--port")
        port = int(sys.argv[i + 1])
        del sys.argv[i:i + 2]
    sys.path.insert(0, str(HERE))
    import live_demo_streaming as streaming

    class VisualAdapter(streaming.StreamingAdapter):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            STATE["adapter"] = self

        def push(self, samples):
            TAP.append("in", samples)
            ready = super().push(samples)
            for out in ready:
                TAP.append("out", out)
            return ready

    streaming.StreamingAdapter = VisualAdapter  # the demo builds its extractor through this name
    STATE["model"] = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "tfmap_context_causal_100"
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{port}"
    print(f"Spectrum view: {url}", flush=True)
    webbrowser.open(url)
    try:
        streaming.main()
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
