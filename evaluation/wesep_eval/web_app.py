"""Web control panel for the live target-speaker demo: everything from the browser.

In the page you can:
1. pick the microphone, headphones, and model;
2. record a stretch of conversation, or use a file, for discovery;
3. listen to each discovered speaker and pick one;
4. start and stop live extraction, toggle bypass, and switch speakers;
5. watch live spectrograms of the input and the output.

The extractor is the streaming WeSep model (live_demo_streaming.py, ~0.17 s delay). Discovery is SepFormer + ECAPA
(live_demo.py).

    .venv/bin/python web_app.py            # opens http://127.0.0.1:8770
    .venv/bin/python web_app.py --port 9000 --no-browser

Served only on this machine (127.0.0.1). Nothing is saved: recordings and discovered voices live in memory until you
quit. Use headphones.
"""

import argparse
import io
import json
import queue
import sys
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import soundfile as sf
import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "upstream/wesep-real-tse"))

import live_demo as base  # noqa: E402
from live_demo_streaming import StreamingAdapter  # noqa: E402
from live_demo_visual import Tap  # noqa: E402
from run_wesep_experiment import enrollment, load  # noqa: E402

RATE = base.RATE
CHECKPOINTS = HERE / "checkpoints/pretrained"
PAGE = HERE / "web_app.html"
DEFAULT_MIX = HERE.parent / "data/user_recordings/mixture/Mix.wav"


def causal_models():
    """Checkpoint folders whose separator is causal (streaming needs that), newest first."""
    names = []
    for folder in CHECKPOINTS.iterdir() if CHECKPOINTS.exists() else []:
        config = folder / "config.yaml"
        if config.exists() and (folder / "avg_model.pt").exists():
            separator = yaml.safe_load(config.read_text())["model_args"]["tse_model"]["separator"]
            if separator.get("causal"):
                names.append((folder.stat().st_mtime, folder.name))
    return [name for _, name in sorted(names, reverse=True)]


class TappedAdapter(StreamingAdapter):
    """StreamingAdapter that also feeds the spectrogram tap."""

    def __init__(self, tap, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tap = tap

    def push(self, samples):
        self.tap.append("in", samples)
        ready = super().push(samples)
        for out in ready:
            self.tap.append("out", out)
        return ready


class LiveEngine:
    """Microphone (or a file played in real time) -> extractor -> headphones, without a terminal."""

    def __init__(self, sd, extractor, input_device, output_device, margin, file_audio=None):
        self.sd, self.extractor, self.margin = sd, extractor, margin
        self.input_device, self.output_device, self.file_audio = input_device, output_device, file_audio
        self.running = threading.Event()
        self.inputs = queue.Queue()
        self.streams = []

    def start(self):
        out_rate = int(self.sd.query_devices(self.output_device, "output")["default_samplerate"])
        self.to_out = base.Resampler(RATE, out_rate)
        self.buffer = base.OutputBuffer()
        self.buffer.write(np.zeros(int((self.extractor.block / RATE + self.margin) * out_rate), dtype="float32"))
        self.running.set()
        self.streams = [self.sd.OutputStream(samplerate=out_rate, channels=1, dtype="float32",
                                             device=self.output_device, callback=self._output)]
        if self.file_audio is None:
            in_rate = int(self.sd.query_devices(self.input_device, "input")["default_samplerate"])
            self.to_16k = base.Resampler(in_rate, RATE)
            self.streams.append(self.sd.InputStream(samplerate=in_rate, channels=1, dtype="float32",
                                                    device=self.input_device, callback=self._input,
                                                    blocksize=int(base.CHUNK_SECONDS * in_rate)))
        else:
            threading.Thread(target=self._feed_file, daemon=True).start()
        threading.Thread(target=self._work, daemon=True).start()
        for stream in self.streams:
            stream.start()

    def _input(self, indata, frames, time_info, status):
        self.inputs.put(self.to_16k(indata[:, 0].copy()))

    def _output(self, outdata, frames, time_info, status):
        outdata[:, 0] = self.buffer.read(frames)

    def _feed_file(self):
        step, position = int(base.CHUNK_SECONDS * RATE), 0
        begin = time.perf_counter()
        while self.running.is_set():  # loops the file so the demo keeps going
            chunk = self.file_audio[position % len(self.file_audio):][:step]
            self.inputs.put(chunk)
            position += len(chunk)
            delay = begin + position / RATE - time.perf_counter()
            if delay > 0:
                time.sleep(delay)

    def _work(self):
        while self.running.is_set():
            try:
                chunk = self.inputs.get(timeout=0.1)
            except queue.Empty:
                continue
            for out in self.extractor.push(chunk):
                self.buffer.write(self.to_out(out))

    def stop(self):
        self.running.clear()
        for stream in self.streams:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass


class App:
    def __init__(self, sd):
        self.sd = sd
        self.lock = threading.RLock()
        self.phase, self.message = "starting", "Loading the discovery models (first time can take ~30 s)…"
        self.recording = None
        self.speakers, self.engine, self.extractor = [], None, None
        self.loaded = {}
        self.live_settings = {}
        self.tap = Tap()
        self.encoders = None
        threading.Thread(target=self._warm_up, daemon=True).start()

    # ---- background helpers
    def _guard(self, work, *args):
        try:
            work(*args)
        except Exception as error:
            traceback.print_exc()
            with self.lock:
                self.phase, self.message = "error", f"{type(error).__name__}: {error}"

    def _warm_up(self):
        def work():
            self.encoders = base.dp.Models()
            self.encoders.sepformer, self.encoders.ecapa  # load both now, not on the first discovery
            with self.lock:
                self.phase, self.message = "idle", "Ready. Pick devices, then record a conversation for discovery."
        self._guard(work)

    def devices(self):
        listed = self.sd.query_devices()
        default_in, default_out = (list(self.sd.default.device) + [None, None])[:2]
        def entries(key, default):
            return [{"index": i, "name": d["name"], "default": i == default}
                    for i, d in enumerate(listed) if d[key] > 0]
        return {"inputs": entries("max_input_channels", default_in),
                "outputs": entries("max_output_channels", default_out)}

    # ---- discovery
    def record(self, seconds, device):
        with self.lock:
            if self.phase not in ("idle", "ready", "error"):
                raise RuntimeError(f"busy ({self.phase})")
            self.stop_live()
            rate = int(self.sd.query_devices(device, "input")["default_samplerate"])
            self.recording = {"chunks": [], "rate": rate, "total": float(seconds), "started": time.time(),
                              "stop": threading.Event(), "level": -120.0}
            self.phase, self.message = "recording", "Recording: let everyone talk, ideally taking turns."
        def callback(indata, frames, time_info, status):
            chunk = indata[:, 0].copy()
            self.recording["chunks"].append(chunk)
            self.recording["level"] = float(20 * np.log10(np.sqrt(np.mean(chunk ** 2)) + 1e-9))
        def work():
            stream = self.sd.InputStream(samplerate=rate, channels=1, dtype="float32", device=device,
                                         callback=callback, blocksize=int(0.05 * rate))
            stream.start()
            rec = self.recording
            while not rec["stop"].is_set() and time.time() - rec["started"] < rec["total"]:
                time.sleep(0.05)
            stream.stop()
            stream.close()
            audio = np.concatenate(rec["chunks"]) if rec["chunks"] else np.zeros(0, dtype="float32")
            self._discover(base.to_mono_16k(audio, rate))
        threading.Thread(target=self._guard, args=(work,), daemon=True).start()

    def stop_recording(self):
        if self.recording:
            self.recording["stop"].set()

    def discover_file(self, path):
        audio = base.read_any(Path(path).expanduser())
        with self.lock:
            if self.phase not in ("idle", "ready", "error"):
                raise RuntimeError(f"busy ({self.phase})")
            self.stop_live()
        threading.Thread(target=self._guard, args=(self._discover, audio), daemon=True).start()

    def _discover(self, audio):
        with self.lock:
            self.phase = "discovering"
            self.message = f"Finding the speakers in {len(audio) / RATE:.0f} s of audio (takes about as long)…"
            self.discovery_audio = audio
        if len(audio) < 4 * RATE:
            raise RuntimeError("recording too short; record at least ~15 s")
        speakers = base.discover_speakers(self.encoders, audio, base.settings())
        with self.lock:
            self.speakers = speakers
            if speakers:
                self.phase = "ready"
                self.message = f"Found {len(speakers)} speaker(s). Listen to each, then start live with one."
            else:
                self.phase = "idle"
                self.message = "No speakers found. Record longer, with each person talking alone for ~5 s."

    def speaker_wav(self, number):
        track = self.speakers[number - 1]["track"]
        speech = base.dp.active_speech(track)
        clip = (speech if len(speech) > RATE else track)[:12 * RATE]
        peak = float(np.max(np.abs(clip))) if len(clip) else 0.0
        out = io.BytesIO()
        sf.write(out, clip * (0.9 / peak if peak > 1e-9 else 1.0), RATE, subtype="PCM_16", format="WAV")
        return out.getvalue()

    # ---- live
    def start_live(self, settings):
        with self.lock:
            if not self.speakers:
                raise RuntimeError("discover speakers first")
            if self.phase not in ("ready", "live", "error"):
                raise RuntimeError(f"busy ({self.phase})")
            self.stop_live()
            self.phase, self.message = "loading", f"Loading {settings['model']}…"
        def work():
            name = settings["model"]
            if name not in self.loaded:
                self.loaded[name] = load(name).model
            torch.set_num_threads(int(settings.get("threads", 2)))
            enrollments = [enrollment(s["track"]) for s in self.speakers]
            self.extractor = TappedAdapter(self.tap, self.loaded[name], enrollments, int(settings["speaker"]) - 1,
                                           float(settings["chunk"]), "cpu")
            file_audio = base.read_any(Path(settings["file"]).expanduser()) if settings.get("source") == "file" else None
            engine = LiveEngine(self.sd, self.extractor, settings.get("input"), settings.get("output"),
                                float(settings["margin"]), file_audio)
            engine.start()
            with self.lock:
                self.engine, self.live_settings = engine, settings
                self.phase = "live"
                self.message = "Live. Use headphones; toggle bypass to compare with the unprocessed sound."
        threading.Thread(target=self._guard, args=(work,), daemon=True).start()

    def stop_live(self):
        with self.lock:
            if self.engine:
                self.engine.stop()
                self.engine = None
                if self.phase == "live":
                    self.phase, self.message = "ready", "Stopped. Start again, or record a new discovery."

    def set_bypass(self, on):
        if self.extractor:
            self.extractor.bypass = bool(on)

    def switch(self, number):
        if self.extractor and 1 <= number <= len(self.speakers):
            self.extractor.target_index = number - 1

    def state(self):
        with self.lock:
            rec = self.recording if self.phase == "recording" else None
            state = {"phase": self.phase, "message": self.message, "models": causal_models(),
                     "default_file": str(DEFAULT_MIX) if DEFAULT_MIX.exists() else "",
                     "speakers": [{"number": i + 1, "windows": s["windows"]} for i, s in enumerate(self.speakers)],
                     "recording": None, "live": None}
            if rec:
                state["recording"] = {"elapsed": round(time.time() - rec["started"], 1), "total": rec["total"],
                                      "level": round(rec["level"], 1)}
            if self.phase == "live" and self.engine and self.extractor:
                ex, s = self.extractor, self.live_settings
                compute = np.array(ex.compute[-200:]) if ex.compute else np.zeros(0)
                state["live"] = {
                    "speaker": ex.target_index + 1, "bypass": ex.bypass, "model": s["model"],
                    "source": s.get("source", "mic"), "chunk_ms": round(ex.chunk / RATE * 1000),
                    "delay_ms": round((ex.chunk / RATE + float(s["margin"])) * 1000),
                    "rtf": round(float(np.median(compute)) * RATE / ex.chunk, 2) if len(compute) else None,
                    "underruns": self.engine.buffer.underruns,
                    "level_in": round(self.tap.level("in"), 1), "level_out": round(self.tap.level("out"), 1)}
            return state


APP = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, body, kind="application/json", status=200):
        data = body if isinstance(body, bytes) else (body if isinstance(body, str) else json.dumps(body)).encode()
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        try:
            if url.path == "/":
                self.reply(PAGE.read_text(), "text/html; charset=utf-8")
            elif url.path == "/api/state":
                self.reply(APP.state())
            elif url.path == "/api/devices":
                self.reply(APP.devices())
            elif url.path.startswith("/api/speaker/") and url.path.endswith(".wav"):
                self.reply(APP.speaker_wav(int(url.path.split("/")[-1][:-4])), "audio/wav")
            elif url.path == "/data":
                query = parse_qs(url.query)
                out = {"rows": 128, "hop_ms": 16.0}
                for kind in ("in", "out"):
                    cols, count, cursor = APP.tap.columns(kind, int(query.get(kind, ["-1"])[0]))
                    out[kind] = {"cols": cols, "count": count, "next": cursor}
                self.reply(out)
            else:
                self.reply({"error": "not found"}, status=404)
        except Exception as error:
            self.reply({"error": f"{type(error).__name__}: {error}"}, status=400)

    def do_POST(self):
        url = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        try:
            if url.path == "/api/record":
                APP.record(float(body.get("seconds", 30)), body.get("input"))
            elif url.path == "/api/stop_record":
                APP.stop_recording()
            elif url.path == "/api/discover_file":
                APP.discover_file(body["path"])
            elif url.path == "/api/start":
                APP.start_live(body)
            elif url.path == "/api/stop":
                APP.stop_live()
            elif url.path == "/api/bypass":
                APP.set_bypass(body.get("on"))
            elif url.path == "/api/switch":
                APP.switch(int(body["speaker"]))
            else:
                return self.reply({"error": "not found"}, status=404)
            self.reply({"ok": True, "state": APP.state()})
        except Exception as error:
            self.reply({"error": f"{type(error).__name__}: {error}"}, status=400)


def serve(sd, port, open_browser=True):
    global APP
    APP = App(sd)
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    print(f"Control panel: {url}  (Ctrl+C to quit)", flush=True)
    if open_browser:
        webbrowser.open(url)
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    server = serve(base.sd_module(), args.port, not args.no_browser)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if APP:
            APP.stop_live()
        server.server_close()


if __name__ == "__main__":
    main()
