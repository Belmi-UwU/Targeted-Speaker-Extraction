"""Live demo: discover the people talking around you, pick one, hear only them.

Flow (PRD next phase, offline research models):
  1. Discovery: record a stretch of conversation from the microphone (or read
     a file), separate it with windowed SepFormer, and group the pieces into
     one track per speaker with ECAPA.
  2. Choose: listen to each discovered speaker and pick one.
  3. Live: microphone -> 1 s blocks with 2 s of past context -> SepFormer ->
     keep the stream whose closest discovered speaker is your pick ->
     20 ms crossfade -> headphones. No future audio is used; total delay is
     about block + safety margin (default about 2.2 s).

Modes:
  python live_demo.py                                  # mic discovery + live mic (use headphones!)
  python live_demo.py --discover-file Mix.wav          # discover from a file, live from mic
  python live_demo.py --input-file Mix.wav             # simulate live: stream a file in real time to headphones
  python live_demo.py --input-file Mix.wav --output-file out.wav --choose 1   # offline test, no audio devices

Live audio is not saved unless --save is given (PRD F8).
"""

import argparse
import json
import queue
import sys
import threading
import time
from collections import deque
from datetime import datetime
from math import gcd
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf
import torch

import discovery_pipeline as dp

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "data/discovery/results"
SESSIONS = ROOT / "data/user_recordings/live_sessions"
RATE = dp.RATE
CHUNK_SECONDS = 0.02


def settings():
    """Thresholds tuned on the synthetic dev set; fall back to the reported values."""
    params = {"distance_threshold": 0.6, "min_windows": 3, "floor": 0.4}
    try:
        config = json.loads((RESULTS / "config_and_tuning.json").read_text())
        chosen = config["discovery_tuning"]["ecapa"]["chosen"]
        params.update(distance_threshold=chosen["distance_threshold"], min_windows=chosen["min_windows"])
        online = json.loads((RESULTS / "block_online_summary.json").read_text())["configs"]
        params["floor"] = online["1.0_2.0"]["thresholds"]["sepformer_competitive"]["discovered"]
    except (OSError, KeyError, json.JSONDecodeError):
        pass
    return params


def to_mono_16k(wave, rate):
    wave = np.asarray(wave, dtype="float32")
    if wave.ndim == 2:
        wave = wave.mean(axis=1)
    if rate != RATE:
        d = gcd(rate, RATE)
        wave = scipy.signal.resample_poly(wave, RATE // d, rate // d).astype("float32")
    return wave


def read_any(path):
    wave, rate = sf.read(path, dtype="float32")
    return to_mono_16k(wave, rate)


def separate_window(models, segment):
    """3-stream SepFormer on one window with mixture-fit gains; kept streams + ECAPA embeddings."""
    if np.sqrt(np.mean(segment ** 2)) < dp.SILENT_WINDOW_RMS:
        return []
    seg8 = scipy.signal.resample_poly(segment, 1, 2).astype("float32")
    with torch.inference_mode():
        estimate = models.sepformer.separate_batch(torch.from_numpy(seg8).unsqueeze(0))
    streams8 = estimate[0].cpu().numpy().astype("float64")[:len(seg8)]
    gains, _, _, _ = np.linalg.lstsq(streams8, seg8.astype("float64"), rcond=None)
    fitted = streams8 * gains
    energy = float(np.sum(segment.astype("float64") ** 2))
    kept = []
    for column in range(fitted.shape[1]):
        wave = scipy.signal.resample_poly(fitted[:, column], 2, 1)[:len(segment)]
        wave = np.pad(wave, (0, len(segment) - len(wave))).astype("float32")
        relative = 10 * np.log10(max(float(np.sum(wave.astype("float64") ** 2)), 1e-20) / max(energy, 1e-20))
        if relative >= dp.STREAM_FLOOR_DB:
            kept.append((wave, models.embed("ecapa", wave)))
    return kept


class LiveExtractor:
    """Streaming block-online competitive selector with crossfade (no future audio)."""

    def __init__(self, models, candidates, target_index, floor, block_s=1.0, window_s=2.0, fade_s=0.02):
        self.models, self.known, self.floor = models, np.stack(candidates), floor
        self.target_index = target_index
        self.block, self.window = int(block_s * RATE), int(window_s * RATE)
        self.fade = int(fade_s * RATE)
        self.ease_in = (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, self.fade))).astype("float32")
        self.history = np.zeros(self.window, dtype="float32")
        self.pending = np.zeros(0, dtype="float32")
        self.held = None  # last fade samples of the previous block, not yet played
        self.bypass = False
        self.compute = []

    def push(self, samples):
        """Feed 16 kHz input; returns the output chunks that became ready."""
        self.pending = np.concatenate([self.pending, np.asarray(samples, dtype="float32")])
        ready = []
        while len(self.pending) >= self.block:
            block, self.pending = self.pending[:self.block], self.pending[self.block:]
            self.history = np.concatenate([self.history, block])[-self.window:]
            ready.append(self._process())
        return ready

    def _process(self):
        start = time.perf_counter()
        span = self.block + (self.fade if self.held is not None else 0)
        if self.bypass:
            piece = self.history[-span:].copy()
        else:
            chosen = []
            for wave, embedding in separate_window(self.models, self.history):
                if embedding is None:
                    continue
                sims = self.known @ embedding
                if int(np.argmax(sims)) == self.target_index and sims[self.target_index] >= self.floor:
                    chosen.append(wave[-span:])
            piece = np.sum(chosen, axis=0).astype("float32") if chosen else np.zeros(span, dtype="float32")
        if self.held is not None:
            blended = self.held * (1 - self.ease_in) + piece[:self.fade] * self.ease_in
            out = np.concatenate([blended, piece[self.fade:-self.fade]])
        else:
            out = piece[:-self.fade]
        self.held = piece[-self.fade:]
        self.compute.append(time.perf_counter() - start)
        return out

    def flush(self):
        return self.held if self.held is not None else np.zeros(0, dtype="float32")

    def stats(self):
        if not self.compute:
            return "no blocks yet"
        c = np.array(self.compute)
        return (f"blocks {len(c)} | compute median {np.median(c):.2f} s, p90 {np.percentile(c, 90):.2f} s "
                f"per {self.block / RATE:.1f} s block (real-time factor {np.median(c) * RATE / self.block:.2f})")


def discover_speakers(models, audio, params):
    print(f"Discovering speakers in {len(audio) / RATE:.1f} s of audio (takes roughly as long as the audio)...",
          flush=True)
    start = time.perf_counter()
    found, _ = dp.discover(models, audio, "ecapa", params["distance_threshold"], params["min_windows"])
    speakers = []
    for speaker in found:
        embedding = models.embed("ecapa", speaker["track"])
        if embedding is not None:
            speakers.append({"track": speaker["track"], "ecapa": embedding,
                             "windows": len(speaker["windows"])})
    print(f"Found {len(speakers)} speaker(s) in {time.perf_counter() - start:.0f} s.", flush=True)
    return speakers


# ---------------------------------------------------------------- audio devices

def sd_module():
    import sounddevice as sd
    return sd


class Resampler:
    """Streaming resampler between a device rate and 16 kHz."""

    def __init__(self, source, target):
        self.same = source == target
        if not self.same:
            import soxr
            self.stream = soxr.ResampleStream(source, target, 1, dtype="float32")

    def __call__(self, chunk):
        chunk = np.asarray(chunk, dtype="float32")
        return chunk if self.same else self.stream.resample_chunk(chunk)


class OutputBuffer:
    def __init__(self):
        self.chunks, self.lock, self.underruns, self.level = deque(), threading.Lock(), 0, 0

    def write(self, data):
        with self.lock:
            self.chunks.append(np.asarray(data, dtype="float32"))
            self.level += len(data)

    def read(self, frames):
        out = np.zeros(frames, dtype="float32")
        filled = 0
        with self.lock:
            while filled < frames and self.chunks:
                head = self.chunks[0]
                take = min(frames - filled, len(head))
                out[filled:filled + take] = head[:take]
                filled += take
                if take == len(head):
                    self.chunks.popleft()
                else:
                    self.chunks[0] = head[take:]
            self.level -= filled
        if filled < frames:
            self.underruns += 1
        return out


def device_rate(sd, device, kind):
    info = sd.query_devices(device, kind)
    return int(info["default_samplerate"]), info["name"]


def play(sd, wave16, output_device):
    rate, _ = device_rate(sd, output_device, "output")
    sd.play(Resampler(RATE, rate)(wave16), rate, device=output_device)
    sd.wait()


def record(sd, seconds, input_device):
    rate, name = device_rate(sd, input_device, "input")
    print(f"Recording {seconds:.0f} s from '{name}'. Let everyone talk, ideally taking turns...", flush=True)
    audio = sd.rec(int(seconds * rate), samplerate=rate, channels=1, dtype="float32", device=input_device)
    for remaining in range(int(seconds), 0, -5):
        print(f"  {remaining} s left", flush=True)
        time.sleep(min(5, remaining))
    sd.wait()
    return to_mono_16k(audio[:, 0], rate)


def choose_speaker(sd, speakers, output_device):
    print("\nDiscovered speakers:")
    for i, s in enumerate(speakers, 1):
        print(f"  {i}) speaker {i}  (heard in {s['windows']} windows)")
    print("Type a number to listen, 'pick N' to choose speaker N, 'q' to quit.")
    while True:
        answer = input("> ").strip().lower()
        if answer in ("q", "quit"):
            sys.exit(0)
        parts = answer.split()
        try:
            number = int(parts[-1])
        except (ValueError, IndexError):
            print("  Type a number, or 'pick N'.")
            continue
        if not 1 <= number <= len(speakers):
            print(f"  Choose 1-{len(speakers)}.")
            continue
        if parts[0] == "pick":
            return number - 1
        print(f"  Playing speaker {number} (first 8 s)...", flush=True)
        play(sd, speakers[number - 1]["track"][:8 * RATE], output_device)


def run_live(sd, args, extractor, speakers, file_audio):
    """Mic (or a real-time paced file) -> extractor -> headphones."""
    out_rate, out_name = device_rate(sd, args.output_device, "output")
    to_out = Resampler(RATE, out_rate)
    buffer = OutputBuffer()
    # Prefill: block + margin of silence, so a block that takes up to `margin` to process still plays on time.
    buffer.write(np.zeros(int((extractor.block / RATE + args.margin) * out_rate), dtype="float32"))
    inputs = queue.Queue()
    running = threading.Event()
    running.set()
    saved_in, saved_out = [], []

    def worker():
        while running.is_set() or not inputs.empty():
            try:
                chunk = inputs.get(timeout=0.1)
            except queue.Empty:
                continue
            for out in extractor.push(chunk):
                if args.save:
                    saved_out.append(out)
                buffer.write(to_out(out))

    def out_callback(outdata, frames, time_info, status):
        outdata[:, 0] = buffer.read(frames)

    streams = [sd.OutputStream(samplerate=out_rate, channels=1, dtype="float32",
                               device=args.output_device, callback=out_callback)]
    if file_audio is None:
        in_rate, in_name = device_rate(sd, args.input_device, "input")
        to_16k = Resampler(in_rate, RATE)

        def in_callback(indata, frames, time_info, status):
            chunk = to_16k(indata[:, 0].copy())
            if args.save:
                saved_in.append(chunk)
            inputs.put(chunk)

        streams.append(sd.InputStream(samplerate=in_rate, channels=1, dtype="float32",
                                      device=args.input_device, callback=in_callback,
                                      blocksize=int(CHUNK_SECONDS * in_rate)))
        source = f"microphone '{in_name}'"
    else:
        def feeder():
            step = int(CHUNK_SECONDS * RATE)
            begin = time.perf_counter()
            for i in range(0, len(file_audio), step):
                if not running.is_set():
                    return
                inputs.put(file_audio[i:i + step])
                delay = begin + (i + step) / RATE - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            print("\n[end of input file - press Enter to stop]", flush=True)

        threading.Thread(target=feeder, daemon=True).start()
        source = "input file (real-time simulation)"
    threading.Thread(target=worker, daemon=True).start()
    delay = extractor.block / RATE + args.margin + extractor.fade / RATE
    print(f"\nLIVE: {source} -> '{out_name}'. Delay about {delay:.1f} s.")
    print("Commands: Enter = stop | b = toggle bypass (hear unprocessed, same delay) | "
          f"s N = switch to speaker N (1-{len(speakers)})", flush=True)

    def status():
        while running.is_set():
            time.sleep(5)
            if running.is_set():
                print(f"  [{extractor.stats()} | underruns {buffer.underruns} | "
                      f"{'BYPASS' if extractor.bypass else f'speaker {extractor.target_index + 1}'}]",
                      flush=True)

    threading.Thread(target=status, daemon=True).start()
    for stream in streams:
        stream.start()
    try:
        while True:
            command = input().strip().lower()
            if command == "":
                break
            if command == "b":
                extractor.bypass = not extractor.bypass
                print("  bypass ON" if extractor.bypass else "  bypass OFF", flush=True)
            elif command.startswith("s ") and command[2:].strip().isdigit():
                n = int(command[2:])
                if 1 <= n <= len(speakers):
                    extractor.target_index = n - 1
                    print(f"  now extracting speaker {n}", flush=True)
    except (KeyboardInterrupt, EOFError):
        pass
    running.clear()
    for stream in streams:
        stream.stop()
        stream.close()
    print(f"Stopped. {extractor.stats()} | underruns {buffer.underruns}")
    return saved_in, saved_out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--input-device", help="microphone index or name (default: system default)")
    parser.add_argument("--output-device", help="headphones index or name (default: system default)")
    parser.add_argument("--discover-seconds", type=float, default=30.0)
    parser.add_argument("--discover-file", type=Path, help="discover speakers from this file instead of the mic")
    parser.add_argument("--input-file", type=Path, help="use this file as the live input (also used for discovery "
                                                        "unless --discover-file is given)")
    parser.add_argument("--output-file", type=Path, help="offline test: write output here, no audio devices")
    parser.add_argument("--choose", type=int, help="pick discovered speaker N without asking")
    parser.add_argument("--margin", type=float, default=1.2,
                        help="extra playback delay in seconds that absorbs slow blocks (default 1.2)")
    parser.add_argument("--save", action="store_true", help="save discovery, chosen track and live audio")
    args = parser.parse_args()
    for name in ("input_device", "output_device"):
        value = getattr(args, name)
        if value is not None and value.isdigit():
            setattr(args, name, int(value))

    offline = args.output_file is not None
    sd = None if offline else sd_module()
    if args.list_devices:
        print(sd_module().query_devices())
        return
    if offline and args.input_file is None:
        parser.error("--output-file needs --input-file")
    if offline and args.choose is None:
        parser.error("--output-file needs --choose")

    params = settings()
    models = dp.Models()
    file_audio = read_any(args.input_file) if args.input_file else None
    if args.discover_file:
        discovery_audio = read_any(args.discover_file)
    elif file_audio is not None:
        discovery_audio = file_audio
    else:
        discovery_audio = record(sd, args.discover_seconds, args.input_device)
    speakers = discover_speakers(models, discovery_audio, params)
    if not speakers:
        print("No speakers found. Record longer, with each person talking alone for at least ~5 s.")
        return
    index = args.choose - 1 if args.choose else choose_speaker(sd, speakers, args.output_device)
    if not 0 <= index < len(speakers):
        sys.exit(f"--choose must be 1-{len(speakers)}")
    extractor = LiveExtractor(models, [s["ecapa"] for s in speakers], index, params["floor"])
    print(f"Speaker {index + 1} chosen. Competitive floor {params['floor']}, block 1 s, context 2 s, crossfade 20 ms.")

    if offline:
        outputs, step = [], int(CHUNK_SECONDS * RATE)
        for i in range(0, len(file_audio), step):
            outputs.extend(extractor.push(file_audio[i:i + step]))
        outputs.append(extractor.flush())
        result = np.concatenate(outputs) if outputs else np.zeros(0, dtype="float32")
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        sf.write(args.output_file, result, RATE, subtype="FLOAT")
        print(f"Wrote {args.output_file} ({len(result) / RATE:.1f} s; output lags input by "
              f"{extractor.fade / RATE * 1000:.0f} ms crossfade, plus block and compute time when live).")
        print(extractor.stats())
        return

    saved_in, saved_out = run_live(sd, args, extractor, speakers, file_audio)
    if args.save:
        folder = SESSIONS / datetime.now().strftime("%Y%m%d_%H%M%S")
        folder.mkdir(parents=True, exist_ok=True)
        sf.write(folder / "discovery_buffer.wav", discovery_audio, RATE, subtype="PCM_16")
        for i, s in enumerate(speakers, 1):
            sf.write(folder / f"discovered_{i}.wav", s["track"], RATE, subtype="FLOAT")
        if saved_in:
            sf.write(folder / "live_input.wav", np.concatenate(saved_in), RATE, subtype="PCM_16")
        if saved_out:
            sf.write(folder / "live_output.wav", np.concatenate(saved_out), RATE, subtype="FLOAT")
        print(f"Saved session to {folder}")


if __name__ == "__main__":
    main()
