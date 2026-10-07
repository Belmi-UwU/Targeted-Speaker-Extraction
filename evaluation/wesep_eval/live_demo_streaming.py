"""Live demo with true streaming WeSep: about 0.16 s total delay instead of 0.9 s (block mode) or 2.2 s (SepFormer).

Discovery, choosing, devices, bypass (`b`), and speaker switching (`s N`) are the
same as ../live_demo.py and live_demo_wesep.py. The extractor is
streaming_wesep.StreamingWesep:
- frame-by-frame causal inference with carried LSTM state;
- the enrollment computed once;
- 32 ms algorithmic latency.

Audio is processed in chunks of --chunk seconds (default 64 ms). The playback
buffer holds one chunk plus --margin to absorb compute jitter.

  .venv/bin/python live_demo_streaming.py --input-file ../data/user_recordings/mixture/Mix.wav   # simulated live
  .venv/bin/python live_demo_streaming.py                                                     # microphone (headphones!)
  .venv/bin/python live_demo_streaming.py --chunk 0.032          # lower delay, needs a faster CPU (RTF ~0.8 here)
  .venv/bin/python live_demo_streaming.py --model retrained_finetune_research   # a model from training/
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE / "upstream/wesep-real-tse"))

import live_demo as base  # noqa: E402
from run_wesep_experiment import enrollment, load  # noqa: E402
from streaming_wesep import StreamingWesep  # noqa: E402

ALGORITHMIC_SECONDS = 0.032


class StreamingAdapter:
    """Exposes StreamingWesep through the interface live_demo.run_live expects."""

    def __init__(self, model, enrollments, target_index, chunk_s, device):
        self.enrollments = enrollments
        self._target = target_index
        self.streamer = StreamingWesep(model, enrollments[target_index], device)
        self.chunk = int(chunk_s * base.RATE)
        for _ in range(4):  # warm up kernels so the first live chunk is not slow, then start clean
            self.streamer.push(np.zeros(self.chunk, dtype="float32"))
        self.streamer.reset()
        self.block, self.fade = self.chunk, 0  # run_live prefills block + margin of playback buffer
        self.bypass = False
        self.pending = np.zeros(0, dtype="float32")
        self.raw = np.zeros(0, dtype="float32")  # unprocessed input with the same delay, for bypass
        self.compute = []

    @property
    def target_index(self):
        return self._target

    @target_index.setter
    def target_index(self, index):
        self._target = index
        self.streamer.set_enrollment(self.enrollments[index])  # LSTM state carries on, no glitch

    def push(self, samples):
        self.pending = np.concatenate([self.pending, np.asarray(samples, dtype="float32")])
        ready = []
        while len(self.pending) >= self.chunk:
            x, self.pending = self.pending[:self.chunk], self.pending[self.chunk:]
            self.raw = np.concatenate([self.raw, x])
            began = time.perf_counter()
            y = self.streamer.push(x)
            self.compute.append(time.perf_counter() - began)
            raw, self.raw = self.raw[:len(y)], self.raw[len(y):]
            ready.append(raw if self.bypass else y)
        return ready

    def flush(self):
        return np.zeros(0, dtype="float32")

    def stats(self):
        if not self.compute:
            return "no chunks yet"
        c = np.array(self.compute)
        return (f"chunks {len(c)} | compute median {np.median(c) * 1000:.0f} ms, p90 {np.percentile(c, 90) * 1000:.0f} ms "
                f"per {self.chunk / base.RATE * 1000:.0f} ms chunk (real-time factor "
                f"{np.median(c) * base.RATE / self.chunk:.2f})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="tfmap_context_causal_100",
                        help="folder under checkpoints/pretrained/ (default: tfmap_context_causal_100)")
    parser.add_argument("--chunk", type=float, default=0.064, help="processing chunk in seconds (default 0.064)")
    parser.add_argument("--margin", type=float, default=0.10,
                        help="extra playback buffer in seconds (default 0.10). It must cover mic-block arrival "
                             "(~20 ms), compute per chunk (~35 ms at 64 ms chunks), and the 24 ms start-up lag of "
                             "the iSTFT; lower it only with smaller chunks or a faster machine")
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps"),
                        help="cpu (default, fastest for small chunks here) or mps (Metal)")
    parser.add_argument("--threads", type=int, default=2, help="CPU threads (default 2, fastest here)")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--input-device")
    parser.add_argument("--output-device")
    parser.add_argument("--discover-seconds", type=float, default=30.0)
    parser.add_argument("--discover-file", type=Path)
    parser.add_argument("--input-file", type=Path)
    parser.add_argument("--output-file", type=Path, help="offline test: write output here, no audio devices")
    parser.add_argument("--choose", type=int)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    for name in ("input_device", "output_device"):
        value = getattr(args, name)
        if value is not None and value.isdigit():
            setattr(args, name, int(value))
    if args.list_devices:
        print(base.sd_module().query_devices())
        return
    offline = args.output_file is not None
    if offline and (args.input_file is None or args.choose is None):
        parser.error("--output-file needs --input-file and --choose")
    sd = None if offline else base.sd_module()

    models = base.dp.Models()
    torch.set_num_threads(args.threads)
    file_audio = base.read_any(args.input_file) if args.input_file else None
    if args.discover_file:
        discovery_audio = base.read_any(args.discover_file)
    elif file_audio is not None:
        discovery_audio = file_audio
    else:
        discovery_audio = base.record(sd, args.discover_seconds, args.input_device)
    speakers = base.discover_speakers(models, discovery_audio, base.settings())
    torch.set_num_threads(args.threads)  # discovery models may have changed it
    if not speakers:
        print("No speakers found. Record longer, with each person talking alone for at least ~5 s.")
        return
    index = args.choose - 1 if args.choose else base.choose_speaker(sd, speakers, args.output_device)
    if not 0 <= index < len(speakers):
        sys.exit(f"--choose must be 1-{len(speakers)}")
    print(f"Loading WeSep checkpoint {args.model} ...", flush=True)
    extractor = StreamingAdapter(load(args.model).model, [enrollment(s["track"]) for s in speakers], index,
                                 args.chunk, args.device)
    print(f"Speaker {index + 1} chosen. Streaming WeSep {args.model} on {args.device}, chunk "
          f"{args.chunk * 1000:.0f} ms; expected delay about "
          f"{(args.chunk + args.margin) * 1000:.0f} ms plus audio-device buffers.")

    if offline:
        outputs, step = [], int(base.CHUNK_SECONDS * base.RATE)
        for i in range(0, len(file_audio), step):
            outputs.extend(extractor.push(file_audio[i:i + step]))
        result = np.concatenate(outputs)
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        sf.write(args.output_file, result, base.RATE, subtype="FLOAT")
        print(f"Wrote {args.output_file} ({len(result) / base.RATE:.1f} s)")
        print(extractor.stats())
        return
    base.run_live(sd, args, extractor, speakers, file_audio)


if __name__ == "__main__":
    main()
