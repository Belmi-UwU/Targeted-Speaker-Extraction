"""Live demo with WeSep causal BSRNN as the extractor (comparison with ../live_demo.py).

Discovery, choosing, devices, bypass, and speaker switching are exactly those
of ../live_demo.py. Only the live extractor changes: instead of SepFormer +
ECAPA stream selection, a causal target-speaker extraction model
(REAL-TSE `tfmap_context_causal_100` by default) is conditioned on the
**audio** of the discovered track. It processes short blocks with past
context only, so it uses no future audio, and its blocks can be much shorter
than SepFormer's 1 s.

Runs in this folder's .venv. Checkpoints have no stated weights licence:
local evaluation only. Usage mirrors ../live_demo.py, e.g.

  .venv/bin/python live_demo_wesep.py --input-file ../data/user_recordings/mixture/Mix.wav
  .venv/bin/python live_demo_wesep.py                       # mic discovery + live mic (headphones!)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE / "upstream/wesep-real-tse"))

import live_demo as base  # noqa: E402
from run_wesep_experiment import enrollment, extract, load  # noqa: E402


class WesepLiveExtractor(base.LiveExtractor):
    """Same block/crossfade/bypass behaviour as the SepFormer extractor, WeSep inside."""

    def __init__(self, model, enrollments, target_index, block_s, window_s, fade_s=0.02):
        self.model, self.enrollments, self.target_index = model, enrollments, target_index
        self.block, self.window = int(block_s * base.RATE), int(window_s * base.RATE)
        self.fade = int(fade_s * base.RATE)
        self.ease_in = (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, self.fade))).astype("float32")
        self.history = np.zeros(self.window, dtype="float32")
        self.pending = np.zeros(0, dtype="float32")
        self.held = None
        self.bypass = False
        self.compute = []

    def _process(self):
        start = time.perf_counter()
        span = self.block + (self.fade if self.held is not None else 0)
        if self.bypass:
            piece = self.history[-span:].copy()
        else:
            piece = extract(self.model, self.history, self.enrollments[self.target_index])[-span:]
        if self.held is not None:
            blended = self.held * (1 - self.ease_in) + piece[:self.fade] * self.ease_in
            out = np.concatenate([blended, piece[self.fade:-self.fade]])
        else:
            out = piece[:-self.fade]
        self.held = piece[-self.fade:]
        self.compute.append(time.perf_counter() - start)
        return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="tfmap_context_causal_100",
                        help="REAL-TSE checkpoint folder name (default: tfmap_context_causal_100)")
    parser.add_argument("--block", type=float, default=0.5, help="block seconds = added delay (default 0.5)")
    parser.add_argument("--window", type=float, default=1.0, help="past context seconds (default 1.0)")
    parser.add_argument("--margin", type=float, default=0.4,
                        help="extra playback delay absorbing slow blocks (default 0.4)")
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
    offline = args.output_file is not None
    if args.list_devices:
        print(base.sd_module().query_devices())
        return
    if offline and (args.input_file is None or args.choose is None):
        parser.error("--output-file needs --input-file and --choose")
    sd = None if offline else base.sd_module()

    models = base.dp.Models()
    file_audio = base.read_any(args.input_file) if args.input_file else None
    if args.discover_file:
        discovery_audio = base.read_any(args.discover_file)
    elif file_audio is not None:
        discovery_audio = file_audio
    else:
        discovery_audio = base.record(sd, args.discover_seconds, args.input_device)
    speakers = base.discover_speakers(models, discovery_audio, base.settings())
    if not speakers:
        print("No speakers found. Record longer, with each person talking alone for at least ~5 s.")
        return
    index = args.choose - 1 if args.choose else base.choose_speaker(sd, speakers, args.output_device)
    if not 0 <= index < len(speakers):
        sys.exit(f"--choose must be 1-{len(speakers)}")
    print(f"Loading WeSep checkpoint {args.model} ...", flush=True)
    extractor = WesepLiveExtractor(load(args.model), [enrollment(s["track"]) for s in speakers], index,
                                   args.block, args.window)
    print(f"Speaker {index + 1} chosen. WeSep {args.model}, block {args.block} s, context {args.window} s, "
          f"crossfade 20 ms.")

    if offline:
        outputs, step = [], int(base.CHUNK_SECONDS * base.RATE)
        for i in range(0, len(file_audio), step):
            outputs.extend(extractor.push(file_audio[i:i + step]))
        outputs.append(extractor.flush())
        result = np.concatenate(outputs)
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        sf.write(args.output_file, result, base.RATE, subtype="FLOAT")
        print(f"Wrote {args.output_file} ({len(result) / base.RATE:.1f} s)")
        print(extractor.stats())
        return
    base.run_live(sd, args, extractor, speakers, file_audio)


if __name__ == "__main__":
    main()
