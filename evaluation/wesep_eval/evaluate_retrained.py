"""Retrained WeSep vs the released checkpoint vs SepFormer on the user's real recording.

Both WeSep models run through the live streaming path (streaming_wesep.StreamingWesep,
64 ms chunks), exactly as live_demo_streaming.py does. Each discovered speaker is
used as enrollment (whole-clip discovery from ../run_user_discovery.py), plus the
separate Nicker/Mond/Sam recordings for comparison. Outputs get an ECAPA voice
match against each person's own recording, and a local listening page is written
to ../data/user_recordings/discovery/retrained_compare.html (Git-ignored).

    .venv/bin/python evaluate_retrained.py [--model retrained_finetune_research]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE / "upstream/wesep-real-tse"))

import discovery_pipeline as dp  # noqa: E402
from live_demo import read_any  # noqa: E402
from run_wesep_experiment import enrollment, load  # noqa: E402
from streaming_wesep import stream  # noqa: E402

USER = HERE.parent / "data/user_recordings"
PAGE = USER / "discovery"
NAMES = ("Nicker", "Mond", "Sam")


def save(wave, name):
    path = PAGE / "retrained" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.max(np.abs(wave)))
    sf.write(path, wave * (0.9 / peak if peak > 1e-9 else 1.0), dp.RATE, subtype="PCM_16")
    return str(path.relative_to(PAGE))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="retrained_finetune_research")
    args = parser.parse_args()
    torch.set_num_threads(2)
    encoders = dp.Models()
    run = json.loads((PAGE / "run.json").read_text())
    full = next(v for v in run["variants"] if v["variant"] == "full")
    mix = read_any(USER / "mixture/Mix.wav")
    clean = {n: read_any(USER / f"enrollments/{n.lower()}/{n}.wav") for n in NAMES}
    voice = {n: encoders.embed("ecapa", clean[n]) for n in NAMES}
    clusters = [c for c in full["clusters"] if c["match"]]
    models = {"released": "tfmap_context_causal_100", "retrained": args.model}
    results, cells = [], {}
    for label, name in models.items():
        model = load(name).model
        for c in clusters:
            who = c["match"]["name"]
            for kind, source in (("discovered", dp.read_wave(PAGE / c["audio"])), ("clean", clean[who])):
                out, stats = stream(model, mix, enrollment(source), "cpu", 1024)
                vector = encoders.embed("ecapa", out)
                sims = {n: float(np.dot(vector, voice[n])) for n in NAMES}
                other = max(v for n, v in sims.items() if n != who)
                results.append({"model": label, "checkpoint": name, "speaker": who, "enrollment": kind,
                                "similarity": sims, "correct": sims[who] > other,
                                "real_time_factor": stats["real_time_factor"]})
                cells[(label, who, kind)] = (save(out, f"{label}_{kind}_{who}.wav"), sims[who], other)
                print(f"{label:9s} {kind:10s} {who:6s} {sims[who]:.2f} vs {other:.2f} "
                      f"{'OK' if sims[who] > other else 'WRONG'}  RTF {stats['real_time_factor']:.2f}", flush=True)
    for c in clusters:
        who = c["match"]["name"]
        live = full["live"][f"cluster:{c['cluster']}"]["sepformer_competitive"]["similarity"]
        cells[("sepformer", who, "discovered")] = (f"listen/full/sepformer_competitive__cluster_{c['cluster']}.wav",
                                                   live[who], max(v for n, v in live.items() if n != who))
        live = full["live"][f"clean:{who}"]["sepformer_competitive"]["similarity"]
        cells[("sepformer", who, "clean")] = (f"listen/full/sepformer_competitive__clean_{who}.wav",
                                              live[who], max(v for n, v in live.items() if n != who))
    titles = {"sepformer": "SepFormer competitive (offline, ~2.2 s live delay)",
              "released": "WeSep released causal checkpoint (streaming, ~0.17 s)",
              "retrained": f"WeSep {args.model} (streaming, ~0.17 s)"}
    order = [c["match"]["name"] for c in clusters]
    rows = ""
    for label in ("sepformer", "released", "retrained"):
        for kind in ("discovered", "clean"):
            rows += f"<tr><td>{titles[label]}<br><small>{kind} enrollment</small></td>"
            for who in order:
                audio, own, other = cells[(label, who, kind)]
                color = "#2f6f3e" if own > other else "#a33a2b"
                rows += (f"<td><audio controls preload='none' src='{audio}'></audio><br>"
                         f"<small style='color:{color}'>{who} {own:.2f} vs other {other:.2f}</small></td>")
            rows += "</tr>"
    (PAGE / "retrained_compare.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Retrained WeSep</title>"
        "<style>body{font:15px -apple-system,sans-serif;margin:16px}td,th{padding:6px;text-align:left;"
        "vertical-align:top;border-bottom:1px solid #ddd}audio{width:210px}</style>"
        "<h1>Retrained WeSep on your recording</h1><p>Whole 23 s Mix.wav. Columns are discovered speakers. "
        "Green: the output sounds most like the chosen person (ECAPA voice match vs the best other person); "
        "red: it sounds more like someone else. Clips are peak-normalized. Both WeSep rows use the live streaming "
        "path. The retrained model is research-only (initialised from unlicensed weights).</p>"
        f"<table><tr><th>Model</th>{''.join(f'<th>{w}</th>' for w in order)}</tr>{rows}</table>")
    (PAGE / "retrained_compare.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {PAGE / 'retrained_compare.html'}")


if __name__ == "__main__":
    main()
