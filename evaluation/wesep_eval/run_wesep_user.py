"""WeSep vs SepFormer on the user's real recording: listening page + voice-similarity check.

Uses the discovered tracks from ../run_user_discovery.py (whole-clip run) as
enrollment, exactly like the live flow, and the separate Nicker/Mond/Sam
recordings for the clean-enrollment comparison. Existing SepFormer outputs
are linked, not regenerated. Writes
../data/user_recordings/discovery/wesep_compare.html (Git-ignored).
"""

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
from run_wesep_experiment import block_online, enrollment, extract, load  # noqa: E402

USER = HERE.parent / "data/user_recordings"
PAGE_DIR = USER / "discovery"
NAMES = ("Nicker", "Mond", "Sam")
WESEP_RUNS = [("tfmap_context_causal_100", None), ("tfmap_context_causal_100", (1.0, 2.0)),
              ("tfmap_context_causal_100", (0.5, 1.0)), ("tfmap_context_100", None)]


def listening(wave, name):
    path = PAGE_DIR / "wesep" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.max(np.abs(wave)))
    sf.write(path, wave * (0.9 / peak if peak > 1e-9 else 1.0), dp.RATE, subtype="PCM_16")
    return str(path.relative_to(PAGE_DIR))


def label(model, block):
    kind = "causal" if "causal" in model else "non-causal"
    return f"WeSep {kind}, " + ("whole clip (offline)" if block is None
                                else f"block {block[0]} s / context {block[1]} s (no future)")


def main():
    torch.set_num_threads(4)
    models = dp.Models()
    run = json.loads((PAGE_DIR / "run.json").read_text())
    full = next(v for v in run["variants"] if v["variant"] == "full")
    mix = read_any(USER / "mixture/Mix.wav")
    clean = {n: read_any(USER / f"enrollments/{n.lower()}/{n}.wav") for n in NAMES}
    enrolled = {n: models.embed("ecapa", clean[n]) for n in NAMES}
    clusters = [c for c in full["clusters"] if c["match"]]
    rows, results = [], []
    loaded = {}
    for model_name, block in WESEP_RUNS:
        model = loaded.setdefault(model_name, load(model_name))
        cells = []
        for c in clusters:
            who = c["match"]["name"]
            cell = {}
            for kind, source in (("discovered", dp.read_wave(PAGE_DIR / c["audio"])), ("clean", clean[who])):
                enroll = enrollment(source)
                out = extract(model, mix, enroll) if block is None else block_online(model, mix, enroll, *block)[0]
                vector = models.embed("ecapa", out)
                sims = {n: float(np.dot(vector, e)) for n, e in enrolled.items()} if vector is not None else None
                tag = f"{model_name}_{'full' if block is None else f'B{block[0]}_W{block[1]}'}_{kind}_{who}.wav"
                cell[kind] = {"audio": listening(out, tag), "similarity": sims}
                results.append({"model": model_name, "block": block, "speaker": who, "enrollment": kind,
                                "similarity": sims})
            cells.append(cell)
        rows.append((label(model_name, block), cells))
        print(f"done {label(model_name, block)}", flush=True)
    sepformer = [("SepFormer competitive, whole clip (offline)",
                  [{"discovered": {"audio": f"listen/full/sepformer_competitive__cluster_{c['cluster']}.wav",
                                   "similarity": full["live"][f"cluster:{c['cluster']}"]["sepformer_competitive"]["similarity"]},
                    "clean": {"audio": f"listen/full/sepformer_competitive__clean_{c['match']['name']}.wav",
                              "similarity": full["live"][f"clean:{c['match']['name']}"]["sepformer_competitive"]["similarity"]}}
                   for c in clusters]),
                 ("SepFormer competitive, block 1 s / context 2 s + 20 ms crossfade (no future)",
                  [{"discovered": {"audio": f"block_online/crossfade_20ms_discovered_{c['cluster']}_{c['match']['name']}.wav",
                                   "similarity": None}} for c in clusters])]

    def fmt(sims, who):
        if not sims:
            return ""
        best_other = max(v for n, v in sims.items() if n != who)
        ok = sims[who] > best_other
        return (f"<br><small style='color:{'#2f6f3e' if ok else '#a33a2b'}'>{who} {sims[who]:.2f} vs "
                f"other {best_other:.2f}</small>")

    head = "".join(f"<th>Discovered {c['cluster']} ({c['match']['name']})</th>" for c in clusters)
    body = ""
    for title, cells in sepformer + rows:
        for kind in ("discovered", "clean"):
            if not any(kind in cell for cell in cells):
                continue
            body += f"<tr><td>{title}<br><small>{kind} enrollment</small></td>"
            for c, cell in zip(clusters, cells):
                item = cell.get(kind)
                body += ("<td></td>" if not item else
                         f"<td><audio controls preload='none' src='{item['audio']}'></audio>"
                         f"{fmt(item['similarity'], c['match']['name'])}</td>")
            body += "</tr>"
    (PAGE_DIR / "wesep_compare.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>WeSep vs SepFormer</title>"
        "<style>body{font:15px -apple-system,sans-serif;margin:16px}td,th{padding:6px;text-align:left;"
        "vertical-align:top;border-bottom:1px solid #ddd}audio{width:210px}</style>"
        "<h1>WeSep vs SepFormer on your recording</h1>"
        "<p>Each column is one discovered speaker (label = voice match from the discovery page). "
        "Green/red text: ECAPA similarity of the output to that person's own recording vs the best other "
        "person, a rough proxy (green means it sounds most like the right person). Clips are "
        "peak-normalized. WeSep checkpoints are REAL-TSE research weights with no stated licence.</p>"
        f"<table><tr><th>Model</th>{head}</tr>{body}</table>")
    (PAGE_DIR / "wesep_compare.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {PAGE_DIR / 'wesep_compare.html'}")


if __name__ == "__main__":
    main()
