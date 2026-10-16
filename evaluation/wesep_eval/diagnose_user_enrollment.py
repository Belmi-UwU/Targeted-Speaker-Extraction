"""Compare original and discovered enrollments using the web app's streaming WeSep.

This is a listening control on the same Mix.wav, not a held-out quality score.
Original recordings are reference examples, not aligned clean source tracks.
"""
import argparse
from datetime import datetime
import html
import json
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from live_demo_streaming import StreamingAdapter, base, enrollment, load
from diagnose_user_discovery import LISTEN_SCRIPT, audio, describe, write_clip


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("diagnostic", type=Path)
    parser.add_argument("--model", default="retrained_finetune_research")
    parser.add_argument("--person", choices=("Nicker", "Sam", "Mond"))
    parser.add_argument("--segment-start", type=float)
    parser.add_argument("--output-name", default="wesep_enrollment")
    args = parser.parse_args()
    if Path(args.output_name).name != args.output_name or args.output_name in (".", ".."):
        parser.error("--output-name must be a folder name")
    if args.segment_start is not None and args.segment_start < 0:
        parser.error("--segment-start must be nonnegative")
    diagnostic = args.diagnostic.resolve()
    original = json.loads((diagnostic / "diagnostics.json").read_text())
    folder = diagnostic / args.output_name
    folder.mkdir(exist_ok=False)
    mix_path = Path(original["inputs"]["Mix"]["path"])
    if describe(mix_path)["sha256"] != original["inputs"]["Mix"]["sha256"]:
        raise ValueError("Mix.wav changed since the discovery run")
    mix = base.read_any(mix_path)
    torch.set_num_threads(2)
    print(f"Loading {args.model}", flush=True)
    model = load(args.model).model
    checkpoint = HERE / "checkpoints/pretrained" / args.model
    record = {
        "created_at": datetime.now(ZoneInfo("Asia/Jakarta")).isoformat(),
        "scope": "Same-recording listening control, no aligned clean sources or quality score",
        "model": args.model,
        "checkpoint": {name: describe_hash(checkpoint / name) for name in ("avg_model.pt", "config.yaml")},
        "mixture": describe(mix_path), "chunk_seconds": 0.064, "threads": 2,
        "enrollment_seconds_cap": 8.0, "enrollment_rms_dbfs": -25.0,
        "source_segment_start_seconds": args.segment_start,
        "pairs": [],
    }
    # All mappings remain hypotheses from ECAPA similarity, not verified labels.
    for profile in original["profiles"]:
        name = profile["match"]["best"]
        if args.person and name != args.person:
            continue
        pair = {"reference_name": name, "discovered_profile": profile["profile"], "runs": {}}
        record["pairs"].append(pair)
        for variant in ("original", "discovered"):
            path = (Path(original["inputs"][name]["path"]) if variant == "original"
                    else diagnostic / profile["clip"]["raw"])
            if variant == "original" and describe(path)["sha256"] != original["inputs"][name]["sha256"]:
                raise ValueError(f"{name} enrollment changed")
            wave = base.read_any(path)
            if args.segment_start is not None:
                start = int(args.segment_start * base.RATE)
                wave = wave[start:start + 8 * base.RATE]
            enroll = enrollment(wave)
            if len(enroll) < base.RATE:
                raise ValueError(f"Insufficient enrollment audio for {name}/{variant}")
            print(f"Processing {name}: {variant}, enrollment {len(enroll)/base.RATE:.2f}s", flush=True)
            adapter = StreamingAdapter(model, [enroll], 0, 0.064, "cpu")
            began = time.perf_counter()
            outputs = []
            # Same adapter and end-of-file behavior as the existing streaming CLI.
            for offset in range(0, len(mix), adapter.chunk):
                outputs.extend(adapter.push(mix[offset:offset + adapter.chunk]))
            elapsed = time.perf_counter() - began
            output = np.concatenate(outputs)
            if len(output) == 0 or not np.isfinite(output).all():
                raise ValueError(f"Invalid output for {name}/{variant}")
            pair["runs"][variant] = {
                "source": describe(path), "enrollment_frames": len(enroll),
                "enrollment_clip": write_clip(folder, f"enrollment_{name.lower()}_{variant}", enroll),
                "output_frames": len(output), "unemitted_tail_frames": len(mix) - len(output),
                "compute_seconds": elapsed, "stats": adapter.stats(),
                "output_rms": float(np.sqrt(np.mean(output.astype("float64") ** 2))),
                "clip": write_clip(folder, f"{name.lower()}_{variant}", output),
            }
            print(f"  {adapter.stats()}", flush=True)
    (folder / "results.json").write_text(json.dumps(record, indent=2) + "\n")
    cards = []
    for pair in record["pairs"]:
        name = html.escape(pair["reference_name"])
        cards.append(f"<section><h2>{name} dibandingkan dengan Profil {pair['discovered_profile']}</h2>"
                     "<div class='grid'>")
        for variant, label in (("original", "Enrollment rekaman asli"),
                               ("discovered", "Enrollment profil discovery")):
            run = pair["runs"][variant]
            if args.segment_start is not None:
                label += f" · sumber {args.segment_start:g}–{args.segment_start + 8:g} detik"
            cards.append(f"<article><h3>{label}</h3><p><b>Keluaran WeSep</b></p>"
                         f"{audio(run['clip']['listen'])}<details><summary>Dengarkan contoh enrollment yang dipakai</summary>"
                         f"{audio(run['enrollment_clip']['listen'])}</details></article>")
        cards.append("</div></section>")
        if args.segment_start is not None and (diagnostic / "wesep_enrollment/results.json").exists():
            baseline = json.loads((diagnostic / "wesep_enrollment/results.json").read_text())
            if baseline["checkpoint"] != record["checkpoint"]:
                raise ValueError("Baseline uses a different checkpoint")
            if baseline["mixture"]["sha256"] != record["mixture"]["sha256"]:
                raise ValueError("Baseline uses a different mixture")
            previous = next(p for p in baseline["pairs"] if p["discovered_profile"] == pair["discovered_profile"])
            cards.append("<h2>Pembanding sebelumnya: 8 detik pertama setelah pemilihan frame aktif</h2><div class='grid'>")
            for variant, label in (("original", "Enrollment asli sebelumnya"),
                                   ("discovered", "Enrollment discovery sebelumnya")):
                relative = "../wesep_enrollment/" + previous["runs"][variant]["clip"]["listen"]
                cards.append(f"<article><h3>{label}</h3>{audio(relative)}</article>")
            cards.append("</div>")
    mix_clip = write_clip(folder, "mix", mix)
    page = """<!doctype html><html lang="id"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Perbandingan enrollment WeSep</title>
<style>body{font:16px/1.5 system-ui;background:#f5f6f8;color:#17212f}main{max-width:1000px;margin:auto;padding:24px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}
article{background:white;padding:16px;border:1px solid #dde2e8;border-radius:12px}
audio{display:block;width:100%;margin-bottom:12px}button{padding:10px;cursor:pointer}
.note{background:#fff2cd;padding:16px;border-radius:12px}</style>
<main><h1>Pengaruh enrollment pada WeSep</h1>
<p class="note">Semua keluaran memakai Mix.wav, checkpoint R1, dan streaming CPU 64 ms yang sama.
Yang berubah hanya rekaman enrollment. Nama pasangan mengikuti dugaan ECAPA pada profil discovery,
bukan bukti profil itu murni. Rekaman asli juga perlu diperiksa kebersihannya.
Setiap klip dinormalisasi sendiri untuk mendengarkan; bandingkan suara yang tersisa dan kejernihan, bukan volume.
Ini pemeriksaan rekaman yang sama, bukan evaluasi data baru. Tidak ada skor SI-SDR karena sumber bersih selaras tidak tersedia.</p>
<p>Gunakan <b>Putar alternatif</b>. Periksa apakah suara target tetap jelas, apakah volumenya naik-turun sampai kata hilang,
dan apakah suara pembicara lain masih terdengar. Contoh enrollment dapat dibuka di bawah setiap keluaran.</p>
<h2>Campuran asli</h2>""" + audio(mix_clip["listen"]) + "".join(cards) + "</main>" + LISTEN_SCRIPT + "</html>"
    (folder / "index.html").write_text(page)
    print(f"Ready: {folder / 'index.html'}", flush=True)


def describe_hash(path):
    return {"path": str(path), "sha256": base.dp.sha256(path)}


if __name__ == "__main__":
    main()
