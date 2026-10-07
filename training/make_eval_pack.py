"""Pack the held-out evaluation conversations for the training server.

Run on the machine that holds the discovery experiment outputs:

    speakerbeam_ss_verification/.venv/bin/python training/make_eval_pack.py

It writes training/eval_pack.zip (Git-ignored). Upload it next to the
notebook and unzip it into the notebook's EVAL_PACK folder. For each of the
six controlled conversations it contains:

- the live mixture and each speaker's clean live source (scoring references);
- the segment timeline, so target-absent spans can be scored;
- each speaker's clean enrollment utterance;
- each speaker's discovered track, as found by SepFormer + ECAPA discovery
  from the 4-round buffer, already trimmed to speech;

plus baselines.json with the SepFormer and released-WeSep numbers.

The user's personal recording is NOT included unless --include-user-recording
is given.
"""

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "evaluation"
DISCOVERY = EVAL / "data/discovery"
RESULTS = DISCOVERY / "results"
RATE = 16000


def active_speech(wave, frame=1600, floor_db=-35.0):
    """Same rule as evaluation/discovery_pipeline.active_speech."""
    frames = len(wave) // frame
    if frames == 0:
        return wave
    blocks = wave[:frames * frame].reshape(frames, frame)
    energy = np.square(blocks).mean(axis=1)
    if energy.max() <= 0:
        return np.zeros(0, dtype="float32")
    return blocks[energy > energy.max() * 10 ** (floor_db / 10)].reshape(-1)


def read(path):
    wave, rate = sf.read(path, dtype="float32")
    if rate != RATE or wave.ndim != 1:
        raise ValueError(f"expected 16 kHz mono: {path}")
    return wave


def baselines():
    out = {}
    live = [json.loads(l) for l in (RESULTS / "live_scores.jsonl").read_text().splitlines() if l]
    for model in ("sepformer_competitive", "sepformer_selector", "speakerbeam_ss"):
        for kind, rounds in (("clean", None), ("discovered", 4)):
            rows = [r for r in live if r["split"] == "test" and r["model"] == model
                    and r["enrollment"] == kind and r["buffer_rounds"] == rounds
                    and (kind == "clean" or r.get("mapped"))]
            if rows:
                out[f"{model} (offline, future audio) | {kind}"] = {
                    "median_si_sdri_db": float(np.median([r["overall_si_sdri_db"] for r in rows])),
                    "median_absent_suppression_db": float(np.median([r["absent_suppression_db"] for r in rows]))}
    wesep = json.loads((RESULTS / "wesep_comparison.json").read_text())
    for row in wesep["full_summary"]:
        out[f"WeSep {row['model']} (released checkpoint) | {row['enrollment']}"] = {
            "median_si_sdri_db": row["median_si_sdri_db"],
            "median_absent_suppression_db": row["median_absent_suppression_db"]}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=ROOT / "training/eval_pack.zip")
    parser.add_argument("--include-user-recording", action="store_true",
                        help="also pack Mix.wav + Nicker/Mond/Sam recordings (personal data)")
    args = parser.parse_args()
    plan = json.loads((DISCOVERY / "mixtures.json").read_text())
    audit = [json.loads(l) for l in (RESULTS / "discovery_audit.jsonl").read_text().splitlines() if l]
    index = {"rate": RATE, "conversations": []}
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as pack:
        def add_wave(name, wave):
            path = Path("/tmp") / "eval_pack_tmp.wav"
            sf.write(path, wave, RATE, subtype="FLOAT")
            pack.write(path, name)

        for split in ("test", "dev"):
            for mixture_id in plan[split]:
                folder = DISCOVERY / split / mixture_id
                provenance = json.loads((folder / "provenance.json").read_text())
                live = provenance["sections"]["live"]
                row = next(r for r in audit if r["mixture"] == mixture_id and r["encoder"] == "ecapa"
                           and r["buffer_rounds"] == 4)
                base = f"eval_pack/{split}/{mixture_id}"
                add_wave(f"{base}/live_mix.wav", read(folder / live["mixture_path"]))
                entry = {"split": split, "id": mixture_id, "speakers": provenance["speakers"],
                         "segments": live["segments"], "discovered": []}
                for s in provenance["speakers"]:
                    add_wave(f"{base}/sources/{s}.wav", read(folder / live["source_paths"][s]))
                    add_wave(f"{base}/enroll_clean/{s}.wav", read(folder / f"audio/{s}_enrollment.wav"))
                    cluster = row["mapping"].get(s)
                    if cluster is not None:
                        track = read(folder / f"outputs/discovery_ecapa/speaker_{cluster + 1}.wav")
                        add_wave(f"{base}/enroll_discovered/{s}.wav", active_speech(track))
                        entry["discovered"].append(s)
                index["conversations"].append(entry)
                print(f"packed {mixture_id}", flush=True)
        if args.include_user_recording:
            user = EVAL / "data/user_recordings"
            for relative in ("mixture/Mix.wav", "enrollments/nicker/Nicker.wav",
                             "enrollments/mond/Mond.wav", "enrollments/sam/Sam.wav"):
                pack.write(user / relative, f"eval_pack/user_recording/{Path(relative).name}")
            index["user_recording"] = True
        pack.writestr("eval_pack/index.json", json.dumps(index, indent=2))
        pack.writestr("eval_pack/baselines.json", json.dumps(baselines(), indent=2))
    print(f"wrote {args.output} ({args.output.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
