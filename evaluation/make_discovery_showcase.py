"""Build the local listening showcase for the discovery enrollment experiment.

Writes data/discovery/showcase/index.html plus PCM16 listening copies (each
clip peak-normalized to 0.9; compare voices and leakage, not loudness).
Open index.html directly in a browser; no server is needed.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/discovery"
RESULTS = DATA / "results"
SHOW = DATA / "showcase"
FULL_BUFFER = 4


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def listening_copy(source, relative):
    wave, rate = sf.read(source, dtype="float32")
    peak = float(np.max(np.abs(wave))) if len(wave) else 0.0
    target = SHOW / "audio" / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(target, wave * (0.9 / peak if peak > 1e-9 else 1.0), rate, subtype="PCM_16")
    return f"audio/{relative}"


def median(values):
    values = [v for v in values if v is not None]
    return float(np.median(values)) if values else None


def summarize(audit, fidelity, live):
    summary = {"discovery": [], "fidelity": [], "live": []}
    for split in ("dev", "test"):
        for encoder in ("ecapa", "resemblyzer"):
            for rounds in (1, 2, 4):
                rows = [r for r in audit if r["split"] == split and r["encoder"] == encoder
                        and r["buffer_rounds"] == rounds]
                if rows:
                    summary["discovery"].append({
                        "split": split, "encoder": encoder, "buffer_rounds": rounds,
                        "mixtures": len(rows), "exact": sum(r["exact"] for r in rows),
                        "speakers": sum(r["true_speakers"] for r in rows),
                        "found": sum(len(r["mapping"]) for r in rows),
                        "missed": sum(len(r["missed"]) for r in rows),
                        "splits": sum(len(r["split_speakers"]) for r in rows),
                        "merged": sum(len(r["merged_clusters"]) for r in rows),
                        "extra": sum(r["discovered"] - len(r["mapping"]) for r in rows)})
        for rounds in (1, 2, 4):
            rows = [r for r in fidelity if r["split"] == split and r["buffer_rounds"] == rounds]
            if rows:
                entry = {"split": split, "buffer_rounds": rounds, "speakers": len(rows),
                         "discovered": sum(r["discovered"] for r in rows)}
                for encoder in ("ecapa", "resemblyzer"):
                    entry[f"{encoder}_cos_disc"] = median(r[f"{encoder}_cos_disc_vs_enroll"] for r in rows)
                    entry[f"{encoder}_cos_inmix"] = median(r[f"{encoder}_cos_inmix_vs_enroll"] for r in rows)
                    margins = [r[f"{encoder}_margin_split"] for r in rows
                               if r[f"{encoder}_margin_split"] is not None]
                    entry[f"{encoder}_margin_split"] = median(margins)
                    entry[f"{encoder}_confusions"] = sum(m <= 0 for m in margins)
                summary["fidelity"].append(entry)
        for model in ("speakerbeam_ss", "sepformer_selector", "sepformer_competitive"):
            clean = {(r["mixture"], r["target"]): r for r in live
                     if r["split"] == split and r["model"] == model and r["enrollment"] == "clean"}
            for label, rounds in (("clean", None), ("discovered 1 round", 1),
                                  ("discovered 2 rounds", 2), ("discovered 4 rounds", 4)):
                rows = [r for r in live if r["split"] == split and r["model"] == model and
                        r["buffer_rounds"] == rounds and (rounds is None or r["mapped"])]
                if not rows:
                    continue
                drops = [clean[(r["mixture"], r["target"])]["overall_si_sdri_db"] - r["overall_si_sdri_db"]
                         for r in rows if rounds is not None and (r["mixture"], r["target"]) in clean]
                summary["live"].append({
                    "split": split, "model": model, "enrollment": label, "runs": len(rows),
                    "median_si_sdri": median(r["overall_si_sdri_db"] for r in rows),
                    "p10_si_sdri": float(np.percentile([r["overall_si_sdri_db"] for r in rows], 10)),
                    "median_drop_vs_clean": median(drops) if drops else None,
                    "median_absent_suppression": median(r["absent_suppression_db"] for r in rows),
                    "median_alone_si_sdr": median(r["target_alone_si_sdr_db"] for r in rows),
                    "by_overlap": {k: median(r["si_sdri_by_overlap_db"].get(k) for r in rows)
                                   for k in ("target_plus_1", "target_plus_2", "target_plus_3")}})
    return summary


def main():
    config = json.loads((RESULTS / "config_and_tuning.json").read_text())
    audit = read_jsonl(RESULTS / "discovery_audit.jsonl")
    fidelity = read_jsonl(RESULTS / "embedding_fidelity.jsonl")
    live = read_jsonl(RESULTS / "live_scores.jsonl")
    plan = json.loads((DATA / "mixtures.json").read_text())
    mixtures = []
    for split, ids in plan.items():
        for mixture_id in ids:
            folder = DATA / split / mixture_id
            if not (folder / "outputs").exists():
                continue
            provenance = json.loads((folder / "provenance.json").read_text())
            speakers = provenance["speakers"]
            sections = provenance["sections"]
            base = f"{split}/{mixture_id}"
            entry = {"split": split, "id": mixture_id, "speakers": speakers,
                     "discovery_mix": listening_copy(folder / sections["discovery"]["mixture_path"],
                                                     f"{base}/discovery_mix.wav"),
                     "live_mix": listening_copy(folder / sections["live"]["mixture_path"],
                                                f"{base}/live_mix.wav"),
                     "live_segments": sections["live"]["segments"],
                     "discovery_seconds": sections["discovery"]["frames"] / 16000,
                     "live_seconds": sections["live"]["frames"] / 16000,
                     "references": {s: listening_copy(folder / sections["live"]["source_paths"][s],
                                                      f"{base}/reference_{s}.wav") for s in speakers},
                     "enrollments": {s: listening_copy(folder / f"audio/{s}_enrollment.wav",
                                                       f"{base}/enrollment_{s}.wav") for s in speakers},
                     "discovery": {}, "live": defaultdict(dict)}
            for encoder in ("ecapa", "resemblyzer"):
                row = next(r for r in audit if r["mixture"] == mixture_id and r["encoder"] == encoder
                           and r["buffer_rounds"] == FULL_BUFFER)
                entry["discovery"][encoder] = [{
                    **cluster,
                    "audio": listening_copy(folder / f"outputs/discovery_{encoder}/speaker_{cluster['cluster'] + 1}.wav",
                                            f"{base}/discovery_{encoder}_speaker_{cluster['cluster'] + 1}.wav")}
                    for cluster in row["clusters"]]
            for r in live:
                if r["mixture"] != mixture_id or not r["output_file"]:
                    continue
                key = (f"clean:{r['target']}" if r["enrollment"] == "clean"
                       else f"cluster:{r['cluster']}")
                entry["live"][key][r["model"]] = {
                    "audio": listening_copy(folder / r["output_file"],
                                            f"{base}/{Path(r['output_file']).name}"),
                    "target": r["target"],
                    "overall_si_sdri_db": r["overall_si_sdri_db"],
                    "absent_suppression_db": r["absent_suppression_db"],
                    "target_alone_si_sdr_db": r["target_alone_si_sdr_db"],
                    "si_sdri_by_overlap_db": r["si_sdri_by_overlap_db"]}
            entry["live"] = dict(entry["live"])
            mixtures.append(entry)
            print(f"showcase: {mixture_id}", flush=True)
    payload = {"config": {k: config[k] for k in ("models", "window_seconds", "hop_seconds",
                                                  "discovery_tuning", "live_tuning")},
               "summary": summarize(audit, fidelity, live), "mixtures": mixtures}
    for tuning in payload["config"]["discovery_tuning"].values():
        tuning.pop("grid", None)
    for model in payload["config"]["live_tuning"].values():
        for tuning in model.values():
            tuning.pop("grid", None)
    template = (ROOT / "discovery_showcase_template.html").read_text()
    html = template.replace("/*__DATA__*/null", json.dumps(payload))
    (SHOW / "index.html").write_text(html)
    print(f"wrote {SHOW / 'index.html'}")


if __name__ == "__main__":
    main()
