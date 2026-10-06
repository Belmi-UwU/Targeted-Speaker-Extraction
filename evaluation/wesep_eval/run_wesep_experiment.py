"""WeSep causal BSRNN (REAL-TSE baselines) as the live model, beside the SepFormer selector.

Runs in evaluation/wesep_eval/.venv (see README.md here). It reuses the
discovery conversations, the discovered tracks (4-round buffer), the clean
enrollments, and the scoring of ../run_discovery_experiment.py, so its numbers
compare directly with the SepFormer results in ../DISCOVERY_REPORT.md.

The REAL-TSE checkpoints come from a Google Drive archive with **no weights
licence stated**: evaluation only, never redistributed or shipped.

Parts:
1. causality check (prefix test) for every checkpoint;
2. full-sequence extraction of each live section, discovered vs clean enrollment;
3. block + past-context processing (no future audio) at small block sizes for
   the causal checkpoints, with compute time per block (real-time factor).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
sys.path.insert(0, str(EVAL))
sys.path.insert(0, str(HERE / "upstream/wesep-real-tse"))

import discovery_pipeline as dp  # noqa: E402
from run_discovery_experiment import DATA, FULL_BUFFER, RESULTS, load_mixture, score_live  # noqa: E402

CHECKPOINTS = HERE / "checkpoints/pretrained"
MODELS = {"spk_emb_causal_100": True, "tfmap_context_causal_100": True,
          "spk_emb_100": False, "tfmap_context_100": False}
# (block, context) pairs whose recompute cost can keep up on a laptop CPU; see README.md.
BLOCK_CONFIGS = [(1.0, 2.0), (0.5, 1.0), (0.25, 1.0), (0.25, 0.5)]
BLOCK_MODELS = ("tfmap_context_causal_100",)
SPLITS = ("test",)  # WeSep has no thresholds to tune, so dev is not needed
ENROLL_SECONDS = 8.0
OUT = HERE / "outputs"


def load(name):
    from wesep.cli.extractor import load_model_local
    model = load_model_local(str(CHECKPOINTS / name))
    model.set_output_norm(False)
    model.set_device("cpu")
    return model


ENROLL_RMS_DBFS = -25.0


def enrollment(wave):
    """Speech-only part of an enrollment or discovered track, capped for compute.

    The level is normalized to -25 dBFS: WeSep's enrollment features are
    level-sensitive, and raw discovered tracks from a real recording can be far
    quieter than the LibriSpeech-level enrollments it was trained with.
    """
    speech = dp.active_speech(wave)[:int(ENROLL_SECONDS * dp.RATE)]
    rms = float(np.sqrt(np.mean(speech.astype("float64") ** 2))) if len(speech) else 0.0
    return speech * (10 ** (ENROLL_RMS_DBFS / 20) / rms) if rms > 1e-8 else speech


def extract(model, mix, enroll):
    with torch.no_grad():
        out = model.extract_speech_from_pcm(torch.from_numpy(mix.astype("float32"))[None], dp.RATE,
                                            torch.from_numpy(enroll.astype("float32"))[None], dp.RATE)
    out = out[0].numpy().astype("float32")
    return np.pad(out, (0, max(0, len(mix) - len(out))))[:len(mix)]


def causality(model, mix, enroll):
    """Largest output change before the cut when the audio after the cut is replaced."""
    base = extract(model, mix[:12 * dp.RATE], enroll)
    worst = 0.0
    for cut in (4, 6, 8):
        altered = mix[:12 * dp.RATE].copy()
        altered[cut * dp.RATE:] = np.random.default_rng(cut).uniform(-0.5, 0.5, len(altered) - cut * dp.RATE)
        out = extract(model, altered, enroll)
        before = slice(0, cut * dp.RATE - 512)  # exclude one STFT window at the cut
        worst = max(worst, float(np.max(np.abs(out[before] - base[before]))) /
                    max(float(np.max(np.abs(base[before]))), 1e-12))
    return worst


def block_online(model, mix, enroll, block_s, window_s):
    block, window = int(block_s * dp.RATE), int(window_s * dp.RATE)
    out = np.zeros(len(mix), dtype="float32")
    times = []
    for start in range(0, len(mix), block):
        stop = min(start + block, len(mix))
        segment = mix[max(0, stop - window):stop]
        began = time.perf_counter()
        estimate = extract(model, segment, enroll)
        times.append(time.perf_counter() - began)
        out[start:stop] = estimate[-(stop - start):]
    return out, times


def runs_for(mixture, audit):
    row = next(r for r in audit if r["mixture"] == mixture["id"] and r["encoder"] == "ecapa"
               and r["buffer_rounds"] == FULL_BUFFER)
    runs = []
    for s in mixture["speakers"]:
        runs.append({"target": s, "enrollment": "clean", "audio": enrollment(mixture["enrollments"][s])})
        c = row["mapping"].get(s)
        if c is not None:
            track = dp.read_wave(mixture["folder"] / f"outputs/discovery_ecapa/speaker_{c + 1}.wav")
            runs.append({"target": s, "enrollment": "discovered", "audio": enrollment(track)})
    return runs


def summarize(rows, keys):
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    table = []
    for key, items in sorted(groups.items()):
        table.append({**dict(zip(keys, key)), "runs": len(items),
                      "median_si_sdri_db": float(np.median([r["overall_si_sdri_db"] for r in items])),
                      "p10_si_sdri_db": float(np.percentile([r["overall_si_sdri_db"] for r in items], 10)),
                      "median_alone_si_sdr_db": float(np.median([r["target_alone_si_sdr_db"] for r in items
                                                                 if r["target_alone_si_sdr_db"] is not None])),
                      "median_absent_suppression_db": float(np.median([r["absent_suppression_db"] for r in items])),
                      **({"median_real_time_factor": float(np.median([r["real_time_factor"] for r in items]))}
                         if "real_time_factor" in items[0] else {})})
    return table


def main():
    torch.set_num_threads(4)
    plan = json.loads((DATA / "mixtures.json").read_text())
    audit = [json.loads(l) for l in (RESULTS / "discovery_audit.jsonl").read_text().splitlines() if l]
    mixtures = {split: [load_mixture(split, m) for m in plan[split]] for split in SPLITS}
    record = {"checkpoints": {}, "full": [], "block": []}
    probe = mixtures["test"][0]
    probe_enroll = enrollment(probe["enrollments"][probe["speakers"][0]])
    for name, causal in MODELS.items():
        model = load(name)
        params = sum(p.numel() for p in model.model.parameters())
        drift = causality(model, probe["sections"]["live"]["mix"], probe_enroll)
        record["checkpoints"][name] = {"declared_causal": causal, "parameters": params,
                                       "prefix_change_before_cut_relative": drift}
        print(f"{name}: {params / 1e6:.2f} M params, prefix change {drift:.2e}", flush=True)
        for split, items in mixtures.items():
            for m in items:
                mix = m["sections"]["live"]["mix"]
                for run in runs_for(m, audit):
                    began = time.perf_counter()
                    out = extract(model, mix, run["audio"])
                    seconds = time.perf_counter() - began
                    record["full"].append({"model": name, "causal": causal, "split": split, "mixture": m["id"],
                                           "target": run["target"], "enrollment": run["enrollment"],
                                           "real_time_factor": seconds / (len(mix) / dp.RATE),
                                           **score_live(out, m, run["target"])})
                    if split == "test" and run["enrollment"] == "discovered":
                        path = OUT / m["id"] / f"{name}__discovered_{run['target']}.wav"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        import soundfile as sf
                        sf.write(path, out, dp.RATE, subtype="FLOAT")
        print(f"  full-sequence done", flush=True)
        if name not in BLOCK_MODELS:
            continue
        for block_s, window_s in BLOCK_CONFIGS:
            for m in mixtures["test"]:
                mix = m["sections"]["live"]["mix"]
                for run in [r for r in runs_for(m, audit) if r["enrollment"] == "discovered"]:
                    out, times = block_online(model, mix, run["audio"], block_s, window_s)
                    record["block"].append({"model": name, "block_seconds": block_s, "window_seconds": window_s,
                                            "mixture": m["id"], "target": run["target"],
                                            "real_time_factor": float(np.median(times)) / block_s,
                                            "compute_p90_s": float(np.percentile(times, 90)),
                                            **score_live(out, m, run["target"])})
            print(f"  block B={block_s} W={window_s} done", flush=True)
    record["full_summary"] = summarize(record["full"], ("model", "split", "enrollment"))
    record["block_summary"] = summarize(record["block"], ("model", "block_seconds", "window_seconds"))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "wesep_comparison.json").write_text(json.dumps(record, indent=2) + "\n")
    for row in record["full_summary"] + record["block_summary"]:
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()})


if __name__ == "__main__":
    main()
