"""Discovery enrollment experiment: discovery audit, embedding fidelity, live extraction.

Thresholds are tuned on the dev speakers only and reported on the disjoint
test speakers. Clean sources are used only after inference, to audit which
speaker each discovered track holds (oracle analysis) and to score outputs.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

import discovery_pipeline as dp
from score_outputs import si_sdr

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/discovery"
RESULTS = DATA / "results"
BUFFERS = {1: "round1", 2: "round2", 4: "round4"}
FULL_BUFFER = 4
ENCODERS = ("ecapa", "resemblyzer")
CLUSTER_GRID = {"ecapa": np.round(np.arange(0.30, 0.86, 0.05), 3),
                "resemblyzer": np.round(np.arange(0.10, 0.51, 0.025), 3)}
MIN_WINDOWS = (1, 2, 3)
SELECTOR_GRID = np.round(np.arange(0.15, 0.61, 0.05), 3)
COMPETITIVE_GRID = np.round(np.arange(0.0, 0.51, 0.05), 3)
LIVE_MODELS = ("speakerbeam_ss", "sepformer_selector", "sepformer_competitive")
PURE_SHARE = 0.8


def load_mixture(split, mixture_id):
    folder = DATA / split / mixture_id
    provenance = json.loads((folder / "provenance.json").read_text())
    speakers = provenance["speakers"]
    sections = {}
    for name in ("discovery", "live"):
        section = provenance["sections"][name]
        sections[name] = {
            "mix": dp.read_wave(folder / section["mixture_path"]),
            "sources": np.stack([dp.read_wave(folder / section["source_paths"][s])
                                 for s in speakers]),
            "segments": section["segments"],
        }
    sections["discovery"]["round_end_frames"] = provenance["sections"]["discovery"]["round_end_frames"]
    enrollments = {s: dp.read_wave(folder / f"audio/{s}_enrollment.wav") for s in speakers}
    return {"split": split, "id": mixture_id, "folder": folder, "speakers": speakers,
            "sections": sections, "enrollments": enrollments}


def audit(found, sources, speakers):
    """Oracle audit: which clean source dominates each discovered track."""
    clusters = []
    basis = sources.T.astype("float64")
    for index, speaker in enumerate(found):
        coef, _, _, _ = np.linalg.lstsq(basis, speaker["track"].astype("float64"), rcond=None)
        energy = np.array([np.sum((coef[k] * basis[:, k]) ** 2) for k in range(len(speakers))])
        share = energy / max(float(energy.sum()), 1e-20)
        dominant = int(np.argmax(share))
        clusters.append({"cluster": index, "dominant_speaker": speakers[dominant],
                         "purity": float(share[dominant]),
                         "share_by_speaker": {s: float(v) for s, v in zip(speakers, share)},
                         "windows": len(speaker["windows"])})
    mapping, splits, missed = {}, [], []
    for s in speakers:
        mine = [c for c in clusters if c["dominant_speaker"] == s]
        if not mine:
            missed.append(s)
            continue
        if len(mine) > 1:
            splits.append(s)
        mapping[s] = max(mine, key=lambda c: c["purity"])["cluster"]
    merged = [c["cluster"] for c in clusters if c["purity"] < PURE_SHARE]
    extra = len(clusters) - len(mapping)
    errors = len(missed) + extra + len(merged)
    return {"clusters": clusters, "mapping": mapping, "missed": missed, "split_speakers": splits,
            "merged_clusters": merged, "discovered": len(clusters), "true_speakers": len(speakers),
            "exact": errors == 0, "errors": errors}


def discovery_runs(models, mixtures):
    """Separate each discovery buffer once and embed streams with both encoders."""
    cache = {}
    for mixture in mixtures:
        section = mixture["sections"]["discovery"]
        for rounds in BUFFERS:
            end = section["round_end_frames"][rounds - 1]
            start = time.perf_counter()
            separation = dp.separate_windows(models, section["mix"][:end])
            for encoder in ENCODERS:
                dp.embed_streams(models, separation, encoder)
            cache[(mixture["id"], rounds)] = {"separation": separation, "end": end,
                                              "seconds": time.perf_counter() - start}
            print(f"separated {mixture['id']} buffer={rounds} rounds", flush=True)
    return cache


def run_discovery(cache, mixture, rounds, encoder, params):
    entry = cache[(mixture["id"], rounds)]
    found = dp.cluster_streams(entry["separation"], encoder, *params)
    sources = mixture["sections"]["discovery"]["sources"][:, :entry["end"]]
    return found, audit(found, sources, mixture["speakers"])


def tune_discovery(cache, dev):
    tuned = {}
    for encoder in ENCODERS:
        table = []
        for threshold in CLUSTER_GRID[encoder]:
            for min_windows in MIN_WINDOWS:
                errors = sum(run_discovery(cache, m, FULL_BUFFER, encoder,
                                           (float(threshold), min_windows))[1]["errors"]
                             for m in dev)
                table.append({"distance_threshold": float(threshold),
                              "min_windows": min_windows, "dev_errors": errors})
        best = min(e["dev_errors"] for e in table)
        ties = [e for e in table if e["dev_errors"] == best]
        choice = ties[len(ties) // 2]
        tuned[encoder] = {"chosen": choice, "grid": table}
        print(f"discovery {encoder}: {choice}", flush=True)
    return tuned


def embedding_fidelity(models, mixture, rounds, discovered, audit_result, split_enrollments):
    entry_end = mixture["sections"]["discovery"]["round_end_frames"][rounds - 1]
    rows = []
    for k, s in enumerate(mixture["speakers"]):
        inmix = mixture["sections"]["discovery"]["sources"][k, :entry_end]
        row = {"mixture": mixture["id"], "speaker": s, "buffer_rounds": rounds,
               "solo_seconds_in_buffer": rounds * 2.5,
               "discovered": s in audit_result["mapping"]}
        track = (discovered[audit_result["mapping"][s]]["track"]
                 if row["discovered"] else None)
        for encoder in ENCODERS:
            enroll = split_enrollments[encoder][s]
            e_inmix = models.embed(encoder, inmix)
            e_disc = models.embed(encoder, track) if track is not None else None
            row[f"{encoder}_cos_inmix_vs_enroll"] = (float(np.dot(e_inmix, enroll))
                                                     if e_inmix is not None else None)
            if e_disc is None:
                row.update({f"{encoder}_cos_disc_vs_enroll": None,
                            f"{encoder}_margin_mixture": None, f"{encoder}_margin_split": None})
                continue
            row[f"{encoder}_cos_disc_vs_enroll"] = float(np.dot(e_disc, enroll))
            others_mix = [float(np.dot(e_disc, split_enrollments[encoder][o]))
                          for o in mixture["speakers"] if o != s]
            others_split = [float(np.dot(e_disc, v)) for o, v in split_enrollments[encoder].items()
                            if o != s]
            row[f"{encoder}_margin_mixture"] = row[f"{encoder}_cos_disc_vs_enroll"] - max(others_mix)
            row[f"{encoder}_margin_split"] = row[f"{encoder}_cos_disc_vs_enroll"] - max(others_split)
        rows.append(row)
    return rows


def spans(segments, predicate):
    return [(seg["start_frame"], seg["end_frame"]) for seg in segments if predicate(seg)]


def gather(wave, ranges):
    return np.concatenate([wave[a:b] for a, b in ranges]) if ranges else np.zeros(0)


def rms_db(wave):
    return 10 * np.log10(max(float(np.mean(np.square(wave.astype("float64")))), 1e-20))


def score_live(output, mixture, target):
    section = mixture["sections"]["live"]
    k = mixture["speakers"].index(target)
    reference, mix, segments = section["sources"][k], section["mix"], section["segments"]
    result = {"overall_si_sdri_db": si_sdr(reference, output) - si_sdr(reference, mix)}
    alone = spans(segments, lambda seg: seg["active_speaker_ids"] == [target])
    # The input is already clean when the target talks alone, so report output
    # SI-SDR there (target preservation) instead of an improvement.
    result["target_alone_si_sdr_db"] = (si_sdr(gather(reference, alone), gather(output, alone))
                                        if alone else None)
    by_category = {}
    for others in range(1, 4):
        ranges = spans(segments, lambda seg: target in seg["active_speaker_ids"]
                       and len(seg["active_speaker_ids"]) == others + 1)
        if ranges:
            ref, out, inp = gather(reference, ranges), gather(output, ranges), gather(mix, ranges)
            by_category[f"target_plus_{others}"] = si_sdr(ref, out) - si_sdr(ref, inp)
    result["si_sdri_by_overlap_db"] = by_category
    active = spans(segments, lambda seg: target in seg["active_speaker_ids"])
    absent = spans(segments, lambda seg: target not in seg["active_speaker_ids"])
    out_ratio = rms_db(gather(output, absent)) - rms_db(gather(output, active))
    in_ratio = rms_db(gather(mix, absent)) - rms_db(gather(mix, active))
    result["absent_vs_active_output_db"] = out_ratio
    result["absent_vs_active_input_db"] = in_ratio
    result["absent_suppression_db"] = in_ratio - out_ratio
    return result


def live_separation(models, mixture):
    separation = dp.separate_windows(models, mixture["sections"]["live"]["mix"])
    dp.embed_streams(models, separation, "ecapa")
    return separation


def run_live(models, model, run, mixture, live_cache, params):
    if model == "speakerbeam_ss":
        return dp.speakerbeam_extract(models, mixture["sections"]["live"]["mix"], run["resemblyzer"])
    separation = live_cache[mixture["id"]]
    kind = run["enrollment"]
    if model == "sepformer_selector":
        return dp.select_streams(separation, run["ecapa"], params[model][kind])
    return dp.select_streams_competitive(separation, run["candidates"], run["target_index"],
                                         params[model][kind])


def tune_live(models, dev_runs, live_cache):
    """Tune each selector's threshold per enrollment type on dev runs only."""
    tuned, params = {}, {}
    for model, grid in (("sepformer_selector", SELECTOR_GRID),
                        ("sepformer_competitive", COMPETITIVE_GRID)):
        tuned[model], params[model] = {}, {}
        for kind in ("clean", "discovered"):
            runs = [(m, r) for m, r in dev_runs if r["enrollment"] == kind and
                    (kind == "clean" or (r["buffer_rounds"] == FULL_BUFFER and r["mapped"]))]
            table = []
            for value in grid:
                trial = {model: {kind: float(value)}}
                scores = [score_live(run_live(models, model, r, m, live_cache, trial), m,
                                     r["target"])["overall_si_sdri_db"] for m, r in runs]
                table.append({"threshold": float(value),
                              "dev_median_si_sdri_db": float(np.median(scores))})
            choice = max(table, key=lambda e: e["dev_median_si_sdri_db"])
            tuned[model][kind] = {"chosen": choice, "grid": table}
            params[model][kind] = choice["threshold"]
            print(f"{model} {kind}: {choice}", flush=True)
    return tuned, params


def write_wave(path, wave):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, wave, dp.RATE, subtype="FLOAT")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="mixtures per split (smoke test)")
    args = parser.parse_args()
    started = time.perf_counter()
    plan = json.loads((DATA / "mixtures.json").read_text())
    splits = {split: [load_mixture(split, m) for m in ids[:args.limit]]
              for split, ids in plan.items()}
    models = dp.Models()
    enrollments = {}
    for split, mixtures in splits.items():
        enrollments[split] = {enc: {s: models.embed(enc, m["enrollments"][s])
                                    for m in mixtures for s in m["speakers"]}
                              for enc in ENCODERS}

    cache = discovery_runs(models, splits["dev"] + splits["test"])
    tuned = tune_discovery(cache, splits["dev"])
    params = {enc: (tuned[enc]["chosen"]["distance_threshold"], tuned[enc]["chosen"]["min_windows"])
              for enc in ENCODERS}

    discovery_report, fidelity, discovered_full = [], [], {}
    for split, mixtures in splits.items():
        for mixture in mixtures:
            for rounds in BUFFERS:
                for encoder in ENCODERS:
                    found, result = run_discovery(cache, mixture, rounds, encoder, params[encoder])
                    discovery_report.append({"split": split, "mixture": mixture["id"],
                                             "speakers": len(mixture["speakers"]),
                                             "buffer_rounds": rounds, "encoder": encoder,
                                             "separation_seconds": cache[(mixture["id"], rounds)]["seconds"],
                                             **result})
                    if encoder == "ecapa":
                        rows = embedding_fidelity(models, mixture, rounds, found, result,
                                                  enrollments[split])
                        fidelity.extend({"split": split, **r} for r in rows)
                        discovered_full[(mixture["id"], rounds)] = (found, result)
                    if rounds == FULL_BUFFER:
                        out = mixture["folder"] / f"outputs/discovery_{encoder}"
                        for c, speaker in enumerate(found):
                            write_wave(out / f"speaker_{c + 1}.wav", speaker["track"])
    del cache

    live_cache = {m["id"]: live_separation(models, m) for ms in splits.values() for m in ms}
    all_runs = []
    for split, mixtures in splits.items():
        for mixture in mixtures:
            clean_ecapa = [enrollments[split]["ecapa"][s] for s in mixture["speakers"]]
            for k, s in enumerate(mixture["speakers"]):
                all_runs.append((mixture, {
                    "split": split, "target": s, "enrollment": "clean", "buffer_rounds": None,
                    "cluster": None, "ecapa": clean_ecapa[k],
                    "resemblyzer": enrollments[split]["resemblyzer"][s],
                    "candidates": clean_ecapa, "target_index": k}))
            for rounds in BUFFERS:
                found, result = discovered_full[(mixture["id"], rounds)]
                ecapa = [models.embed("ecapa", sp["track"]) for sp in found]
                usable = [c for c, e in enumerate(ecapa) if e is not None]
                for c, speaker in enumerate(found):
                    dominant = result["clusters"][c]["dominant_speaker"]
                    mapped = result["mapping"].get(dominant) == c
                    if (rounds != FULL_BUFFER and not mapped) or ecapa[c] is None:
                        continue
                    all_runs.append((mixture, {
                        "split": split, "target": dominant, "enrollment": "discovered",
                        "buffer_rounds": rounds, "cluster": c + 1,
                        "cluster_purity": result["clusters"][c]["purity"], "mapped": mapped,
                        "ecapa": ecapa[c], "resemblyzer": models.embed("resemblyzer", speaker["track"]),
                        "candidates": [ecapa[u] for u in usable], "target_index": usable.index(c)}))
    tuned_live, params = tune_live(models, [(m, r) for m, r in all_runs if r["split"] == "dev"],
                                   live_cache)

    live_rows = []
    for mixture, run in all_runs:
        save = run["enrollment"] == "clean" or run["buffer_rounds"] == FULL_BUFFER
        for model in LIVE_MODELS:
            if model == "speakerbeam_ss" and run["resemblyzer"] is None:
                continue
            start = time.perf_counter()
            output = run_live(models, model, run, mixture, live_cache, params)
            seconds = time.perf_counter() - start
            tag = (f"{model}__clean__{run['target']}" if run["enrollment"] == "clean"
                   else f"{model}__discovered_cluster{run['cluster']}")
            if save:
                write_wave(mixture["folder"] / f"outputs/live/{tag}.wav", output)
            live_rows.append({
                "mixture": mixture["id"], "model": model,
                **{k: v for k, v in run.items()
                   if k not in ("ecapa", "resemblyzer", "candidates", "target_index")},
                "known_speakers": len(run["candidates"]),
                "output_file": f"outputs/live/{tag}.wav" if save else None,
                "processing_seconds": seconds,
                **score_live(output, mixture, run["target"])})
    print("live extraction done", flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    record = {
        "scope": "Controlled clean LibriSpeech conversations; offline research models; not streaming or device timing",
        "models": {
            "separator": "SpeechBrain SepFormer Libri3Mix rev 1f2d824e2333aec9e60d0030fb144885c1afc426 (8 kHz, 3 streams)",
            "ecapa": "SpeechBrain ECAPA-TDNN VoxCeleb rev 0f99f2d0ebe89ac095bcc5903c4dd8f72b367286",
            "resemblyzer": "Resemblyzer VoiceEncoder 0.1.4",
            "speakerbeam_ss": f"OpenSpeakerBeam-SS independent checkpoint sha256 {dp.SPEAKERBEAM_HASH}",
            "sepformer_selector": "windowed SepFormer; keep streams whose ECAPA cosine to the target is above a threshold",
            "sepformer_competitive": "windowed SepFormer; keep streams whose closest known speaker (all discovered, or all enrolled) is the target",
        },
        "window_seconds": dp.WINDOW_SECONDS, "hop_seconds": dp.HOP_SECONDS,
        "stream_floor_db": dp.STREAM_FLOOR_DB, "pure_share": PURE_SHARE,
        "discovery_tuning": tuned, "live_tuning": tuned_live,
        "runtime_seconds": time.perf_counter() - started,
    }
    (RESULTS / "config_and_tuning.json").write_text(json.dumps(record, indent=2) + "\n")
    for name, rows in (("discovery_audit", discovery_report), ("embedding_fidelity", fidelity),
                       ("live_scores", live_rows)):
        (RESULTS / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"done in {record['runtime_seconds']:.0f} s")


if __name__ == "__main__":
    main()
