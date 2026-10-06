"""Block-online SepFormer selector: how much quality is left without future audio?

Each block of B seconds is separated together with past context only (a
window of W seconds ending at the block's last sample), and only that newest
block is output. Worst-case algorithmic latency is therefore B; compute time
per block comes on top. The pretrained SepFormer itself is unchanged and
noncausal *within* its window, but no window ever contains audio later than
the block being output, so the system as a whole uses no future audio.

Targets are the discovered ECAPA embeddings (4-round buffer) and the clean
enrollments from the discovery experiment. Thresholds are tuned per
configuration on dev and reported on test. The user's real recording gets
listening outputs for every configuration.
"""

import json
import time
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf
import torch

import discovery_pipeline as dp
from run_discovery_experiment import (COMPETITIVE_GRID, DATA, FULL_BUFFER, RESULTS,
                                      SELECTOR_GRID, load_mixture, score_live)

CONFIGS = [(0.25, 1.0), (0.25, 2.0), (0.5, 1.0), (0.5, 2.0), (1.0, 2.0), (1.0, 4.0), (2.0, 4.0)]
RULES = ("sepformer_selector", "sepformer_competitive")
USER = Path(__file__).resolve().parent / "data/user_recordings/discovery"


def block_online_separation(models, mix16, block_s, window_s):
    block, window = int(block_s * dp.RATE), int(window_s * dp.RATE)
    blocks = []
    for block_start in range(0, len(mix16), block):
        stop = min(block_start + block, len(mix16))
        start = stop - window
        segment = mix16[max(0, start):stop]
        if start < 0:  # stream start: pad with silence instead of reading ahead
            segment = np.pad(segment, (-start, 0))
        began = time.perf_counter()
        streams = []
        if np.sqrt(np.mean(segment ** 2)) >= dp.SILENT_WINDOW_RMS:
            seg8 = scipy.signal.resample_poly(segment, 1, 2).astype("float32")
            with torch.inference_mode():
                estimate = models.sepformer.separate_batch(torch.from_numpy(seg8).unsqueeze(0))
            streams8 = estimate[0].cpu().numpy().astype("float64")[:len(seg8)]
            gains, _, _, _ = np.linalg.lstsq(streams8, seg8.astype("float64"), rcond=None)
            fitted = streams8 * gains
            energy = float(np.sum(segment.astype("float64") ** 2))
            for column in range(fitted.shape[1]):
                wave = scipy.signal.resample_poly(fitted[:, column], 2, 1)[:len(segment)]
                wave = np.pad(wave, (0, len(segment) - len(wave))).astype("float32")
                relative = 10 * np.log10(max(float(np.sum(wave.astype("float64") ** 2)), 1e-20) /
                                         max(energy, 1e-20))
                if relative < dp.STREAM_FLOOR_DB:
                    continue
                streams.append({"tail": wave[-(stop - block_start):],
                                "ecapa": models.embed("ecapa", wave)})
        blocks.append({"start": block_start, "stop": stop, "streams": streams,
                       "seconds": time.perf_counter() - began})
    return blocks


def select(blocks, frames, rule, run, threshold):
    out = np.zeros(frames, dtype="float32")
    known = np.stack(run["candidates"])
    for b in blocks:
        chosen = []
        for stream in b["streams"]:
            if stream["ecapa"] is None:
                continue
            if rule == "sepformer_selector":
                keep = float(np.dot(stream["ecapa"], run["ecapa"])) >= threshold
            else:
                sims = known @ stream["ecapa"]
                keep = int(np.argmax(sims)) == run["target_index"] and sims[run["target_index"]] >= threshold
            if keep:
                chosen.append(stream["tail"])
        if chosen:
            out[b["start"]:b["stop"]] = np.sum(chosen, axis=0)
    return out


def runs_for(models, mixture, audit, enrollments):
    row = next(r for r in audit if r["mixture"] == mixture["id"] and r["encoder"] == "ecapa"
               and r["buffer_rounds"] == FULL_BUFFER)
    folder = mixture["folder"] / "outputs/discovery_ecapa"
    disc = [models.embed("ecapa", dp.read_wave(folder / f"speaker_{c['cluster'] + 1}.wav"))
            for c in row["clusters"]]
    usable = [i for i, e in enumerate(disc) if e is not None]
    clean = [enrollments[s] for s in mixture["speakers"]]
    runs = []
    for k, s in enumerate(mixture["speakers"]):
        runs.append({"target": s, "enrollment": "clean", "ecapa": clean[k],
                     "candidates": clean, "target_index": k})
        c = row["mapping"].get(s)
        if c is not None and disc[c] is not None:
            runs.append({"target": s, "enrollment": "discovered", "ecapa": disc[c],
                         "candidates": [disc[i] for i in usable], "target_index": usable.index(c)})
    return runs


def median(values):
    return float(np.median(values)) if values else None


def user_outputs(models, summary):
    """Listening outputs on the real recording for every configuration."""
    run_json = json.loads((USER / "run.json").read_text())
    full = next(v for v in run_json["variants"] if v["variant"] == "full")
    mix = dp.read_wave(USER / full["live_mix"])
    clusters = [(c["cluster"], c["match"]["name"] if c["match"] else None,
                 models.embed("ecapa", dp.read_wave(USER / c["audio"]))) for c in full["clusters"]]
    clusters = [c for c in clusters if c[2] is not None]
    candidates = [e for _, _, e in clusters]
    page = []
    for block_s, window_s in CONFIGS:
        blocks = block_online_separation(models, mix, block_s, window_s)
        floor = summary[f"{block_s}_{window_s}"]["thresholds"]["sepformer_competitive"]["discovered"]
        for i, (cluster, name, emb) in enumerate(clusters):
            out = select(blocks, len(mix), "sepformer_competitive",
                         {"ecapa": emb, "candidates": candidates, "target_index": i}, floor)
            path = USER / "block_online" / f"B{block_s}_W{window_s}_discovered_{cluster}_{name}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            peak = float(np.max(np.abs(out)))
            sf.write(path, out * (0.9 / peak if peak > 1e-9 else 1.0), dp.RATE, subtype="PCM_16")
            page.append({"block": block_s, "window": window_s, "cluster": cluster, "name": name,
                         "audio": str(path.relative_to(USER))})
        print(f"user recording B={block_s} W={window_s} done", flush=True)
    offline = {c["cluster"]: c["audio"].replace("discovered_", "sepformer_competitive__cluster_")
               for c in full["clusters"]}
    rows = "".join(
        f"<tr><td>{'offline 4 s windows, future audio' if cfg is None else f'B {cfg[0]} s / W {cfg[1]} s'}</td>"
        + "".join(f"<td><audio controls preload='none' src='{src}'></audio></td>" for src in srcs) + "</tr>"
        for cfg, srcs in [(None, [offline[c] for c, _, _ in clusters])] +
        [((b, w), [p["audio"] for p in page if p["block"] == b and p["window"] == w])
         for b, w in CONFIGS])
    head = "".join(f"<th>Discovered {c} ({n})</th>" for c, n, _ in clusters)
    (USER / "block_online.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>No-Future Selector Outputs</title>"
        "<style>body{font:15px -apple-system,sans-serif;margin:16px}td,th{padding:4px;text-align:left}"
        "audio{width:220px}</style><h1>Competitive selector without future audio</h1>"
        "<p>B = block length (worst-case added delay); W = window of past audio the separator sees. "
        "Each clip is peak-normalized. Labels are the voice match from the discovery page.</p>"
        f"<table><tr><th>Setting</th>{head}</tr>{rows}</table>")


def main():
    plan = json.loads((DATA / "mixtures.json").read_text())
    splits = {split: [load_mixture(split, m) for m in ids] for split, ids in plan.items()}
    audit = [json.loads(l) for l in (RESULTS / "discovery_audit.jsonl").read_text().splitlines() if l]
    models = dp.Models()
    runs = {}
    for split, mixtures in splits.items():
        enrollments = {s: models.embed("ecapa", m["enrollments"][s]) for m in mixtures for s in m["speakers"]}
        for m in mixtures:
            runs[m["id"]] = runs_for(models, m, audit, enrollments)

    summary, rows = {}, []
    for block_s, window_s in CONFIGS:
        key = f"{block_s}_{window_s}"
        blocks = {}
        for split, mixtures in splits.items():
            for m in mixtures:
                blocks[m["id"]] = block_online_separation(models, m["sections"]["live"]["mix"],
                                                          block_s, window_s)
        thresholds = {}
        for rule, grid in (("sepformer_selector", SELECTOR_GRID),
                           ("sepformer_competitive", COMPETITIVE_GRID)):
            thresholds[rule] = {}
            for kind in ("clean", "discovered"):
                best = None
                for value in grid:
                    scores = [score_live(select(blocks[m["id"]], len(m["sections"]["live"]["mix"]),
                                                rule, r, float(value)), m, r["target"])["overall_si_sdri_db"]
                              for m in splits["dev"] for r in runs[m["id"]] if r["enrollment"] == kind]
                    if best is None or np.median(scores) > best[1]:
                        best = (float(value), float(np.median(scores)))
                thresholds[rule][kind] = best[0]
        compute = [b["seconds"] for bl in blocks.values() for b in bl]
        entry = {"block_seconds": block_s, "window_seconds": window_s,
                 "worst_case_algorithmic_latency_ms": block_s * 1000,
                 "compute_per_block_median_s": median(compute),
                 "compute_per_block_p90_s": float(np.percentile(compute, 90)),
                 "real_time_factor_median": median(compute) / block_s,
                 "thresholds": thresholds, "results": {}}
        for split, mixtures in splits.items():
            for rule in RULES:
                for kind in ("clean", "discovered"):
                    scored = []
                    for m in mixtures:
                        for r in runs[m["id"]]:
                            if r["enrollment"] != kind:
                                continue
                            out = select(blocks[m["id"]], len(m["sections"]["live"]["mix"]), rule, r,
                                         thresholds[rule][kind])
                            score = score_live(out, m, r["target"])
                            scored.append(score)
                            rows.append({"block_seconds": block_s, "window_seconds": window_s,
                                         "split": split, "mixture": m["id"], "rule": rule,
                                         "enrollment": kind, "target": r["target"], **score})
                    entry["results"][f"{split}|{rule}|{kind}"] = {
                        "runs": len(scored),
                        "median_si_sdri_db": median([s["overall_si_sdri_db"] for s in scored]),
                        "p10_si_sdri_db": float(np.percentile([s["overall_si_sdri_db"] for s in scored], 10)),
                        "median_absent_suppression_db": median([s["absent_suppression_db"] for s in scored]),
                        "median_alone_si_sdr_db": median([s["target_alone_si_sdr_db"] for s in scored
                                                          if s["target_alone_si_sdr_db"] is not None])}
        summary[key] = entry
        test = entry["results"]["test|sepformer_competitive|discovered"]
        print(f"B={block_s} W={window_s}: RTF {entry['real_time_factor_median']:.2f}, "
              f"test competitive discovered median {test['median_si_sdri_db']:.1f} dB, "
              f"suppression {test['median_absent_suppression_db']:.1f} dB, thresholds {thresholds}",
              flush=True)
        del blocks
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "block_online_summary.json").write_text(json.dumps({
        "scope": "Block-online SepFormer selection on the discovery conversations' live sections; no audio later than the output block is used; desktop CPU, 4 torch threads",
        "thresholds_note": "tuned per configuration and per enrollment type on dev runs only",
        "configs": summary}, indent=2) + "\n")
    (RESULTS / "block_online_scores.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    if (USER / "run.json").exists():
        user_outputs(models, summary)


if __name__ == "__main__":
    main()
