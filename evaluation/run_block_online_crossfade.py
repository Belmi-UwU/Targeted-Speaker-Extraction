"""Crossfade between output blocks for the best block-online setting (B = 1 s, W = 2 s).

Each window is separated once with a 50 ms lead kept before its block; the
same separation is then assembled with no crossfade, 20 ms, and 50 ms
raised-cosine crossfades. Thresholds come from block_online_summary.json.
"Boundary jump" is the median sample-to-sample jump at block boundaries
divided by the median jump elsewhere in the output; about 1 means no audible
step, much larger means clicks.
"""

import json

import numpy as np
import soundfile as sf

import discovery_pipeline as dp
from run_block_online_selector import USER, block_online_separation, runs_for, select
from run_discovery_experiment import DATA, RESULTS, load_mixture, score_live

BLOCK, WINDOW = 1.0, 2.0
FADES = (0.0, 0.02, 0.05)
RULE = "sepformer_competitive"


def boundary_jump(wave, blocks):
    diffs = np.abs(np.diff(wave.astype("float64")))
    edges = [b["start"] - 1 for b in blocks if b["start"] > 0]
    active = diffs[diffs > 1e-6]
    if not edges or not len(active):
        return None
    return float(np.median(diffs[edges]) / np.median(active))


def main():
    summary = json.loads((RESULTS / "block_online_summary.json").read_text())["configs"]
    thresholds = summary[f"{BLOCK}_{WINDOW}"]["thresholds"][RULE]
    plan = json.loads((DATA / "mixtures.json").read_text())
    audit = [json.loads(l) for l in (RESULTS / "discovery_audit.jsonl").read_text().splitlines() if l]
    models = dp.Models()
    rows = []
    for split, ids in plan.items():
        mixtures = [load_mixture(split, m) for m in ids]
        enrollments = {s: models.embed("ecapa", m["enrollments"][s]) for m in mixtures for s in m["speakers"]}
        for m in mixtures:
            mix = m["sections"]["live"]["mix"]
            blocks = block_online_separation(models, mix, BLOCK, WINDOW, fade_s=max(FADES))
            for run in runs_for(models, m, audit, enrollments):
                for fade in FADES:
                    out = select(blocks, len(mix), RULE, run, thresholds[run["enrollment"]], fade)
                    rows.append({"split": split, "mixture": m["id"], "target": run["target"],
                                 "enrollment": run["enrollment"], "fade_ms": int(fade * 1000),
                                 "boundary_jump": boundary_jump(out, blocks),
                                 **score_live(out, m, run["target"])})
            print(f"done {m['id']}", flush=True)
    table = {}
    for split in plan:
        for kind in ("clean", "discovered"):
            for fade in FADES:
                sel = [r for r in rows if r["split"] == split and r["enrollment"] == kind
                       and r["fade_ms"] == int(fade * 1000)]
                jumps = [r["boundary_jump"] for r in sel if r["boundary_jump"] is not None]
                table[f"{split}|{kind}|{int(fade * 1000)}ms"] = {
                    "runs": len(sel),
                    "median_si_sdri_db": float(np.median([r["overall_si_sdri_db"] for r in sel])),
                    "median_alone_si_sdr_db": float(np.median([r["target_alone_si_sdr_db"] for r in sel
                                                               if r["target_alone_si_sdr_db"] is not None])),
                    "median_boundary_jump": float(np.median(jumps)) if jumps else None}
    (RESULTS / "block_online_crossfade.json").write_text(json.dumps(
        {"block_seconds": BLOCK, "window_seconds": WINDOW, "rule": RULE, "thresholds": thresholds,
         "added_delay_ms": {f"{int(f * 1000)}ms": int(f * 1000) for f in FADES},
         "summary": table, "runs": rows}, indent=2) + "\n")
    for key, value in table.items():
        print(key, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in value.items()})

    if not (USER / "run.json").exists():
        return
    full = next(v for v in json.loads((USER / "run.json").read_text())["variants"]
                if v["variant"] == "full")
    mix = dp.read_wave(USER / full["live_mix"])
    clusters = [(c["cluster"], c["match"]["name"] if c["match"] else "?",
                 models.embed("ecapa", dp.read_wave(USER / c["audio"]))) for c in full["clusters"]]
    clusters = [c for c in clusters if c[2] is not None]
    candidates = [e for _, _, e in clusters]
    blocks = block_online_separation(models, mix, BLOCK, WINDOW, fade_s=max(FADES))
    cells = {}
    for i, (cluster, name, emb) in enumerate(clusters):
        run = {"ecapa": emb, "candidates": candidates, "target_index": i}
        for fade in FADES:
            out = select(blocks, len(mix), RULE, run, thresholds["discovered"], fade)
            path = USER / "block_online" / f"crossfade_{int(fade * 1000)}ms_discovered_{cluster}_{name}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            peak = float(np.max(np.abs(out)))
            sf.write(path, out * (0.9 / peak if peak > 1e-9 else 1.0), dp.RATE, subtype="PCM_16")
            cells[(cluster, fade)] = (str(path.relative_to(USER)), boundary_jump(out, blocks))
    head = "".join(f"<th>Discovered {c} ({n})</th>" for c, n, _ in clusters)
    body = "".join(
        f"<tr><td>{'No crossfade' if not fade else f'Crossfade {int(fade * 1000)} ms'}</td>"
        + "".join(f"<td><audio controls preload='none' src='{cells[(c, fade)][0]}'></audio>"
                  f"<br><small>boundary jump {cells[(c, fade)][1]:.1f}×</small></td>" for c, _, _ in clusters)
        + "</tr>" for fade in FADES)
    (USER / "crossfade.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Crossfade Comparison</title>"
        "<style>body{font:15px -apple-system,sans-serif;margin:16px}td,th{padding:6px;text-align:left}"
        "audio{width:220px}</style><h1>Block 1 s, context 2 s: crossfade comparison</h1>"
        "<p>Competitive selector with the discovered speaker, no future audio. Listen at the block "
        "boundaries (every second) for clicks or jumps. Boundary jump about 1× means smooth. "
        "Each clip is peak-normalized.</p>"
        f"<table><tr><th>Assembly</th>{head}</tr>{body}</table>")
    print(f"wrote {USER / 'crossfade.html'}")


if __name__ == "__main__":
    main()
