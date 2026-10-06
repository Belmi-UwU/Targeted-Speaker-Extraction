"""Diagnostic only: use clean sources to locate separator versus matcher errors."""

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf
from scipy.optimize import linear_sum_assignment

from score_outputs import si_sdr
from validate_manifest import audio_path, read_manifest


def wave8(path):
    audio, rate = sf.read(path, dtype="float64")
    if rate == 16000:
        return scipy.signal.resample_poly(audio, 1, 2)
    if rate == 8000:
        return audio
    raise ValueError(f"unsupported rate {rate}: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    args = parser.parse_args()
    data = args.data_dir.resolve()
    manifest = data / "single_trials.jsonl"
    trials = read_manifest(manifest)
    sources = trials[0]["sources"]
    speaker_ids = [source["speaker_id"] for source in sources]
    base_id = trials[0]["trial_id"].split("__speaker_")[0]
    clean = [wave8(audio_path(manifest, source["source_path"])) for source in sources]
    separated = [wave8(data / "sepformer_outputs" / f"{base_id}__stream_{i}.wav")
                 for i in range(3)]
    score_matrix = np.array([[si_sdr(source, stream) for stream in separated]
                             for source in clean])
    rows, cols = linear_sum_assignment(-score_matrix)
    oracle_map = {speaker_ids[row]: int(col) for row, col in zip(rows, cols)}
    run = json.loads((data / "sepformer_run.json").read_text())
    actual = {tuple(record["selected_speaker_ids"]): record["matched_stream_by_speaker"]
              for record in run["trials"]}
    # Recover the signed gain per anonymous stream from the run record directly.
    gains = run["mixture_fit_gains_by_stream"]
    calibrated_streams = [stream * gain for stream, gain in zip(separated, gains)]
    mix = wave8(audio_path(manifest, trials[0]["mixture_path"]))
    pairs = []
    for trial in read_manifest(data / "selected_set_trials.jsonl"):
        ids = trial["selected_speaker_ids"]
        if len(ids) != 2:
            continue
        clean_sum = sum((clean[speaker_ids.index(sid)] for sid in ids),
                        np.zeros_like(mix))
        oracle_sum = sum((calibrated_streams[oracle_map[sid]] for sid in ids),
                         np.zeros_like(mix))
        oracle_score = si_sdr(clean_sum, oracle_sum) - si_sdr(clean_sum, mix)
        actual_map = actual[tuple(ids)]
        pairs.append({"selected": ids,
                      "actual_match": actual_map,
                      "actual_matches_clean_oracle": all(actual_map[sid] == oracle_map[sid] for sid in ids),
                      "clean_oracle_si_sdri_db": oracle_score})
    record = {
        "warning": "clean references used only for post-hoc diagnosis; oracle mappings and scores are not deployable or model outputs",
        "speaker_ids": speaker_ids,
        "stream_si_sdr_to_clean_source_db": score_matrix.tolist(),
        "clean_oracle_mapping": oracle_map,
        "pairs": pairs,
    }
    (data / "matching_diagnostic.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"data_dir": str(data), "oracle_mapping": oracle_map,
                      "pairs": pairs}, indent=2))


if __name__ == "__main__":
    main()
