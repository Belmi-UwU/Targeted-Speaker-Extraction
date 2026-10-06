"""Score target-only, target-absent, and overlap segments in controlled M2 clip."""

import json
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf

from score_outputs import si_sdr

DATA = Path(__file__).resolve().parent / "data/m2_turns"
METHODS = {
    "speakerbeam_ss_sum": DATA / "model_outputs",
    "sepformer_raw_sum": DATA / "sepformer_outputs",
    "sepformer_mixfit": DATA / "sepformer_mixfit_outputs",
}
GUARD_8K_FRAMES = 800  # 0.1 second from either end of each segment


def wave8(path):
    wave, rate = sf.read(path, dtype="float64")
    if rate != 16000 or wave.ndim != 1:
        raise ValueError(f"expected 16 kHz mono WAV: {path}")
    return scipy.signal.resample_poly(wave, 1, 2)


def rms(wave):
    return float(np.sqrt(np.mean(wave**2)))


def db_ratio(a, b):
    return float(20 * np.log10(max(a, 1e-12) / max(b, 1e-12)))


def main():
    trial = next(json.loads(line) for line in
                 (DATA / "selected_set_trials.jsonl").read_text().splitlines()
                 if json.loads(line)["selected_speaker_ids"] == ["4077", "5142"])
    mix = wave8(DATA / trial["mixture_path"])
    sources = {source["speaker_id"]: wave8(DATA / source["source_path"])
               for source in trial["sources"]}
    reference = sources["4077"] + sources["5142"]
    segments = json.loads((DATA / "provenance.json").read_text())["segments"]
    ranges = {segment["label"]: slice(segment["start_frame"] // 2 + GUARD_8K_FRAMES,
                                      segment["end_frame"] // 2 - GUARD_8K_FRAMES)
              for segment in segments}
    solo = np.concatenate((mix[ranges["4077_only"]], mix[ranges["5142_only"]]))
    mix_absent = rms(mix[ranges["selected_absent_5683_only"]])
    records = []
    for method, folder in METHODS.items():
        output = wave8(folder / f"{trial['trial_id']}.wav")
        absent = rms(output[ranges["selected_absent_5683_only"]])
        target_solo = rms(np.concatenate((output[ranges["4077_only"]],
                                          output[ranges["5142_only"]])))
        overlap = ranges["all_three_overlap"]
        records.append({
            "method": method, "selected_speaker_ids": ["4077", "5142"],
            "segment_guard_seconds": 0.1,
            "target_absent_output_rms": absent,
            "target_solo_output_rms": target_solo,
            "absent_to_target_solo_output_db": db_ratio(absent, target_solo),
            "input_absent_to_target_solo_db": db_ratio(mix_absent, rms(solo)),
            "absent_output_to_input_db": db_ratio(absent, mix_absent),
            "4077_solo_si_sdr_db": si_sdr(reference[ranges["4077_only"]],
                                          output[ranges["4077_only"]]),
            "5142_solo_si_sdr_db": si_sdr(reference[ranges["5142_only"]],
                                          output[ranges["5142_only"]]),
            "all_overlap_si_sdri_db":
                si_sdr(reference[overlap], output[overlap]) -
                si_sdr(reference[overlap], mix[overlap]),
        })
    (DATA / "segment_scores.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
