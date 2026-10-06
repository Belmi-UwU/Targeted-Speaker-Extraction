"""Score all M2 methods after the same 16-to-8 kHz resampling step."""

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf

from score_outputs import si_sdr
from validate_manifest import audio_path, read_manifest, validate_trial

DEFAULT_DATA = Path(__file__).resolve().parent / "data/m2_three_speakers"


def wave8(path):
    audio, rate = sf.read(path, dtype="float64")
    if rate != 16000 or audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError(f"expected finite mono 16 kHz WAV: {path}")
    return scipy.signal.resample_poly(audio, 1, 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    data = args.data_dir.resolve()
    methods = {
        "speakerbeam_ss_sum": data / "model_outputs",
        "sepformer_raw_sum": data / "sepformer_outputs",
        "sepformer_mixfit": data / "sepformer_mixfit_outputs",
        "sepformer_ecapa_mixfit": data / "sepformer_ecapa_outputs",
    }
    methods = {name: folder for name, folder in methods.items() if folder.is_dir()}
    records = []
    for name in ("single_trials.jsonl", "selected_set_trials.jsonl"):
        manifest = data / name
        for trial in read_manifest(manifest):
            validate_trial(manifest, trial, check_mix_sum=True)
            mix = wave8(audio_path(manifest, trial["mixture_path"]))
            reference = np.zeros_like(mix)
            for source in trial["sources"]:
                if source["speaker_id"] in trial["selected_speaker_ids"]:
                    reference += wave8(audio_path(manifest, source["source_path"]))
            input_score = si_sdr(reference, mix)
            for method, output_dir in methods.items():
                estimate = wave8(output_dir / f"{trial['trial_id']}.wav")
                if len(estimate) != len(mix):
                    raise ValueError("resampled length mismatch")
                output_score = si_sdr(reference, estimate)
                records.append({
                    "trial_id": trial["trial_id"], "selected_speaker_ids": trial["selected_speaker_ids"],
                    "method": method, "score_sample_rate_hz": 8000,
                    "si_sdr_input_db": input_score, "si_sdr_output_db": output_score,
                    "si_sdri_db": output_score - input_score,
                })
    out = data / "common8k_scores.jsonl"
    out.write_text("\n".join(json.dumps(record, allow_nan=False) for record in records) + "\n")
    for record in records:
        if len(record["selected_speaker_ids"]) == 2:
            print(",".join(record["selected_speaker_ids"]), record["method"],
                  f"{record['si_sdri_db']:+.2f} dB")
    print(f"wrote {len(records)} scores to {out}")


if __name__ == "__main__":
    main()
