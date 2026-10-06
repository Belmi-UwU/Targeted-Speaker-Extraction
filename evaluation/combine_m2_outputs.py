"""Sum independently extracted voices for selected-set offline trials."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from validate_manifest import audio_path, read_manifest, validate_trial

ROOT = Path(__file__).resolve().parent / "data/m2_three_speakers"


def read_wave(path):
    signal, rate = sf.read(path, dtype="float32")
    if rate != 16000 or signal.ndim != 1 or not np.isfinite(signal).all():
        raise ValueError(f"invalid mono 16 kHz waveform: {path}")
    return signal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    manifest = data_dir / "selected_set_trials.jsonl"
    output_dir = data_dir / "model_outputs"
    listening_dir = data_dir / "listening_comparisons"
    output_dir.mkdir(parents=True, exist_ok=True)
    listening_dir.mkdir(parents=True, exist_ok=True)
    trials = read_manifest(manifest)
    combined = {}
    max_peak = 0.0
    for trial in trials:
        validate_trial(manifest, trial, check_mix_sum=True)
        mixture = read_wave(audio_path(manifest, trial["mixture_path"]))
        selected = trial["selected_speaker_ids"]
        base_id = trial["trial_id"].split("__selected_")[0]
        raw_sum = np.zeros_like(mixture)
        for speaker_id in selected:
            raw = read_wave(output_dir / f"{base_id}__speaker_{speaker_id}.wav")
            if len(raw) != len(raw_sum):
                raise ValueError("output length mismatch")
            raw_sum += raw
        if not np.isfinite(raw_sum).all():
            raise ValueError("nonfinite combined output")
        out = output_dir / f"{trial['trial_id']}.wav"
        sf.write(out, raw_sum, 16000, subtype="FLOAT")
        max_peak = max(max_peak, float(np.max(np.abs(raw_sum))))
        clean_sum = np.zeros_like(mixture)
        for source in trial["sources"]:
            if source["speaker_id"] in selected:
                clean_sum += read_wave(audio_path(manifest, source["source_path"]))
        combined[trial["trial_id"]] = (mixture, clean_sum, raw_sum)
    gain = min(0.014, 0.9 / max(max_peak, 1e-12))
    for trial_id, (mixture, clean_sum, raw_sum) in combined.items():
        sf.write(listening_dir / f"{trial_id}__mix.wav", mixture, 16000, subtype="PCM_16")
        sf.write(listening_dir / f"{trial_id}__selected_clean.wav", clean_sum, 16000, subtype="PCM_16")
        sf.write(listening_dir / f"{trial_id}__model_fixed_gain.wav", raw_sum * gain, 16000,
                 subtype="PCM_16")
    metadata = {
        "method": "direct sum of raw single-target outputs; no clean-reference calibration",
        "preview_gain": gain,
        "preview_gain_policy": "M1 fixed gain 0.014, reduced if needed to keep combined PCM16 peaks about 0.9 or lower",
        "max_raw_combined_peak": max_peak,
        "trials": list(combined),
        "warning": "preview gain is for listening only; scoring uses raw float outputs",
    }
    (listening_dir / "comparison_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
