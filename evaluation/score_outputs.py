"""Score raw model WAV outputs for validated clean-mixture trials."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from validate_manifest import audio_info, audio_path, read_manifest, validate_trial


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Scale-invariant SDR in dB, capped at +/-120 dB for degenerate outputs."""
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    if reference.shape != estimate.shape or reference.ndim != 1:
        raise ValueError("reference and estimate must be aligned mono arrays")
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    target_energy = float(np.dot(reference, reference))
    if target_energy <= 1e-12:
        raise ValueError("reference has no usable energy")
    projection = np.dot(estimate, reference) / target_energy * reference
    signal_energy = float(np.dot(projection, projection))
    distortion = estimate - projection
    error_energy = float(np.dot(distortion, distortion))
    floor = target_energy * 1e-12
    score = 10 * np.log10(max(signal_energy, floor) / max(error_energy, floor))
    return float(np.clip(score, -120.0, 120.0))


def score_trial(manifest: Path, trial: dict, outputs: Path):
    validate_trial(manifest, trial)
    mixture, _ = sf.read(audio_path(manifest, trial["mixture_path"]), dtype="float64")
    selected = set(trial["selected_speaker_ids"])
    reference = np.zeros_like(mixture)
    for source in trial["sources"]:
        if source["speaker_id"] in selected:
            signal, _ = sf.read(audio_path(manifest, source["source_path"]), dtype="float64")
            reference += signal
    output_path = outputs / f'{trial["trial_id"]}.wav'
    if not output_path.is_file():
        raise ValueError(f"model output missing: {output_path}")
    if audio_info(output_path, 16000).frames != len(mixture):
        raise ValueError(f"output length differs from mixture: {output_path}")
    estimate, _ = sf.read(output_path, dtype="float64")
    input_score = si_sdr(reference, mixture)
    output_score = si_sdr(reference, estimate)
    per_source_scores = {}
    for source in trial["sources"]:
        clean, _ = sf.read(audio_path(manifest, source["source_path"]), dtype="float64")
        per_source_scores[source["speaker_id"]] = si_sdr(clean, estimate)
    return {
        "trial_id": trial["trial_id"], "dataset": trial["dataset"],
        "dataset_revision": trial["dataset_revision"],
        "selected_speaker_ids": trial["selected_speaker_ids"],
        "si_sdr_input_db": input_score, "si_sdr_output_db": output_score,
        "si_sdri_db": output_score - input_score,
        "output_si_sdr_by_source_db": per_source_scores,
        "output_peak": float(np.max(np.abs(estimate))),
        "output_rms": float(np.sqrt(np.mean(estimate**2))),
        "output_fraction_outside_unit_range": float(np.mean(np.abs(estimate) > 1.0)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    for trial in read_manifest(args.manifest):
        print(json.dumps(score_trial(args.manifest, trial, args.output_dir), allow_nan=False))


if __name__ == "__main__":
    main()
