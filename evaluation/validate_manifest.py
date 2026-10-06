"""Validate the clean 16 kHz LibriMix trial contract before scoring."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf

TRIAL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
REQUIRED_TRIAL = {"trial_id", "dataset", "dataset_revision", "speech_license",
                  "split", "condition", "sample_rate_hz",
                  "mixture_path", "selected_speaker_ids", "sources"}
REQUIRED_SOURCE = {"speaker_id", "source_path", "source_utterance_id",
                   "enrollment_path", "enrollment_utterance_id"}


def read_manifest(path: Path):
    trials = []
    ids = set()
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            trial = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_no}: invalid JSON: {exc}") from exc
        if not isinstance(trial, dict) or set(trial) != REQUIRED_TRIAL:
            raise ValueError(f"line {line_no}: trial fields must be {sorted(REQUIRED_TRIAL)}")
        trial_id = trial["trial_id"]
        if not isinstance(trial_id, str) or not TRIAL_ID.fullmatch(trial_id):
            raise ValueError(f"line {line_no}: unsafe or empty trial_id")
        if trial_id in ids:
            raise ValueError(f"line {line_no}: duplicate trial_id {trial_id}")
        ids.add(trial_id)
        trials.append(trial)
    if not trials:
        raise ValueError("manifest contains no trials")
    return trials


def audio_path(manifest: Path, value: str):
    if not isinstance(value, str) or not value:
        raise ValueError("audio path must be a nonempty string")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"audio path must stay relative to manifest directory: {value}")
    path = manifest.parent / relative
    if not path.is_file():
        raise ValueError(f"audio file missing: {path}")
    return path


def audio_info(path: Path, sample_rate: int):
    info = sf.info(path)
    if info.samplerate != sample_rate or info.channels != 1 or info.frames <= 0:
        raise ValueError(f"expected nonempty mono {sample_rate} Hz WAV: {path}; got {info}")
    return info


def validate_trial(manifest: Path, trial: dict, check_mix_sum=False):
    trial_id = trial["trial_id"]
    if trial["dataset"] not in ("Libri2Mix", "Libri3Mix", "ControlledLibri3Mix"):
        raise ValueError(f"{trial_id}: unsupported dataset")
    for key in ("dataset_revision", "speech_license"):
        if not isinstance(trial[key], str) or not trial[key]:
            raise ValueError(f"{trial_id}: {key} must be a nonempty string")
    allowed = {("Libri2Mix", "mix_clean"), ("Libri3Mix", "mix_clean"),
               ("ControlledLibri3Mix", "turns_clean")}
    if trial["split"] != "test" or (trial["dataset"], trial["condition"]) not in allowed:
        raise ValueError(f"{trial_id}: unsupported split/dataset/condition combination")
    if trial["sample_rate_hz"] != 16000:
        raise ValueError(f"{trial_id}: M1 requires 16 kHz audio")
    sources = trial["sources"]
    expected = 2 if trial["dataset"] == "Libri2Mix" else 3
    if not isinstance(sources, list) or len(sources) != expected:
        raise ValueError(f"{trial_id}: expected {expected} sources")
    selected = trial["selected_speaker_ids"]
    if not isinstance(selected, list) or not selected or any(
        not isinstance(s, str) or not s for s in selected
    ) or len(set(selected)) != len(selected):
        raise ValueError(f"{trial_id}: selected_speaker_ids must be distinct nonempty strings")
    mix_path = audio_path(manifest, trial["mixture_path"])
    mix_info = audio_info(mix_path, 16000)
    speaker_ids = set()
    source_paths = []
    source_utterance_ids = set()
    for source in sources:
        if not isinstance(source, dict) or set(source) != REQUIRED_SOURCE:
            raise ValueError(f"{trial_id}: source fields must be {sorted(REQUIRED_SOURCE)}")
        speaker_id = source["speaker_id"]
        if not isinstance(speaker_id, str) or not speaker_id or speaker_id in speaker_ids:
            raise ValueError(f"{trial_id}: speaker IDs must be distinct nonempty strings")
        speaker_ids.add(speaker_id)
        source_id = source["source_utterance_id"]
        enroll_id = source["enrollment_utterance_id"]
        if not isinstance(source_id, str) or not isinstance(enroll_id, str) or not source_id or not enroll_id or source_id == enroll_id:
            raise ValueError(f"{trial_id}: source and enrollment need different utterance IDs")
        if source_id in source_utterance_ids:
            raise ValueError(f"{trial_id}: duplicate source utterance ID")
        source_utterance_ids.add(source_id)
        source_path = audio_path(manifest, source["source_path"])
        enroll_path = audio_path(manifest, source["enrollment_path"])
        if source_path in source_paths:
            raise ValueError(f"{trial_id}: duplicate clean source path")
        if source_path.resolve() == enroll_path.resolve():
            raise ValueError(f"{trial_id}: source and enrollment are the same file")
        if audio_info(source_path, 16000).frames != mix_info.frames:
            raise ValueError(f"{trial_id}: source length differs from mixture: {source_path}")
        audio_info(enroll_path, 16000)
        source_paths.append(source_path)
    if not set(selected).issubset(speaker_ids):
        raise ValueError(f"{trial_id}: selected speaker is not in sources")
    if check_mix_sum:
        mixture, _ = sf.read(mix_path, dtype="float64")
        summed = np.zeros_like(mixture)
        for source_path in source_paths:
            signal, _ = sf.read(source_path, dtype="float64")
            summed += signal
        relative_error = np.linalg.norm(mixture - summed) / max(np.linalg.norm(mixture), 1e-12)
        if relative_error > 0.01:
            raise ValueError(f"{trial_id}: mixture/source sum relative error {relative_error:.4g} exceeds 0.01")
    return {"trial_id": trial_id, "frames": mix_info.frames, "speakers": len(sources),
            "selected": len(selected)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--check-mix-sum", action="store_true")
    args = parser.parse_args()
    trials = read_manifest(args.manifest)
    for trial in trials:
        print(json.dumps(validate_trial(args.manifest, trial, args.check_mix_sum)))
    print(f"validated {len(trials)} trials")


if __name__ == "__main__":
    main()
