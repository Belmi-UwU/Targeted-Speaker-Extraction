"""Run the pinned independent OpenSpeakerBeam-SS checkpoint on clean trials."""

import hashlib
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from resemblyzer import VoiceEncoder

from validate_manifest import audio_path, read_manifest, validate_trial

ROOT = Path(__file__).resolve().parent
MODEL_ROOT = ROOT.parent / "speakerbeam_ss_verification"
sys.path.insert(0, str(MODEL_ROOT / "upstream"))
from model import SpeakerBeamSS  # noqa: E402
from tools import get_speaker_embeddings_batch  # noqa: E402

CHECKPOINT_HASH = "41c250587af2e947dc3e265cd6e7aa7de230048d408e93c53d8348d63cf75896"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_wave(path):
    audio, rate = sf.read(path, dtype="float32")
    if rate != 16000 or audio.ndim != 1:
        raise ValueError(f"expected mono 16 kHz: {path}")
    return torch.from_numpy(audio).reshape(1, 1, -1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/trials/trials.jsonl")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else manifest.parent / "model_outputs"
    preview_dir = manifest.parent / "listening_previews"
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = MODEL_ROOT / "checkpoints/best_model.pth"
    if sha256(checkpoint) != CHECKPOINT_HASH:
        raise ValueError("checkpoint SHA-256 differs from verified value")
    torch.set_num_threads(4)
    model = SpeakerBeamSS().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    speaker_encoder = VoiceEncoder(device="cpu")
    summaries = []
    for trial in read_manifest(manifest):
        validate_trial(manifest, trial, check_mix_sum=True)
        selected = set(trial["selected_speaker_ids"])
        if len(selected) != 1:
            raise ValueError("this checkpoint runner supports only one selected speaker per trial")
        source = next(item for item in trial["sources"] if item["speaker_id"] in selected)
        mixture = load_wave(audio_path(manifest, trial["mixture_path"]))
        enrollment = load_wave(audio_path(manifest, source["enrollment_path"]))
        with torch.inference_mode():
            embedding = get_speaker_embeddings_batch(speaker_encoder, enrollment)
            start = time.perf_counter()
            result = model(mixture, embedding)
            inference_seconds = time.perf_counter() - start
        if result.shape != mixture.shape or not torch.isfinite(result).all():
            raise ValueError(f"invalid output for {trial['trial_id']}: {result.shape}")
        raw = result[0, 0].numpy()
        raw_path = output_dir / f"{trial['trial_id']}.wav"
        sf.write(raw_path, raw, 16000, subtype="FLOAT")
        peak = float(np.max(np.abs(raw)))
        gain = 0.9 / max(peak, 1e-12)
        sf.write(preview_dir / f"{trial['trial_id']}.wav", raw * gain, 16000,
                 subtype="PCM_16")
        summaries.append({"trial_id": trial["trial_id"],
                          "selected_speaker_id": source["speaker_id"],
                          "embedding_shape": list(embedding.shape),
                          "offline_model_forward_seconds": inference_seconds,
                          "raw_peak": peak, "listening_preview_gain": gain,
                          "raw_output_sha256": sha256(raw_path)})
    record = {"model": "OpenSpeakerBeam-SS independent variant",
              "model_code_commit": "bbbae73f74fc14eafdf7abb93f9094eafccd21df",
              "checkpoint_sha256": CHECKPOINT_HASH,
              "manifest_sha256": sha256(manifest),
              "device": "cpu", "torch_threads": torch.get_num_threads(),
              "scope": "offline single-target trials; not streaming or mobile timing",
              "trials": summaries}
    (manifest.parent / "baseline_run.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
