"""Offline Libri3Mix blind separation followed by enrollment-based stream selection."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import scipy
import scipy.signal
import soundfile as sf
import speechbrain
import torch
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.optimize import linear_sum_assignment
from speechbrain.inference.separation import SepformerSeparation

from validate_manifest import audio_path, read_manifest, validate_trial

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/m2_three_speakers"
MODEL = ROOT / "sepformer_libr3mix_checkpoint"
LOADED = ROOT / "sepformer_libr3mix_loaded"
OUTPUTS = DATA / "sepformer_outputs"
LISTENING = DATA / "sepformer_listening"
MIXFIT_OUTPUTS = DATA / "sepformer_mixfit_outputs"
MIXFIT_LISTENING = DATA / "sepformer_mixfit_listening"
MODEL_REVISION = "1f2d824e2333aec9e60d0030fb144885c1afc426"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_wave(path):
    wave, rate = sf.read(path, dtype="float32")
    if rate != 16000 or wave.ndim != 1 or not np.isfinite(wave).all():
        raise ValueError(f"expected finite mono 16 kHz audio: {path}")
    return wave


def embedding(encoder, wave):
    prepared = preprocess_wav(wave, source_sr=16000)
    if len(prepared) == 0:
        raise ValueError("speaker matching got silent stream")
    vector = encoder.embed_utterance(prepared)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def matching(similarity, selected_indices):
    """One-to-one mapping of selected enrollments onto anonymous output streams."""
    rows, cols = linear_sum_assignment(-similarity[selected_indices, :])
    return {selected_indices[row]: int(col) for row, col in zip(rows, cols)}


def main():
    global DATA, OUTPUTS, LISTENING, MIXFIT_OUTPUTS, MIXFIT_LISTENING
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    args = parser.parse_args()
    DATA = args.data_dir.resolve()
    OUTPUTS = DATA / "sepformer_outputs"
    LISTENING = DATA / "sepformer_listening"
    MIXFIT_OUTPUTS = DATA / "sepformer_mixfit_outputs"
    MIXFIT_LISTENING = DATA / "sepformer_mixfit_listening"
    torch.set_num_threads(4)
    single_manifest = DATA / "single_trials.jsonl"
    set_manifest = DATA / "selected_set_trials.jsonl"
    single_trials = read_manifest(single_manifest)
    set_trials = read_manifest(set_manifest)
    for trial in single_trials:
        validate_trial(single_manifest, trial, check_mix_sum=True)
    for trial in set_trials:
        validate_trial(set_manifest, trial, check_mix_sum=True)
    if len({trial["mixture_path"] for trial in single_trials + set_trials}) != 1:
        raise ValueError("this runner expects one common mixture")
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    LISTENING.mkdir(parents=True, exist_ok=True)
    MIXFIT_OUTPUTS.mkdir(parents=True, exist_ok=True)
    MIXFIT_LISTENING.mkdir(parents=True, exist_ok=True)
    mix16 = read_wave(audio_path(single_manifest, single_trials[0]["mixture_path"]))
    mix8 = scipy.signal.resample_poly(mix16, 1, 2).astype("float32")
    separator = SepformerSeparation.from_hparams(
        source=str(MODEL), savedir=str(LOADED), run_opts={"device": "cpu"}
    )
    if separator.hparams.sample_rate != 8000 or separator.hparams.num_spks != 3:
        raise ValueError("unexpected checkpoint sample rate or speaker count")
    with torch.inference_mode():
        start = time.perf_counter()
        estimated = separator.separate_batch(torch.from_numpy(mix8).unsqueeze(0))
        inference_seconds = time.perf_counter() - start
    separated8 = estimated[0].cpu().numpy().astype("float32")
    if separated8.shape != (len(mix8), 3) or not np.isfinite(separated8).all():
        raise ValueError(f"invalid separation shape or values: {separated8.shape}")
    base_id = single_trials[0]["trial_id"].split("__speaker_")[0]
    for col in range(3):
        sf.write(OUTPUTS / f"{base_id}__stream_{col}.wav", separated8[:, col], 8000,
                 subtype="FLOAT")
    separated16 = [scipy.signal.resample_poly(separated8[:, col], 2, 1).astype("float32")
                   for col in range(3)]
    separated16 = [np.pad(w, (0, max(0, len(mix16) - len(w))))[:len(mix16)]
                   for w in separated16]
    mixfit_gains, residuals, rank, singular_values = np.linalg.lstsq(
        separated8.astype("float64"), mix8.astype("float64"), rcond=None
    )
    if rank != 3 or not np.isfinite(mixfit_gains).all():
        raise ValueError("mixture-fit gains could not be estimated")
    mixfit_relative_error = float(np.linalg.norm(
        separated8 @ mixfit_gains - mix8) / max(np.linalg.norm(mix8), 1e-12))
    calibrated16 = [wave * float(gain) for wave, gain in zip(separated16, mixfit_gains)]
    speaker_ids = [source["speaker_id"] for source in single_trials[0]["sources"]]
    speaker_encoder = VoiceEncoder(device="cpu")
    enroll_embeddings = [embedding(speaker_encoder,
                                   read_wave(audio_path(single_manifest, source["enrollment_path"])))
                         for source in single_trials[0]["sources"]]
    stream_embeddings = [embedding(speaker_encoder, wave) for wave in separated16]
    similarities = np.stack(enroll_embeddings) @ np.stack(stream_embeddings).T
    trials = single_trials + set_trials
    selected_waves = {}
    calibrated_waves = {}
    records = []
    for trial in trials:
        selected = trial["selected_speaker_ids"]
        indices = [speaker_ids.index(sid) for sid in selected]
        mapping = matching(similarities, indices)
        combined = np.zeros_like(mix16)
        calibrated = np.zeros_like(mix16)
        for col in mapping.values():
            combined += separated16[col]
            calibrated += calibrated16[col]
        output_path = OUTPUTS / f"{trial['trial_id']}.wav"
        sf.write(output_path, combined, 16000, subtype="FLOAT")
        selected_waves[trial["trial_id"]] = combined
        mixfit_path = MIXFIT_OUTPUTS / f"{trial['trial_id']}.wav"
        sf.write(mixfit_path, calibrated, 16000, subtype="FLOAT")
        calibrated_waves[trial["trial_id"]] = calibrated
        records.append({"trial_id": trial["trial_id"], "selected_speaker_ids": selected,
                        "matched_stream_by_speaker": {speaker_ids[i]: col for i, col in mapping.items()},
                        "raw_peak": float(np.max(np.abs(combined))),
                        "output_sha256": sha256(output_path),
                        "mixfit_output_sha256": sha256(mixfit_path)})
    max_peak = max(float(np.max(np.abs(w))) for w in selected_waves.values())
    preview_gain = min(1.0, 0.9 / max(max_peak, 1e-12))
    mixfit_peak = max(float(np.max(np.abs(w))) for w in calibrated_waves.values())
    mixfit_preview_gain = min(1.0, 0.9 / max(mixfit_peak, 1e-12))
    for trial in set_trials:
        trial_id = trial["trial_id"]
        clean = np.zeros_like(mix16)
        for source in trial["sources"]:
            if source["speaker_id"] in trial["selected_speaker_ids"]:
                clean += read_wave(audio_path(set_manifest, source["source_path"]))
        sf.write(LISTENING / f"{trial_id}__mix.wav", mix16, 16000, subtype="PCM_16")
        sf.write(LISTENING / f"{trial_id}__selected_clean.wav", clean, 16000,
                 subtype="PCM_16")
        sf.write(LISTENING / f"{trial_id}__model_safe_gain.wav",
                 selected_waves[trial_id] * preview_gain, 16000, subtype="PCM_16")
        sf.write(MIXFIT_LISTENING / f"{trial_id}__mix.wav", mix16, 16000,
                 subtype="PCM_16")
        sf.write(MIXFIT_LISTENING / f"{trial_id}__selected_clean.wav", clean, 16000,
                 subtype="PCM_16")
        sf.write(MIXFIT_LISTENING / f"{trial_id}__model_mixfit.wav",
                 calibrated_waves[trial_id] * mixfit_preview_gain, 16000,
                 subtype="PCM_16")
    record = {
        "architecture": "SpeechBrain SepFormer, blind 3-speaker separation",
        "model_source": "https://huggingface.co/speechbrain/sepformer-libri3mix",
        "model_revision": MODEL_REVISION, "model_card_license": "apache-2.0",
        "model_file_sha256": {p.name: sha256(p) for p in sorted(MODEL.glob("*.ckpt"))},
        "speechbrain_version": speechbrain.__version__, "torch_version": torch.__version__,
        "scipy_version": scipy.__version__, "device": "cpu", "torch_threads": 4,
        "sample_rate_hz": 8000, "input_mix_16k_frames": len(mix16),
        "resampled_mix_8k_frames": len(mix8),
        "offline_model_forward_seconds": inference_seconds,
        "speaker_matching": "Resemblyzer embedding cosine similarity; Hungarian one-to-one assignment across selected enrollments and 3 streams; no clean sources or unselected enrollment used",
        "speaker_ids_for_diagnostics": speaker_ids,
        "enrollment_to_stream_cosine": similarities.tolist(),
        "preview_gain": preview_gain,
        "preview_gain_policy": "one common gain for all selected-set model listening files; keeps PCM16 peak about 0.9 or lower",
        "mixture_fit_gains_by_stream": mixfit_gains.tolist(),
        "mixture_fit_relative_error_8k": mixfit_relative_error,
        "mixture_fit_matrix_condition_number": float(singular_values[0] / singular_values[-1]),
        "mixture_fit_method": "unconstrained least squares of original 8 kHz mixture onto 3 separator streams; signed gains can fix inverted polarity; no clean source used",
        "mixfit_preview_gain": mixfit_preview_gain,
        "warnings": ["offline/noncausal model", "8 kHz input bandwidth",
                     "no absent-speaker threshold; matching always assigns a stream"],
        "trials": records,
    }
    (DATA / "sepformer_run.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": "completed", "forward_seconds": inference_seconds,
                      "preview_gain": preview_gain,
                      "mixture_fit_gains": mixfit_gains.tolist(),
                      "mixture_fit_relative_error": mixfit_relative_error,
                      "similarity": similarities.tolist(),
                      "matches": [r["matched_stream_by_speaker"] for r in records]}, indent=2))


if __name__ == "__main__":
    main()
