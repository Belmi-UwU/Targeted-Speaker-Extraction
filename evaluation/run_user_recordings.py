"""Offline SepFormer + ECAPA separation of a user's three-voice WAV recording."""

import argparse
import hashlib
import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf
import torch
from scipy.optimize import linear_sum_assignment
from speechbrain.inference.classifiers import EncoderClassifier
from speechbrain.inference.separation import SepformerSeparation

ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "data/user_recordings"
SEP_MODEL = ROOT / "sepformer_libr3mix_checkpoint"
ECAPA_MODEL = ROOT / "ecapa_checkpoint"
PROFILES = {
    "nicker": "enrollments/nicker/Nicker.wav",
    "mond": "enrollments/mond/Mond.wav",
    "sam": "enrollments/sam/Sam.wav",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_mono(path):
    wave, rate = sf.read(path, dtype="float32")
    if wave.ndim == 2:
        wave = wave.mean(axis=1)
    if wave.ndim != 1 or not np.isfinite(wave).all() or len(wave) == 0:
        raise ValueError(f"invalid audio: {path}")
    return wave, rate


def resample(wave, original, target):
    from math import gcd
    divisor = gcd(original, target)
    return scipy.signal.resample_poly(wave, target // divisor,
                                      original // divisor).astype("float32")


def embed(model, wave):
    with torch.inference_mode():
        encoded = model.encode_batch(torch.from_numpy(wave).unsqueeze(0))
    vector = encoded.reshape(-1).cpu().numpy()
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUTS)
    args = parser.parse_args()
    inputs = args.input_dir.resolve()
    output = inputs / "results"
    output.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    mix_path = inputs / "mixture/Mix.wav"
    mix, input_rate = read_mono(mix_path)
    mix8 = resample(mix, input_rate, 8000)
    mix16 = resample(mix, input_rate, 16000)
    sf.write(output / "original_mix_mono_16k.wav", mix16, 16000,
             subtype="PCM_16")
    print(f"Loaded {len(mix)/input_rate:.2f}s input; running SepFormer", flush=True)
    separator = SepformerSeparation.from_hparams(
        source=str(SEP_MODEL.resolve()),
        savedir=str((ROOT / "sepformer_libr3mix_loaded").resolve()),
        run_opts={"device": "cpu"},
    )
    with torch.inference_mode():
        start = time.perf_counter()
        estimated = separator.separate_batch(torch.from_numpy(mix8).unsqueeze(0))
        forward_seconds = time.perf_counter() - start
    streams8 = estimated[0].cpu().numpy().astype("float32")
    if streams8.shape != (len(mix8), 3) or not np.isfinite(streams8).all():
        raise ValueError(f"invalid separator result: {streams8.shape}")
    gains, _, rank, singular = np.linalg.lstsq(streams8.astype("float64"),
                                                mix8.astype("float64"), rcond=None)
    if rank != 3 or not np.isfinite(gains).all():
        raise ValueError("mixture-fit gains could not be calculated")
    fit_error = float(np.linalg.norm(streams8 @ gains - mix8) /
                      max(np.linalg.norm(mix8), 1e-12))
    streams16 = [resample(streams8[:, i] * float(gains[i]), 8000, 16000)
                 for i in range(3)]
    streams16 = [np.pad(w, (0, max(0, len(mix16) - len(w))))[:len(mix16)]
                 for w in streams16]
    matcher = EncoderClassifier.from_hparams(
        source=str(ECAPA_MODEL.resolve()),
        savedir=str((ROOT / "ecapa_loaded").resolve()),
        overrides={"pretrained_path": str(ECAPA_MODEL.resolve())},
        run_opts={"device": "cpu"},
    )
    names = list(PROFILES)
    enroll = []
    for name in names:
        wave, rate = read_mono(inputs / PROFILES[name])
        enroll.append(resample(wave, rate, 16000))
    enrolled_vectors = np.stack([embed(matcher, wave) for wave in enroll])
    stream_vectors = np.stack([embed(matcher, wave) for wave in streams16])
    similarity = enrolled_vectors @ stream_vectors.T
    print("Speaker matching complete; exporting selections", flush=True)
    selected = [(name,) for name in names]
    selected += list(combinations(names, 2))
    mappings = {}
    waveforms = {}
    for speakers in selected:
        indices = [names.index(name) for name in speakers]
        rows, cols = linear_sum_assignment(-similarity[indices, :])
        mapping = {names[indices[row]]: int(col) for row, col in zip(rows, cols)}
        combined = np.zeros_like(mix16)
        for col in mapping.values():
            combined += streams16[col]
        label = "only_" + speakers[0] if len(speakers) == 1 else "_and_".join(speakers)
        mappings[label] = mapping
        waveforms[label] = combined
    peak = max(float(np.max(np.abs(wave))) for wave in waveforms.values())
    preview_gain = min(1.0, 0.9 / max(peak, 1e-12))
    files = {}
    for label, wave in waveforms.items():
        path = output / f"{label}.wav"
        sf.write(path, wave * preview_gain, 16000, subtype="PCM_16")
        files[label] = {"path": str(path.relative_to(inputs)),
                        "speakers": list(mappings[label]),
                        "matched_stream_by_speaker": mappings[label],
                        "peak_before_preview_gain": float(np.max(np.abs(wave))),
                        "sha256": sha256(path)}
    record = {
        "scope": "offline local-recording exploratory inference; no clean references available",
        "input_mixture_path": str(mix_path.relative_to(inputs)),
        "input_file_sha256": {str(path.relative_to(inputs)): sha256(path)
                              for path in [mix_path] + [inputs / p for p in PROFILES.values()]},
        "input_sample_rate_hz": input_rate,
        "input_seconds": len(mix) / input_rate,
        "input_channels_downmixed_to_mono": True,
        "separator_model_revision": "1f2d824e2333aec9e60d0030fb144885c1afc426",
        "matcher_model_revision": "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286",
        "separator_input_rate_hz": 8000,
        "matcher_input_rate_hz": 16000,
        "offline_separator_forward_seconds": forward_seconds,
        "mixture_fit_gains": gains.tolist(),
        "mixture_fit_relative_error": fit_error,
        "mixture_fit_condition_number": float(singular[0] / singular[-1]),
        "enrollment_to_stream_cosine": similarity.tolist(),
        "speaker_order": names,
        "preview_gain": preview_gain,
        "warnings": ["no clean-reference score", "fixed three-stream offline separator",
                     "selected-speaker absence is not rejected", "not a real-time result"],
        "outputs": files,
    }
    (output / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": "complete", "output_dir": str(output),
                      "forward_seconds": forward_seconds,
                      "mixture_fit_relative_error": fit_error,
                      "similarity": similarity.tolist(),
                      "mappings": mappings,
                      "preview_gain": preview_gain}, indent=2))


if __name__ == "__main__":
    main()
