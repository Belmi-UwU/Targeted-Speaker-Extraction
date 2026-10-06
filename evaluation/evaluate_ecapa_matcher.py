"""Compare ECAPA enrollment-to-stream matching using cached SepFormer streams."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf
import torch
from scipy.optimize import linear_sum_assignment
from speechbrain.inference.classifiers import EncoderClassifier

from validate_manifest import audio_path, read_manifest, validate_trial

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "ecapa_checkpoint"
LOADED = ROOT / "ecapa_loaded"
REVISION = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_wave(path, rate):
    wave, actual = sf.read(path, dtype="float32")
    if actual != rate or wave.ndim != 1 or not np.isfinite(wave).all():
        raise ValueError(f"invalid {rate} Hz mono waveform: {path}")
    return wave


def ecapa_embedding(model, wave):
    with torch.inference_mode():
        encoded = model.encode_batch(torch.from_numpy(wave).unsqueeze(0))
    vector = encoded.reshape(-1).cpu().numpy()
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    args = parser.parse_args()
    data = args.data_dir.resolve()
    torch.set_num_threads(4)
    singles_manifest = data / "single_trials.jsonl"
    sets_manifest = data / "selected_set_trials.jsonl"
    singles = read_manifest(singles_manifest)
    sets = read_manifest(sets_manifest)
    for trial in singles:
        validate_trial(singles_manifest, trial, check_mix_sum=True)
    for trial in sets:
        validate_trial(sets_manifest, trial, check_mix_sum=True)
    model = EncoderClassifier.from_hparams(
        source=str(MODEL.resolve()), savedir=str(LOADED.resolve()),
        overrides={"pretrained_path": str(MODEL.resolve())},
        run_opts={"device": "cpu"},
    )
    sources = singles[0]["sources"]
    speakers = [source["speaker_id"] for source in sources]
    base_id = singles[0]["trial_id"].split("__speaker_")[0]
    mixture = read_wave(audio_path(singles_manifest, singles[0]["mixture_path"]), 16000)
    run = json.loads((data / "sepformer_run.json").read_text())
    gains = run["mixture_fit_gains_by_stream"]
    if len(gains) != 3:
        raise ValueError("expected three separator-stream gains")
    streams = [read_wave(data / "sepformer_outputs" / f"{base_id}__stream_{i}.wav", 8000)
               for i in range(3)]
    streams16 = [scipy.signal.resample_poly(stream * gain, 2, 1).astype("float32")
                 for stream, gain in zip(streams, gains)]
    streams16 = [np.pad(wave, (0, max(0, len(mixture) - len(wave))))[:len(mixture)]
                 for wave in streams16]
    enroll = [read_wave(audio_path(singles_manifest, source["enrollment_path"]), 16000)
              for source in sources]
    enroll_vectors = np.stack([ecapa_embedding(model, wave) for wave in enroll])
    stream_vectors = np.stack([ecapa_embedding(model, wave) for wave in streams16])
    similarity = enroll_vectors @ stream_vectors.T
    outdir = data / "sepformer_ecapa_outputs"
    outdir.mkdir(exist_ok=True)
    trial_records = []
    for trial in singles + sets:
        indices = [speakers.index(sid) for sid in trial["selected_speaker_ids"]]
        rows, cols = linear_sum_assignment(-similarity[indices, :])
        mapping = {speakers[indices[row]]: int(col) for row, col in zip(rows, cols)}
        combined = np.zeros_like(mixture)
        for col in mapping.values():
            combined += streams16[col]
        output = outdir / f"{trial['trial_id']}.wav"
        sf.write(output, combined, 16000, subtype="FLOAT")
        trial_records.append({"trial_id": trial["trial_id"],
                              "selected_speaker_ids": trial["selected_speaker_ids"],
                              "matched_stream_by_speaker": mapping,
                              "output_sha256": sha256(output)})
    record = {
        "matcher": "SpeechBrain ECAPA-TDNN speaker embedding cosine similarity plus one-to-one selected-speaker assignment",
        "matcher_source": "https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb",
        "matcher_revision": REVISION, "model_card_license": "apache-2.0",
        "checkpoint_sha256": {p.name: sha256(p) for p in MODEL.glob("*.ckpt")},
        "separator_streams_source": "cached SepFormer 8 kHz output streams; no separator rerun",
        "stream_gain_source": "input-mixture least-squares fit in sepformer_run.json; no clean source used",
        "sample_rate_for_matcher_hz": 16000,
        "matching_enrollment_to_stream_cosine": similarity.tolist(),
        "speakers_in_source_order": speakers,
        "warnings": ["no absent-speaker rejection threshold", "offline/noncausal pipeline"],
        "trials": trial_records,
    }
    (data / "ecapa_matcher_run.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"data_dir": str(data), "similarity": similarity.tolist(),
                      "pair_matches": [r["matched_stream_by_speaker"]
                                       for r in trial_records if len(r["selected_speaker_ids"]) == 2]},
                     indent=2))


if __name__ == "__main__":
    main()
