"""Build one pinned Libri2Mix 16 kHz/min/mix_clean evaluation mixture."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data/raw/LibriSpeech"
OUT = ROOT / "data/trials"
MIXTURE_ID = "4077-13754-0001_5142-33396-0065"
ENROLLMENTS = {
    "4077": "test-clean/4077/13751/4077-13751-0013.flac",
    "5142": "test-clean/5142/33396/5142-33396-0056.flac",
}
LIBRIMIX_REVISION = "b898ca375c03e4039cdb3dbe3dd485aad58ec7d1"
ARCHIVE_MD5 = "32fa31d27d2e1cad72775fee3f4849a9"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_mono(path):
    audio, rate = sf.read(path, dtype="float32")
    if rate != 16000 or audio.ndim != 1 or len(audio) == 0:
        raise ValueError(f"expected nonempty 16 kHz mono speech: {path}")
    return audio


def main():
    rows = list(csv.DictReader((ROOT / "librimix_test_clean.csv").open()))
    row = next(item for item in rows if item["mixture_ID"] == MIXTURE_ID)
    audio_dir = OUT / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    sources = []
    processed = []
    original_paths = []
    for i in (1, 2):
        source_relative = row[f"source_{i}_path"]
        source_path = RAW / source_relative
        speaker_id = source_relative.split("/")[1]
        enroll_relative = ENROLLMENTS[speaker_id]
        enroll_path = RAW / enroll_relative
        source = read_mono(source_path)
        enrollment = read_mono(enroll_path)
        gain = float(row[f"source_{i}_gain"])
        processed.append((source * gain).astype("float32"))
        source_wav = f"audio/{MIXTURE_ID}_s{i}.wav"
        enroll_wav = f"audio/{speaker_id}_enrollment.wav"
        sf.write(OUT / enroll_wav, enrollment, 16000, subtype="PCM_16")
        sources.append({
            "speaker_id": speaker_id,
            "source_path": source_wav,
            "source_utterance_id": Path(source_relative).stem,
            "enrollment_path": enroll_wav,
            "enrollment_utterance_id": Path(enroll_relative).stem,
        })
        original_paths.extend([source_path, enroll_path])
    length = min(len(source) for source in processed)
    processed = [source[:length] for source in processed]
    for source, metadata in zip(processed, sources):
        sf.write(OUT / metadata["source_path"], source, 16000, subtype="PCM_16")
    mixture = np.zeros_like(processed[0])
    for source in processed:
        mixture += source
    mix_wav = f"audio/{MIXTURE_ID}_mix.wav"
    sf.write(OUT / mix_wav, mixture, 16000, subtype="PCM_16")
    trials = []
    for speaker_id in ("4077", "5142"):
        trials.append({
            "trial_id": f"{MIXTURE_ID}__speaker_{speaker_id}",
            "dataset": "Libri2Mix", "dataset_revision": LIBRIMIX_REVISION,
            "speech_license": "CC-BY-4.0", "split": "test", "condition": "mix_clean",
            "sample_rate_hz": 16000, "mixture_path": mix_wav,
            "selected_speaker_ids": [speaker_id], "sources": sources,
        })
    manifest = OUT / "trials.jsonl"
    manifest.write_text("\n".join(json.dumps(trial) for trial in trials) + "\n")
    files = original_paths + [OUT / mix_wav] + [OUT / item["source_path"] for item in sources]
    files += [OUT / item["enrollment_path"] for item in sources]
    provenance = {
        "dataset": "Libri2Mix 16k/min/test/mix_clean subset",
        "mixture_id": MIXTURE_ID,
        "librimix_metadata_revision": LIBRIMIX_REVISION,
        "librimix_metadata_source": f"https://github.com/JorisCos/LibriMix/tree/{LIBRIMIX_REVISION}/metadata/Libri2Mix",
        "librimix_mixing_source": f"https://github.com/JorisCos/LibriMix/blob/{LIBRIMIX_REVISION}/scripts/create_librimix_from_metadata.py",
        "librispeech_source": "https://www.openslr.org/12",
        "librispeech_test_clean_archive_md5": ARCHIVE_MD5,
        "speech_license": "CC-BY-4.0",
        "noise_used": False,
        "generation": "metadata gains, 16 kHz, min length, source sum, PCM16 WAV; clean branch only",
        "source_gains": [float(row[f"source_{i}_gain"]) for i in (1, 2)],
        "file_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in files},
        "manifest_sha256": sha256(manifest),
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"status": "built", "trials": len(trials), "frames": length,
                      "manifest": str(manifest.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
