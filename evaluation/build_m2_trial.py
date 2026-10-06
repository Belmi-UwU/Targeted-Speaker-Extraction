"""Build one pinned Libri3Mix clean mixture and single/set-selection manifests."""

import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import soundfile as sf

from build_first_trial import ARCHIVE_MD5, LIBRIMIX_REVISION, RAW, ROOT, read_mono, sha256

OUT = ROOT / "data/m2_three_speakers"
MIXTURE_ID = "4077-13754-0001_5142-33396-0065_5683-32866-0012"
ENROLLMENTS = {
    "4077": "test-clean/4077/13751/4077-13751-0013.flac",
    "5142": "test-clean/5142/33396/5142-33396-0056.flac",
    "5683": "test-clean/5683/32865/5683-32865-0016.flac",
}


def main():
    row = next(r for r in csv.DictReader((ROOT / "librimix3_test_clean.csv").open())
               if r["mixture_ID"] == MIXTURE_ID)
    audio_dir = OUT / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    processed, sources, originals = [], [], []
    for i in (1, 2, 3):
        source_rel = row[f"source_{i}_path"]
        source_path = RAW / source_rel
        speaker_id = source_rel.split("/")[1]
        enrollment_rel = ENROLLMENTS[speaker_id]
        enrollment_path = RAW / enrollment_rel
        if Path(source_rel).stem == Path(enrollment_rel).stem:
            raise ValueError("enrollment must be a different utterance")
        processed.append((read_mono(source_path) * float(row[f"source_{i}_gain"])).astype("float32"))
        enrollment_wav = f"audio/{speaker_id}_enrollment.wav"
        sf.write(OUT / enrollment_wav, read_mono(enrollment_path), 16000, subtype="PCM_16")
        sources.append({
            "speaker_id": speaker_id,
            "source_path": f"audio/{MIXTURE_ID}_s{i}.wav",
            "source_utterance_id": Path(source_rel).stem,
            "enrollment_path": enrollment_wav,
            "enrollment_utterance_id": Path(enrollment_rel).stem,
        })
        originals.extend([source_path, enrollment_path])
    length = min(map(len, processed))
    processed = [source[:length] for source in processed]
    for source, info in zip(processed, sources):
        sf.write(OUT / info["source_path"], source, 16000, subtype="PCM_16")
    mixture = np.sum(np.stack(processed), axis=0)
    mixture_wav = f"audio/{MIXTURE_ID}_mix.wav"
    sf.write(OUT / mixture_wav, mixture, 16000, subtype="PCM_16")
    common = {
        "dataset": "Libri3Mix", "dataset_revision": LIBRIMIX_REVISION,
        "speech_license": "CC-BY-4.0", "split": "test", "condition": "mix_clean",
        "sample_rate_hz": 16000, "mixture_path": mixture_wav, "sources": sources,
    }
    speaker_ids = [source["speaker_id"] for source in sources]
    single = [{**common, "trial_id": f"{MIXTURE_ID}__speaker_{sid}",
               "selected_speaker_ids": [sid]} for sid in speaker_ids]
    sets = [{**common, "trial_id": f"{MIXTURE_ID}__selected_{'_'.join(ids)}",
             "selected_speaker_ids": list(ids)}
            for size in (2, 3) for ids in combinations(speaker_ids, size)]
    for filename, trials in (("single_trials.jsonl", single), ("selected_set_trials.jsonl", sets)):
        (OUT / filename).write_text("\n".join(json.dumps(trial) for trial in trials) + "\n")
    files = originals + [OUT / mixture_wav]
    files += [OUT / s["source_path"] for s in sources]
    files += [OUT / s["enrollment_path"] for s in sources]
    record = {
        "dataset": "Libri3Mix 16k/min/test/mix_clean subset", "mixture_id": MIXTURE_ID,
        "librimix_metadata_revision": LIBRIMIX_REVISION,
        "librimix_metadata_source": f"https://github.com/JorisCos/LibriMix/tree/{LIBRIMIX_REVISION}/metadata/Libri3Mix",
        "librimix_mixing_source": f"https://github.com/JorisCos/LibriMix/blob/{LIBRIMIX_REVISION}/scripts/create_librimix_from_metadata.py",
        "librispeech_source": "https://www.openslr.org/12",
        "librispeech_test_clean_archive_md5": ARCHIVE_MD5,
        "speech_license": "CC-BY-4.0", "noise_used": False,
        "generation": "metadata gains, 16 kHz, min length, source sum, PCM16 WAV; clean branch only",
        "source_gains": [float(row[f"source_{i}_gain"]) for i in (1, 2, 3)],
        "file_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in files},
        "manifest_sha256": {name: sha256(OUT / name) for name in
                            ("single_trials.jsonl", "selected_set_trials.jsonl")},
    }
    (OUT / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": "built", "single_trials": len(single),
                      "selected_set_trials": len(sets), "frames": length}, indent=2))


if __name__ == "__main__":
    main()
