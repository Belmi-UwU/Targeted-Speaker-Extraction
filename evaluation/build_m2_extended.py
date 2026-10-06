"""Build three additional speaker-disjoint Libri3Mix clean test examples."""

import argparse
import csv
import json
import tarfile
from itertools import combinations
from pathlib import Path

import numpy as np
import soundfile as sf

from build_first_trial import ARCHIVE_MD5, LIBRIMIX_REVISION, RAW, ROOT, read_mono, sha256

MIXTURE_IDS = (
    "672-122797-0044_8463-294828-0013_4507-16021-0043",
    "61-70970-0033_4446-2271-0021_1089-134691-0025",
    "7176-88083-0000_4970-29095-0006_2094-142345-0034",
)
HOLDOUT_IDS = (
    "6930-81414-0027_7729-102255-0036_7127-75947-0015",
    "1188-133604-0013_1320-122612-0009_3570-5695-0000",
    "4992-41797-0010_2300-131720-0005_6829-68771-0014",
    "2961-961-0007_237-134493-0013_2830-3980-0011",
)
OUT = ROOT / "data/m2_extended"
ARCHIVE = ROOT / "data/archives/test-clean.tar.gz"


def get_enrollment(speaker, source, enrollments):
    return next(path for path, frames in enrollments[speaker]
                if path != source and 32000 <= frames <= 144000)


def ensure_audio(paths):
    needed = {f"LibriSpeech/{path}" for path in paths if not (RAW / path).is_file()}
    if not needed:
        return
    for member_name in needed:
        relative = Path(member_name)
        if relative.parts[:2] != ("LibriSpeech", "test-clean") or ".." in relative.parts:
            raise ValueError(f"unsafe archive path: {member_name}")
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        for name in sorted(needed):
            member = archive.getmember(name)
            if not member.isfile():
                raise ValueError(f"not a regular audio file: {name}")
            output = ROOT / "data/raw" / name
            output.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as src, output.open("wb") as dst:
                while block := src.read(1024 * 1024):
                    dst.write(block)


def build(row, enrollments):
    mixture_id = row["mixture_ID"]
    out = OUT / mixture_id
    (out / "audio").mkdir(parents=True, exist_ok=True)
    source_rels = [row[f"source_{i}_path"] for i in (1, 2, 3)]
    speakers = [path.split("/")[1] for path in source_rels]
    if len(set(speakers)) != 3:
        raise ValueError("expected three distinct speakers")
    enrollment_rels = [get_enrollment(speaker, path, enrollments)
                       for speaker, path in zip(speakers, source_rels)]
    ensure_audio(source_rels + enrollment_rels)
    processed, sources = [], []
    for i, (speaker, source_rel, enrollment_rel) in enumerate(
        zip(speakers, source_rels, enrollment_rels), 1
    ):
        source = read_mono(RAW / source_rel)
        enrollment = read_mono(RAW / enrollment_rel)
        gain = float(row[f"source_{i}_gain"])
        processed.append((source * gain).astype("float32"))
        sf.write(out / f"audio/{speaker}_enrollment.wav", enrollment, 16000,
                 subtype="PCM_16")
        sources.append({
            "speaker_id": speaker,
            "source_path": f"audio/{mixture_id}_s{i}.wav",
            "source_utterance_id": Path(source_rel).stem,
            "enrollment_path": f"audio/{speaker}_enrollment.wav",
            "enrollment_utterance_id": Path(enrollment_rel).stem,
        })
    length = min(map(len, processed))
    processed = [p[:length] for p in processed]
    for source, info in zip(processed, sources):
        sf.write(out / info["source_path"], source, 16000, subtype="PCM_16")
    mixture_path = f"audio/{mixture_id}_mix.wav"
    sf.write(out / mixture_path, np.sum(np.stack(processed), axis=0), 16000,
             subtype="PCM_16")
    common = {
        "dataset": "Libri3Mix", "dataset_revision": LIBRIMIX_REVISION,
        "speech_license": "CC-BY-4.0", "split": "test", "condition": "mix_clean",
        "sample_rate_hz": 16000, "mixture_path": mixture_path, "sources": sources,
    }
    singles = [{**common, "trial_id": f"{mixture_id}__speaker_{sid}",
                "selected_speaker_ids": [sid]} for sid in speakers]
    sets = [{**common, "trial_id": f"{mixture_id}__selected_{'_'.join(ids)}",
             "selected_speaker_ids": list(ids)}
            for size in (2, 3) for ids in combinations(speakers, size)]
    for name, trials in (("single_trials.jsonl", singles),
                         ("selected_set_trials.jsonl", sets)):
        (out / name).write_text("\n".join(json.dumps(t) for t in trials) + "\n")
    record = {
        "source": "official Libri3Mix test-clean metadata, derived 16k/min/mix_clean",
        "mixture_id": mixture_id, "metadata_revision": LIBRIMIX_REVISION,
        "metadata_url": f"https://github.com/JorisCos/LibriMix/tree/{LIBRIMIX_REVISION}/metadata/Libri3Mix",
        "speech_archive": "https://www.openslr.org/12",
        "archive_md5": ARCHIVE_MD5, "speech_license": "CC-BY-4.0",
        "source_gains": [float(row[f"source_{i}_gain"]) for i in (1, 2, 3)],
        "original_paths": source_rels + enrollment_rels,
        "file_sha256": {str(p.relative_to(out)): sha256(p) for p in sorted((out / "audio").glob("*.wav"))},
    }
    (out / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    return {"mixture_id": mixture_id, "speakers": speakers, "frames": length}


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", action="store_true",
                        help="build four new speaker-disjoint rows reserved for matcher validation")
    args = parser.parse_args()
    mixture_ids = HOLDOUT_IDS if args.holdout else MIXTURE_IDS
    if args.holdout:
        OUT = ROOT / "data/m2_holdout"
    rows = {row["mixture_ID"]: row for row in
            csv.DictReader((ROOT / "librimix3_test_clean.csv").open())}
    enrollments = {}
    for row in csv.DictReader((ROOT / "librispeech_test_clean.csv").open()):
        enrollments.setdefault(row["speaker_ID"], []).append(
            (row["origin_path"], int(row["length"])))
    results = [build(rows[mixture_id], enrollments) for mixture_id in mixture_ids]
    expected = 3 * len(mixture_ids)
    result_speakers = {s for result in results for s in result["speakers"]}
    if len(result_speakers) != expected:
        raise ValueError("extended examples reuse a speaker")
    if args.holdout:
        earlier_ids = ("4077-13754-0001_5142-33396-0065_5683-32866-0012",) + MIXTURE_IDS
        earlier_speakers = {utterance.split("-")[0]
                            for mixture_id in earlier_ids for utterance in mixture_id.split("_")}
        if result_speakers & earlier_speakers:
            raise ValueError("holdout speakers overlap earlier examples")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
