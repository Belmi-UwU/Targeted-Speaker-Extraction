"""Build one controlled turn-taking and selected-absence test from M2 sources."""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import soundfile as sf

from build_first_trial import LIBRIMIX_REVISION, ROOT, sha256

BASE = ROOT / "data/m2_three_speakers"
OUT = ROOT / "data/m2_turns"
SOURCE_ID = "4077-13754-0001_5142-33396-0065_5683-32866-0012"
MIXTURE_ID = SOURCE_ID + "_controlled_turns"
RATE = 16000
SEGMENT_FRAMES = int(0.9 * RATE)
SEGMENTS = [
    {"label": "4077_only", "active_speaker_ids": ["4077"]},
    {"label": "selected_absent_5683_only", "active_speaker_ids": ["5683"]},
    {"label": "all_three_overlap", "active_speaker_ids": ["4077", "5142", "5683"]},
    {"label": "5142_only", "active_speaker_ids": ["5142"]},
]


def read_wave(path):
    wave, rate = sf.read(path, dtype="float32")
    if rate != RATE or wave.ndim != 1:
        raise ValueError(f"expected mono 16 kHz audio: {path}")
    return wave


def fade_mask(active, frames):
    mask = np.zeros(frames, dtype="float32")
    for index, segment in enumerate(SEGMENTS):
        if active in segment["active_speaker_ids"]:
            start, end = index * SEGMENT_FRAMES, (index + 1) * SEGMENT_FRAMES
            mask[start:end] = 1.0
    fade = int(0.01 * RATE)
    changes = np.flatnonzero(np.diff(mask) != 0) + 1
    for frame in changes:
        if mask[frame] > mask[frame - 1]:
            mask[frame:frame + fade] = np.linspace(0.0, 1.0, fade,
                                                      dtype="float32")
        else:
            mask[frame - fade:frame] = np.linspace(1.0, 0.0, fade,
                                                      dtype="float32")
    return mask


def main():
    (OUT / "audio").mkdir(parents=True, exist_ok=True)
    base_trial = json.loads((BASE / "single_trials.jsonl").read_text().splitlines()[0])
    frames = 4 * SEGMENT_FRAMES
    sources = []
    processed = []
    for index, original in enumerate(base_trial["sources"], 1):
        speaker = original["speaker_id"]
        signal = read_wave(BASE / original["source_path"])[:frames]
        gated = signal * fade_mask(speaker, frames)
        source_path = f"audio/{MIXTURE_ID}_s{index}.wav"
        enrollment_path = f"audio/{speaker}_enrollment.wav"
        sf.write(OUT / source_path, gated, RATE, subtype="PCM_16")
        sf.write(OUT / enrollment_path, read_wave(BASE / original["enrollment_path"]),
                 RATE, subtype="PCM_16")
        sources.append({**original, "source_path": source_path,
                        "enrollment_path": enrollment_path})
        processed.append(gated)
    mixture_path = f"audio/{MIXTURE_ID}_mix.wav"
    sf.write(OUT / mixture_path, np.sum(np.stack(processed), axis=0), RATE,
             subtype="PCM_16")
    speakers = [source["speaker_id"] for source in sources]
    common = {
        "dataset": "ControlledLibri3Mix",
        "dataset_revision": f"{LIBRIMIX_REVISION}+controlled-turns-v1",
        "speech_license": "CC-BY-4.0", "split": "test", "condition": "turns_clean",
        "sample_rate_hz": RATE, "mixture_path": mixture_path, "sources": sources,
    }
    singles = [{**common, "trial_id": f"{MIXTURE_ID}__speaker_{sid}",
                "selected_speaker_ids": [sid]} for sid in speakers]
    sets = [{**common, "trial_id": f"{MIXTURE_ID}__selected_{'_'.join(ids)}",
             "selected_speaker_ids": list(ids)}
            for size in (2, 3) for ids in combinations(speakers, size)]
    for name, trials in (("single_trials.jsonl", singles),
                         ("selected_set_trials.jsonl", sets)):
        (OUT / name).write_text("\n".join(json.dumps(t) for t in trials) + "\n")
    segments = []
    for index, segment in enumerate(SEGMENTS):
        segments.append({**segment, "start_frame": index * SEGMENT_FRAMES,
                         "end_frame": (index + 1) * SEGMENT_FRAMES})
    provenance = {
        "type": "controlled derivative, not an official Libri3Mix metadata row",
        "base_mixture_id": SOURCE_ID,
        "source": "official LibriSpeech test-clean utterances, gains from pinned Libri3Mix metadata",
        "metadata_revision": LIBRIMIX_REVISION,
        "construction": "four 0.9-second intervals with 10 ms source on/off fades, additive clean sources; no new speech or noise",
        "segments": segments,
        "selected_pair_of_interest": ["4077", "5142"],
        "file_sha256": {str(p.relative_to(OUT)): sha256(p)
                        for p in sorted((OUT / "audio").glob("*.wav"))},
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"status": "built", "frames": frames, "segments": segments},
                     indent=2))


if __name__ == "__main__":
    main()
