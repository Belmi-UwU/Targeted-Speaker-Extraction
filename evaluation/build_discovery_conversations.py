"""Build controlled 4-6 speaker conversations for discovery enrollment tests.

Each mixture has two sections built from different LibriSpeech utterances:

* discovery: rounds of solo turns plus 2- and 3-speaker overlaps, so the
  discovery pipeline can be run on the first 1, 2, or 4 rounds;
* live: 2-second segments with 1-4 simultaneous speakers, giving every
  speaker target-active and target-absent time.

A third, separate utterance per speaker is the clean enrollment. Speakers are
split into disjoint dev (threshold tuning) and test (reporting) sets.
"""

import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from build_first_trial import ROOT, sha256

RAW = ROOT / "data/raw/LibriSpeech"
OUT = ROOT / "data/discovery"
RATE = 16000
SEED = 2026
TURN_SECONDS = 2.5
ROUNDS = 4
LIVE_SEGMENT_SECONDS = 2.0
# Simultaneous speakers per live segment; one 4-speaker stress segment.
LIVE_SIZES = [1, 2, 1, 3, 2, 1, 3, 1, 2, 4, 1, 2, 3, 1, 2, 1]
GROUP_SIZES = (4, 5, 6)
FADE = int(0.01 * RATE)
TARGET_RMS = 0.05
REVISION = "LibriSpeech-test-clean+controlled-conversation-v1"


def read_mono(path):
    audio, rate = sf.read(path, dtype="float32")
    if rate != RATE or audio.ndim != 1 or len(audio) == 0:
        raise ValueError(f"expected nonempty 16 kHz mono speech: {path}")
    return audio


def trim_silence(audio, frame=320, floor_db=-40.0):
    """Drop leading/trailing frames quieter than floor_db below the peak frame."""
    frames = len(audio) // frame
    if frames == 0:
        return audio
    energy = np.square(audio[:frames * frame]).reshape(frames, frame).mean(axis=1)
    keep = np.flatnonzero(energy > energy.max() * 10 ** (floor_db / 10))
    return audio[keep[0] * frame:(keep[-1] + 1) * frame]


class Streamer:
    """Contiguous speech from a speaker's utterances; never reuses an utterance."""

    def __init__(self, paths):
        self.paths = list(paths)
        self.buffer = np.zeros(0, dtype="float32")
        self.used = []

    def take(self, frames):
        while len(self.buffer) < frames:
            if not self.paths:
                raise ValueError("speaker ran out of utterances")
            path = self.paths.pop(0)
            self.used.append(Path(path).stem)
            self.buffer = np.concatenate([self.buffer, trim_silence(read_mono(RAW / path))])
        chunk, self.buffer = self.buffer[:frames], self.buffer[frames:]
        return chunk

    def start_new_section(self):
        self.buffer = np.zeros(0, dtype="float32")
        used, self.used = self.used, []
        return used


def faded(chunk):
    chunk = chunk.copy()
    chunk[:FADE] *= np.linspace(0.0, 1.0, FADE, dtype="float32")
    chunk[-FADE:] *= np.linspace(1.0, 0.0, FADE, dtype="float32")
    return chunk


def render(segments, speakers, streamers, frames_per_segment):
    """Place each active speaker's next chunk in each segment; returns per-speaker tracks."""
    tracks = {s: np.zeros(len(segments) * frames_per_segment, dtype="float32")
              for s in speakers}
    for index, segment in enumerate(segments):
        start = index * frames_per_segment
        for speaker in segment["active_speaker_ids"]:
            chunk = streamers[speaker].take(frames_per_segment)
            tracks[speaker][start:start + frames_per_segment] = faded(chunk)
    return tracks


def discovery_schedule(speakers):
    segments, round_ends = [], []
    n = len(speakers)
    for r in range(ROUNDS):
        for speaker in speakers:
            segments.append({"label": f"round{r + 1}_solo", "active_speaker_ids": [speaker]})
        pair = [speakers[(r + k) % n] for k in (0, 1)]
        triple = [speakers[(r + k) % n] for k in (2, 3, 4)]
        segments.append({"label": f"round{r + 1}_overlap2", "active_speaker_ids": pair})
        segments.append({"label": f"round{r + 1}_overlap3", "active_speaker_ids": triple})
        round_ends.append(len(segments))
    return segments, round_ends


def live_schedule(speakers, rng):
    counts = {s: 0 for s in speakers}
    segments = []
    for size in LIVE_SIZES:
        order = sorted(speakers, key=lambda s: (counts[s], rng.random()))
        active = order[:size]
        for s in active:
            counts[s] += 1
        segments.append({"label": f"live_{size}_active",
                         "active_speaker_ids": [s for s in speakers if s in active]})
    if min(counts.values()) < 3:
        raise ValueError(f"live schedule leaves a speaker with too little speech: {counts}")
    return segments


def speaker_utterances():
    rows = list(csv.DictReader((ROOT / "librispeech_test_clean.csv").open()))
    by_speaker = {}
    for row in rows:
        by_speaker.setdefault(row["speaker_ID"], []).append(row)
    for items in by_speaker.values():
        items.sort(key=lambda r: r["origin_path"])
    return by_speaker


def build_mixture(split, speakers, by_speaker, rng):
    mixture_id = f"{split}_n{len(speakers)}_" + "-".join(speakers)
    out = OUT / split / mixture_id
    (out / "audio").mkdir(parents=True, exist_ok=True)
    streamers, enrollments = {}, {}
    for speaker in speakers:
        rows = by_speaker[speaker]
        # Enrollment: first utterance of 5-15 s; everything else feeds the sections.
        enroll = next(r for r in rows if 5 * RATE <= int(r["length"]) <= 15 * RATE)
        enrollments[speaker] = enroll["origin_path"]
        streamers[speaker] = Streamer(r["origin_path"] for r in rows if r is not enroll)

    gains_db = {s: float(rng.uniform(-3.0, 3.0)) for s in speakers}
    sections = {}
    disc_segments, round_ends = discovery_schedule(speakers)
    disc_frames = int(TURN_SECONDS * RATE)
    disc_tracks = render(disc_segments, speakers, streamers, disc_frames)
    disc_utts = {s: streamers[s].start_new_section() for s in speakers}
    live_segments = live_schedule(speakers, rng)
    live_frames = int(LIVE_SEGMENT_SECONDS * RATE)
    live_tracks = render(live_segments, speakers, streamers, live_frames)
    live_utts = {s: streamers[s].start_new_section() for s in speakers}

    for name, tracks, segments, seg_frames, utts in (
        ("discovery", disc_tracks, disc_segments, disc_frames, disc_utts),
        ("live", live_tracks, live_segments, live_frames, live_utts),
    ):
        for s in speakers:
            active = tracks[s][np.abs(tracks[s]) > 0]
            tracks[s] *= TARGET_RMS * 10 ** (gains_db[s] / 20) / np.sqrt(np.mean(active ** 2))
        mix = np.sum(np.stack([tracks[s] for s in speakers]), axis=0)
        scale = min(1.0, 0.9 / float(np.max(np.abs(mix))))
        mix *= scale
        sf.write(out / f"audio/{name}_mix.wav", mix, RATE, subtype="FLOAT")
        for k, s in enumerate(speakers, 1):
            sf.write(out / f"audio/{name}_s{k}_{s}.wav", tracks[s] * scale, RATE,
                     subtype="FLOAT")
        sections[name] = {
            "mixture_path": f"audio/{name}_mix.wav",
            "source_paths": {s: f"audio/{name}_s{k}_{s}.wav"
                             for k, s in enumerate(speakers, 1)},
            "utterance_ids": utts,
            "segment_seconds": seg_frames / RATE,
            "segments": [{**seg, "start_frame": i * seg_frames,
                          "end_frame": (i + 1) * seg_frames}
                         for i, seg in enumerate(segments)],
            "frames": len(mix),
            "common_scale": scale,
        }
    sections["discovery"]["round_end_frames"] = [e * disc_frames for e in round_ends]

    for s in speakers:
        sf.write(out / f"audio/{s}_enrollment.wav",
                 read_mono(RAW / enrollments[s]), RATE, subtype="PCM_16")

    live = sections["live"]
    sources = [{"speaker_id": s,
                "source_path": live["source_paths"][s],
                "source_utterance_id": "+".join(live["utterance_ids"][s]),
                "enrollment_path": f"audio/{s}_enrollment.wav",
                "enrollment_utterance_id": Path(enrollments[s]).stem}
               for s in speakers]
    trials = [{"trial_id": f"{mixture_id}__speaker_{s}",
               "dataset": "ControlledLibriNMix", "dataset_revision": REVISION,
               "speech_license": "CC-BY-4.0", "split": "test",
               "condition": "conversation_clean", "sample_rate_hz": RATE,
               "mixture_path": live["mixture_path"], "selected_speaker_ids": [s],
               "sources": sources}
              for s in speakers]
    (out / "single_trials.jsonl").write_text("\n".join(json.dumps(t) for t in trials) + "\n")
    provenance = {
        "type": "controlled conversation derivative of LibriSpeech test-clean; not an official LibriMix row",
        "mixture_id": mixture_id, "split_role": split, "speakers": speakers,
        "seed": SEED, "speaker_gain_db": gains_db,
        "enrollment_utterances": {s: Path(enrollments[s]).stem for s in speakers},
        "construction": "per-section speaker RMS normalized to 0.05 times a +/-3 dB speaker gain; 10 ms fades per turn; common scale keeps mixture peak <= 0.9; sources sum exactly to mixture",
        "sections": sections,
        "file_sha256": {str(p.relative_to(out)): sha256(p)
                        for p in sorted((out / "audio").glob("*.wav"))},
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return mixture_id


def main():
    by_speaker = speaker_utterances()
    rng = np.random.default_rng(SEED)
    speakers = sorted(by_speaker)
    rng.shuffle(speakers)
    built = {}
    offset = 0
    for split in ("dev", "test"):
        built[split] = []
        for size in GROUP_SIZES:
            group = speakers[offset:offset + size]
            offset += size
            built[split].append(build_mixture(split, group, by_speaker, rng))
    (OUT / "mixtures.json").write_text(json.dumps(built, indent=2) + "\n")
    print(json.dumps(built, indent=2))


if __name__ == "__main__":
    main()
