# Where to place your recordings

Put the **original files** in `data/user_recordings/` using this layout:

```text
data/user_recordings/
├── mixture/
│   └── three_voice_mix.wav
├── enrollments/
│   ├── nicker/
│   │   ├── 01.wav
│   │   ├── 02.wav
│   │   └── 03.wav
│   ├── mond/
│   │   ├── 01.wav
│   │   ├── 02.wav
│   │   └── 03.wav
│   └── sam/
│       ├── 01.wav
│       ├── 02.wav
│       └── 03.wav
└── clean_references/          # optional; see below
```

The three separate recordings for each person are **enrollment examples**: they tell the speaker matcher what Nicker, Mond, or Sam sounds like. They do not need to say the same words as the mixture. If a file is `.m4a`, `.flac`, or another format, keep its real extension; do not rename it to `.wav` without converting it. Keep the original sample rate and channels for now; the evaluation pipeline will make working copies at the rates needed by SepFormer and the speaker matcher.

Use `clean_references/` only if you have an isolated, time-aligned track of one person's voice **from the exact same mixed recording**. Separate example recordings are not a clean reference for the mixture. We can still run separation and listen without these reference tracks; objective SI-SDR scoring requires them.

The LibriMix dataset scripts do **not** automatically pick up this folder. Use the dedicated [run_user_recordings.py](run_user_recordings.py) script for the pinned SepFormer + ECAPA offline pipeline after checking the input files. The local audio folder is ignored by Git so personal recordings are not added to source control by accident.

## First local run

The uploaded `Mix.wav`, `Nicker.wav`, `Mond.wav`, and `Sam.wav` were processed with [run_user_recordings.py](run_user_recordings.py). The original files remain untouched. The script downmixes stereo 48 kHz input to mono, feeds 8 kHz audio to SepFormer, uses 16 kHz ECAPA enrollment matching, and exports 16 kHz PCM16 listening WAVs in `data/user_recordings/results/`:

- `original_mix_mono_16k.wav` — downmixed original mixture.
- `only_nicker.wav`, `only_mond.wav`, `only_sam.wav` — one selected speaker each.
- `nicker_and_mond.wav`, `nicker_and_sam.wav`, `mond_and_sam.wav` — two selected speakers each.

The run's `results/run.json` records input hashes, model revisions, speaker-to-stream matches, gain fitting, and output hashes. There are no clean references from this mixed recording, so objective source-separation scores are unavailable. The model's selected-speaker outputs require listening checks, especially for unwanted-speaker leakage. This is offline processing, not a live-app latency test.
