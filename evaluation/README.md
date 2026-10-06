# Offline TSE evaluation

This folder implements the offline evaluation steps from the [PRD](../PRD.md). It contains one official-metadata-derived Libri2Mix clean mixture, eight Libri3Mix three-speaker mixtures, and one controlled turns/absence derivative, with separate enrollment for every speaker. See the [M1 report](M1_REPORT.md), [M2 interim report](M2_INTERIM_REPORT.md), and [M2 model comparison](M2_MODEL_COMPARISON.md). These are exploratory checks, not a benchmark or proof of generalization.

## First trial set

The first set uses a 16 kHz, mono **Libri2Mix `mix_clean` test** mixture with two speakers. Each trial has the mixture, aligned clean source WAVs, and a different utterance from each source speaker for enrollment. The matching Libri3Mix row adds a third speaker and tests each single selection, every two-speaker selection, and all three. Keep the published test metadata and source IDs; do not create arbitrary speaker pairs and then call them the official LibriMix test set.

The manifest is JSON Lines: one trial object per line. Paths are relative to the manifest's directory. All source WAVs must already be aligned and scaled as they appear in the mixture. `selected_speaker_ids` may contain one or more IDs. `source_utterance_id` must differ from `enrollment_utterance_id`; comparing filenames alone is insufficient for derived audio. Each row records the dataset revision and speech license. The minimum fields and meaning are in [manifest.schema.json](manifest.schema.json). An example with intentionally nonexistent paths is in [manifest.example.jsonl](manifest.example.jsonl).

### Validation and scoring

```sh
../speakerbeam_ss_verification/.venv/bin/python validate_manifest.py path/to/trials.jsonl --check-mix-sum
../speakerbeam_ss_verification/.venv/bin/python score_outputs.py path/to/trials.jsonl path/to/model_outputs/
```

Reproduce the current small set from the already-downloaded archive and pinned metadata:

```sh
mkdir -p data/raw
tar -xzf data/archives/test-clean.tar.gz -C data/raw \
  LibriSpeech/LICENSE.TXT LibriSpeech/README.TXT \
  LibriSpeech/test-clean/4077/13754/4077-13754-0001.flac \
  LibriSpeech/test-clean/5142/33396/5142-33396-0065.flac \
  LibriSpeech/test-clean/4077/13751/4077-13751-0013.flac \
  LibriSpeech/test-clean/5142/33396/5142-33396-0056.flac
../speakerbeam_ss_verification/.venv/bin/python build_first_trial.py
../speakerbeam_ss_verification/.venv/bin/python validate_manifest.py data/trials/trials.jsonl --check-mix-sum
../speakerbeam_ss_verification/.venv/bin/python run_baseline.py
../speakerbeam_ss_verification/.venv/bin/python score_outputs.py data/trials/trials.jsonl data/trials/model_outputs > data/trials/scores.jsonl
../speakerbeam_ss_verification/.venv/bin/python make_listening_comparisons.py
```

Reproduce the three-speaker selected-set trial with the same archive and pinned `librimix3_test_clean.csv` metadata:

```sh
tar -xzf data/archives/test-clean.tar.gz -C data/raw \
  LibriSpeech/test-clean/5683/32866/5683-32866-0012.flac \
  LibriSpeech/test-clean/5683/32865/5683-32865-0016.flac
../speakerbeam_ss_verification/.venv/bin/python build_m2_trial.py
../speakerbeam_ss_verification/.venv/bin/python validate_manifest.py data/m2_three_speakers/single_trials.jsonl --check-mix-sum
../speakerbeam_ss_verification/.venv/bin/python validate_manifest.py data/m2_three_speakers/selected_set_trials.jsonl --check-mix-sum
../speakerbeam_ss_verification/.venv/bin/python run_baseline.py --manifest data/m2_three_speakers/single_trials.jsonl
../speakerbeam_ss_verification/.venv/bin/python combine_m2_outputs.py
../speakerbeam_ss_verification/.venv/bin/python score_outputs.py data/m2_three_speakers/single_trials.jsonl data/m2_three_speakers/model_outputs > data/m2_three_speakers/single_scores.jsonl
../speakerbeam_ss_verification/.venv/bin/python score_outputs.py data/m2_three_speakers/selected_set_trials.jsonl data/m2_three_speakers/model_outputs > data/m2_three_speakers/selected_set_scores.jsonl
```

The subsequent [M2 comparison](M2_MODEL_COMPARISON.md) adds [three speaker-disjoint official mixtures](build_m2_extended.py), [four further holdout mixtures](data/m2_holdout), and [one controlled speaking-turns clip](build_m2_turns.py). The [SpeechBrain Libri3Mix SepFormer model](https://huggingface.co/speechbrain/sepformer-libri3mix) is pinned at revision `1f2d824e2333aec9e60d0030fb144885c1afc426`; only its `encoder.ckpt`, `masknet.ckpt`, and `decoder.ckpt` are needed. Their SHA-256 hashes, code/dependency versions, and exact per-trial matches are in each `sepformer_run.json`. This model runs at 8 kHz and is an offline comparator, not a streaming TSE checkpoint. [ECAPA matching](evaluate_ecapa_matcher.py) replaces the original voice matcher for the current offline reference; its checkpoint is pinned at `0f99f2d0ebe89ac095bcc5903c4dd8f72b367286`. Use [run_sepformer_comparison.py](run_sepformer_comparison.py), [score_m2_common8k.py](score_m2_common8k.py), and [analyze_m2_matching.py](analyze_m2_matching.py) for inference, common-bandwidth scores, and clean-source diagnostics respectively. The [numbered listening set](data/listen_next) gives the easiest A/B comparison.

`validate_manifest.py` checks IDs, enrollment separation, route-independent audio properties, sample alignment, and optional mixture/source arithmetic. Run the sum check only on clean additive mixtures. Corpus metadata must establish that each enrollment really belongs to the stated speaker; the validator cannot determine identity from a filename. `score_outputs.py` expects `<trial_id>.wav` files and writes one result per trial to stdout in JSON Lines. It scores the **raw float model output**, not a clipped or peak-normalized listening copy.

For one selected speaker, SI-SDR improvement compares that speaker's clean source with the estimate versus the same source with the mixture. For multiple selected speakers, the reference is the sum of selected clean sources. That combined score can hide the loss of a quieter selected speaker, so the three-speaker review also asks the listener to check each selected speaker's audibility. The per-source SI-SDR diagnostic is meaningful for one-target cases; in a multi-selected trial it treats the *other selected speaker* as distortion and must not be read as an individual retention score. A standard Libri3Mix full-overlap clip alone cannot establish turn-taking or target-absence behavior; [SparseLibriMix](https://github.com/popcornell/SparseLibriMix) or controlled recordings are needed for those conditions.

The scorer reports input/output peak, RMS, and the fraction of raw samples outside [-1, 1], in addition to SI-SDRi. The current model's demo has many such samples; clipping before scoring would throw away information and make results difficult to interpret.

### Data handling and estimated storage

The [LibriMix generator](https://github.com/JorisCos/LibriMix) can generate many variants and warns that its default run uses hundreds of GB. For M1, work was limited to the 16 kHz `test`, `mix_clean`, two-speaker condition and one fixed mixture. The source [LibriSpeech `test-clean` archive](https://www.openslr.org/12) is listed as **346 MB compressed**; only four FLAC utterances were extracted for M1. The **3 GB temporary working-storage cap** was respected: the evaluation `data/` folder used about **338 MB** after M1 acquisition. M2 reused that archive and extracted only two additional utterances.

Record dataset source URL, version/commit, original split, speech/noise license, generation command, and file hashes in a separate acquisition log. LibriSpeech is [CC BY 4.0](https://www.openslr.org/12). Noisy LibriMix uses [WHAM! noise with CC BY-NC 4.0 terms](https://wham.whisper.ai/), which is a separate licensing consideration. Avoid training or tuning on the same test trials used for reporting. Because the current checkpoint's training speakers are unknown, label initial results as an exploratory evaluation rather than proven independent generalization.

### M1 exit criteria

- At least one **real clean two-speaker trial** has a mixture, two aligned clean sources, and separate enrollment for the selected speaker.
- The validator passes for that trial and detects intentionally broken ID/path/audio examples.
- Scoring on a known synthetic output reproduces the expected direction of SI-SDR improvement; the real trial produces a per-trial score and matched listening files.
- Dataset provenance, file hashes, license, and storage use are recorded.

**M1 complete:** the validator, synthetic mechanics check, clean real-speech trial, raw checkpoint inference, per-trial SI-SDRi scores, and listening comparisons are available. The [M1 report](M1_REPORT.md) states what these two trials do and do not show.
