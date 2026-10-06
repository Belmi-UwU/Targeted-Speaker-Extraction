# M1 report: first clean target-speaker trial

**Date:** 6 October 2026  
**Decision:** M1's first-trial and scoring contract is complete. Quality is uneven, and this checkpoint is still unsuitable for live streaming. Proceed to M2 with more speakers and conditions before choosing an app model.

## What was evaluated

One official [Libri2Mix](https://github.com/JorisCos/LibriMix) `test-clean` metadata row, `4077-13754-0001_5142-33396-0065`, was used to build a 16 kHz `min/mix_clean` mixture. The pinned metadata tree is `b898ca375c03e4039cdb3dbe3dd485aad58ec7d1`. The source utterances are from the official [LibriSpeech test-clean](https://www.openslr.org/12) archive (CC BY 4.0), whose MD5 matched the publisher's `32fa31d27d2e1cad72775fee3f4849a9`. Only four FLAC files were extracted: two mixture sources and one distinct enrollment utterance for each speaker. Metadata gains were applied, sources were trimmed to the shorter 3.77-second length, and sources plus their sum were saved as PCM16 WAV. No WHAM noise was used.

The [provenance record](data/trials/provenance.json) contains source and output hashes, gains, URLs, revision, and manifest hash. The [manifest](data/trials/trials.jsonl) has two trials on the same mixture, selecting speaker 4077 and speaker 5142 in turn. The [validator](validate_manifest.py) passed ID, sample-rate, length, and mixture-sum checks. Its [self-check](self_check.py) also covered two- and three-speaker arithmetic and rejected wrong IDs, same enrollment utterance IDs, missing files, and wrong sample rates.

The independently verified [OpenSpeakerBeam-SS checkpoint](../speakerbeam_ss_verification/VERIFICATION_REPORT.md) ran unchanged on each trial using separate Resemblyzer enrollment embeddings. Raw 32-bit float outputs were retained; [run metadata](data/trials/baseline_run.json) pins code and weight hashes. These runs use a computer CPU, not a phone.

## Results

| Selected speaker | Mixture SI-SDR to target | Output SI-SDR to target | Improvement | Raw peak | Raw samples beyond ±1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4077 | 1.12 dB | 11.69 dB | **+10.57 dB** | 27.54 | 32.00% |
| 5142 | -1.22 dB | -0.07 dB | **+1.14 dB** | 23.10 | 18.19% |

These are **two selections of one mixture**, not two independent samples or a dataset average. SI-SDR is scale-invariant, so large raw peaks do not change these scores; they still matter for playback. The output for speaker 5142 is closer to that speaker than to the interferer by the same diagnostic, but retains much less clean target structure than the 4077 selection. Do not treat either result as a public SpeakerBeam-SS benchmark: the checkpoint's training speakers are not documented, and possible train/test overlap is unresolved. Full [per-trial scores](data/trials/scores.jsonl) include each source diagnostic and output statistics.

For listening, [comparison WAVs](data/trials/listening_comparisons) include the same mixture, selected clean source, and model output for each selection. A **single fixed gain of 0.014** was applied to both model outputs, calibrated from the first trial's clean reference. This is an evaluation listening aid; it is not a deployable gain policy, and it does not affect the raw scores. The earlier repository demo mismatch was separately resolved: its supplied output matches clipping the raw model waveform, losing about 38.32% of samples. [Verification details](../speakerbeam_ss_verification/VERIFICATION_REPORT.md).

**Listener feedback (6 October 2026):** The user reported that the 4077 `model_fixed_gain.wav` sounds cleaner than the 5142 version. This is one subjective comparison on this mixture. It agrees with the larger SI-SDR improvement for 4077, but does not establish general model quality or intelligibility.

## What remains for M2

1. Repeat on a larger speaker-disjoint set or label results exploratory until checkpoint training provenance is known. Quantify the distribution, not just this one mixture.
2. Test three-speaker mixtures with **two selected speakers**. The current checkpoint accepts only one enrollment at a time; compare per-speaker extraction plus careful combination against a model that directly supports a selected set.
3. Add target-only, selected-speaker-absent, turns, and noisy conditions. Use SparseLibriMix or controlled recordings for turns/absence; ordinary full-overlap LibriMix cannot test those behaviors alone.
4. Choose a causal candidate with compatible, redistributable weights and measure real device routing and delay after the first platform is selected.

The evaluation dataset occupies about **338 MB**, within the planned 3 GB cap. No full LibriMix corpus or WHAM noise was downloaded.
