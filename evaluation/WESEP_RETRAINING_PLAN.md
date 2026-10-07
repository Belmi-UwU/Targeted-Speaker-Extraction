# Retraining plan: causal WeSep for the live stage

**Status:** Draft for decision · 7 October 2026  
**Why:** The [WeSep comparison](WESEP_COMPARISON.md) found that the REAL-TSE causal checkpoint has the delay we want but separates poorly in our conditions:
- about +3 dB median SI-SDRi in 4–6 speaker conversations, against about +12 dB for the SepFormer selector;
- about 2 dB of suppression while the target is silent;
- on the real recording, it never extracted Nicker.

The likely cause is its training data (clean **two-speaker** Libri2Mix), not causality: the non-causal checkpoint is just as weak with more speakers. The checkpoint also has no weights licence. This plan trains our own causal model with the WeSep recipe on data that looks like the product.

## 1. Goal and exit gates

A causal target-speaker extraction model that is conditioned on the **audio of a discovered track** and runs live on a laptop, later a phone. Gates are measured with the existing harness on held-out speakers:

| Area | Gate | Today: SepFormer selector / WeSep checkpoint |
| --- | --- | --- |
| Quality, 4–6 speakers | Median SI-SDRi ≥ **+10 dB** with discovered enrollment; p10 > 0 | +11.6 (2.2 s delay) / +2.8 |
| Target absent | Absent suppression ≥ **10 dB** | 13–34 / ~2 |
| Target preservation | Target-alone SI-SDR ≥ **20 dB** | 12–21 / ~15 |
| Discovered vs clean enrollment | Median drop ≤ **1 dB** (PRD §8) | 0.0 / ~1.1 |
| Causality | Strict prefix test: future audio changes earlier output by < 1e-5 relative | — / 3e-3 |
| Delay | ≤ **150 ms** capture-to-playback on the laptop with stateful streaming; ≤ 0.5 s with block mode as an interim | 2.2 s / 0.9 s |
| Compute | Real-time factor ≤ 0.5 on the laptop CPU (Apple M5, 4 threads) | 0.85 / 0.55 |
| Real recording | Right person for Nicker, Mond, and Sam (voice match), plus the user's listening ratings | 3/3 / 2/3 |

## 2. Starting point

- **Code:** upstream [WeSep](https://github.com/wenet-e2e/wesep) (Apache-2.0), with its BSRNN separator in causal mode:
  - cumulative/channel-wise layer norm;
  - unidirectional time LSTMs;
  - 512-sample window and 128 hop (32 ms algorithmic latency).

  We use upstream rather than the `wesep-real-tse` fork, which has no LICENSE file. If the fork's TF-map + context code is not upstream, re-implement it from the paper.
- **Conditioning:** TF-map + frame-level context from enrollment audio, the best causal variant in the WeSep paper (12.98 dB on Libri2Mix vs 6.87 dB for a plain ECAPA embedding). Keep a speaker-embedding variant using our SpeechBrain ECAPA as a cheaper baseline.
- **Context speaker encoder:** WeSpeaker ECAPA (VoxCeleb). The survey records the WeSpeaker weights as CC BY 4.0; verify before use.
- **Initialization:** train **from scratch** for anything shippable. A fine-tune from the unlicensed REAL-TSE checkpoint is allowed only as a labelled research pilot (R1). Its weights never ship.

## 3. Training data

Mix on the fly from source audio, recording every licence. Use no WHAM! noise (CC BY-NC).

| Component | Source | Licence | Notes |
| --- | --- | --- | --- |
| Speech | LibriSpeech `train-clean-100`, `train-clean-360`, `train-other-500` (~2,300 speakers) | CC BY 4.0 | Disjoint from `test-clean`, so our dev/test conversations stay held out |
| Language coverage (optional) | Common Voice, Indonesian and other languages the users speak | CC0 | Read speech; checks for language mismatch |
| Noise | MUSAN noise subset, or another licence-clean set | Verify per file | Room noise, babble, music at low level |
| Reverb | Synthetic room impulse responses (pyroomacoustics) | MIT (tool) | Phone-distance rooms, RT60 0.2–0.8 s |

**Conversation simulator.** This extends `build_discovery_conversations.py`, randomized for each example:
- 2–6 speakers per mixture, with mostly 1–3 talking at once and occasional 4;
- random turn lengths of 0.5–4 s, overlaps, and pauses;
- speaker levels ±10 dB, overall level −35 to −15 dBFS;
- noise at 5–30 dB SNR on some examples; reverb on some examples;
- 4–8 s training chunks, so each chunk has turn changes.

**Target-absent examples.** About 25% of chunks have the target silent for long stretches, and about 5% have the target absent throughout. The released model barely suppresses in this case, so this is the main fix for "leaks others when the target is silent".

**Enrollment that matches the live flow.** For each example, one of:
- 40% clean: another utterance by the target;
- 40% simulated discovered track: band-limit to 8 kHz, add another speaker's leakage at −25 to −12 dB, insert gaps, apply random level;
- 20% real discovered tracks from a pool built by running our SepFormer discovery on simulated conversations. This needs a GPU or an overnight CPU job.

Enrollment level is randomized to ±15 dB to remove the level sensitivity we found. Inference keeps the −25 dBFS normalization.

**Storage.** LibriSpeech training sets are about 60 GB, plus noise. That is far above the 3 GB local cap used so far, so training data lives on the training machine, not this laptop.

## 4. Model and loss

- **Main model:** BSRNN causal at the released size (feature dim 128, 6 repeats, ~27 M parameters), so results compare directly with the checkpoint.
- **Phone candidate:** a smaller model (feature dim 64, 4 repeats, about a quarter of the compute), trained once the main model passes R2.
- **Loss:**
  - SI-SDR on target-present regions;
  - an energy penalty on target-absent regions (output power relative to input), because SI-SDR is undefined for a silent target;
  - optionally a multi-resolution STFT loss for artifacts.
- **Curriculum:** start with 2–3 speakers and clean enrollment, then add more speakers, target absence, discovered-style enrollment, noise, and reverb.

## 5. Streaming implementation

Today's live demo re-runs the model on block + context. Real streaming needs:
1. a stateful step function: streaming STFT/iSTFT with overlap buffers, LSTM hidden states carried between hops, cumulative norm statistics carried forward;
2. enrollment features computed once when the user picks a speaker;
3. a strict causality test (prefix invariance < 1e-5). First find and remove the small future dependence measured in the released model (0.3% for TF-map, about 13% for speaker embedding);
4. a third live demo beside `live_demo.py` and `wesep_eval/live_demo_wesep.py`, with measured capture-to-playback delay;
5. later, ONNX export, then Core ML or TFLite for the phone milestones (PRD M3/M4).

## 6. Evaluation

Each candidate goes through the existing harness unchanged. That gives direct comparison with SepFormer and the released checkpoint:
- `run_discovery_experiment.py`-style scoring on the six controlled conversations, plus a **larger held-out set** of about 30 conversations built from `test-clean` with the same builder, with noise/reverb variants;
- the Nicker/Mond/Sam recording (voice match plus the user's ratings exported from the listening pages);
- the REAL-T DEV split (real conversations, transcript/timing metrics). Check its use terms first;
- unknown-speaker and target-absent recordings: a person who joins after discovery, and a selected person who never talks.

## 7. Milestones

| Step | Work | Compute | Go / no-go |
| --- | --- | --- | --- |
| **R0 Data** | Simulator, enrollment simulation, licence log, small sample validated with `validate_manifest.py` | Laptop | Mixtures sum exactly; licence log complete |
| **R1 Pilot (research only)** | Fine-tune the released causal checkpoint on R0 data. Answers whether data alone fixes groups and absence. | ~1 GPU-day (estimate) | Test median ≥ +8 dB and suppression ≥ 8 dB, otherwise rethink the model/conditioning before R2 |
| **R2 From scratch** | Licence-clean training, main model size | Several GPU-days per run (estimate; the released model trained 150 epochs on 2 GPUs) | All quality gates in §1 |
| **R3 Streaming** | Stateful inference, strict causality, live demo, latency measurement | Laptop | Delay ≤ 150 ms, RTF ≤ 0.5 |
| **R4 Phone** | Small model, export, on-device benchmark | Laptop + phone | PRD M3/M4 gates |

## 8. Risks

- **Domain gap.** Read English audiobooks are not spontaneous conversation in the users' language. Mitigate with Common Voice, real recordings in evaluation, and REAL-T.
- **Discovered-enrollment mismatch.** Simulated artifacts may not match real SepFormer tracks. Mitigate with the real discovered-track pool and by tracking the clean-vs-discovered drop at every step.
- **Absent-target behaviour** may trade off against target preservation. Watch both gates together.
- **Licences.** Speaker-encoder weights (VoxCeleb), noise sources, and any optional data each need a recorded grant before a model is shipped.
- **Compute and cost.** Full training does not fit on the laptop.

## 9. Decisions needed

1. **Where to train:** cloud GPU (budget?) or an existing machine. This sets the R2 timeline.
2. **Languages:** add Indonesian (or others) to training now, or evaluate English-only first.
3. **R1 pilot:** OK to fine-tune the unlicensed REAL-TSE checkpoint as a research-only signal, or go straight to from-scratch training?
4. **Storage cap:** raise the 3 GB local cap for training data on the training machine only.
