# M2 model comparison: SpeakerBeam-SS versus three-stream SepFormer

**Date:** 6 October 2026  
**Decision:** SepFormer plus mixture-based gain fitting and an ECAPA enrollment matcher is the strongest **offline candidate tested so far**. ECAPA matched the correct streams in all 12 original pairs and all 12 new holdout pairs, but the sample is still small and clean. No live-app model is selected: target absence, noise, real recordings, streaming causality, and device routing remain open.

## What was compared

The earlier independent OpenSpeakerBeam-SS checkpoint extracts one enrolled speaker at a time; two selected outputs are summed. The new comparator is [SpeechBrain's pretrained Libri3Mix SepFormer](https://huggingface.co/speechbrain/sepformer-libri3mix) at model revision `1f2d824e2333aec9e60d0030fb144885c1afc426`, loaded using [SpeechBrain](https://github.com/speechbrain/speechbrain) 1.1.1. The model card labels the weights **Apache 2.0**, and SpeechBrain code is Apache 2.0. The model is a **blind separator**, not a TSE model: it produces three anonymous streams at 8 kHz. We use the same Resemblyzer voice encoder as in the SpeakerBeam-SS baseline to match the enrolled selected speakers to distinct streams by cosine similarity and one-to-one assignment. Matching uses **only selected enrollments**, not clean sources or the unselected speaker's enrollment. It has no confidence threshold, so it would still assign a stream if a selected speaker were absent.

We compared four clean three-speaker mixtures: the first 4077/5142/5683 clip, plus [three pinned official Libri3Mix test-clean rows](data/m2_extended) with nine additional speakers who do not overlap the first three or each other. The new rows are 3.42–4.47 seconds at their shortest-source lengths. All speech comes from the already verified LibriSpeech test-clean archive, using different enrollment utterances; no new dataset archive or noise was downloaded. [The builder](build_m2_extended.py), per-mixture manifests, provenance, output hashes, and scores make the exact subset reproducible. This is a 12-pair exploratory set, not a published benchmark.

SepFormer streams have inconsistent raw gain and one stream's polarity is inverted. Summing raw streams can therefore sound wrong even when the speakers are separated. We tested a second combination: least-squares signed gains that reconstruct the **observed input mixture** from all three streams at 8 kHz, then sum only selected streams. This uses no clean reference. Across the five clips including the turns clip, relative mixture-fit error was about 0.8–1.0%; the signed gains were roughly -0.0367, +0.0398, +0.0398. [The run code](run_sepformer_comparison.py) saves raw and fitted outputs separately. The first [run record](data/m2_three_speakers/sepformer_run.json) pins model hashes, runtime versions, matching similarities, gains, and output hashes. These gains are empirical for this checkpoint, not a general SepFormer rule.

## Same-clip result and four-mixture summary

All figures below use [the same 8 kHz scoring path](score_m2_common8k.py): mixture, clean references, and both models' outputs are downsampled from the saved 16 kHz WAVs before SI-SDR is calculated. This avoids counting SepFormer's lower bandwidth as an extraction error. The listening WAVs are 16 kHz containers, but SepFormer has only 8 kHz input bandwidth.

| Selected pair | SpeakerBeam-SS sum SI-SDRi | SepFormer raw sum | SepFormer mixture-fit + original matcher | SepFormer mixture-fit + ECAPA |
| --- | ---: | ---: | ---: | ---: |
| 4077 + 5142 (user's problem clip) | +0.48 dB | +19.68 dB | +19.68 dB | **+19.68 dB** |
| Median of original 12 pairs | +0.04 dB | -5.52 dB | +12.94 dB | **+15.98 dB** |
| Pairs with positive improvement | 6/12 | 3/12 | 7/12 | **12/12** |
| Range across original 12 pairs | -3.46 to +8.32 dB | -20.37 to +19.68 dB | -8.02 to +19.68 dB | **+12.13 to +20.20 dB** |

The original Resemblyzer matching pipeline had **five failing pair selections**. ECAPA matching fixed those cases on this set while using the same SepFormer streams and mixture-fit gains. The first clip's 16 kHz score is +15.46 dB for SepFormer mixture-fit versus +0.46 dB for SpeakerBeam-SS; the 8 kHz figures above are the fairer extraction comparison. Exact per-trial scores live in each mixture's `common8k_scores.jsonl` under [the first clip](data/m2_three_speakers) and [the extended clips](data/m2_extended).

**Listener feedback (6 October 2026):** After hearing the [4077 + 5142 SepFormer output](data/listen_next/04_sepformer_mixfit_4077_5142.wav) and the [5142-only SepFormer output](data/listen_next/09_sepformer_only_5142.wav), the user reported that SepFormer seems to split the voices well for both one- and two-speaker selection on this three-voice example. The 5142-only output improved SI-SDR by **+23.23 dB** under the common 8 kHz scoring path. This supports SepFormer as the stronger offline separator on this clip; it does not resolve the matching failures on the other mixtures.

To diagnose the bottleneck, [a separate script](analyze_m2_matching.py) used clean references *after inference* to identify which anonymous stream actually belongs to each speaker. This is **oracle analysis only** and is never used to form the model outputs. The original Resemblyzer matcher agreed with the clean-source mapping for **6 of 12 selected pairs**. With clean-source mapping, the mixture-fit separator's 12 pair improvements were all positive, about **+12.5 to +20.2 dB**. Thus the separator can isolate these voices, while the original identity matcher often chose the wrong ones. The [diagnostic records](data/m2_three_speakers/matching_diagnostic.json) and corresponding records in the extended mixture folders show each mapping.

### Stronger speaker matching and new holdout

We replaced **only** the enrollment-to-stream matcher with [SpeechBrain ECAPA-TDNN speaker embeddings](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb), pinned at revision `0f99f2d0ebe89ac095bcc5903c4dd8f72b367286`. Its model card labels the checkpoint Apache 2.0. [The matcher code](evaluate_ecapa_matcher.py) uses the same cached SepFormer streams and the same mixture-fit gains; it embeds each enrollment and stream at 16 kHz, computes cosine similarity, and assigns distinct streams to the selected speakers. It uses no clean source or unselected-speaker enrollment to make outputs. Each [run record](data/m2_three_speakers/ecapa_matcher_run.json) pins the ECAPA weight hashes and decisions. The oracle mapping is used *only afterward* to assess accuracy.

Because the original 12 pairs motivated this change, we reserved [four additional official Libri3Mix mixtures](data/m2_holdout), with **12 new speakers and 12 new selected pairs**, as a separate holdout. The builders, source/enrollment IDs, hashes, output decisions, and common 8 kHz scores are saved per mixture. The holdout contains clean full-overlap speech only; it is not a large or real-world benchmark.

| Pair-selection evaluation | Original Resemblyzer matcher | ECAPA matcher |
| --- | ---: | ---: |
| Correct stream pair on original 12 (clean-source audit) | 6/12 | **12/12** |
| Correct stream pair on new holdout 12 | 7/12 | **12/12** |
| New holdout pairs with positive SI-SDR improvement | 7/12 | **12/12** |
| New holdout median SI-SDR improvement | +13.09 dB | **+15.25 dB** |
| New holdout improvement range | -11.79 to +20.99 dB | **+12.99 to +20.99 dB** |

This supports ECAPA as the better **offline matching choice within this small clean test**. It still forces a match even when a selected speaker is absent from the entire recording, so a rejection/confidence policy remains necessary. The SepFormer separator remains noncausal and fixed to three output streams.

## Speaking turns and selected-speaker absence

We also built a **controlled derivative**, clearly separate from the official Libri3Mix rows, from the same three source recordings. It has four 0.9-second intervals: 4077 alone, 5683 alone, all three overlapping, then 5142 alone. During the 5683-only interval, **neither selected speaker is active**. Source gating uses 10 ms fades and the mixture is the sum of the gated sources. [Construction and segment boundaries](data/m2_turns/provenance.json), [run files](data/m2_turns), and [segment scoring code](score_m2_turns.py) are saved.

For the 4077 + 5142 selection, whole-clip 8 kHz SI-SDRi was **+0.71 dB** for SpeakerBeam-SS sum and **+24.06 dB** for SepFormer mixture-fit. On the interior 0.7 second of the 5683-only interval, the output RMS relative to the target-only intervals was **+7.1 dB** for SpeakerBeam-SS and **-37.1 dB** for SepFormer mixture-fit. The corresponding original-mixture ratio was +7.6 dB. SepFormer mixture-fit attenuated the 5683-only interval about **44.7 dB** relative to its input RMS on this constructed example. The unscaled SpeakerBeam-SS output has arbitrary amplitude, so its absolute output/input attenuation is not comparable; the within-output ratio is the useful diagnostic. [Exact segment results](data/m2_turns/segment_scores.json).

This one controlled clip does not establish real conversation behavior. It has fixed speaker count, known clean sources, synthetic on/off boundaries, and no room noise. The SepFormer model is noncausal and the model forward time on a desktop CPU (~1.1–1.4 seconds for ~3–4 seconds of audio) excludes capture, resampling, voice matching, and playback. It is **not** a streaming latency measurement or phone benchmark.

## Listening and next gate

The short [listening set](data/listen_next) contains numbered original mix, ideal selected reference, SpeakerBeam-SS output, and SepFormer mixture-fit output for both the original full-overlap clip and the controlled turns clip. It also contains a 5142-only SepFormer extraction and clean reference. Different model outputs have different playback gains; compare **which voices remain, clarity, and unwanted leakage**, not absolute loudness. The source files and gain methods remain in their respective run folders.

Next, test complete target absence and add an explicit confidence/rejection policy. Then evaluate several controlled turns and noisy/real recordings under a recorded acquisition and license plan. A causal selected-speaker system and device-route proof remain required before real-time phone integration. SepFormer + ECAPA is the current **offline reference pipeline**, not the chosen live app model.
