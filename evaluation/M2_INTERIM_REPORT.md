# M2 interim report: three voices, selected sets

**Date:** 6 October 2026  
**Decision:** Directly summing this checkpoint's single-speaker outputs is not a reliable two-selected-speaker solution on this trial. Keep M2 open and evaluate another candidate and more conditions before choosing an app model.

**Later work:** The [M2 model comparison](M2_MODEL_COMPARISON.md) evaluates a pinned SepFormer separator, three more speaker-disjoint mixtures, and a controlled turns/absence clip. Use that report for the current M2 status.

## Trial and method

One official [Libri3Mix test-clean metadata row](https://github.com/JorisCos/LibriMix/tree/b898ca375c03e4039cdb3dbe3dd485aad58ec7d1/metadata/Libri3Mix) was used: `4077-13754-0001_5142-33396-0065_5683-32866-0012`. It extends the M1 mixture with speaker 5683, so the change in the 4077/5142 results is easier to inspect. The 16 kHz `min/mix_clean` mixture is 3.77 seconds. Its three sources and three *different-utterance* enrollments came from the already verified [LibriSpeech test-clean archive](https://www.openslr.org/12); only two additional FLAC files were extracted. No noise dataset was downloaded. [Provenance and hashes](data/m2_three_speakers/provenance.json), [single-target manifest](data/m2_three_speakers/single_trials.jsonl), and [selected-set manifest](data/m2_three_speakers/selected_set_trials.jsonl) are saved.

The same independently verified OpenSpeakerBeam-SS checkpoint extracted each of the three speakers once. For every pair and the all-three control, the candidate output was the **unweighted sum of the relevant raw single-target outputs**. No clean reference was used to tune the combination. The [baseline run record](data/m2_three_speakers/baseline_run.json) pins checkpoint and code; the validator passed all seven trial rows with the mixture-sum check. Scores use raw float WAVs. [Listening WAVs](data/m2_three_speakers/listening_comparisons) use one common gain of 0.0127804 across the four model combinations, reduced from the M1 0.014 gain to avoid preview clipping. That gain is for listening only, not a phone gain policy.

## Results

| Selected | Mixture SI-SDR | Output SI-SDR | Improvement |
| --- | ---: | ---: | ---: |
| 4077 only | -6.19 dB | 1.79 dB | +7.98 dB |
| 5142 only | -10.92 dB | -15.70 dB | **-4.78 dB** |
| 5683 only | 4.34 dB | 10.10 dB | +5.76 dB |
| 4077 + 5142 | -4.32 dB | -3.86 dB | +0.46 dB |
| 4077 + 5683 | 10.85 dB | 10.23 dB | -0.62 dB |
| 5142 + 5683 | 6.17 dB | 7.07 dB | +0.90 dB |
| All three | 71.25 dB | 8.26 dB | -63.00 dB |

The single-target output requested for **5142** is much closer to **unselected 5683** by the per-source SI-SDR diagnostic: +0.48 dB to 5683 versus -15.70 dB to 5142. This indicates a wrong-speaker extraction on this clip. The two-selected scores are near zero or negative improvement; the selected-set sum did not produce a dependable benefit here. When all three are selected, the unprocessed mixture is already essentially the ideal reference, so the large negative score simply confirms extraction damages that case. See the exact [single-target scores](data/m2_three_speakers/single_scores.jsonl) and [selected-set scores](data/m2_three_speakers/selected_set_scores.jsonl).

**Listener feedback (6 October 2026):** For the 4077 + 5142 `model_fixed_gain.wav`, the user hears 5683 as overpowering both selected speakers and finds 5142 unclear. This confirms that the pair fails the intended listening outcome for this example. The preview gain multiplies the entire waveform uniformly; changing that gain would change volume, not restore 5142 or selectively remove 5683. The single-target 5142 diagnostic supports the same failure mode. This is a subjective observation on one clip, not a failure rate across the dataset.

Raw outputs have peaks of 20.5–35.9 for single targets and 36.3–70.4 for selected sets; 35–63% of samples, depending on the trial, exceed ±1. A gain or limiter is essential for playback. This report does **not** claim the checkpoint's original paper or other TSE models share these errors. It is one exploratory mixture from an independent implementation with unknown training-speaker overlap. The full-overlap clip cannot test selected-speaker turns or absence. Combined-reference SI-SDR can also hide loss of one selected speaker; the 5142 wrong-speaker diagnostic is why listening and source-specific checks remain necessary.

## What to listen to

In [listening comparisons](data/m2_three_speakers/listening_comparisons), compare the files ending `__mix.wav`, `__selected_clean.wav`, and `__model_fixed_gain.wav` for the same `__selected_...` pair. The priority is `__selected_4077_5142`, then `__selected_4077_5683`. Note whether **both chosen voices** are audible and whether unselected 5683 or 5142 leaks through. The clean file is the intended target, not model output. These previews share one gain for model outputs, while the original mixture and clean references retain their dataset levels, so absolute loudness alone should not be treated as a quality score.

## Next M2 work

1. Add several speaker-disjoint clean mixtures, target-only, target-absent, and turn-taking conditions; do not infer rates from this one example. Resolve checkpoint training provenance where possible.
2. Compare a candidate designed for selected sets or a causal single-target model with measured multi-output gain control. Keep the current checkpoint as an offline diagnostic reference. The 4077 + 5142 pair is a useful regression example because the unwanted 5683 voice is clearly audible to the listener.
3. Test noise only after selecting an appropriately licensed noise source and specifying a small acquisition plan. Measure physical route and latency after choosing the first phone platform/device.
