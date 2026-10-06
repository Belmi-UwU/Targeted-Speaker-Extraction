# Causal SepFormer inference experiment

Date: 6 October 2026. This is a diagnostic conversion of the existing
SpeechBrain Libri3Mix SepFormer checkpoint, not a trained streaming model.
The original checkpoint files were not changed.

## Reproduce

From this directory:

```bash
../speakerbeam_ss_verification/.venv/bin/python experiment_causal_sepformer.py
```

The script uses checkpoint revision
`1f2d824e2333aec9e60d0030fb144885c1afc426` and the existing 3.77 s
clean Libri3Mix mixture with speakers 4077, 5142, and 5683. It writes
`data/causal_sepformer_experiment/results.json` and nine short WAV previews.

## Variants

1. **Original:** unmodified checkpoint inference.
2. **Masked only:** add an upper-triangular future attention mask to all four
   intra/inter `SBTransformerBlock` instances.
3. **Masked + local norm:** also replace five `GroupNorm(1, channels)` layers
   in the mask network with per-time-step `CumulativeLayerNorm`, copying the
   learned affine weights and biases. This changes normalization behavior.

No variant was retrained or fine-tuned. The mask does not add parameters, but
the modified computation differs from what the pretrained weights saw in
training. SpeechBrain's `TransformerEncoder(causal=True)` alone does not
construct a future mask for this checkpoint's `regularMHA`; its attention
receives `src_mask`, which the stock `SBTransformerBlock` does not supply.

## Results

Two inputs were used for the prefix check: the first 1.5 s of the mixture and
the first 2.5 s. We compared their outputs over the common 1.5 s. The table's
prefix error excludes the final eight samples, which are reported separately.
SI-SDR uses the full 3.77 s mixture and an *oracle* best assignment of the
three output streams to their clean sources; it does not depend on ECAPA.
Times are medians of three warmed CPU runs with four PyTorch threads and omit
audio I/O, matching, and playback.

| Variant | Relative RMS prefix error before final 8 samples | Mean SI-SDR across 3 sources | 250 ms input | 500 ms input | 1 s input |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original | 2.86% | 15.44 dB | 324 ms | 384 ms | 511 ms |
| Masked only | 4.79% | 0.03 dB | 329 ms | 389 ms | 509 ms |
| Masked + local norm | 0.000264% | −4.28 dB | 343 ms | 417 ms | 531 ms |

The masked + local norm variant changed by more than `1e-4` in precisely the
last eight samples of this prefix comparison. Before those samples, maximum
absolute difference was `2.41e-5`, consistent with floating-point noise.
The final-sample difference reached `0.136`. The convolutional encoder has a
16-sample kernel and eight-sample stride, so this observed boundary effect is
about 1 ms at 8 kHz. This **does not pass strict sample-level prefix
invariance**. Other inputs and cuts have not been checked for a global bound.

Per-speaker SI-SDR for original versus masked + local norm:

| Speaker | Original | Modified |
| --- | ---: | ---: |
| 4077 | 13.81 dB | −4.99 dB |
| 5142 | 12.36 dB | −10.58 dB |
| 5683 | 20.15 dB | 2.75 dB |

The attention mask alone fails the future-dependence test because global
normalization still reads the entire segment. After replacing normalization,
future dependence is confined to the observed boundary, but separation
quality collapses on this example. The modified model also lacks a persistent
streaming cache and has not been exercised through a live audio route.
The 250 ms segment takes longer than 250 ms to compute on this machine, so
independent 250 ms calls would accumulate a backlog even before I/O.

## Decision

Do not use this converted checkpoint as the app's real-time model. Preserve
the original for offline quality comparison. The next model experiment should
start from a **trained causal/streaming checkpoint**, then verify prefix
behavior, separation quality, sustained chunk timing, speaker selection, and
physical capture-to-playback delay on the chosen device.
