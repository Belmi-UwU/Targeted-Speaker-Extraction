# WeSep vs SepFormer as the live extractor

**Date:** 7 October 2026  
**Summary:** The REAL-TSE causal WeSep checkpoint (`tfmap_context_causal_100`) does what SepFormer cannot: it runs live with **about 0.5–0.9 s total delay**, comfortably faster than real time, and uses no future audio. It is much weaker in our 4–6 speaker conversations, though: about **+3 dB** median SI-SDRi vs about **+12 dB** for the SepFormer selector, with hardly any suppression of other voices while the chosen person is silent. On the real Nicker/Mond/Sam recording it extracted Sam and Mond but never Nicker. It was trained on clean **two-speaker** Libri2Mix, and the numbers match that: it is reasonable only when the target overlaps with one other person.

Setup, sources, and licence notes are in [wesep_eval/README.md](wesep_eval/README.md). The checkpoints have no stated weights licence and are for local evaluation only. All earlier SepFormer files and results are unchanged.

## Synthetic conversations (test set, 15 speakers, 4–6 per conversation)

Same live sections, discovered tracks (4-round buffer), clean enrollments and scoring as the [discovery report](DISCOVERY_REPORT.md):
- **SI-SDRi** is over the whole 32 s live section.
- **+1/+2/+3** score only spans where the target talks with that many others.
- **Absent suppression** is how much more the output drops than the input while the target is silent.

The WeSep rows come from `data/discovery/results/wesep_comparison.json`.

| Live model | Uses future audio? | Delay (live) | Enrollment | Median SI-SDRi | +1 | +2 | +3 | Absent suppression | Compute ÷ audio |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| SepFormer competitive, 4 s windows | yes | – (offline) | clean | 12.1 | 16.0 | 10.3 | 6.9 | 14.4 dB | – |
| | | | discovered | 11.9 | 15.8 | 10.3 | 5.6 | 14.1 dB | – |
| SepFormer competitive, block 1 s / context 2 s + 20 ms crossfade | no | ≈2.2 s | clean | 12.3 | | | | | 0.85 |
| | | | discovered | 11.6 | | | | 34 dB | 0.85 |
| **WeSep tfmap causal**, whole clip | no (≤0.3% leak) | – | clean | 3.9 | 8.8 | 4.7 | 3.3 | 2.0 dB | 0.33 |
| | | | discovered | 2.8 | 5.5 | 2.9 | 2.2 | 2.2 dB | 0.33 |
| **WeSep tfmap causal**, block 0.5 s / context 1 s | no | **≈0.9 s** | discovered | 3.0 | | | | 2.2 dB | 0.53 |
| **WeSep tfmap causal**, block 0.25 s / context 0.5 s | no | **≈0.5 s** | discovered | 2.8 | | | | 2.1 dB | 0.59 |
| WeSep tfmap non-causal, whole clip | yes | – | clean | 3.9 | 14.0 | 5.2 | 4.0 | 1.3 dB | 0.34 |
| | | | discovered | 4.0 | 11.6 | 5.1 | 3.5 | 1.2 dB | 0.34 |
| WeSep spk_emb causal, whole clip | no (≈13% leak) | – | clean | 3.0 | | | | 1.6 dB | 0.32 |
| | | | discovered | 1.8 | | | | 1.1 dB | 0.32 |

Empty cells were not computed for that row.

- **Where WeSep holds up.** With one other talker it gives +8.8 dB (causal) and +14 dB (non-causal) using a clean enrollment. That matches its two-speaker training. With two or three others it falls to about +3–5 dB. It also mostly passes the mixture through when the chosen person is silent: absent suppression is about 2 dB, against 14 dB or more for SepFormer.
- **Target preservation is good.** While the target talks alone, causal WeSep keeps them at about 37 dB SI-SDR with a clean enrollment. The problem is removing others, not damaging the target.
- **Discovered enrollment costs about 1 dB** of median for causal WeSep (2.8 vs 3.9 dB). That is near the PRD's ≤1 dB gate, which these medians cannot settle.
- **Speed and delay.** Each re-run on block + context costs about 0.27–0.31 s for 1 s of audio, 0.58 s for 2 s, and 1.1 s for 4 s on this CPU. So 0.5 s blocks keep up with 1 s of context, and 0.25 s blocks with 0.5 s. Shorter context barely changed the scores here: 2.8–3.0 dB across all four block settings.
- **Causality.** Replacing future audio changed earlier `tfmap_context_causal_100` output by at most 0.3% of peak, and `spk_emb_causal_100` output by up to about 13%. Block processing never gives the model future audio, so the live system is causal either way.

## Real recording (Nicker, Mond, Sam)

[run_wesep_user.py](wesep_eval/run_wesep_user.py) extracts each discovered speaker from the whole 23 s `Mix.wav`. Each output gets an ECAPA voice match against that person's own recording: ✅ means the output sounds most like the chosen person, ❌ means it sounds most like someone else.

| Live model | Nicker | Sam | Mond |
|---|---|---|---|
| SepFormer competitive (offline, earlier run) | ✅ 0.46 vs 0.24 | ✅ 0.54 vs 0.36 | ✅ 0.47 vs 0.20 |
| WeSep causal, whole clip | ❌ 0.40 vs 0.46 | ✅ 0.56 vs 0.40 | ✅ 0.43 vs 0.21 |
| WeSep causal, block 1 s / context 2 s | ❌ 0.40 vs 0.47 | ✅ 0.54 vs 0.37 | ✅ 0.42 vs 0.24 |
| WeSep causal, block 0.5 s / context 1 s | ❌ 0.40 vs 0.51 | ✅ 0.57 vs 0.39 | ✅ 0.40 vs 0.18 |
| WeSep non-causal, whole clip | ✅ 0.47 vs 0.24 | ❌ 0.24 vs 0.66 | ✅ 0.42 vs 0.29 |

All rows use discovered enrollment. With Nicker's own recording as enrollment, causal WeSep still failed on Nicker (0.24 vs 0.55). The local page `data/user_recordings/discovery/wesep_compare.html` has every output for listening.

**Enrollment level matters for WeSep.** Its enrollment features depend on loudness. The raw discovered track for Mond from the real recording was quiet, and with it WeSep failed on Mond (0.15). The same track normalized in level worked (0.40). `enrollment()` therefore normalizes enrollments to −25 dBFS. On the synthetic set this changed nothing (median change 0.01 dB), because those tracks were already at about that level.

## Live demo with WeSep

[wesep_eval/live_demo_wesep.py](wesep_eval/live_demo_wesep.py) is the same flow as [live_demo.py](live_demo.py) (discover → choose → live), with WeSep as the extractor:
- defaults are 0.5 s blocks, 1 s context, and a 0.4 s margin, for **about 0.9 s total delay**;
- `--block 0.25 --window 0.5` gives about 0.5 s;
- a simulated real-time run produced the same output as the offline path, delayed by exactly 0.900 s, with no underruns while audio was flowing.

Compare it directly with the SepFormer demo, which has about 2.2 s of delay but better separation.

## What this means

Neither model is ready for the product:
- **SepFormer** separates well but needs about 2 s of delay.
- **This WeSep checkpoint** has the delay we want, but separates poorly with three or more people and leaks other voices when the target is silent.

The shortfall comes from WeSep's training data (clean two-speaker mixtures), not from causality: the non-causal checkpoint is just as weak with more speakers.

The next step from [CAUSAL_CANDIDATES.md](CAUSAL_CANDIDATES.md) still stands: **train or fine-tune the causal WeSep recipe**, whose code is Apache-2.0. Train it on mixtures that look like our use:
- 3–6 speakers;
- target-absent stretches;
- discovered-track style (separated, 8 kHz band-limited) enrollments;
- level augmentation;
- licence-clean data.

A stateful streaming implementation, which carries LSTM states instead of recomputing context, would cut delay towards its 32 ms algorithmic latency.
