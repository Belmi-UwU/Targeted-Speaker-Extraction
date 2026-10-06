# Project progress — 6 October 2026

The [PRD](PRD.md) defines the goal: hear **one or more selected speakers** from a live mixture through headphones. The phone platform remains open. Work proceeds by milestone; a green check below means its stated exit criterion is met.

| Milestone | Status | Evidence / next gate |
| --- | --- | --- |
| M0 — model feasibility | **Complete for compatibility only** | Independent OpenSpeakerBeam-SS checkpoint loads strictly and produces offline output. Bundled result is consistent with clipping of raw output. Causality failed, quality is unmeasured, and weight terms are unresolved. See [verification](speakerbeam_ss_verification/VERIFICATION_REPORT.md). |
| M1 — dataset and scoring contract | **Complete** | [M1 report](evaluation/M1_REPORT.md): pinned clean Libri2Mix mixture, two separate enrollments, validated manifest, raw checkpoint outputs, SI-SDRi scores, and listening comparisons. |
| M2 — offline model decision | **In progress** | [Model comparison](evaluation/M2_MODEL_COMPARISON.md): SepFormer + ECAPA matched all selected pairs on four initial and four new holdout clean mixtures. Complete target absence, noise, real recordings, and causal operation remain open. No live app model selected. |
| M3 — phone audio route | Waiting for platform/device choice | Measure actual microphone/output route and physical delay. |
| M4–M6 — streaming and app pilot | Later | Require a causal licensed model and a viable device route. |

## Findings from this pass

- The upstream example does reproduce after clipping our raw inference result to [-1, 1]: correlation is above 0.99999999999, RMSE about 0.00000122. Roughly 38.32% of raw samples exceed that range. Clipping is a serious playback/quality risk, though the match resolves the previous checkpoint/demo discrepancy.
- The model is still noncausal and supports a single enrollment target per inference. Two selected speakers require a separate, measured system design.
- The scoring contract supports a list of selected speaker IDs and a combined clean reference; its synthetic check covers both two-speaker/single-selection and three-speaker/two-selection cases.
- One 16 kHz clean Libri2Mix test mixture was generated from pinned official metadata and verified LibriSpeech audio. Selecting speaker 4077 yielded **+10.57 dB SI-SDRi**; selecting speaker 5142 yielded **+1.14 dB**. Both raw outputs exceed playback range. These two trials share one mixture and do not establish a general performance score.
- The user heard the 4077 fixed-gain output as cleaner than the 5142 fixed-gain output. This is one listening impression, consistent with the measured difference on that mixture.
- A matching clean Libri3Mix trial added speaker 5683. The checkpoint requested for 5142 produced an output much closer to 5683 than 5142. Directly summing two extracted voices yielded SI-SDR improvements of **+0.46, -0.62, and +0.90 dB** across the three possible selected pairs. These are all selections of one mixture; see the [M2 interim report](evaluation/M2_INTERIM_REPORT.md).
- In the 4077+5142 pair output, the user heard unselected 5683 overpowering the chosen voices and found 5142 unclear. This pair fails the desired listening outcome on this example; changing the uniform preview gain cannot repair that speaker confusion.
- A pinned Apache-2.0 SpeechBrain Libri3Mix SepFormer checkpoint was added as a blind-separation-plus-enrollment-matching comparator. On the user's 4077+5142 clip, its mixture-fit combination reached **+19.68 dB SI-SDRi at a common 8 kHz score rate**, versus **+0.48 dB** for the SpeakerBeam-SS sum. It is offline, 8 kHz, fixed to three output streams, and noncausal.
- The user listened to the SepFormer 4077+5142 and 5142-only outputs and reported that SepFormer seems to split and select the voices well on this original three-voice example. The 5142-only extraction reached **+23.23 dB SI-SDRi** at the common 8 kHz score rate. See [listening files 04 and 09](evaluation/data/listen_next).
- On four initial three-speaker clips (12 selected pairs), SepFormer with the original Resemblyzer matcher had a **+12.94 dB median** but improved only **7/12** pairs; matching was correct for **6/12**. A pinned ECAPA matcher, using the same cached separated streams, matched **12/12** and improved **12/12** on that set.
- On four new speaker-disjoint holdout mixtures (12 new selected pairs), ECAPA matched **12/12** correct streams and all **12/12** pairs improved, with **+15.25 dB median** SI-SDRi at 8 kHz. The original matcher matched 7/12 on this holdout. This is still a small clean-speech sample.
- The user's real stereo 48 kHz recordings (`Mix.wav` plus one separate enrollment for Nicker, Mond, and Sam) were located and run through the offline SepFormer + ECAPA pipeline. Six 23.13-second single- and two-speaker listening outputs are in [local results](evaluation/data/user_recordings/results). The originals were not altered. No same-take clean references exist for objective SI-SDR scoring; user listening is the next quality check.
- A controlled turn/absence clip gave **-37.1 dB** output energy during the 5683-only interval relative to selected-only intervals for SepFormer mixture-fit, versus **+7.1 dB** for SpeakerBeam-SS sum. This is one synthetic speaking pattern, not real-room validation.
- A [causal SepFormer diagnostic](evaluation/CAUSAL_SEPFORMER_EXPERIMENT.md) tested attention masking alone and masking plus local normalization without retraining. Masking alone still depended on future audio; masking plus local normalization reduced prefix dependence to an observed 8-sample boundary effect, but mean oracle SI-SDR on one clean three-speaker mixture fell from **15.44 dB to −4.28 dB**. A 250 ms segment took **343 ms** to compute on this machine. This converted checkpoint is not suitable for the live app.
- Evaluation audio/data plus SepFormer and ECAPA inference weights use about **641 MB**, within the 3 GB storage cap. No full LibriMix corpus or WHAM noise was downloaded.

## Next action

Ask the user to listen to the Nicker, Mond, and Sam single-speaker outputs and at least one pair output from their real recording, noting speaker identity, clarity, and unwanted leakage. Test a full recording where the selected speaker never appears; the current ECAPA matcher still forces a stream assignment. Add a confidence/rejection rule. For real-time work, test a trained causal/streaming checkpoint rather than directly converting the offline SepFormer weights. A device-route proof is still required before app integration.
