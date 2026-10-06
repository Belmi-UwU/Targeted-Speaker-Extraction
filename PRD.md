# Product requirements document: Live target-speaker listening app

**Status:** Draft for project alignment  
**Version:** 0.3 — 6 October 2026  
**Owner:** Project team  
**Current phase:** M1 complete; M2 offline model evaluation before app implementation

## 1. Product in one paragraph

Build a phone app that lets a listener hear **one or more selected people** when several people are speaking nearby. The listener supplies a short, separate recording of each desired speaker (an **enrollment** recording). The app captures live sound from the phone microphone or, where the audio route allows it, a true wireless earbud microphone. A speaker-selective system keeps the chosen voice or voices and plays them to headphones while reducing unselected voices. The first release uses a **selected-speaker list**: only speakers on that list should be passed through.

The immediate deliverable is an evidence-based offline evaluation, then a causal streaming prototype, then a usable phone app. **The model is not chosen for release yet.** The verified SpeakerBeam-SS variant handles one target at a time; support for multiple selected speakers is an explicit research gate.

**Example:** Three people A, B, and C are talking. If the listener selects A and C, both should remain audible, even when they overlap, while B is reduced. If neither A nor C is speaking, the processed signal should contain little or no speech. The app changes what the listener hears; it cannot prevent anyone from speaking.

## 2. Why this exists

In a conversation, café, classroom, or meeting, a listener may want to follow particular people despite simultaneous speech. Ordinary noise suppression can reduce background sounds but does not reliably identify which voices to retain. TSE uses enrollment recordings to condition separation on speaker identities.

### Intended users and jobs

| User | Job | Success looks like |
| --- | --- | --- |
| Listener in a two-person overlap | Follow one selected voice | Target words remain understandable while the competing voice is reduced. |
| Listener in a group conversation | Follow one or several selected participants | Selected speakers remain audible when they take turns or overlap; unselected voices are reduced, including when no selected speaker is active. |
| Researcher/developer | Compare models and audio routes | Reproducible scores, listening samples, latency traces, and a clear release decision. |

The first app is a listening aid and research prototype. Clinical hearing-assistance claims, safety-critical listening, and transcription are outside the first release scope.

## 3. Current evidence and constraints

We verified one public independent [OpenSpeakerBeam-SS implementation](https://github.com/helloooideeeeea/OpenSpeakerBeam-SS) with its [published checkpoint](https://huggingface.co/datasets/helloidea/OpenSpeakerBeam-SS-dataset). It loads strictly and runs offline inference on the repository's 10-second, 16 kHz mono demo. The model's output scale is abnormally high; the repository's bundled example is consistent with **clipping** that output, which loses a large fraction of its waveform detail. A future-audio perturbation changed earlier output by about 20% relative L2, so that implementation is **not causal in its tested form**. Its separate checkpoint license and complete training provenance remain unclear. See the [verification report](speakerbeam_ss_verification/VERIFICATION_REPORT.md) and [measured results](speakerbeam_ss_verification/outputs/verification_results.json).

The [SpeakerBeam-SS paper](https://www.isca-archive.org/interspeech_2024/sato24_interspeech.pdf) describes a causal real-time design. The verified public implementation differs from it, including separate Resemblyzer enrollment embeddings, global normalization, and no carried streaming state in the tested inference path. Paper latency and quality results must not be attributed to this checkpoint.

**Working product assumption:** one mono input is sufficient for the TSE research path. A microphone array and beamforming are optional later work. The first platform is **open**. Input/output routing must be checked on the platform selected for the prototype. For example, Apple documents that selecting a Bluetooth HFP microphone can also switch output to the corresponding HFP route, so earbud microphone support needs a device-level route and quality test before it is promised. [Apple audio-session guidance](https://developer.apple.com/documentation/avfaudio/avaudiosession/categoryoptions-swift.struct/allowbluetoothhfp)

## 4. Product goals and boundaries

### Goals

1. Let a user enroll multiple speakers from separate, reasonably clean voice samples and select one or more of them for listening.
2. Produce a listenable signal that retains selected speakers and suppresses unselected speakers.
3. Work with an actual live phone audio route at a delay suitable for conversation.
4. Explain input/output route, model status, and interruptions in the interface.
5. Keep evaluation reproducible and separate offline quality from device latency.

### First release boundaries

- One or more selected speakers at a time, from profiles created for each speaker; one mono mixture channel supplied to the processing system.
- Live monitoring through headphones, with an obvious Start/Stop control and bypass comparison.
- A short enrollment capture for each speaker and stored speaker profiles on device, subject to user delete actions.
- Phone microphone as the primary capture route for the first hardware feasibility test. True wireless earbud microphone is a required investigated route, with support decided by measured audio routing and quality.
- No server processing in the intended app. Research notebooks may run on a computer.

### Later candidates

- Remove a specified unwanted speaker while preserving all others; would require a model designed or validated for that task.
- Microphone-array beamforming, background recording, transcription, cloud sync, and platform expansion.

## 5. Core user flow

1. **Choose or create profiles.** User records approximately 10–20 seconds of each desired speaker talking alone, or imports permitted recordings. The app checks speech presence and usable level, then creates embeddings/profiles. Exact minimum duration will be set by testing.
2. **Select input and output.** App displays the actual active microphone and headphone route. The initial flow favors phone microphone plus headphones. Route changes must be visible.
3. **Select who to hear and start listening.** User checks one or more speaker profiles, then presses Start. The app captures audio, processes it, and plays the selected voices. A Bypass control lets the user compare it with the unprocessed input at matched safe loudness.
4. **Adjust and stop.** User controls output level, sees processing/route status, and stops immediately with one tap. No automatic storage of live conversation audio.
5. **Recover from interruption.** If microphone permission, route, or processing fails, the app stops processed playback and clearly indicates the problem. It does not silently claim filtering is active.

### Next phase: discover speakers from the live mix

The next phase replaces the separate enrollment recording with **discovery enrollment**: the app learns voices from the conversation itself, and the user picks who to hear by listening.

1. **Capture and separate.** The app buffers a short stretch of the live mixture, which may contain **any number of speakers, including more than three**. A blind separator splits it into one track per speaker, with no enrollment or target embedding.
2. **Choose.** The user plays the discovered tracks and picks one speaker (several speakers is later work).
3. **Embed.** A **separate speaker-encoder model**, such as the pinned ECAPA-TDNN, turns the chosen track into the target embedding. The extraction model does not create this embedding itself.
4. **Listen live.** A causal TSE model conditioned on that embedding processes the live input and outputs only the chosen speaker.

Only step 4 must run in real time. Steps 1–3 run once on a buffered clip, so the offline SepFormer + mixture-fit + ECAPA pipeline from M2 is the starting discovery stage. Because the Libri3Mix SepFormer always returns exactly three streams, discovery with more than three speakers needs short overlapping windows, separation per window, removal of near-silent streams, and clustering of stream embeddings into speakers. This handles any total number of speakers as long as at most three talk at once in a window.

Requirements this adds:

- The TSE model must be trained on, or adapted to, the **same embedding space** as the chosen encoder. An encoder cannot be swapped without retraining or a validated adapter.
- A discovered track is a degraded enrollment: it can carry separation artifacts, leakage from other voices, and reduced bandwidth (8 kHz for the current SepFormer). Its effect must be measured with the separated-track enrollment test in Section 8 before this flow is adopted.
- F2 applies to discovered tracks. If a speaker has too little clean solo speech in the buffer, the app keeps listening and says so instead of offering a weak profile.
- A discovered embedding is a voiceprint of someone who did not enroll themselves. Keep it on the device, for the current session only by default; saving it requires an explicit user action.

## 6. Functional requirements

| ID | Requirement | Acceptance evidence |
| --- | --- | --- |
| F1 | Capture and manage speaker profiles; allow one or more active selections. | Profiles can be created, selected together, renamed, and deleted; enrollment audio and derived embedding lifecycle are documented. |
| F2 | Reject obviously invalid enrollment (silence, clipping, too little speech) and explain what to rerecord. | Automated checks and user-visible feedback on representative clips. |
| F3 | Capture and display the **actual** active input/output routes. | Physical device test for phone mic, wired route, and selected Bluetooth devices; route changes update the UI. |
| F4 | Pass selected speakers and reduce unselected speakers while monitoring. | Test one selected speaker, two selected speakers, overlap between selected speakers, and periods with no selected speaker. |
| F5 | Provide Start, Stop, and level-matched Bypass. | Controls respond consistently; Bypass has no unexpected gain jump. |
| F6 | Prevent digital clipping and handle silence, invalid samples, and buffer overruns. | Logged signal-level checks and long-run playback test. |
| F7 | Handle calls, app backgrounding, unplugging, and Bluetooth handoff. | Route/interruption test matrix; user can recover without reinstalling. |
| F8 | Process voice on-device for the app and save no live audio by default. | Data-flow inspection and a user-facing delete action for profiles. |
| F9 | Keep each selected speaker audible when another selected speaker starts talking. | Three-speaker test with two selected voices and one unselected voice; listen and measure each selected source separately. |

## 7. Nonfunctional requirements and proposed gates

These are **initial decision thresholds**, not measured performance or claims about the current checkpoint. Revisit them after the first dataset baseline and after choosing a specific phone and earbud model.

| Area | Proposed gate | How to measure |
| --- | --- | --- |
| Identity and separation | Provisional single-target gate: median SI-SDR improvement over the unprocessed mixture ≥ 3 dB on the held-out two-speaker clean set; report the distribution and failure examples. | Same target and mixture alignment, unmodified model output, per-trial SI-SDRi. Test noisy data separately. |
| Multiple selected speakers | Both selected speakers must remain audible during their individual turns and overlap; set a numeric quality gate after the first three-speaker baseline. | Compare processed audio against the sum of selected clean references, and inspect each selected source's audibility. |
| Target preservation | Report target-only degradation and listening quality; do not accept a model that earns separation gains mainly by muting or distorting target speech. | Target-only controls, clean reference when available, listening panel. |
| Selected speakers absent | Output when none of the selected speakers is active should be strongly suppressed; set a numeric leakage threshold after measuring a reference baseline. | Energy/leakage and speaker-similarity analysis on explicit absence segments. |
| Conversation delay | Provisional p95 capture-to-playback delay ≤ 150 ms on the first wired route; establish a separate measured Bluetooth target. | Physical loopback timing on selected device; include I/O buffers and processing. |
| Sustained operation | No recurring buffer underruns or progressive latency growth during a 30-minute run. | Device logs and captured loopback trace. |
| Privacy | No cloud audio transfer; no live audio file retained by default. | Data-flow inspection and on-device storage audit. |

Report median, p10/p90, and relevant failure counts alongside any aggregate. A 10-second offline forward-pass time is not a live delay measurement. Output gain or normalization must be applied consistently for playback, and scored raw output must be retained for analysis.

## 8. Evaluation data plan

Do not download datasets until the evaluation workflow and storage budget are selected. Keep sources, mixtures, enrollment, and any clean target in a manifest with speaker IDs, recording IDs, sample rates, license, and split. Enrollment must be from a different utterance than the one in the mixture. No speaker or recording leakage between evaluation and any subsequent fine-tuning.

| Stage | Dataset | Purpose and scoring |
| --- | --- | --- |
| Smoke test | Existing repository demo, already in `speakerbeam_ss_verification/samples` | Verify load, shape, finite output, gain. It has no confirmed clean target and cannot establish separation quality. |
| First measured test | [Libri2Mix](https://github.com/JorisCos/LibriMix), 16 kHz, two-speaker `mix_clean` test subset | Mixture and isolated sources support SI-SDRi; choose a separate LibriSpeech utterance for target enrollment. Start with a small manifest before full test scale. |
| Multiple selected voices | Libri3Mix three-speaker test subset | Enroll each speaker separately; compare selecting one, two, or all three. Include intervals where a selected speaker is absent. |
| Noise robustness | Libri2Mix `mix_both` | Compare clean versus noise plus interfering speaker. The WHAM noise component brings its own noncommercial license; review before redistribution/product use. [WHAM terms](https://wham.whisper.ai/) |
| Speaker turns | [SparseLibriMix](https://github.com/popcornell/SparseLibriMix) | Variable overlap and target-silent spans; generated from source data using its published metadata. |
| Real conversation | [REAL-T](https://huggingface.co/datasets/REAL-TSE/REAL-T) | Supplied mixture–enrollment trials from natural conversations. No isolated clean source is provided, so use its transcript/speaker/timing/quality evaluation rather than SI-SDR. Respect DEV/EVAL use rules. |
| Product reality | Small, consented recordings using the exact phone and earbud routes under test | Check route artifacts, indoor/outdoor noise, distance, latency, and subjective usefulness. Ground truth for quantitative separation needs separately recorded or controlled-playback sources. |

[MiniLibriMix](https://zenodo.org/records/3871592) is convenient for pipeline smoke tests but is 8 kHz; resampling it to 16 kHz does not restore missing high-frequency content. [LibriCSS](https://github.com/chenzhuo1011/libri_css) is a later option for long, continuous speech and enrollment-based evaluation.

The current checkpoint's training split is unknown. Dataset overlap must be checked before any public or comparative score is presented as independent generalization.

### Separated-track enrollment test (next phase)

This test measures how much worse an embedding from a discovered track is than one from a clean enrollment recording. It reuses the existing Libri3Mix trials and their clean sources, and later needs mixtures with four to six speakers. For each speaker in each mixture, compute encoder embeddings from three inputs:

| Input | What it isolates |
| --- | --- |
| **Clean enrollment:** a separate LibriSpeech utterance by the same speaker (the current method) | Reference |
| **Clean in-mix source:** that speaker's clean source for the mixture utterance | Effect of using a different utterance, with no separation damage |
| **Discovered track:** the separator stream assigned to that speaker by the discovery pipeline | Effect of separation artifacts and leakage, plus the utterance change |

Report the following, with the distribution and failures, not only the median:

- **Embedding fidelity:** cosine similarity of the discovered-track embedding to the clean-enrollment embedding.
- **Identification margin:** cosine to the correct clean enrollment minus the highest cosine to any other speaker's clean enrollment. A margin ≤ 0 means the discovered voice would be confused with someone else.
- **Discovery correctness:** whether each discovered track contains exactly one speaker. Use the clean sources after inference only to audit this (oracle analysis). Count splits (one speaker over several tracks) and merges (several speakers in one track).
- **Downstream cost:** once a TSE candidate exists, its SI-SDRi with discovered-track enrollment versus clean enrollment on the same trials. Provisional gate: median drop ≤ 1 dB, with no trial falling below 0 dB SI-SDRi only because of the enrollment source.
- **Speech needed:** repeat the measurements using 2, 5, and 10 seconds of the speaker's solo speech in the discovery buffer, to set the minimum length used by F2.

Record encoder and separator revisions and sample rates for every run. The current separator's streams are 8 kHz upsampled to 16 kHz before ECAPA, which is a confound that must be reported.

### Metrics and review artifacts

- **Synthetic mixtures with clean references:** SI-SDRi relative to the mixture for one selected target; combined selected-source reference and per-speaker audibility for multiple targets; unselected-source leakage, target-only degradation, output clipping rate; optional intelligibility and perceptual quality measures when methods are validated.
- **Real conversations without clean references:** target-speaker similarity, target activity precision/recall, intelligibility or transcript error where references exist, and blinded listening ratings.
- **Every condition:** matched input/output audio examples, model/version hashes, audio route, sample rate, duration, runtime, and failure notes.

## 9. Model selection plan

SpeakerBeam-SS is the starting research family, not a locked deployment choice. The verified OpenSpeakerBeam-SS checkpoint is an offline **single-target** comparison candidate. A streaming candidate must satisfy all of the following: checkpoint/source compatibility; permitted use of code and weights; causal prefix-invariance test; explicit state or bounded-history chunk inference; acceptable quality on held-out mixtures and target-silent spans; mobile conversion path; and measured on-device timing. It must also support the selected-speaker list, directly or through a separately validated composition method.

For multiple selected speakers, two approaches need comparison: (a) a model trained to produce the selected set, or (b) running one-target extraction for each selected speaker and combining the outputs with gain control. Approach (b) can duplicate sound, distort overlap, leak unselected voices, and multiply compute cost. It cannot be accepted just because each single-target output sounds plausible. The evaluation must include three-speaker mixtures where **two speakers are selected and one is unselected**, plus selected-speaker overlap and silence.

Decision options after the offline benchmark:

1. Keep the independent checkpoint as an **offline single-target reference only** and select a causal approach capable of supporting multiple selected speakers for the app.
2. Reimplement the paper's causal design and train/fine-tune compatible weights, then re-evaluate.
3. Choose another checkpointed causal TSE family if it reaches the same quality, license, and mobile gates sooner.

Changing `gLN` to causal normalization in the existing model is an architecture change; old weights must not be assumed to preserve quality. Mixing audio in short chunks without correct state and context is also not proof of causality or consistent output.

## 10. Technical design at a glance

```text
Enrollment voices -> quality checks -> speaker encoder -> selected profile set

Mic route -> capture buffer -> mono/resample -> causal selection model -> gain/limiter -> output buffer -> headphones
                              |               |                 |
                         route monitor   timing logs      clipping/underrun logs
```

Next-phase discovery enrollment replaces the first line (see Section 5):

```text
Buffered mix -> windowed blind separation -> drop silent streams -> speaker encoder -> cluster per speaker
             -> user picks a discovered speaker -> its embedding becomes the selected profile
```

For the first offline evaluation, a Python runner/notebook reads a trial manifest, invokes a pinned model, saves raw output plus a separate normalized listening copy, computes metrics, and writes a CSV/JSON result with per-trial errors. The notebook should call reusable functions; dataset handling, model inference, and scoring should not exist only as notebook cells.

For the app, keep enrollment processing off the live audio callback. The streaming processor needs bounded memory, state reset on route change, deterministic frame accounting, and an explicit output gain/limiter. If multiple per-speaker streams are combined, the combiner needs its own clipping and leakage tests. Audio capture, resampling, model inference, and playback must each be measured separately. The phone microphone and Bluetooth microphone routes require separate validation; on Apple devices, selecting an HFP earbud mic can change the playback route too. [Apple documentation](https://developer.apple.com/documentation/avfaudio/avaudiosession/categoryoptions-swift.struct/allowbluetoothhfp)

## 11. Milestones and exit criteria

| Milestone | Deliverable | Exit criterion |
| --- | --- | --- |
| M0 — feasibility evidence | Current checkpoint report | Completed: strict load and offline inference. Noncausality, output scale, and license provenance remain blockers. |
| M1 — dataset and scoring contract | Small manifest, reproducible loader/metrics, listening examples | Completed on one Libri2Mix clean mixture with separate enrollments for both speakers; see [M1 report](evaluation/M1_REPORT.md). |
| M2 — offline model decision | Results on clean, noisy, target-only, target-absent, and multiple-selected-speaker sets | Single-target and two-selected-speaker failures understood; one or more candidates selected for causal work, or the project records a no-go decision. |
| M3 — audio-route proof | Minimal phone capture-to-playback app without TSE | Actual input/output routes, sample rates, and physical loopback delay documented on named devices. |
| M4 — causal streaming proof | Chunk/state inference integrated with audio path | Prefix-invariance check, sustained operation, and latency measured with model enabled. |
| M5 — user prototype | Profile flow and live listening controls | Users can enroll several people, select one or more to hear, listen, bypass, recover from route changes, and delete profiles. |
| M6 — pilot review | Structured listening study and issue log | Product gates reviewed with real device recordings; release scope and supported routes stated honestly. |
| M7 — discovery enrollment (next phase) | Offline discovery pipeline for any number of speakers; separated-track enrollment test report | Discovery finds the correct speakers on mixtures with four or more speakers. The separated-track enrollment gate in Section 8 passes with the chosen TSE candidate. The user flow lets a listener pick a discovered speaker and listen live. |

M1 and M3 can proceed in parallel after the basic contracts are fixed. M4 depends on selecting a causally valid model and validating the chosen audio route. The offline parts of M7 (discovery and the embedding-fidelity measurements) reuse the M2 pipeline and can start alongside M2–M4. Its downstream gate waits for a TSE candidate.

## 12. Risks and decisions to close

| Risk or decision | Current position | How to close it |
| --- | --- | --- |
| Multiple selected speakers | Required user outcome; current checkpoint handles one target at a time. | Compare a selected-set model against per-speaker extraction plus combination on three-speaker test mixtures. |
| First platform and route | Platform open; primary research route phone mic plus headphones. | Name target phone, OS, wired/Bluetooth output, and earbud models; run M3 route test. |
| Current checkpoint quality | Inference succeeds; quality unknown. | Run M1/M2 with clean target references and blinded listening. |
| Current checkpoint causality | Failed prefix-invariance check. | Choose or train a causal candidate; repeat test on its streaming API. |
| Output amplitude/playback | Peak about 29; supplied result matches the raw output after clipping. | Keep raw outputs for scoring; determine a safe, fixed gain/limiter strategy and measure its effect on listening quality. |
| Checkpoint license | Code MIT; weights have no verified separate grant. | Obtain authoritative weight terms/provenance or exclude from distributed app. |
| Dataset rights | Licenses differ across clean/noisy/real datasets. | Track rights per artifact and review before packaging or redistribution. |
| Earbud mic route and fidelity | Platform route behavior may force HFP input/output pairing. | Physical device measurements, listening tests, and route fallback rules. |
| User privacy | Live bystanders' voices are captured. Discovery enrollment would also create voiceprints of people who did not enroll. | On-device processing, no default recording, session-only discovered profiles unless the user saves them, explicit profile deletion, clear microphone indicator. |
| Enrollment from separated tracks (next phase) | First offline test on clean controlled conversations: embeddings are clearly degraded but still identify the speaker. Median live quality matched clean enrollment once thresholds were calibrated per enrollment type. About 5 s of solo speech per speaker is the provisional minimum. See the [discovery report](evaluation/DISCOVERY_REPORT.md). | Run the separated-track enrollment test in Section 8; set the minimum solo-speech duration from it. |
| More than three speakers in the input | Required; the current separator outputs exactly three streams, and the evaluation manifest allows at most three sources. | Extend the manifest and builders to four–six speaker mixtures; test windowed separation with clustering for discovery and TSE robustness to four or more interferers. |
| Monitoring comfort | Playback delay and speaker leakage can be distracting; open-air playback may cause feedback. | Test with headphones at safe levels and monitor for feedback, clipping, and delay discomfort. |

## 13. Immediate next work

1. Choose the first platform/route for the hardware prototype and document the selected-speaker list UX.
2. Investigate the current demo gain path; the bundled output mismatch was explained by clipping, and playback quality remains to be measured.
3. Build the smallest 16 kHz Libri2Mix evaluation manifest with a separate enrollment utterance and clean target, then add a three-speaker/two-selected manifest. Download only the selected test assets once the manifest and storage estimate are final.
4. Produce a per-trial score report and listening examples, test a multiple-selected-speaker strategy, then compare a causal candidate before starting live integration.
5. For the next phase, extend the manifest beyond three sources, build four–six speaker mixtures from the verified LibriSpeech test-clean audio, and run the embedding-fidelity part of the separated-track enrollment test with the existing SepFormer + ECAPA pipeline.

## References

- [Local SpeakerBeam-SS checkpoint verification](speakerbeam_ss_verification/VERIFICATION_REPORT.md)
- [SpeakerBeam-SS paper](https://www.isca-archive.org/interspeech_2024/sato24_interspeech.pdf)
- [LibriMix project](https://github.com/JorisCos/LibriMix) and [SpeakerBeam enrollment recipe](https://github.com/BUTSpeechFIT/speakerbeam/blob/main/README.md)
- [SparseLibriMix](https://github.com/popcornell/SparseLibriMix) and [REAL-T dataset card](https://huggingface.co/datasets/REAL-TSE/REAL-T)
- [Apple audio routing reference](https://developer.apple.com/documentation/avfaudio/avaudiosession/categoryoptions-swift.struct/allowbluetoothhfp)
