# Product requirements document: Live target-speaker listening app

**Status:** Draft for project alignment  
**Version:** 0.4 — 9 October 2026<br>
**Owner:** Project team  
**Current phase:** Local browser prototype; mixed calibration + microphone capture + headphone playback

## Current implementation contract — local website

**This file is the single source of truth for the project.** The local website
requirements below define the current iteration. The phone release requirements
in sections 1–11 remain the longer-term product plan. Experiment reports are
evidence, and README files explain operation; they do not replace this contract.
Update this PRD when scope or architecture changes and update PROGRESS.md with
what passed, what remains unverified, and the next user action.

### A. Goal, evidence, and fixed decisions

Build a website served on **127.0.0.1**, with all processing on this Mac,
to try hearing one or several selected speakers through headphones. UI language
is Indonesian. Preferred first route: **Mac microphone → Mac M5 → AirPods Max 2
USB-C output**. Browser routing must be shown rather than assumed. Phone and
Bluetooth microphone performance are not inferred from Mac benchmarks.

Use the existing pinned **SepFormer Libri3Mix + ECAPA** models. Separation on
MPS, CPU ECAPA features and MPS embedding layers; **1,500 ms retained context
and 200 ms new audio**. This is bounded-window noncausal inference, not a
trained streaming model. Unsupported MPS ops must not silently fall back to CPU.
Explicit CPU mode is for debugging and must display its device accurately.

**Website measurement update:** the first browser run overloaded at hop 200 ms;
loopback-only checks measured approximately 232–240 ms on some full-context
blocks under the current host load. Keep 200 ms as the accepted-quality default,
and offer **300 ms as an explicit comparison option**, retaining 1,500 ms context.
Do not call 300 ms quality accepted or <500 ms until measured. Bypass uses 20 ms
blocks for an audio-route baseline. Report the actual hop in every session.

Evidence: the user accepted MPS 1,500/200 ms listening quality on Mix.wav.
The previous virtual p95 was 424 ms; automatic enrollment from the first 10 s
gave 403 ms on the disjoint 13.125 s suffix, with identity differences versus
separate-enrollment control. **These are not browser or physical-route latency.**
The separator always produces three candidate streams; arbitrary speaker count,
target absence, and unseen speakers remain unverified.

### B. User flow and required controls

1. **Open local site.** Show model loading/ready/error, actual compute device,
   workflow steps, and local-only processing. No microphone access on page load.
2. **Choose audio route.** Request mic access through a user action; list browser
   audio input devices, show the active track settings, and request echo
   cancellation/noise suppression/automatic gain control off. Show effective
   settings because browsers may ignore requests. Choose output where supported;
   otherwise instruct the user to select AirPods in macOS Sound settings.
3. **Calibrate.** Capture exactly 30 s of mono PCM. Countdown uses captured sample
   count. Stop capture at the boundary; show profile-processing progress. All
   participants must speak. Incoming audio while profiles are being made is
   intentionally not buffered into a delayed conversation. Calibration and
   warm-up are startup time, separate from live metrics.
4. **Review/select.** Play each anonymous profile preview; allow renaming, selecting
   one/multiple voices, reusing saved profiles, and deleting web-created profiles.
   Explain three candidates and possible mixed/duplicate voices. Do not identify
   people automatically or use future live audio to build initial profiles.
5. **Listen.** Start actual microphone capture and incremental playback before
   recording ends. Provide volume, Stop, and unfiltered Bypass. Selection/mode
   changes reset pending audio and context so prior outputs cannot leak into a
   new selection. Stop releases mic tracks and silences scheduled playback.
6. **Test without a room recording.** Include a clearly labelled Mix.wav demo and
   previously generated 10 s profiles when available. File audio passes through
   the same paced capture/transport/model/playback path as the mic, without being
   acoustically played into the mic. Never describe that demo as a 30 s calibration.

### C. Audio and timing architecture

Browser Web Audio + **AudioWorklet** captures mono PCM in 200/300 ms filtered blocks
(20 ms for Bypass; 200 ms during calibration) at the
actual AudioContext rate. A same-origin WebSocket sends sequence numbers,
capture timestamps and binary float32 audio to a Python **aiohttp** backend.
A single model executor owns GPU work; the event loop stays responsive to Stop.
Per-session rolling history is capped at 1.5 s. Only the latest chosen hop is emitted
at 16 kHz and scheduled through Web Audio into headphones. The browser handles
output-rate conversion. Use the existing checkpoint/embedding code, not a
replacement network, or pre-rendered WAVs masquerading as live inference.

Bound both input backlog and browser playback backlog. On overload, stop and
explain it instead of accumulating unlimited delay or silently dropping chunks.
No model work occurs inside the audio callback. Gain and limiting are applied
online, using only current/past audio; count limiting events. Bypass uses the
same output volume and separate timing, without claiming perfect loudness match.

Display model processing p50/p95, server pending blocks, browser queued audio,
playback gaps and an **estimated application delay** from browser capture to
scheduled playback. Do not subtract clocks across Python/browser. Export a
bounded JSON timing report with mode, selected IDs, sample rates, requested and
effective input settings, compute device, drops/overload/gaps, and configuration.
Browser timing excludes unknown acoustic/input/output hardware delay; a physical
loopback test is needed for microphone-to-ear measurements. Target <500 ms is
exploratory, not a guarantee or replacement for the eventual phone PRD gate.

### D. Local storage and errors

Use only loopback HTTP/WS, same-origin controls, local static assets, no CDN,
analytics, cloud inference, automatic downloads, or deployment. Keep audio and
profile artifacts below the existing Git-ignored user-recordings folder. Store
calibration and derived profiles from the explicit calibration action; **do not
store live conversation audio by default**. Reports contain metrics, not PCM.
Do not alter original Mix.wav, manual enrollments, or previous experiment outputs.
One active browser audio session; extra clients get a visible busy response.
Validate message sizes, rates, sequence continuity, finite PCM, profile ID paths
and embedding hashes. Reject empty selection, silent calibration, too-short
uploads and incompatible profiles. Permission denial, disconnect, route ending,
closed/background-suspended audio context, overload and model failure stop
monitoring visibly. Retry/recalibration must be possible without restarting the app.

### E. Deliverables and acceptance gates

Deliver `local_web/` server, UI, audio worklet, launch script, usage instructions,
and verification evidence. Canonical requirements remain in **this PRD**.

| Gate | Required evidence |
| --- | --- |
| Website and local model | Local URL responds, models load on displayed device, UI works at desktop/mobile widths; no placeholder inference. |
| Calibration | Real 30 s boundary by sample count; saved previews/embeddings reload; silent/short input rejected. Short existing demo is labelled correctly. |
| Incremental inference | Real checkpoint handles multiple successive packets and returns audio before EOF; bounded context and contiguous samples. |
| Selection | One/two profiles, reuse and rename work; empty selection rejected; switching/Stop prevent stale audio. |
| Timing and overload | Browser/server metrics named honestly; overload stops bounded queues; metrics downloadable. |
| Privacy/recovery | Loopback binding, origin/path checks, no live PCM retention, disconnect/Stop cleanup, repeat start works. |
| Hardware listening | **User test required** for real microphone permission, AirPods route, identity quality and physical delay; do not mark passed from file replay. |

Implementation sequence: PRD → local transport and audio route → reuse saved
profiles and live inference → 30 s calibration + speaker controls → integration
verification → user microphone/AirPods listening → sustained/physical latency tests.
Do not wait for optional design choices to complete the agreed local prototype.

## 1. Product in one paragraph

Build a phone app that lets a listener hear **one or more selected people** when several people are speaking nearby. The listener creates speaker profiles either from separate enrollment recordings or, in the new prototype flow, from an initial **30-second mixed calibration recording**. The app captures live sound from the phone microphone or, where the audio route allows it, a true wireless earbud microphone. A speaker-selective system keeps the chosen voice or voices and plays them to headphones while reducing unselected voices. The first release uses a **selected-speaker list**: only speakers on that list should be passed through.

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

1. Let a user create profiles from separate voice samples or from mixed-audio calibration, then select one or more speakers for listening. Automatic profiles require a listening review and a separate quality gate.
2. Produce a listenable signal that retains selected speakers and suppresses unselected speakers.
3. Work with an actual live phone audio route at a delay suitable for conversation.
4. Explain input/output route, model status, and interruptions in the interface.
5. Keep evaluation reproducible and separate offline quality from device latency.

### First release boundaries

- One or more selected speakers at a time, from profiles created for each speaker; one mono mixture channel supplied to the processing system.
- Live monitoring through headphones, with an obvious Start/Stop control and bypass comparison.
- Separate enrollment capture or an initial mixed-audio calibration, with stored speaker profiles on device, subject to user delete actions.
- Phone microphone as the primary capture route for the first hardware feasibility test. True wireless earbud microphone is a required investigated route, with support decided by measured audio routing and quality.
- No server processing in the intended app. Research notebooks may run on a computer.

### Later candidates

- Remove a specified unwanted speaker while preserving all others; would require a model designed or validated for that task.
- Microphone-array beamforming, background recording, transcription, cloud sync, and platform expansion.

## 5. Core user flow

1. **Choose or create profiles.** User can record approximately 10–20 seconds of each desired speaker talking alone, import recordings, or record a 30-second mixed calibration. In the calibration flow, everyone to be selected must speak during that interval. The system separates the calibration recording, creates anonymous candidate profiles, and presents voice previews to label and select. Processing after capture takes additional time; show progress before announcing profiles ready. Exact minimum speech duration and quality thresholds will be set by testing.
2. **Select input and output.** App displays the actual active microphone and headphone route. The initial flow favors phone microphone plus headphones. Route changes must be visible.
3. **Select who to hear and start listening.** User checks one or more speaker profiles, then presses Start. The app captures audio, processes it, and plays the selected voices. A Bypass control lets the user compare it with the unprocessed input at matched safe loudness.
4. **Adjust and stop.** User controls output level, sees processing/route status, and stops immediately with one tap. No automatic storage of live conversation audio.
5. **Recover from interruption.** If microphone permission, route, or processing fails, the app stops processed playback and clearly indicates the problem. It does not silently claim filtering is active.

### Mixed-calibration prototype gate

The current SepFormer Libri3Mix checkpoint supplies **three output streams**, not an estimate of the number of people. The first automatic-enrollment experiment is scoped to the existing three-person recording. Fewer speakers, more than three speakers, duplicate profiles, identities changing within a separated stream, noise, and a person who did not speak during calibration remain explicit research cases. Generated enrollment is an estimate, not a clean reference.

Calibration must use only the initial interval. Evaluate subsequent filtering on a disjoint later interval or a new recording, and compare it with the same pipeline using separate enrollments. Check each selected voice, pairs, identity swaps, and target absence. Keep calibration processing, warm-up, per-block latency, and physical route latency separate. Initial file replay pauses input while profiles and warm-up are prepared; continuous microphone capture/buffering during this transition is not implemented yet. A new person requires recalibration in this prototype.

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
| F10 | Generate selectable anonymous profiles from an initial mixed calibration without requiring separate enrollment files. | Profile previews, reusable embeddings, input boundaries and processing time recorded; disjoint subsequent filtering compared with separate-enrollment control. Unknown speaker count and profile purity must be validated before general release. |

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

M1 and M3 can proceed in parallel after the basic contracts are fixed. M4 depends on selecting a causally valid model and validating the chosen audio route.

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
| User privacy | Live bystanders' voices are captured. | On-device processing, no default recording, explicit profile deletion, clear microphone indicator. |
| Monitoring comfort | Playback delay and speaker leakage can be distracting; open-air playback may cause feedback. | Test with headphones at safe levels and monitor for feedback, clipping, and delay discomfort. |

## 13. Immediate next work

1. Choose the first platform/route for the hardware prototype and document the selected-speaker list UX.
2. Investigate the current demo gain path; the bundled output mismatch was explained by clipping, and playback quality remains to be measured.
3. Build the smallest 16 kHz Libri2Mix evaluation manifest with a separate enrollment utterance and clean target, then add a three-speaker/two-selected manifest. Download only the selected test assets once the manifest and storage estimate are final.
4. Produce a per-trial score report and listening examples, test a multiple-selected-speaker strategy, then compare a causal candidate before starting live integration.

## References

- [Local SpeakerBeam-SS checkpoint verification](speakerbeam_ss_verification/VERIFICATION_REPORT.md)
- [SpeakerBeam-SS paper](https://www.isca-archive.org/interspeech_2024/sato24_interspeech.pdf)
- [LibriMix project](https://github.com/JorisCos/LibriMix) and [SpeakerBeam enrollment recipe](https://github.com/BUTSpeechFIT/speakerbeam/blob/main/README.md)
- [SparseLibriMix](https://github.com/popcornell/SparseLibriMix) and [REAL-T dataset card](https://huggingface.co/datasets/REAL-TSE/REAL-T)
- [Apple audio routing reference](https://developer.apple.com/documentation/avfaudio/avaudiosession/categoryoptions-swift.struct/allowbluetoothhfp)
