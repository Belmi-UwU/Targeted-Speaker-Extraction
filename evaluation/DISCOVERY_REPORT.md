# Discovery enrollment: first offline results

**Date:** 6 October 2026  
**Summary:** On six controlled clean conversations with 4–6 speakers, discovery from the live mix found every speaker once each speaker had about 5 seconds of solo speech. Embeddings from the discovered tracks are clearly worse than clean enrollments, but they still identified the right person in every held-out case at that length. Used as the live target, a discovered track cost **no median quality** relative to a clean enrollment recording, **provided each selector's threshold is calibrated for discovered embeddings**. The best live model, a competitive SepFormer selector, reached about **+12 dB median SI-SDRi** with either enrollment. None of these models is causal; this answers the PRD's separated-track enrollment question offline only.

## What was tested

The flow follows PRD §5 "Next phase". A blind separator splits a buffered stretch of conversation into one track per speaker. A listener picks one, a speaker encoder turns that track into an embedding, and a live model extracts that speaker from a later stretch of the same conversation.

**Data.** [build_discovery_conversations.py](build_discovery_conversations.py) builds six conversations from LibriSpeech `test-clean` (CC BY 4.0), with 4, 5, and 6 speakers in each of two speaker-disjoint sets:

- **dev** (15 speakers): used to tune every threshold;
- **test** (15 speakers): used only for reporting.

Each conversation has two sections built from **different utterances**:

- **Discovery section (60–80 s).** Four rounds of 2.5 s solo turns, plus one 2-speaker and one 3-speaker overlap per round. Discovery was run on the first 1, 2, or 4 rounds, about 2.5, 5, or 10 s of solo speech per speaker.
- **Live section (32 s).** Sixteen 2 s segments with 1, 2, 3, or 4 simultaneous speakers. Every speaker has both target-active and target-absent time. The live sections are manifest rows of the new `ControlledLibriNMix` / `conversation_clean` type and pass `validate_manifest.py --check-mix-sum`.

A third, separate utterance per speaker is the **clean enrollment** used for comparison. Speaker levels vary by ±3 dB. There is no noise and no room acoustics.

**Discovery models** ([discovery_pipeline.py](discovery_pipeline.py)):

1. Pinned Libri3Mix SepFormer applied in 4 s windows with a 2 s hop.
2. Mixture-fit gains per window.
3. Streams more than 15 dB below the window mixture are dropped.
4. Each remaining stream is embedded and clustered by average-linkage agglomerative clustering on cosine distance. Clusters seen in fewer than 3 windows are dropped.
5. Each cluster's streams are overlap-added into one track.

Two encoders were compared for clustering: **ECAPA-TDNN** and **Resemblyzer**. The ECAPA tracks feed the live stage. This handles any total number of speakers as long as at most three talk inside one window.

**Live models.** None is causal; all are offline references for the live stage.

| Live model | How it uses the chosen speaker |
| --- | --- |
| **SpeakerBeam-SS** (pinned independent checkpoint) | Resemblyzer embedding of the chosen track conditions a single forward pass. |
| **SepFormer selector** | Windowed SepFormer on the live input. In each window, keep the streams whose ECAPA cosine to the chosen speaker exceeds a threshold. |
| **SepFormer competitive** | Same windows. Keep a stream only if the chosen speaker is its **closest** known speaker: all discovered speakers in the discovery case, all enrolled speakers in the clean case. |

Clean sources are used only after inference: to audit which speaker each discovered track holds (oracle analysis) and to score the outputs. Full results are in [data/discovery/results](data/discovery/results).

## Discovery: are the right speakers found?

Exact = every speaker found once, no extra or mixed tracks. Merged = a track with less than 80% of its energy from one speaker. Split = a speaker who dominates more than one track.

| Split | Encoder | Buffer rounds (solo s/speaker) | Exact mixtures | Speakers found | Missed | Split | Merged | Extra tracks |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| test | ECAPA | 1 (2.5) | 2/3 | 14/15 | 1 | 0 | 1 | 0 |
| test | ECAPA | 2 (5) | 1/3 | 15/15 | 0 | 2 | 0 | 2 |
| test | ECAPA | 4 (10) | 2/3 | 15/15 | 0 | 1 | 0 | 1 |
| test | Resemblyzer | 1 (2.5) | 2/3 | 13/15 | 2 | 0 | 0 | 0 |
| test | Resemblyzer | 2 (5) | 3/3 | 15/15 | 0 | 0 | 0 | 0 |
| test | Resemblyzer | 4 (10) | 2/3 | 15/15 | 0 | 1 | 0 | 1 |
| dev | ECAPA | 1 (2.5) | 1/3 | 13/15 | 2 | 0 | 1 | 0 |
| dev | ECAPA | 2 (5) | 2/3 | 15/15 | 0 | 0 | 1 | 0 |
| dev | ECAPA | 4 (10) | 2/3 | 15/15 | 0 | 1 | 0 | 1 |
| dev | Resemblyzer | 4 (10) | 0/3 | 15/15 | 0 | 3 | 1 | 3 |

**About 2.5 s of solo speech per person is not enough.** In the six-speaker conversations it missed one speaker and produced a mixed track. From 5 s per speaker upward, ECAPA discovery found every speaker in both sets.

The remaining errors are mostly **splits**: one person appears as two tracks, both pure. The extra track is usable. In the test five-speaker conversation, picking speaker 4970's extra track still gave +14.4 dB with both SepFormer selectors and +10.8 dB with SpeakerBeam-SS. It is a user-interface problem, not a wrong-person problem.

The two encoders are close on test. ECAPA was clearly better on dev, where Resemblyzer split three speakers at the longest buffer. ECAPA stays the discovery encoder for now.

Discovery runs offline. Separating and embedding the 80 s six-speaker buffer took about 65 s of desktop CPU time, including both encoders.

## Separated-track enrollment test (PRD §8)

Each column compares embeddings from two sources:

- **discovered**: the ECAPA-discovered track assigned to that speaker by the audit;
- **clean in-mix**: the speaker's clean source over the same buffer, with no separation.

Both are compared with the clean enrollment. Margin = similarity to the right speaker's enrollment minus the highest similarity to any of the other 14 speakers in the same split. A margin of 0 or less is a confusion.

| Split | Buffer | Discovered | ECAPA cos: discovered / clean in-mix | ECAPA median margin | ECAPA confusions | Resemblyzer cos: discovered / clean in-mix | Resemblyzer median margin | Resemblyzer confusions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| test | 1 | 14/15 | 0.61 / 0.81 | 0.36 | 0 | 0.85 / 0.91 | 0.14 | 1 |
| test | 2 | 15/15 | 0.63 / 0.85 | 0.41 | 0 | 0.87 / 0.92 | 0.16 | 0 |
| test | 4 | 15/15 | 0.64 / 0.89 | 0.39 | 0 | 0.88 / 0.93 | 0.14 | 0 |
| dev | 1 | 13/15 | 0.53 / 0.76 | 0.27 | 1 | 0.81 / 0.90 | 0.14 | 1 |
| dev | 2 | 15/15 | 0.63 / 0.83 | 0.34 | 0 | 0.84 / 0.92 | 0.13 | 0 |
| dev | 4 | 15/15 | 0.65 / 0.87 | 0.33 | 1 | 0.85 / 0.94 | 0.13 | 0 |

**Separation damage is clearly visible.** ECAPA similarity to the clean enrollment falls from about 0.85–0.89 (clean in-mix) to about 0.63–0.64 (discovered). The comparison isolates separation artifacts and the 8 kHz bandwidth: the in-mix source uses the same utterances without separation.

Identity still survives. ECAPA margins stay around 0.4 on test, with no confusions at any buffer. Resemblyzer has smaller margins (about 0.14). The only dev confusion at 4 rounds was speaker 3729, who was also the split speaker in that conversation.

## Live extraction of the chosen speaker

The live section is a later, separate stretch of the conversation. SI-SDRi is measured over the whole 32 s; target-absent time counts, so leakage is penalized.

- **Alone** = output SI-SDR while the target talks alone (target preservation; the input is already clean there).
- **+1/+2/+3** = SI-SDRi on spans where the target overlaps with that many others. +3 is the 4-speaker stress segment, which exceeds the 3-stream separator.
- **Absent suppression** = how much more the output drops than the input when the target is silent.

"Drop vs clean" pairs each discovered run with the same speaker's clean-enrollment run.

**Test set (held out):**

| Live model | Enrollment | Runs | Median SI-SDRi | p10 | Median drop vs clean | Alone SI-SDR | +1 | +2 | +3 | Absent suppression |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SpeakerBeam-SS | clean | 15 | 1.9 | -4.1 | – | 21.3 | 1.9 | 2.5 | 1.0 | 0.9 |
| SpeakerBeam-SS | discovered 1 round | 14 | 2.3 | -0.9 | 0.0 | 22.7 | 3.6 | 1.8 | 1.4 | 1.5 |
| SpeakerBeam-SS | discovered 2 rounds | 15 | 2.9 | -0.8 | -0.3 | 21.1 | 4.0 | 2.4 | 1.3 | 1.4 |
| SpeakerBeam-SS | discovered 4 rounds | 15 | 3.2 | 0.3 | -0.4 | 20.8 | 6.5 | 2.6 | 1.7 | 1.5 |
| SepFormer selector | clean | 15 | 12.2 | 7.2 | – | 21.2 | 14.4 | 10.3 | 5.8 | 13.1 |
| SepFormer selector | discovered 1 round | 14 | 12.6 | 5.8 | 0.1 | 20.9 | 13.5 | 9.9 | 3.4 | 13.8 |
| SepFormer selector | discovered 2 rounds | 15 | 12.2 | 8.0 | 0.0 | 21.2 | 14.4 | 9.5 | 6.2 | 13.3 |
| SepFormer selector | discovered 4 rounds | 15 | 12.0 | 5.5 | 0.0 | 21.2 | 14.4 | 11.2 | 5.6 | 13.2 |
| SepFormer competitive | clean | 15 | 12.1 | 7.7 | – | 19.3 | 16.0 | 10.3 | 6.9 | 14.4 |
| SepFormer competitive | discovered 1 round | 14 | 13.0 | 8.0 | 0.0 | 17.7 | 16.1 | 10.8 | 8.3 | 15.1 |
| SepFormer competitive | discovered 2 rounds | 15 | 12.2 | 6.3 | 0.0 | 19.3 | 15.9 | 9.5 | 6.9 | 15.8 |
| SepFormer competitive | discovered 4 rounds | 15 | 11.9 | 8.2 | 0.0 | 21.2 | 15.8 | 10.3 | 5.6 | 14.1 |

Dev results follow the same pattern and are in the showcase summary.

- **SpeakerBeam-SS barely separates** in these conversations, with +2–3 dB median and almost no suppression when the target is silent. This matches its M2 result. A discovered enrollment did not hurt it and slightly improved it. One possible reason is that a discovered track sounds more like the live mixture than a separate clean recording does, but this is not established.
- **Both SepFormer selectors reach about +12 dB median** and suppress target-absent time by about 13–16 dB. Quality falls as more people overlap, to about +6 dB beside three others. At that point the 3-stream separator is over capacity.
- **The competitive rule has the best worst cases.** It never turned a positive clean-enrollment result negative at any buffer. The plain threshold selector did once at 1 round (speaker 2961: +12.2 → −4.1 dB).

### Thresholds must be calibrated for discovered embeddings

The first run used one selector threshold for every enrollment type: 0.3, tuned on dev clean enrollments. With that threshold, discovered enrollment lost about **4.5 dB** median on test (+5.8 vs +12.2 dB at 4 rounds) and absent suppression fell from 13 to about 4 dB. Discovered tracks share the separator's artifacts and bandwidth, so their embeddings sit closer to *every* separated stream, and a fixed threshold lets other voices through.

The dev grid shows the size of the effect. For discovered embeddings, the median was +4.3 dB at threshold 0.3 and +11.1 dB at 0.5. For clean embeddings, 0.3 was best at +10.5 dB. Tuning a separate threshold on dev removed the loss on test.

The competitive selector avoids most of this sensitivity: its dev median stayed between +11.5 and +11.8 dB across floors 0–0.5 for discovered embeddings. A real app must calibrate any absolute threshold for the enrollment source it actually uses.

## Against the PRD's provisional gate

PRD §8 asks for a median drop of at most 1 dB, and no trial falling below 0 dB only because of the enrollment source. The table below is for the test set. On dev, SpeakerBeam-SS also flipped two trials at 4 rounds, both from small clean scores (+0.2 and +1.2 dB).

| Live model | 1 round (≈2.5 s solo) | 2 rounds (≈5 s) | 4 rounds (≈10 s) |
| --- | --- | --- | --- |
| SpeakerBeam-SS | Median passes (0.0 dB); one trial flips (4077: +1.5 → −1.2 dB) | Passes | Passes |
| SepFormer selector (per-type threshold) | Median passes; one trial flips (2961: +12.2 → −4.1 dB) | Passes | Passes |
| SepFormer competitive | Passes | Passes | Passes |

Individual trials still lose up to 4.8 dB with the competitive selector at 4 rounds, so the median hides real per-speaker variation. Based on discovery accuracy and these results, the provisional minimum for F2 is **about 5 seconds of solo speech per speaker** before offering a discovered profile. This should be retested on real recordings.

## Listening showcase (manual test)

`make_discovery_showcase.py` writes [data/discovery/showcase/index.html](data/discovery/showcase/index.html). Its audio stays local: about 430 MB of PCM16 WAV, ignored by Git. Open the file in a browser; no server is needed. The page follows the product flow:

1. Listen to the conversation.
2. Listen to each discovered speaker blind and rate whether it is one clean voice.
3. Choose one.
4. Hear what each of the three live models outputs for that choice.

**Reveal ground truth and scores** then shows who each track really was, the timeline of who talks when, the scores, the same speaker's results with a clean enrollment, and the ideal clean output. Ratings stay in the browser and **Export my ratings** saves them as JSON. Clips are peak-normalized individually, so compare clarity and leakage, not loudness. SepFormer outputs are band-limited to 8 kHz input.

## Real recording check (Nicker, Mond, Sam)

[run_user_discovery.py](run_user_discovery.py) ran the same flow on the user's own 23.1 s `Mix.wav`, a real 48 kHz stereo recording with three people. It used the thresholds tuned above, unchanged. Each person's separate recording (`Nicker.wav`, `Mond.wav`, `Sam.wav`) is used only to label discovered tracks and for the clean-enrollment comparison. There is no isolated track for each person in the mix, so the numbers below are ECAPA voice similarities, not SI-SDR. The audio, `run.json`, and a listening page stay in the Git-ignored `data/user_recordings/discovery/`.

**Discovery.** Similarities to each person's own recording:

| Run | Tracks found | Best match (cosine; margin over the next person) |
| --- | ---: | --- |
| Whole clip (23.1 s) | 3 | Nicker 0.51 (0.24), Sam 0.55 (0.17), Mond 0.46 (0.26) |
| First half only (11.6 s) | 2 | Nicker 0.37 (0.23), Sam 0.54 (0.22); **Mond not found** |

On the whole clip, each discovered track matched a different person with a clear margin. Mond's track appeared in only 4 of 11 windows. In the first half, Mond did not talk long enough to pass the 3-window minimum. That fits the provisional minimum of about 5 s of solo speech per person.

Absolute similarities are lower than on LibriSpeech (about 0.5 vs 0.64). That is expected when the separate recordings and the mix come from different takes and rooms.

**Live extraction on the whole clip.** Output similarity to the chosen person's recording vs the highest other person:

| Chosen | SepFormer selector: discovered / clean recording | SepFormer competitive: discovered / clean recording | SpeakerBeam-SS: discovered / clean recording |
| --- | --- | --- | --- |
| Nicker | 0.50 vs 0.28 / 0.49 vs 0.35 | 0.46 vs 0.24 / 0.49 vs 0.27 | 0.17 vs 0.18 / 0.33 vs 0.28 |
| Sam | 0.53 vs 0.35 / 0.53 vs 0.35 | 0.54 vs 0.36 / 0.56 vs 0.36 | 0.40 vs 0.19 / 0.14 vs 0.20 |
| Mond | 0.46 vs 0.19 / 0.51 vs 0.19 | 0.47 vs 0.20 / 0.47 vs 0.19 | 0.11 vs 0.16 / 0.17 vs 0.14 |

**Both SepFormer selectors** produce outputs that sound most like the chosen person, whether the target came from discovery or from the separate recording. The two enrollment types give similar similarities, consistent with the synthetic result.

**SpeakerBeam-SS** outputs often resemble the chosen person no more than the others, and sometimes less. This matches its weak synthetic scores.

In the half-and-half run, extraction from the second half also kept the chosen person most similar, but with more Nicker leakage in the Sam output (0.42 vs 0.36).

Similarity only shows *who the output sounds like*, not intelligibility or artifacts. The user's ratings from the listening page are the real check.

## Without future audio (block-online check)

The live results above use 4 s windows with overlap-add, so each output sample depends on up to about 4 s of later audio. That is impossible in a live app.

[run_block_online_selector.py](run_block_online_selector.py) removes the future:
- each block of **B** seconds is separated together with past context only, a window of **W** seconds ending at the block's last sample;
- only that newest block is output;
- worst-case algorithmic delay is **B**, plus compute time.

The pretrained SepFormer is unchanged. Thresholds were re-tuned for each setting on dev and reported on test. Blocks are joined without crossfades, so boundary clicks are possible. Timing is a desktop CPU (Apple M5, 4 PyTorch threads), not a phone.

**Real-time factor (RTF)** is median compute per block ÷ block length; above 1 means the system falls behind. All quality numbers are median SI-SDRi with discovered enrollment, test set:

| Block B (worst-case delay) | Past window W | RTF | Competitive | Selector | Competitive: target-alone SI-SDR | Competitive: absent suppression |
|---:|---:|---:|---:|---:|---:|---:|
| 0.25 s | 1 s | 2.99 | 7.5 dB | 6.4 dB | 5 dB | 18 dB |
| 0.25 s | 2 s | 3.40 | 8.2 dB | 5.7 dB | 9 dB | 9 dB |
| 0.5 s | 1 s | 1.18 | 11.0 dB | 8.0 dB | 4 dB | 40 dB |
| 0.5 s | 2 s | 1.67 | 9.2 dB | 8.4 dB | 11 dB | 10 dB |
| 1 s | 2 s | **0.85** | **11.6 dB** | 8.2 dB | 12 dB | 34 dB |
| 1 s | 4 s | 1.42 | 7.5 dB | 5.7 dB | 14 dB | 8 dB |
| 2 s | 4 s | **0.70** | 9.8 dB | 8.4 dB | 21 dB | 17 dB |
| *Offline reference (4 s windows, uses future audio)* | | – | *11.9 dB* | *12.0 dB* | *21 dB* | *14 dB* |

What this shows:

- **Short blocks cost too much.** At 0.25 s, quality roughly halves (about +7.5 dB) and the target is badly distorted even when talking alone (target-alone SI-SDR about 5 dB vs 21 dB offline). It also runs 3× slower than real time on this CPU.
- **Only blocks of 1 s or more keep up in real time here.** The best setting (B = 1 s, W = 2 s) nearly matches the offline median (+11.6 vs +11.9 dB). Its target still sounds more distorted (target-alone SI-SDR 12 vs 21 dB). Its total delay is about 1 s of block plus up to about 0.9 s of compute (p90), so roughly **2 s**: usable as delayed listening, not as live conversation (PRD target ≤150 ms). A phone would be slower.
- **A longer past window does not reliably help, and often hurts.** One likely reason, not tested here: in 4–6 speaker conversations a longer window contains more than three voices, which exceeds the 3-stream separator. The very high suppression in some short-window settings partly comes from blocks where no stream passed the threshold, so those blocks are silent.

Conclusion: the SepFormer selector cannot meet a conversational delay. The live stage needs a model built to be causal. The [survey of causal candidates](CAUSAL_CANDIDATES.md) found no option with public causal weights under a clear licence, so a training run is needed. It ranks first the WeSep causal BSRNN recipe: 32 ms latency, published causal research checkpoints with no weights licence, Apache-2.0 code. In the WeSep paper, causal conditioning on a plain ECAPA embedding fell to 6.87 dB SI-SDRi on Libri2Mix, while conditioning on features of the enrollment audio kept 12.98 dB. The discovered track's **audio**, not just its embedding, should therefore condition the live model.

Listening outputs from the real recording for every setting are on a local page, `data/user_recordings/discovery/block_online.html`, next to the offline output.

## Limitations

- **Clean, controlled audio.** All speech is clean read LibriSpeech with synthetic turn-taking, ±3 dB levels, no noise, no reverberation, and no real conversational overlap. Real recordings are the next check.
- **Small sample.** Six conversations and 15 test speakers. Treat the numbers as exploratory.
- **Offline only.** Discovery uses 4 s windows and the noncausal SepFormer. The live models are offline references with seconds of look-ahead, not streaming models. A causal TSE model conditioned on discovered embeddings still has to be found or trained.
- **Everyone was discovered.** Every live speaker was present during discovery. A person who joins later is never a "known speaker", so the competitive rule may hand their voice to the closest known one. Its dev-tuned floor was 0.0, the lowest grid value. Floors up to 0.2 (clean) and 0.35 (discovered) scored within 0.2 dB of the best on dev, so a stricter floor is affordable, but unknown-speaker rejection was not tested.
- **More than three at once is over capacity.** The window approach assumes at most three simultaneous talkers; the 4-speaker segment shows the drop.
- **SI-SDR at 16 kHz.** Scores are computed at 16 kHz, so SepFormer's 8 kHz bandwidth counts against it. The M2 report's common 8 kHz scoring was not applied here.

## Next steps

1. Run discovery and live selection on real conversation recordings, starting with the Nicker/Mond/Sam mix, and collect listening ratings from the showcase.
2. Add unknown-speaker tests (a speaker who joins after discovery) and set the competitive floor or a rejection rule from them.
3. Show split tracks in the interface (for example, offer to merge two tracks that sound like the same person) and enforce the 5 s minimum from F2.
4. Find or train a causal TSE model that accepts ECAPA embeddings, and re-run this test with it as the live model.

## Reproduce

From this directory, with the pinned models restored by `fetch_models.py` and `test-clean.tar.gz` extracted to `data/raw/`:

```sh
../speakerbeam_ss_verification/.venv/bin/python build_discovery_conversations.py
for f in data/discovery/*/*/single_trials.jsonl; do ../speakerbeam_ss_verification/.venv/bin/python validate_manifest.py "$f" --check-mix-sum; done
../speakerbeam_ss_verification/.venv/bin/python run_discovery_experiment.py   # about 15 min on a desktop CPU
../speakerbeam_ss_verification/.venv/bin/python make_discovery_showcase.py
```
