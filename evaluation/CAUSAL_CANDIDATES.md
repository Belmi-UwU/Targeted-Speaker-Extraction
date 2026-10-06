# Causal live-stage candidates (TSE and causal separation)

Survey date: 2026-10-07. Scope: models that could run the **live** stage causally on a phone (≤150 ms capture-to-playback, mono, on-device), given an enrollment from the discovery stage (the separated track audio, a SpeechBrain ECAPA 192-d embedding and a Resemblyzer 256-d d-vector).

- **Type A**: causal TSE conditioned on the target (embedding or enrollment audio).
- **Type B**: causal blind separator plus an embedding tracker that picks the target stream.

Every fact below was read from a paper, repo README/LICENSE/config or model card (links in each section). "Unverified" means I could not confirm it from a primary source. No weights were downloaded.

A finding that applies to all candidates: **no candidate combines public causal TSE weights, a clear weights licence and licence-clean training data.** Every route to a shippable model includes a training run. The candidates differ in how much code exists and how much of the work has already been shown to work.

## Comparison table

| # | Candidate | Type | Causal / algorithmic latency | Conditioning | SR | Size / compute | Reported quality | Code licence | Weights | Train-data licence |
|---|---|---|---|---|---|---|---|---|---|---|
| A1 | WeSep causal BSRNN (REAL-TSE baselines) | A | Yes. 32 ms (STFT window 512 / hop 128) | WeSpeaker ECAPA-TDNN (192-d) embedding **or** TF-map + contextual features from enrollment audio. Config accepts precomputed embeddings (`input: embedding`). | 16 kHz | unverified | Libri2Mix-100 SI-SDRi, causal: 6.87 dB (ECAPA emb.), 12.98 dB (TF-map+context), 14.15 dB (USEF+context) | Apache-2.0 (wesep); the wesep-real-tse fork has no LICENSE file | Causal ckpts on Google Drive (`spk_emb_causal_100`, `tfmap_context_causal_100`). **Licence not stated.** | Libri2Mix-100 (LibriSpeech CC BY 4.0; noise variant unverified) |
| A2 | ESPnet TD-SpeakerBeam, `causal: true` | A | Yes with `causal=True, norm_type=cLN`. Latency = encoder window (32 samples) | Internal enrollment net, or external embedding via `use_spk_emb` + `spk_emb_dim` | 8 / 16 kHz | ~8.7 M (Conv-TasNet TSE, per SB-SS paper) | Causal Conv-TasNet TSE baseline: 11.10 dB SDR, 2 spk + noise, 16 kHz (SB-SS paper row b1) | Apache-2.0 | Only **non-causal** ckpts published (CC BY 4.0) | Your choice (train) |
| A3 | SpeakerBeam-SS (Sato et al., Interspeech 2024) | A | Yes. 20 ms (+40/120 ms lookahead variants) | Internal learnable speaker encoder | 16 kHz | 7.93 M params, RTF 0.36 (1 core AMD EPYC 7502P, C++) | 11.58 dB SDR (20 ms), 11.88 (60 ms), 12.12 (140 ms) | **No official code** | None official | Paper: LibriSpeech + DNS4 noise |
| A4 | BUT SpeakerBeam (TD-SpeakerBeam, Asteroid) | A | `causal` flag in code | Internal enrollment net | 8 kHz | unverified | unverified | **Evaluation-only licence (BUT/NTT)** | `example/model.pth` under same licence | Libri2Mix |
| A5 | Look Once to Hear (UW, CHI 2024) | A | Yes. 12 ms (8 ms chunk + 4 ms lookahead), 18.24 ms end-to-end measured | **External Resemblyzer 256-d d-vector** | 16 kHz | 2.04 M params, 4.63 GMAC/s; 8 ms chunk in 6.24 ms on Orange Pi 5B | 7.40 dB SI-SNRi (clean enrollment, own binaural test set) | **Non-commercial (UW)** | Not linked in current README (unverified) | LibriSpeech + **WHAM! (CC BY-NC)** + HRTF/BRIR sets |
| A6 | APSR (2026) | A | Neural ops causal. Centered STFT adds 16 ms lookahead. Attention cost and state grow with sequence length. | Internal enrollment tokens from audio | 8 kHz | 0.31 M params | 13.81 dB SI-SDRi, WSJ0-2mix test | MIT | MIT (GitHub release) | **WSJ0 (LDC-licensed)** |
| A7 | pDCCRN / E3Net / pDeepFilterNet2 / VoiceFilter-Lite / Personalized PercepNet | A | Causal by design | d-vector / ECAPA | 16 kHz | see section | see section | **No public code found** | None | n/a |
| A8 | ClearerVoice-Studio | A | Audio-only TSE is SpEx+ **non-causal** | Internal | 8 kHz | n/a | 17.1 dB SI-SDRi WSJ0-2mix (non-causal) | Apache-2.0 | HF, non-causal | WSJ0 (LDC) |
| A9 | SpEx+ (official repos) | A | No causal option in SpEx_Plus code | Internal ResNet speaker encoder | 8 kHz | n/a | 16.9 dB SI-SDRi WSJ0-2mix (paper, non-causal) | MIT (SpEx_Plus) / GPL-3.0 (SpEx) | Baidu pan link, non-causal | WSJ0 (LDC) |
| A10 | Anchored Fast Weights (2026) | A | Chunked, 250 ms chunks, up to 4 s recomputed context | Internal encoder + FiLM | 16 kHz | 41k-param state + Conv-TasNet | Not reproduced in release | MIT | **None** | LibriSpeech (you train) |
| B1 | ESPnet SkiM causal (`lichenda/wsj0_2mix_skim_small_causal`) | B | Yes. Conv encoder kernel 8 / stride 4 at 8 kHz (1 ms) | n/a | 8 kHz | unverified (SkiM: 3.6 M in TFACM table) | SI-SNR 13.92 dB on WSJ0-2mix tt (absolute, not improvement) | Apache-2.0 | CC BY 4.0 (HF) | **WSJ0 (LDC)**, 2 speakers only |
| B2 | TFACM (Tsinghua, 2025) | B | Yes. 8 ms window / 1 ms hop | n/a | 8 kHz | 0.5 M / 19.4 GMAC/s (S); 1.0 M / 36.5 GMAC/s (L) | Libri2Mix noisy SI-SNRi 9.3 (S) / 11.0 (L) | README badge says MIT; **no LICENSE file** | None | LibriMix / WHAM! (you train) |
| B3 | Asteroid / SpeechBrain causal Conv-TasNet / DPRNN recipes | B | `causal=True` + cLN / unidirectional RNN in code | n/a | 8/16 kHz | ConvTasNet ~5 M (unverified for causal cfg) | Published LibriMix ckpts are **non-causal** (e.g. 11.98 dB SI-SDRi Libri3Mix-clean 8k) | MIT / Apache-2.0 | Only non-causal (CC BY-SA 3.0) | Libri3Mix clean = LibriSpeech CC BY 4.0 |

## Type A candidates

### A1. WeSep causal BSRNN (and the SLT 2026 REAL-TSE baselines)
- **Type:** A.
- **Causality:** `causal: true` switches the BSRNN separator to unidirectional temporal LSTMs and cumulative LN (`norm_type = "cLN" if causal else "GN"` in `wesep/modules/separator/bsrnn.py`). The band-to-band RNN stays bidirectional, but it runs across frequency, so causality holds. The repo includes a `check_causal()` perturbation test. The WeSep 2026 paper reports a "theoretical latency of 32 ms (corresponding to the FFT window length)". The causal models were exported to ONNX and validated in a streaming pipeline.
- **Enrollment:** modular. Options are an utterance-level speaker embedding (default config: WeSpeaker `ECAPA_TDNN_GLOB_c512`, 192-d), TF-map, contextual embedding, USEF or LISTEN. The `spkemb` block takes `input: waveform | embedding`, so precomputed embeddings can be fed in. SpeechBrain ECAPA would need retraining because it lives in a different embedding space from WeSpeaker ECAPA, even though both are 192-d. TF-map and USEF use the enrollment **audio**, which the discovery stage already provides.
- **Sample rate:** 16 kHz.
- **Size/compute:** unverified for the released causal checkpoints.
- **Quality (WeSep paper, Libri2Mix-100):** non-causal with ECAPA embedding 13.17 dB SI-SDRi. Causal: speaker embedding 6.87 dB, USEF 13.05 dB, TF-map+context 12.98 dB, USEF+context 14.15 dB. **Plain embedding conditioning loses about half its gain under causality. Audio-feature conditioning mostly survives.** On REAL-T (real conversations) the causal TF-map model matches or beats the non-causal one (DEV TER 0.652 vs 0.703).
- **Availability:** WeSep code is Apache-2.0. WeSep's own pretrained packages are "under validation" (roadmap). The REAL-TSE repos publish four BSRNN checkpoints on Google Drive, two of them causal, with **no licence statement for the weights**. The `wesep-real-tse` fork has no LICENSE file. REAL-TSE-Challenge itself is MIT.
- **Training data:** Libri2Mix-100. LibriSpeech is CC BY 4.0. Whether WHAM! noise was included is unverified. The WeSpeaker VoxCeleb encoders are CC BY 4.0 (WeSpeaker "Model License").
- **Sources:** https://github.com/wenet-e2e/wesep · https://arxiv.org/abs/2607.27436 · https://github.com/REAL-TSE/REAL-TSE-Challenge · https://github.com/REAL-TSE/wesep-real-tse · https://arxiv.org/abs/2607.15198 · https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md

### A2. ESPnet TD-SpeakerBeam in causal mode
- **Type:** A.
- **Causality:** `TDSpeakerBeamExtractor(causal=True, norm_type="cLN")` makes the Conv-TasNet stack causal. The enrollment branch stays non-causal, which is fine because enrollment is offline. Algorithmic latency equals the encoder window: 32 samples in the published configs, i.e. 4 ms at 8 kHz and 2 ms at 16 kHz.
- **Enrollment:** internal auxiliary network, **or** `use_spk_emb=True, spk_emb_dim=N` with `load_spk_embedding`. With that option an external embedding is the conditioning input, so SpeechBrain ECAPA (192) or Resemblyzer (256) can be used directly.
- **Sample rate:** 8 kHz and 16 kHz recipes (`egs2/librimix/tse1`, ESPnet3 `egs3/librimix/tse`).
- **Size/compute:** the SB-SS paper's causal Conv-TasNet TSE (same family) is 8.69 M params with RTF 1.67 on one EPYC core, too slow for real time. The paper's SB-SS changes (A3) cut RTF to 0.36.
- **Quality:** published ESPnet checkpoints are non-causal. Libri2Mix 8 kHz gives 10.74 dB SI-SNR on test. The 16 kHz ESPnet3 model gives 12.66 dB SI-SNR. Expected causal quality is about 11.1 dB SDR (SB-SS paper b1, 2 spk + noise, 16 kHz).
- **Availability:** code is Apache-2.0. The non-causal checkpoints `espnet/Wangyou_Zhang_librimix_train_enh_tse_td_speakerbeam_raw` are CC BY 4.0. `wyz/librimix_tse_clean_2mix_training_td_speakerbeam_16k` has no licence in its card. **No causal TSE checkpoint is published**, so this model has to be trained from scratch. gLN→cLN changes the normalisation statistics, so a non-causal checkpoint at best gives a weight initialisation.
- **Sources:** https://github.com/espnet/espnet/blob/master/espnet2/enh/extractor/td_speakerbeam_extractor.py · https://huggingface.co/espnet/Wangyou_Zhang_librimix_train_enh_tse_td_speakerbeam_raw · https://huggingface.co/wyz/librimix_tse_clean_2mix_training_td_speakerbeam_16k

### A3. SpeakerBeam-SS (NTT, Interspeech 2024)
- **Type:** A.
- **Causality:** S4D blocks (causal) + causal convs + channel-wise LN. Max algorithmic latency equals the encoder window: 320 samples = 20 ms at 16 kHz. Lookahead variants are 60 ms and 140 ms.
- **Enrollment:** learnable internal speaker encoder (one dilated conv block). Footnote 3 says that for the pDCCRN/E3Net baselines "the d-vector extractor [Resemblyzer] performed better than ECAPA-TDNN and the learnable speaker encoder".
- **Sample rate:** 16 kHz.
- **Size/compute (Table 1):** d1 has 7.93 M params, RTF 0.36 (C++, one AMD EPYC 7502P core) and 11.58 dB SDR. d2 (+40 ms lookahead) gives 11.88 dB, d3 (+120 ms) 12.12 dB. Comparison rows: pDCCRN 10.70 dB at 32 ms, E3Net 9.83 dB at 20 ms, non-causal 13.63 dB. Test set is LibriSpeech 2 spk + DNS4 noise. DNSMOS is also reported.
- **Availability:** no official code or weights found. Independent re-implementations:
  - `helloooideeeeea/OpenSpeakerBeam-SS` (MIT, Resemblyzer conditioning) was already tested by this project and rejected.
  - Its fork `linsanzhu/OpenSpeakerBeam-SS` (MIT) adds ONNX/INT8 export, and its README warns of "modest" performance.
  - `gabrielmbmb/voce-speakerbeam-ss-librimix-train360` on HF has checkpoints but **no model card, no licence and no source code**. Excluded.
- **Sources:** https://www.isca-archive.org/interspeech_2024/sato24_interspeech.pdf · https://arxiv.org/abs/2407.01857 · https://github.com/linsanzhu/OpenSpeakerBeam-SS · https://huggingface.co/gabrielmbmb/voce-speakerbeam-ss-librimix-train360

### A4. BUT SpeakerBeam (BUTSpeechFIT/speakerbeam)
- **Type:** A. TD-SpeakerBeam on Asteroid with a `causal` flag (default gLN, non-causal). 8 kHz Libri2Mix/Libri3Mix recipes. Ships `example/model.pth`.
- **Licence:** "SOFTWARE LICENSE AGREEMENT FOR EVALUATION" (BUT + NTT). It allows internal evaluation only and forbids modification and redistribution. **Excluded** for the app.
- **Source:** https://github.com/BUTSpeechFIT/speakerbeam

### A5. Look Once to Hear (Veluri et al., CHI 2024)
- **Type:** A.
- **Causality:** streaming TF-GridNet (`chunk_causal: true`) with 12 ms algorithmic latency: 8 ms hop, 12 ms window and 4 ms lookahead. Measured end-to-end latency is 18.24 ms.
- **Enrollment:** **external 256-d Resemblyzer d-vector** (`embed_dim: 256`). This is the same d-vector the project already computes. A separate network estimates the d-vector from a noisy binaural clip.
- **Sample rate:** 16 kHz. Input is **binaural (`num_ch: 2`)**, so the mono phone case needs retraining with `num_ch: 1`.
- **Size/compute:** 2.04 M params, 4.63 GMAC/s. An 8 ms chunk takes 6.24 ms on an Orange Pi 5B CPU.
- **Quality:** 7.40 dB SI-SNRi with clean enrollment (7.01 dB with noisy enrollment) on the authors' synthetic binaural set.
- **Availability:** "Look Once to Hear Non-commercial License" (UW). It allows noncommercial use only; commercial use requires a UW CoMotion licence. The current README does not link checkpoints (unverified). Training uses WHAM! noise (CC BY-NC 4.0).
- **Sources:** https://github.com/vb000/LookOnceToHear · https://arxiv.org/abs/2405.06289

### A6. APSR (Xiang-Lin75/APSR, 2026)
- **Type:** A. Causal TSE with enrollment-anchored memory.
- **Causality:** "Neural processing is causal; centered STFT analysis adds 16 ms of lookahead" (256-sample window, 64 hop, 8 kHz). The model card warns that "dense attention cost and state grow with sequence length" and that per-hop streaming latency has not been established.
- **Enrollment:** internal, from enrollment audio (8 frequency-resolved tokens). Not ECAPA.
- **Size:** 306,842 trainable params.
- **Quality:** 13.81 dB SI-SDRi, 6,000 queries on WSJ0-2mix test, clean, 2 speakers.
- **Availability:** code **and weights MIT**. Training data is WSJ0-2mix, and **WSJ0 is an LDC-licensed corpus**. The MIT grant on the weights does not settle the corpus terms. A WHAM! recipe exists without a checkpoint.
- **Sources:** https://github.com/Xiang-Lin75/APSR · https://github.com/Xiang-Lin75/APSR/blob/main/docs/MODEL_CARD.md

### A7. Personalized speech enhancement family (no public code)
- **pDCCRN / pDCATTUNET** (Eskimez et al., ICASSP 2022, Microsoft). Causal and d-vector conditioned. The paper links only an audio-samples page. https://arxiv.org/abs/2110.09625
- **E3Net** (Thakker et al., Interspeech 2022, Microsoft). No code found (unverified absence).
- **pDeepFilterNet2** (Serre et al., 2024, Orosound). Conditioned on **ECAPA-TDNN** (TaoRuijie implementation). Uses 20 ms windows with 2-frame lookahead, giving 40 ms lookahead. Only a sample page exists, with no code or weights. A follow-up (arXiv 2601.16235) distils a 150k-param "TinyECAPA". https://arxiv.org/abs/2404.08022 · https://pdeepfilternet2.github.io/
  - Its base model, **DeepFilterNet**, has code and weights dual-licensed MIT/Apache-2.0 (48 kHz). That makes it a plausible backbone to personalise yourself. https://github.com/Rikorose/DeepFilterNet
- **VoiceFilter-Lite** (Google, 2020). Streaming, d-vector conditioned. Its output is **enhanced filterbank features for ASR, not a waveform**. Google's original is not open-sourced (unverified for Lite specifically). Not usable for listening. https://arxiv.org/abs/2009.04323
- **DNS Challenge 5 (ICASSP 2023) personalized track:** the baseline speaker embedding was SpeechBrain `spkrec-ecapa-voxceleb`, the same model this project uses (Apache-2.0). ECAPA-conditioned causal PSE is therefore an established setup, but no participant weights were found. The DNS data is per-source licensed, and the repo code is MIT. https://github.com/microsoft/DNS-Challenge

### A8. ClearerVoice-Studio
- Audio-only TSE supports only **SpEx+ (non-causal)**, trained on WSJ0-2mix with 17.1 dB SI-SDRi. Of the audio-visual TSE models only AV-ConvTasNet is marked "Causal/Non-causal", and it needs video. Separation models (MossFormer2) are non-causal. Apache-2.0. **Not usable live.**
- https://github.com/modelscope/ClearerVoice-Studio/tree/main/train/target_speaker_extraction

### A9. SpEx / SpEx+ official code
- `gemengtju/SpEx_Plus` (MIT) has gLN/cLN norm builders but no causal conv option. Weights are on Baidu pan and trained on WSJ0-2mix. `xuchenglin28/speaker_extraction_SpEx` is GPL-3.0. WeSep also implements SpEx+ (Apache-2.0), but its causal path is documented only for BSRNN. **Not a live candidate** without your own causal rewrite.
- https://github.com/gemengtju/SpEx_Plus · https://github.com/xuchenglin28/speaker_extraction_SpEx

### A10. Others seen, excluded
- **SCI-LFC** (`yuxu93528-ops/SCI-LFC`): causal TSE but released only as an ONNX demo with fixed 4 s input at 8 kHz, 2 speakers. No licence and no training code.
- **Anchored Fast Weights** (`ym2976/anchor-fast-weight`, MIT): no weights, and 250 ms chunks exceed the 150 ms budget. Its idea of an adaptive speaker state anchored to the enrollment is relevant to embedding drift.
- **Dual-mode TSE** (CyberAgent, APSIPA 2025): causal TF-GridNet with 12 ms delay conditioned on Resemblyzer. 7-mic array, no code.
- **UW Proactive Hearing Assistants** (2025): binaural, conditioned on the wearer's own voice. Different task.
- **NVIDIA Real-time RE-USE**: enhancement only, NVIDIA non-commercial licence.

## Type B candidates

### B1. ESPnet SkiM causal
- `lichenda/wsj0_2mix_skim_small_causal`: `separator: skim, causal: true`, conv encoder kernel 8 / stride 4 at 8 kHz. SI-SNR 13.92 dB on WSJ0-2mix tt. Weights are **CC BY 4.0**, but the training corpus is WSJ0 (LDC). **2 speakers only.**
- ESPnet also publishes causal **speech-enhancement** BSRNN checkpoints (`wyz/vctk_bsrnn_{xtiny…large}_causal`, CC BY 4.0, VCTK+DEMAND, 48 kHz training). These are a possible initialisation for a causal BSRNN extractor, not a separator.
- https://huggingface.co/lichenda/wsj0_2mix_skim_small_causal · https://huggingface.co/wyz/vctk_bsrnn_small_causal

### B2. TFACM
- Causal TF-domain separator with cache memory and causal attention. 8 ms window / 1 ms hop at 8 kHz.
- Results at 8 kHz on Libri2Mix: SI-SNRi 9.3 dB for Small (0.5 M params, 19.4 GMAC/s) and 11.0 dB for Large (1.0 M, 36.5 GMAC/s). Reference points from the same table: causal TF-GridNet 10.8 dB (11.4 M, 179.2 GMAC/s), SkiM 9.2 dB, DPRNN 9.0 dB.
- Code only, with no weights. The README shows an MIT badge, but the repo has **no LICENSE file**.
- https://github.com/JusperLee/TFACM · https://arxiv.org/abs/2505.13094

### B3. Asteroid / SpeechBrain causal recipes
- Both toolkits expose `causal=True` + cLN for Conv-TasNet (SpeechBrain `MaskNet`, Asteroid `ConvTasNet`), and Asteroid has unidirectional DPRNN. Every published LibriMix checkpoint is **non-causal**. For example, `JorisCos/ConvTasNet_Libri3Mix_sepclean_8k` (CC BY-SA 3.0, Libri3Mix clean = LibriSpeech only) uses default gLN and scores 11.98 dB SI-SDRi. Code licences are MIT (Asteroid) and Apache-2.0 (SpeechBrain).
- https://huggingface.co/JorisCos/ConvTasNet_Libri3Mix_sepclean_8k · https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/lobes/models/conv_tasnet.py

### Structural limits of route B (from the sources above)
- All available causal separators have a fixed output count (2, or 3 if retrained), which conflicts with conversations of 4–6 speakers.
- Even the best causal 2-speaker blind separators give about 9–11 dB SI-SNRi on noisy LibriMix (TFACM table). That is lower than causal TSE with audio-feature conditioning (about 13 dB, WeSep).
- Identifying the target stream from embeddings needs seconds of audio. A recent streaming diarization + verification pipeline needed two 500 ms chunks, giving about 1 s identification latency. The tracker can run off the audio path, but stream permutations in the meantime leak the wrong speaker. https://arxiv.org/abs/2608.17972

## Recommendation (live stage)

1. **WeSep causal BSRNN (A1).** It is the only candidate that has published causal TSE checkpoints for both ECAPA-embedding and enrollment-audio conditioning, 32 ms algorithmic latency, ONNX export, a working causality test, and Apache-2.0 code. The causal checkpoints can be evaluated right away on the project's Nicker/Mond/Sam recording. They should be treated as evaluation-only until the authors state a weights licence, since none is given today.
   - Key data point: causal conditioning on a plain embedding drops to 6.87 dB SI-SDRi, while TF-map+context reaches 12.98 dB. The live model should therefore be fed the **discovery track audio**, not only the ECAPA vector.
   - For shipping: retrain the causal config on licence-clean data (LibriSpeech + CC0/CC-BY noise, no WHAM!), with more than 2 speakers per mix, optionally adding the SpeechBrain ECAPA embedding via `input: embedding`.
   - Effort: an evaluation now, then one training run.
2. **ESPnet TD-SpeakerBeam with `causal=True`, `use_spk_emb=True` (A2), extended with the SpeakerBeam-SS changes (A3).**
   - This is the most direct way to condition on the existing SpeechBrain ECAPA (192-d) or Resemblyzer (256-d) vectors. The SB-SS footnote suggests also trying Resemblyzer.
   - Code is Apache-2.0, and you control the training data, so licensing is clean end to end.
   - Expected quality is about 11 dB SDR at ≤20 ms latency. The SB-SS changes (320-sample window, N=2048 encoder, S4D blocks) bring RTF from 1.67 to 0.36 on one server core.
   - Effort: train from scratch, since no causal checkpoint exists, plus porting S4D into the ESPnet extractor.
3. **APSR (A6), as a zero-training baseline only.**
   - It is the only causal TSE with an explicit MIT grant on weights, at 0.31 M params and 13.81 dB SI-SDRi on WSJ0-2mix. It is useful to check quickly whether causal TSE with audio enrollment survives this project's real 4–6-speaker audio better than OpenSpeakerBeam-SS did.
   - Not shippable as is: it is 8 kHz, trained only on clean 2-speaker WSJ0 (LDC corpus), uses attention whose cost grows with sequence length, and has 16 ms STFT lookahead.

Not in the top 3:
- **Look Once to Hear (A5)** is technically the closest to the product (12 ms latency, 2 M params, proven on an embedded CPU, conditioned on Resemblyzer d-vectors). Its non-commercial code licence and binaural input put it behind A1/A2. It is worth using as a design reference.
- **Route B** has no licence-clean causal weights beyond 2 speakers (and those are WSJ0-trained). It has a fixed stream count and adds target-identification latency. Keep it as a fallback, built on Asteroid/SpeechBrain causal recipes or TFACM code, trained on Libri3Mix-style data with ≥4 sources.
