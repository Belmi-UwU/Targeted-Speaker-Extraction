# Graph Report - Targeted-Speaker-Extraction  (2026-10-09)

## Corpus Check
- 33 files · ~109,529 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1017 nodes · 2285 edges · 66 communities (45 shown, 21 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 140 edges (avg confidence: 0.86)
- Token cost: 142,999 input · 0 output

## Community Hubs (Navigation)
- M2 Matching & Analysis Scripts
- Discovery Pipeline Code
- WeSep Retraining Script
- Data Assets & Listening Comparisons
- LibriMix Data Generation
- WeSep Comparison & Candidates
- Live Spectrum HTTP Handler
- SepFormer Live Demo
- Discovery Enrollment Report
- Discovery Conversation Builder
- Causal TSE Candidate Models
- WeSep Live Extractor
- SpeakerBeam-SS Inference
- Retrained Eval & Streaming Demo
- Causal SepFormer & M2 Interim
- Causal WeSep Retraining Plan
- Web Control Panel & Visual Demo
- SI-SNR Training & Eval
- Causal SepFormer Experiment
- Manifest Schema: Sources
- Progress & User Recordings
- SpeakerBeam Encoder/Decoder
- Offline Eval Assets & M1
- Discovery Showcase Page
- Streaming Adapter
- WeSep Checkpoints & Conditioning
- PRD Milestones & Delay Gate
- Discovery Evaluation Metrics
- SepFormer + ECAPA Reference Pipeline
- Spectrogram Comparison Page
- SepFormer LiveExtractor
- Manifest Schema: Enums
- Browser Spectrogram Rendering
- StreamingWesep Core
- Live App Product Concepts
- Adaptation Layers
- S4D Dropout
- Web LiveEngine
- S4D Block
- Upstream Verification Script
- S4D Kernel
- Real-Data Gates & Datasets
- Manifest Schema Root
- Manifest Schema: Sources Array
- Manifest Schema: Selected Speakers
- M2 Model Comparison Gates
- Schema: dataset_revision
- Schema: speech_license
- Schema: trial_id
- Schema: enrollment_utterance_id
- Schema: mixture_path
- Schema: sample_rate_hz
- Discovery vs Live Sections
- 5 s Solo-Speech Minimum
- Live Extraction Results
- Oracle Audit
- Resemblyzer Encoder
- Quiet Stream Dropping
- Split Tracks Failure
- Separator Over-Capacity
- Windowed SepFormer Discovery
- M7 Status
- Real Recordings

## God Nodes (most connected - your core abstractions)
1. `Discovery Enrollment Report` - 30 edges
2. `Causal Live-Stage Candidates Survey` - 27 edges
3. `PRD: Live target-speaker listening app` - 25 edges
4. `Causal WeSep Retraining Plan` - 23 edges
5. `read_manifest()` - 22 edges
6. `validate_trial()` - 21 edges
7. `SpeakerBeam-SS checkpoint verification report` - 20 edges
8. `audio_path()` - 19 edges
9. `enrollment()` - 18 edges
10. `App` - 17 edges

## Surprising Connections (you probably didn't know these)
- `SpeakerBeam-SS Live Model` --semantically_similar_to--> `OpenSpeakerBeam-SS checkpoint`  [INFERRED] [semantically similar]
  evaluation/DISCOVERY_REPORT.md → PRD.md
- `User real 48 kHz stereo recordings (Mix + Nicker/Mond/Sam enrollments)` --semantically_similar_to--> `REAL-T dataset`  [INFERRED] [semantically similar]
  evaluation/USER_RECORDINGS.md → PRD.md
- `Controlled turns/absence clip (4 x 0.9 s intervals)` --semantically_similar_to--> `SparseLibriMix`  [INFERRED] [semantically similar]
  evaluation/M2_MODEL_COMPARISON.md → PRD.md
- `SepFormer + mixture-fit + ECAPA offline reference pipeline` --semantically_similar_to--> `Per-speaker extraction plus combination (approach b)`  [INFERRED] [semantically similar]
  evaluation/M2_MODEL_COMPARISON.md → PRD.md
- `GroupNorm -> per-time-step CumulativeLayerNorm replacement` --semantically_similar_to--> `SpeakerBeam-SS paper (Interspeech 2024)`  [INFERRED] [semantically similar]
  evaluation/CAUSAL_SEPFORMER_EXPERIMENT.md → PRD.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Causal conversion of offline SepFormer without retraining** — evaluation_m2_model_comparison_sepformer_libri3mix, evaluation_causal_sepformer_experiment_future_attention_mask, evaluation_causal_sepformer_experiment_cumulative_layer_norm, prd_causal_prefix_invariance_test, evaluation_causal_sepformer_experiment_trained_causal_checkpoint [EXTRACTED 1.00]
- **Discovery enrollment flow: separate, cluster, embed, pick, extract live** — prd_discovery_enrollment, evaluation_discovery_report_windowed_sepformer_discovery, evaluation_discovery_report_silent_stream_drop, evaluation_discovery_report_agglomerative_cosine_clustering, prd_ecapa_tdnn, prd_target_speaker_extraction [EXTRACTED 1.00]
- **PRD evaluation data plan dataset ladder** — prd_libri2mix, prd_libri3mix, prd_wham_noise, prd_sparselibrimix, prd_real_t, prd_librispeech_test_clean [EXTRACTED 1.00]
- **Separated-track enrollment test measurements** — prd_separated_track_enrollment_test, prd_embedding_fidelity_metric, prd_identification_margin, prd_discovery_correctness, prd_discovered_enrollment_downstream_gate, prd_speech_needed_measurement [EXTRACTED 1.00]
- **SepFormer encoder-masknet-decoder pipeline** — evaluation_sepformer_libr3mix_checkpoint_hyperparams_encoder, evaluation_sepformer_libr3mix_checkpoint_hyperparams_masknet, evaluation_sepformer_libr3mix_checkpoint_hyperparams_sbtransformerblock, evaluation_sepformer_libr3mix_checkpoint_hyperparams_decoder [EXTRACTED 1.00]
- **SepFormer + mixture-fit + ECAPA offline reference pipeline** — evaluation_m2_model_comparison_sepformer_libri3mix, evaluation_m2_model_comparison_mixture_fit_gains, evaluation_m2_model_comparison_ecapa_matcher, evaluation_m2_model_comparison_common_8k_scoring, evaluation_m2_model_comparison_sepformer_ecapa_offline_pipeline [EXTRACTED 1.00]
- **Target-speaker extraction method comparison (SpeakerBeam-SS vs SepFormer+embedding matching vs ideal)** — evaluation_data_listen_next_readme_speakerbeam_ss_extraction, evaluation_data_listen_next_readme_sepformer_mixfit, evaluation_data_listen_next_readme_ideal_reference, evaluation_data_listen_next_readme_three_voice_mix, evaluation_data_listen_next_readme_interferer_leakage [EXTRACTED 1.00]
- **SpeakerBeam-SS deployment blockers** — speakerbeam_ss_verification_verification_report_causality_failure, speakerbeam_ss_verification_verification_report_output_scaling_clipping, speakerbeam_ss_verification_verification_report_weight_license_provenance, speakerbeam_ss_verification_verification_report_paper_architecture_differences, speakerbeam_ss_verification_verification_report_decision [EXTRACTED 1.00]
- **Live extractor models evaluated on discovered enrollment** — evaluation_discovery_report_speakerbeam_ss_live_model, evaluation_discovery_report_sepformer_selector, evaluation_discovery_report_sepformer_competitive_selector, evaluation_wesep_comparison_tfmap_context_causal_100 [INFERRED 0.85]
- **Retraining milestones R0-R4** — evaluation_wesep_retraining_plan_r0_data, evaluation_wesep_retraining_plan_r1_pilot, evaluation_wesep_retraining_plan_r2_from_scratch, evaluation_wesep_retraining_plan_r3_streaming, evaluation_wesep_retraining_plan_r4_phone [EXTRACTED 1.00]
- **Training data design matching the live product flow** — evaluation_wesep_retraining_plan_conversation_simulator, evaluation_wesep_retraining_plan_target_absent_examples, evaluation_wesep_retraining_plan_discovered_enrollment_simulation, evaluation_wesep_retraining_plan_si_sdr_plus_absent_energy_loss, evaluation_wesep_retraining_plan_curriculum [INFERRED 0.85]
- **Streaming live extraction pipeline** — evaluation_wesep_eval_streaming_streaming_wesep, evaluation_wesep_eval_live_demo_streaming, evaluation_wesep_eval_web_app, evaluation_wesep_eval_web_app_local_server_api, evaluation_wesep_eval_live_spectrum [INFERRED 0.85]
- **WeSep retraining R1-R2 workflow** — training_readme_wesep_retrain_notebook, training_readme_mode_finetune_research, training_readme_mode_scratch_small, training_readme_eval_pack, training_results_readme_r1_results, training_results_readme_r2_takeaways [EXTRACTED 1.00]
- **Research-only weights licence constraint** — evaluation_wesep_eval_readme_weights_licence_restriction, evaluation_wesep_eval_readme_tfmap_context_causal_100, training_readme_mode_finetune_research, training_results_readme_avg_model [INFERRED 0.85]

## Communities (66 total, 21 thin omitted)

### Community 0 - "M2 Matching & Analysis Scripts"
Cohesion: 0.05
Nodes (62): main(), wave8(), main(), read_mono(), sha256(), build(), ensure_audio(), get_enrollment() (+54 more)

### Community 1 - "Discovery Pipeline Code"
Cohesion: 0.05
Nodes (43): active_speech(), cluster_streams(), discover(), embed_streams(), Models, overlap_add(), read_wave(), select_streams() (+35 more)

### Community 2 - "WeSep Retraining Script"
Cohesion: 0.05
Nodes (33): active_speech(), atomic(), build_index(), causality_check(), coloured_noise(), discovered_style(), ensure_librispeech(), evaluate_pack() (+25 more)

### Community 3 - "Data Assets & Listening Comparisons"
Cohesion: 0.06
Nodes (52): LibriSpeech archive MD5 checksums, LibriSpeech corpus archives (dev/test/train splits), test-clean.tar.gz, Listening comparison: selected speakers 4077 + 5142, Ideal clean-sum reference (4077+5142), M2_MODEL_COMPARISON.md (methods and measured results), SepFormer single-speaker output (only 5142), SpeakerBeam-SS per-speaker extraction output (+44 more)

### Community 4 - "LibriMix Data Generation"
Cohesion: 0.06
Nodes (29): add_to_metrics_metadata(), add_to_mixture_metadata(), compute_snr_list(), create_empty_metrics_md(), create_empty_mixture_md(), create_librimix(), extend_noise(), fit_lengths() (+21 more)

### Community 5 - "WeSep Comparison & Candidates"
Cohesion: 0.06
Nodes (36): WeSep comparison README, CAUSAL_CANDIDATES.md, Libri2Mix train-100 (clean, 2 speakers), Real Nicker/Mond/Sam recording (Mix.wav), REAL-TSE BSRNN baselines, SepFormer selector (live model of discovery-enrollment), spk_emb_causal_100 checkpoint, tfmap_context_causal_100 checkpoint (+28 more)

### Community 6 - "Live Spectrum HTTP Handler"
Cohesion: 0.08
Nodes (11): Handler, __init__(), push(), status(), Tap, App, callback(), work() (+3 more)

### Community 7 - "SepFormer Live Demo"
Cohesion: 0.12
Nodes (15): choose_speaker(), device_rate(), discover_speakers(), main(), OutputBuffer, play(), read_any(), record() (+7 more)

### Community 8 - "Discovery Enrollment Report"
Cohesion: 0.12
Nodes (26): Absent suppression metric, Average-Linkage Agglomerative Clustering on Cosine Distance, Discovery Enrollment Flow, Discovery Listening Showcase (make_discovery_showcase.py), Discovery Pipeline (windowed SepFormer + clustering), Discovery Enrollment Report, ~5 s Solo Speech Minimum per Speaker (F2), Nicker/Mond/Sam Real Recording (Mix.wav) (+18 more)

### Community 9 - "Discovery Conversation Builder"
Cohesion: 0.13
Nodes (14): build_mixture(), discovery_schedule(), faded(), live_schedule(), main(), read_mono(), render(), speaker_utterances() (+6 more)

### Community 10 - "Causal TSE Candidate Models"
Cohesion: 0.19
Nodes (22): A10 Anchored Fast Weights, A2 ESPnet TD-SpeakerBeam (causal), A3 SpeakerBeam-SS (NTT), A4 BUT SpeakerBeam, A5 Look Once to Hear, A6 APSR, A7 Personalized Speech Enhancement Family, A8 ClearerVoice-Studio (+14 more)

### Community 11 - "WeSep Live Extractor"
Cohesion: 0.17
Nodes (10): WesepLiveExtractor, block_online(), causality(), extract(), main(), runs_for(), summarize(), label() (+2 more)

### Community 12 - "SpeakerBeam-SS Inference"
Cohesion: 0.18
Nodes (3): main(), SpeakerBeamSS, get_speaker_embeddings_batch()

### Community 13 - "Retrained Eval & Streaming Demo"
Cohesion: 0.24
Nodes (8): main(), save(), enrollment(), load(), deterministic(), main(), stream(), work()

### Community 14 - "Causal SepFormer & M2 Interim"
Cohesion: 0.15
Nodes (16): Causal SepFormer inference experiment, GroupNorm -> per-time-step CumulativeLayerNorm replacement, Upper-triangular future attention mask in SBTransformerBlock, SpeakerBeam-SS Live Model, Fixed listening gain (0.014 M1, 0.0127804 M2), Libri2Mix trial 4077/5142 (3.77 s, 16 kHz), M2 interim report: three voices, selected sets, Libri3Mix trial 4077/5142/5683 (+8 more)

### Community 15 - "Causal WeSep Retraining Plan"
Cohesion: 0.15
Nodes (16): Stateful Streaming Implementation, Common Voice (CC0), Training Curriculum, Causal WeSep Retraining Plan, Domain Gap Risk, LibriSpeech Training Sets, MUSAN Noise, Small Phone Candidate Model (+8 more)

### Community 16 - "Web Control Panel & Visual Demo"
Cohesion: 0.20
Nodes (5): main(), Live Spectrum page, Local control server HTTP API (/api/*, /data), main(), serve()

### Community 17 - "SI-SNR Training & Eval"
Cohesion: 0.16
Nodes (5): evaluate(), si_snr_loss(), SpeechDataset, test_model(), train_and_validate()

### Community 18 - "Causal SepFormer Experiment"
Cohesion: 0.27
Nodes (9): add_attention_masks(), benchmark(), main(), masked_forward(), prefix_test(), quality(), replace_global_norms(), visit() (+1 more)

### Community 19 - "Manifest Schema: Sources"
Cohesion: 0.15
Nodes (13): minLength, type, properties, enrollment_path, source_path, source_utterance_id, speaker_id, minLength (+5 more)

### Community 20 - "Progress & User Recordings"
Cohesion: 0.22
Nodes (13): Clean References (time-aligned), User Recordings Guide, Enrollment Examples (per-speaker recordings), results/run.json Provenance Record, data/user_recordings Folder Layout, Project Progress, M1 Dataset and Scoring Contract Milestone, M2 Offline Model Decision Milestone (+5 more)

### Community 21 - "SpeakerBeam Encoder/Decoder"
Cohesion: 0.15
Nodes (3): Decoder, Encoder, Separator

### Community 22 - "Offline Eval Assets & M1"
Cohesion: 0.24
Nodes (11): ASSETS.md - Local data and model weights, Offline TSE evaluation README, 3 GB temporary working-storage cap, Trial manifest (JSON Lines), Libri2Mix, LibriSpeech test-clean, M1 - dataset and scoring contract, MiniLibriMix (8 kHz) (+3 more)

### Community 23 - "Discovery Showcase Page"
Cohesion: 0.33
Nodes (11): Discovery enrollment showcase template, audioCard(), DATA placeholder (injected showcase JSON), init(), ratingWidget(), renderLive(), renderMixture(), renderSummary() (+3 more)

### Community 25 - "WeSep Checkpoints & Conditioning"
Cohesion: 0.29
Nodes (11): A1 WeSep Causal BSRNN, Plain Speaker Embedding Conditioning, TF-Map + Context Conditioning on Enrollment Audio, WeSep vs SepFormer Comparison, Enrollment Level Normalization (-25 dBFS), Libri2Mix Two-Speaker Training Data, WeSep spk_emb_causal_100 Checkpoint, WeSep tfmap_context_causal_100 Checkpoint (+3 more)

### Community 26 - "PRD Milestones & Delay Gate"
Cohesion: 0.22
Nodes (10): PRD: Live target-speaker listening app, Apple Bluetooth HFP input/output route pairing, Conversation delay gate: p95 <= 150 ms, M0 - feasibility evidence, M3 - audio-route proof, M4 - causal streaming proof, M5 - user prototype, M6 - pilot review (+2 more)

### Community 27 - "Discovery Evaluation Metrics"
Cohesion: 0.20
Nodes (11): Speaker-disjoint Libri3Mix holdout (4 mixtures, 12 pairs), 8 kHz separator bandwidth confound, Discovered track as degraded enrollment, Discovery correctness (splits/merges, oracle audit), Embedding fidelity (cosine to clean enrollment), F2 - reject invalid enrollment, Identification margin, Libri3Mix (+3 more)

### Community 28 - "SepFormer + ECAPA Reference Pipeline"
Cohesion: 0.33
Nodes (10): Pinned model weight restore (fetch_models.py, SHA-256 checked), SpeechBrain ECAPA-TDNN speaker matcher (VoxCeleb), Mixture-fit signed least-squares stream gains, Oracle clean-source stream mapping analysis, Resemblyzer enrollment-to-stream matcher, SepFormer + mixture-fit + ECAPA offline reference pipeline, SpeechBrain SepFormer Libri3Mix (blind 3-stream separator, 8 kHz, Apache 2.0), Per-speaker extraction plus combination (approach b) (+2 more)

### Community 29 - "Spectrogram Comparison Page"
Cohesion: 0.40
Nodes (5): default_files(), load(), main(), spectrogram_png(), wav_b64()

### Community 31 - "Manifest Schema: Enums"
Cohesion: 0.20
Nodes (10): enum, type, enum, type, properties, condition, dataset, split (+2 more)

### Community 32 - "Browser Spectrogram Rendering"
Cohesion: 0.27
Nodes (9): draw() spectrogram column renderer, poll() (/data spectrogram polling), api() fetch helper, draw() spectrogram renderer, pollSpectrum(), pollState(), render(), renderSpeakers() (+1 more)

### Community 34 - "Live App Product Concepts"
Cohesion: 0.29
Nodes (7): Discovery enrollment (learn voices from the live mix), ECAPA-TDNN speaker encoder (pinned), Enrollment recording, Live target-speaker listening app, M2 - offline model decision, M7 - discovery enrollment (next phase), Target speaker extraction (TSE)

### Community 35 - "Adaptation Layers"
Cohesion: 0.27
Nodes (4): into_orig_type(), into_tuple(), main(), MulAddAdaptLayer

### Community 39 - "Upstream Verification Script"
Cohesion: 0.32
Nodes (3): main(), load_audio(), sha256()

### Community 41 - "Real-Data Gates & Datasets"
Cohesion: 0.29
Nodes (6): User real 48 kHz stereo recordings (Mix + Nicker/Mond/Sam enrollments), Downstream cost gate: median drop <= 1 dB, LibriCSS, REAL-T dataset, SI-SDR improvement (SI-SDRi), Single-target gate: median SI-SDRi >= 3 dB

### Community 42 - "Manifest Schema Root"
Cohesion: 0.29
Nodes (6): additionalProperties, description, required, $schema, title, type

### Community 43 - "Manifest Schema: Sources Array"
Cohesion: 0.29
Nodes (7): additionalProperties, required, sources, items, maxItems, minItems, type

### Community 44 - "Manifest Schema: Selected Speakers"
Cohesion: 0.29
Nodes (7): minLength, type, selected_speaker_ids, items, minItems, type, uniqueItems

### Community 45 - "M2 Model Comparison Gates"
Cohesion: 0.50
Nodes (5): M2 model comparison: SpeakerBeam-SS vs three-stream SepFormer, Common 8 kHz SI-SDR scoring path, Controlled turns/absence clip (4 x 0.9 s intervals), Selected-speakers-absent suppression gate, SparseLibriMix

### Community 46 - "Schema: dataset_revision"
Cohesion: 0.50
Nodes (4): description, minLength, type, dataset_revision

### Community 47 - "Schema: speech_license"
Cohesion: 0.50
Nodes (4): speech_license, description, minLength, type

### Community 48 - "Schema: trial_id"
Cohesion: 0.50
Nodes (4): trial_id, description, minLength, type

### Community 49 - "Schema: enrollment_utterance_id"
Cohesion: 0.67
Nodes (3): minLength, type, enrollment_utterance_id

### Community 50 - "Schema: mixture_path"
Cohesion: 0.67
Nodes (3): minLength, type, mixture_path

### Community 51 - "Schema: sample_rate_hz"
Cohesion: 0.67
Nodes (3): sample_rate_hz, const, type

## Ambiguous Edges - Review These
- `SepFormer mixfit: anonymous streams matched via enrollment embeddings` → `ECAPA-TDNN speaker embedding model`  [AMBIGUOUS]
  evaluation/data/listen_next/README.md · relation: conceptually_related_to

## Knowledge Gaps
- **88 isolated node(s):** `Controlled turns clip (05-08)`, `M2_MODEL_COMPARISON.md (methods and measured results)`, `minLength`, `type`, `minLength` (+83 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 314 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **21 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `SepFormer mixfit: anonymous streams matched via enrollment embeddings` and `ECAPA-TDNN speaker embedding model`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `Discovery Enrollment Report` connect `Discovery Enrollment Report` to `M2 Matching & Analysis Scripts`, `Discovery Pipeline Code`, `Discovery Conversation Builder`, `Causal TSE Candidate Models`, `Causal SepFormer & M2 Interim`, `Progress & User Recordings`, `Offline Eval Assets & M1`, `WeSep Checkpoints & Conditioning`, `Discovery Evaluation Metrics`?**
  _High betweenness centrality (0.069) - this node is a cross-community bridge._
- **What connects `Controlled turns clip (05-08)`, `M2_MODEL_COMPARISON.md (methods and measured results)`, `minLength` to the rest of the system?**
  _88 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `M2 Matching & Analysis Scripts` be split into smaller, more focused modules?**
  _Cohesion score 0.05111434108527132 - nodes in this community are weakly interconnected._
- **Why does `SpeakerBeam-SS checkpoint verification report` connect `Data Assets & Listening Comparisons` to `M2 Matching & Analysis Scripts`, `SpeakerBeam-SS Inference`, `Offline Eval Assets & M1`, `Causal SepFormer & M2 Interim`?**
  _High betweenness centrality (0.062) - this node is a cross-community bridge._
- **Should `Discovery Pipeline Code` be split into smaller, more focused modules?**
  _Cohesion score 0.05297297297297297 - nodes in this community are weakly interconnected._
- **Why does `Causal Live-Stage Candidates Survey` connect `Causal TSE Candidate Models` to `Discovery Enrollment Report`, `WeSep Checkpoints & Conditioning`, `Progress & User Recordings`?**
  _High betweenness centrality (0.046) - this node is a cross-community bridge._