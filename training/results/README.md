# Training results

## R1: `finetune_research` (7–8 October 2026)

**Research only.** This model was fine-tuned from the REAL-TSE checkpoint, which has no stated weights licence ([NOTICE](../../evaluation/wesep_eval/checkpoints/pretrained/retrained_finetune_research/NOTICE.txt), local only).

**Run:**
- 20,000 steps, batch 8, 4 s chunks, on the on-prem Mac Studio (Metal) with Python 3.9 and torch 2.8;
- data: LibriSpeech `train-clean-100` + `train-clean-360` conversation simulator;
- validation: `dev-clean` speakers.

Files: [console.log](finetune_research/console.log), [log.jsonl](finetune_research/log.jsonl), [eval_pack_scores.json](finetune_research/eval_pack_scores.json).

**Exported weights** are `retrained_finetune_research/avg_model.pt`, SHA-256 `55706de208459a496644bd775f08bfdc6a4af6ad00032698cfe320cd64f07502`. They load in tensor-only mode. Laptop scores match the server's to 0.01 dB.

### Validation during training (noisy/reverberant simulated conversations, `dev-clean` speakers)

| Step | 2k | 4k | 8k | 12k | 14k | 16k | 18k | 20k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Loss | −2.0 | −3.4 | −4.2 | −5.2 | −5.5 | −5.4 | −5.8 | −5.7 |
| SI-SDRi, target active (dB) | −0.7 | 0.6 | 2.9 | 3.4 | 4.4 | 3.8 | 4.1 | 4.1 |
| Absent suppression (dB) | 1.0 | 9.3 | 9.4 | 8.9 | 12.4 | 13.0 | 12.2 | 13.0 |

Progress flattened after about 14k steps, while the cosine learning rate reached its floor.

### Held-out test pack (clean 4–6 speaker conversations, 15 unseen speakers)

| Model | Live? | Clean enrollment SI-SDRi | Discovered SI-SDRi (p10) | Discovered: alone SI-SDR | Discovered: absent suppression |
|---|---|---:|---:|---:|---:|
| **WeSep R1 (this run)** | causal, ~0.17 s streaming | **12.3** | **10.7 (5.5)** | 24.5 | 12.8 |
| SepFormer competitive | offline, ~2.2 s | 12.1 | 11.9 | 21.2 | 14.1 |
| Released WeSep causal | causal | 3.9 | 2.8 | 14.7 | 2.2 |

**Against the plan's gates (discovered enrollment):**
- quality ≥ +10 dB: ✅ (10.7 dB, p10 > 0);
- absent suppression ≥ 10 dB: ✅ (12.8 dB);
- target alone ≥ 20 dB: ✅ (24.5 dB);
- strict causality: ✅ (0.00);
- clean-vs-discovered drop ≤ 1 dB: ❌ (1.6 dB).

### Real recording (Nicker/Mond/Sam, whole 23 s `Mix.wav`, streaming path at 64 ms chunks)

ECAPA voice match of the output to the chosen person vs the best other person ([evaluate_retrained.py](../../evaluation/wesep_eval/evaluate_retrained.py); listening page `evaluation/data/user_recordings/discovery/retrained_compare.html`, local only):

| Enrollment | Model | Nicker | Sam | Mond |
|---|---|---|---|---|
| Discovered track | released checkpoint | ❌ 0.39 vs 0.45 | ✅ 0.56 vs 0.41 | ✅ 0.44 vs 0.21 |
| Discovered track | **R1** | ✅ **0.60 vs 0.27** | ✅ **0.62 vs 0.31** | ✅ **0.52 vs 0.23** |
| Separate recording | released checkpoint | ❌ 0.24 vs 0.55 | ✅ 0.57 vs 0.30 | ✅ 0.47 vs 0.26 |
| Separate recording | **R1** | ✅ 0.29 vs 0.23 | ❌ 0.35 vs 0.53 | ✅ 0.26 vs 0.21 |

Streaming ran at a real-time factor of 0.53 on the laptop CPU (2 threads).

**What the real recording shows:**
- With discovered enrollment, which is the product flow, R1 extracts all three people. Its margins are clearer than SepFormer's offline 0.46/0.54/0.47.
- With the separate recordings as enrollment it is weaker than before, and Sam goes wrong. On the clean synthetic test pack, clean enrollment was fine (12.3 dB). The likely causes:
  - those recordings come from a different session and microphone than the mix;
  - training used discovered-style enrollments 60% of the time.

  R2 should keep more clean/other-session enrollments and add channel augmentation to enrollments.

### Takeaways for R2

1. **The recipe works.** Training on simulated groups and on target-absent examples, with discovered-style enrollment, took the causal model from about +3 dB to about +11 dB. It also learned to stay silent.
2. **Open gaps:**
   - the discovered-vs-clean drop (1.6 dB);
   - enrollments from a different channel;
   - noisy/reverberant conditions (only 4.1 dB on validation).

   Evaluate on a noisy test pack too.
3. **R2 (`scratch_small`) is the licence-clean path.** Apply the same recipe with these adjustments.
