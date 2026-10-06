# WeSep comparison (causal target-speaker extraction)

This folder compares the [REAL-TSE](https://github.com/REAL-TSE/REAL-TSE-Challenge) BSRNN baselines, built on [WeSep](https://github.com/wenet-e2e/wesep), with the SepFormer selector as the **live** model of the discovery-enrollment flow. It only adds files: the SepFormer scripts, results, and pages elsewhere in `evaluation/` are unchanged. [CAUSAL_CANDIDATES.md](../CAUSAL_CANDIDATES.md) explains why WeSep was the first causal candidate to try.

## What is local only

`.venv/`, `upstream/`, `checkpoints/`, and `outputs/` are Git-ignored ([.gitignore](.gitignore)).

| Item | Source | Notes |
| --- | --- | --- |
| `upstream/REAL-TSE-Challenge` | GitHub, commit `8bde4f147d452de43834b5c0a4efdb85811fd5dd` | MIT. Only `tse_baseline/tse_inference.py` was read for the loading recipe. |
| `upstream/wesep-real-tse` | GitHub, commit `2a540977a348fbaa92e623210505430e2cec608d` | Model code. **No LICENSE file** in this fork; upstream WeSep is Apache-2.0. |
| `checkpoints/pretrained/*` | REAL-TSE Google Drive archive `1M4UqK2A2EeHmQ0pCevYqBgaYn3RvklgC` | **No weights licence stated: evaluation only, never redistributed or shipped.** SHA-256 values are in `checkpoints/SHA256SUMS`. All four load with PyTorch's tensor-only `weights_only=True`. |

The four checkpoints are BSRNN models trained on **clean Libri2Mix train-100** (two speakers, no noise), at 16 kHz, with a 32 ms STFT window and 8 ms hop:

| Checkpoint | Conditioning | Declared causal | Parameters |
| --- | --- | --- | ---: |
| `tfmap_context_causal_100` | TF-map + frame-level context from the enrollment **audio** | yes | 27.2 M |
| `spk_emb_causal_100` | WeSpeaker ECAPA embedding of the enrollment audio | yes | 25.0 M |
| `tfmap_context_100` | as above | no | — |
| `spk_emb_100` | as above | no | — |

The models take enrollment **audio**, not our SpeechBrain ECAPA vectors. Each discovered track, or clean recording, is trimmed to its speech, capped at 8 s, and normalized to −25 dBFS. The normalization is needed because WeSep's enrollment features are level-sensitive: a quiet real-recording track made it fail.

## Setup

```sh
cd evaluation/wesep_eval
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python torch==2.6.0 torchaudio==2.6.0 "numpy<2" scipy soundfile pyyaml \
  silero-vad==5.1.2 torchmetrics thop tableprint pesq pystoi tqdm requests gdown \
  "git+https://github.com/wenet-e2e/wespeaker.git" speechbrain==1.1.1 scikit-learn soxr sounddevice
mkdir -p upstream && cd upstream
git clone https://github.com/REAL-TSE/REAL-TSE-Challenge.git && git clone https://github.com/REAL-TSE/wesep-real-tse.git
cd .. && mkdir -p checkpoints/pretrained
.venv/bin/gdown 1M4UqK2A2EeHmQ0pCevYqBgaYn3RvklgC -O checkpoints/real_tse.zip
unzip checkpoints/real_tse.zip -d checkpoints/pretrained
```

This environment also has SpeechBrain, so the live WeSep demo can run SepFormer discovery and WeSep extraction in one process.

## Scripts

- [run_wesep_experiment.py](run_wesep_experiment.py) runs three things on the six controlled conversations, using the same scoring as `../run_discovery_experiment.py`:
  - a causality check;
  - whole-clip extraction with discovered vs clean enrollment;
  - block + past-context processing with no future audio.

  Results go to `../data/discovery/results/wesep_comparison.json`.
- [run_wesep_user.py](run_wesep_user.py) compares WeSep and SepFormer on the real Nicker/Mond/Sam recording. It writes the local page `../data/user_recordings/discovery/wesep_compare.html`.
- [live_demo_wesep.py](live_demo_wesep.py) is the live demo with WeSep as the extractor. It uses the same flow, controls, and options as [../live_demo.py](../live_demo.py), plus `--model`, `--block` (default 0.5 s) and `--window` (default 1 s), giving about 0.9 s total delay.

```sh
.venv/bin/python live_demo_wesep.py --input-file ../data/user_recordings/mixture/Mix.wav   # simulated live
.venv/bin/python live_demo_wesep.py                                                       # microphone (headphones!)
```

## Speed and causality on this laptop

Measured on an Apple M5 CPU with 4 threads:

- **Speed.** A 32 s clip takes about 11 s, so a whole pass runs at about 0.35× real time. The checkpoints have no stateful streaming API, so live use re-runs the model on block + context each time:

  | Window | Compute |
  | --- | --- |
  | 0.35 s | ≈0.14 s |
  | 1 s | ≈0.31 s |
  | 2 s | ≈0.58 s |
  | 4 s | ≈1.1 s |

  A 0.25 s block therefore keeps up only with about 0.5 s of context, and a 0.5 s block with 1 s.
- **Causality.** Replacing the audio after a cut still changed earlier output slightly, across the whole past:

  | Checkpoint | Largest change |
  | --- | --- |
  | `tfmap_context_causal_100` | ≤0.18% of peak |
  | `spk_emb_causal_100` | up to 8.8% |

  A fully causal model would show only floating-point noise. Block processing never gives the model future audio, so the live system stays causal either way.

Results are summarised in [WESEP_COMPARISON.md](../WESEP_COMPARISON.md).
