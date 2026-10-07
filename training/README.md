# Training: causal WeSep on a Mac Studio (Metal)

This folder implements R0–R2 of the [retraining plan](../evaluation/WESEP_RETRAINING_PLAN.md) as one Jupyter notebook for an on-prem Apple Silicon machine using PyTorch's Metal (MPS) backend. The training data is English LibriSpeech only.

| File | Purpose |
| --- | --- |
| [wesep_retrain.ipynb](wesep_retrain.ipynb) | The notebook to upload: setup → data → simulator → training → evaluation → export |
| [wesep_retrain.py](wesep_retrain.py) | Same notebook in percent format (source of truth). Rebuild with `python training/build_notebook.py`. |
| [make_eval_pack.py](make_eval_pack.py) | Builds `eval_pack.zip` (64 MB) with the held-out test conversations and baseline numbers |
| [build_notebook.py](build_notebook.py) | Converts the `.py` source to `.ipynb` |
| [run_until_done.sh](run_until_done.sh) | Optional unattended runner: restarts training after any crash until it is done |

## Which mode first, with limited compute

Run **R1 first, then R2 small**:

1. **`MODE = "finetune_research"` (R1).** About 20k steps from the released REAL-TSE checkpoint.
   - It converges fastest, so it answers the key question cheaply: does training on groups, absent targets, and discovered-style enrollments fix what the released model gets wrong?
   - Its weights inherit the checkpoint's missing licence, so they are **evaluation only** and never ship.
2. **`MODE = "scratch_small"` (R2).** If R1 shows the recipe works, train the 11.5 M-parameter model from scratch on licence-clean data.
   - This is the shippable path.
   - It is also the size that suits a phone.
   - The context speaker encoder stays frozen at its pretrained ECAPA weights, which makes training cheaper.
3. **`MODE = "scratch_base"`** (27 M parameters) only if small falls short and compute allows.

## Steps

**On the laptop**, build the evaluation pack:

```sh
speakerbeam_ss_verification/.venv/bin/python training/make_eval_pack.py   # writes training/eval_pack.zip
```

Your personal recording is left out unless you add `--include-user-recording`.

**Upload to the server:**
- `wesep_retrain.ipynb` and `eval_pack.zip`;
- for R1 only, the folder `evaluation/wesep_eval/checkpoints/pretrained/tfmap_context_causal_100/`.

**On the server** (Mac Studio, Python 3.10–3.12, Jupyter):

1. Put the files under the notebook's `WORK` folder (default `~/tse_training`):
   - unzip `eval_pack.zip` there, giving `WORK/eval_pack/index.json`;
   - for R1, place the checkpoint folder at `WORK/pretrained/tfmap_context_causal_100/`.
2. In the **Config** cell, choose `MODE`. Run once with `SMOKE = True` to check everything (a few minutes), then set `SMOKE = False` and **Run All**.
   - The first run installs packages, clones WeSep at a pinned commit, downloads LibriSpeech with MD5 checks, and indexes it.
   - Data needs about 60 GB of disk: archives plus extracted `train-clean-100`, `train-clean-360` and `dev-clean`. Delete `WORK/data/archives` after extraction to reclaim about 30 GB.
3. Keep the machine awake during training, e.g. run `caffeinate -dims` in a terminal.
4. Training checkpoints to `WORK/runs/<MODE>/last.pt` every 1,000 steps and at least every 10 minutes. If anything stops it, re-run all cells and it resumes (see *Crash safety* below).
   - Progress and validation are logged to `log.jsonl` and printed every 50 steps (validation every 2,000).
   - Each line includes `eta_hours`.
5. When training finishes, the notebook:
   - scores the best checkpoint on the evaluation pack, beside the SepFormer and released-WeSep baselines;
   - checks causality and inference speed;
   - exports to `WORK/runs/<MODE>/export/retrained_<MODE>/`.

**Back on the laptop**, copy the exported folder into `evaluation/wesep_eval/checkpoints/pretrained/` and run:

```sh
cd evaluation/wesep_eval
.venv/bin/python live_demo_wesep.py --model retrained_finetune_research --input-file ../data/user_recordings/mixture/Mix.wav
```

## Crash safety

A crash, kernel death, reboot, or power loss costs at most about 10 minutes of training:
- **Checkpoints:** `last.pt` is written every 1,000 steps and at least every 10 minutes, via a temporary file and an atomic rename, so it is never half-written. The previous one is kept as `prev.pt`. If `last.pt` is ever unreadable, resume falls back to `prev.pt`.
- **Resume:** re-running all cells skips finished work (downloads, extraction, index, reverb bank, validation set) and resumes training at the saved step. Log lines from lost steps are dropped.
- **Downloads and extraction** land under a temporary name and appear only when complete. A corrupt archive is deleted automatically, so a re-run downloads it again.
- **Divergence:** a batch with a non-finite loss or gradient is skipped. After 20 in a row, training stops without saving the bad weights and writes `RUN/DIVERGED`.

**Unattended runs.** [run_until_done.sh](run_until_done.sh) runs the same code as a script, keeps the Mac awake, and restarts it after any crash until `RUN/DONE` exists. It never restarts a diverged run.

```sh
cd training && TSE_MODE=finetune_research PYTHON=$(which python) ./run_until_done.sh
```

**Tested on the laptop:**
1. Training was killed with `kill -9` at step 15, and `last.pt` was then deliberately truncated.
2. The runner restarted on its own, reported `last.pt` as unreadable, and resumed from `prev.pt` at step 10.
3. It finished all steps with a clean log.

## Time estimates

Measured on an Apple M5 laptop (MPS):

| Mode | Seconds per step (batch 8, 4 s) | Steps | Laptop total |
| --- | ---: | ---: | ---: |
| `finetune_research` | ≈4 | 20k | ≈1 day |
| `scratch_small` | ≈1.4 | 150k | ≈2.5 days |
| `scratch_base` | ≈4 | 200k | ≈9 days |

Step times are extrapolated from batch-4 and batch-2 runs; data simulation runs in parallel CPU workers. A Mac Studio Max has several times the laptop's GPU cores and should be noticeably faster; the `eta_hours` printout gives the real number after the first 50 steps.

## Checked locally

Both `finetune_research` and `scratch_small` ran end to end in smoke mode on the laptop (MPS, 20 steps, `test-clean` speakers split into train/validation), with no errors:
- data simulation, training, resume, validation, pack evaluation with baselines, the causality check, and export all ran;
- `live_demo_wesep.py` loaded the exported model.

Smoke scores mean nothing: 20 steps is far too few.

The from-scratch small model passed the strict causality check: exactly 0 change once Kaldi fbank dither is off for evaluation, which the notebook now does. Its live inference on MPS took 0.036 s per 1 s of audio. Exported models load in the streaming live demo, `evaluation/wesep_eval/live_demo_streaming.py --model <name>`, with about 0.17 s delay.

When fine-tuning, validation may first dip below the released model while it adapts to the new scale-dependent loss, which also teaches silence when the target is absent. Judge R1 at the later validation points.

## Licences

- **Code and data:** WeSep code is Apache-2.0 (pinned commit `dc11cb1`). LibriSpeech is CC BY 4.0.
- **Speaker encoder:** WeSpeaker VoxCeleb ECAPA512 (Hugging Face revision `89426e1`). Its model card says Apache-2.0; WeSpeaker's docs say CC BY 4.0. Keep attribution either way.
- **R1 weights** come from the REAL-TSE checkpoint, which has no stated licence: research only. The export writes this into `NOTICE.txt`.
