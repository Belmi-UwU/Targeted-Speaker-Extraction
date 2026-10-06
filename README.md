# Targeted Speaker Extraction research

This repository holds an offline evaluation of speaker extraction and
separation for a proposed phone/TWS listening app. The product goal and gates
are in the [PRD](PRD.md), with current findings in [PROGRESS.md](PROGRESS.md).
No model has been approved for real-time use yet.

- [Evaluation code and methods](evaluation/README.md)
- [SpeakerBeam-SS compatibility check](speakerbeam_ss_verification/VERIFICATION_REPORT.md)
- [Causal SepFormer experiment](evaluation/CAUSAL_SEPFORMER_EXPERIMENT.md)
- [Local model and audio setup](ASSETS.md)

Quick setup on a compatible Python 3.12 system:

```sh
python3.12 -m venv speakerbeam_ss_verification/.venv
speakerbeam_ss_verification/.venv/bin/python -m pip install -r speakerbeam_ss_verification/requirements.lock.txt
speakerbeam_ss_verification/.venv/bin/python fetch_models.py
```

The model weights and evaluation recordings are kept outside Git. Follow
[ASSETS.md](ASSETS.md) to restore the pinned models and build the evaluation
audio. Personal recordings are local-only.
