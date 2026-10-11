# Targeted Speaker Extraction research

This repository holds model evaluations and a **local browser prototype** for
selected-speaker listening. The longer-term goal is a phone/headphone app. The product goal and gates
are in the [PRD](PRD.md), with current findings in [PROGRESS.md](PROGRESS.md).
No model has been approved for phone deployment yet.

## Local website

The current prototype captures a microphone in the browser, creates profiles
from a 30-second calibration, and streams selected-speaker output from local
SepFormer + ECAPA inference. It also includes a paced Mix.wav demo, Bypass,
profile controls and timing reports. Run on this configured Mac:

```sh
./run_local_web.sh
```

Open **http://127.0.0.1:8765**. See [website instructions](local_web/README.md)
and [measured limitations](local_web/VERIFICATION.md). Default hop 200 ms
overloaded in the website test; optional hop 300 ms completed the short browser
demo at about 601 ms estimated application p95 delay. Microphone/AirPods
latency, automatic-profile quality and sustained operation remain user-test gates.

## Research and setup

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
