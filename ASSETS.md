# Local data and model weights

This repository stores the project code, manifests, model configuration,
provenance, scores, and reports. Large model weights, source audio, generated
WAVs, Python environments, and the user's Nicker/Mond/Sam recordings remain
on the local computer. They were removed **only from Git tracking**, not from
the working folder. This prevents GitHub size failures and avoids publishing
personal recordings. The original model and dataset sources are linked below.

## Model weights

After installing the dependencies in
[`speakerbeam_ss_verification/requirements.lock.txt`](speakerbeam_ss_verification/requirements.lock.txt),
run from the repository root:

```sh
speakerbeam_ss_verification/.venv/bin/python fetch_models.py
```

This restores the [SpeechBrain SepFormer Libri3Mix](https://huggingface.co/speechbrain/sepformer-libri3mix)
and [SpeechBrain ECAPA VoxCeleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
weights to the paths used by the evaluation scripts. It pins revisions and
checks SHA-256. For the optional
[OpenSpeakerBeam-SS checkpoint](https://huggingface.co/datasets/helloidea/OpenSpeakerBeam-SS-dataset),
add `--speakerbeam`. Its weight licensing remains unresolved, as documented
in the [verification report](speakerbeam_ss_verification/VERIFICATION_REPORT.md).
Use `--offline-check` to verify weights already on disk without downloading.

The small model cards, `hyperparams.yaml`, and config files remain in Git.
The inference programs still require the model weights locally.

## Evaluation audio

The clean evaluation trials were derived from the official
[LibriSpeech `test-clean` archive](https://www.openslr.org/12) and pinned
LibriMix metadata, as described in the [evaluation README](evaluation/README.md).
Download `test-clean.tar.gz` from OpenSLR into
`evaluation/data/archives/` to regenerate the evaluation audio. The archive's
published MD5 is `32fa31d27d2e1cad72775fee3f4849a9eca928`. The existing
build scripts regenerate the trial audio from it; the tracked manifests and
reports preserve the experiment results. The archive and generated audio are
intentionally not committed.

`evaluation/data/user_recordings/` contains personal recordings and their
derived outputs. A new clone cannot recreate them without the user's files.
The [user-recording guide](evaluation/USER_RECORDINGS.md) gives the expected
input layout.
