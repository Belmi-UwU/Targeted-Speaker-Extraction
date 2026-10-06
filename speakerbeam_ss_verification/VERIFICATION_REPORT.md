# SpeakerBeam-SS checkpoint verification

## Decision

**Checkpoint compatibility and offline inference passed. Paper reproduction, extraction quality, and streaming readiness are not established.** The tested project is an independent reimplementation, not an official release from the paper's authors. Use it only as an explicitly labeled experimental offline baseline for now.

No evaluation datasets were downloaded. Inputs are four small WAV files bundled with the code repository: a mixture, two enrollment recordings, and an existing output. The checkpoint was downloaded individually from a Hugging Face dataset repository; its audio datasets were not downloaded. No evaluation notebook was created.

## Sources and licensing

| Component | Source / pinned version | License finding |
| --- | --- | --- |
| Implementation | [OpenSpeakerBeam-SS](https://github.com/helloooideeeeea/OpenSpeakerBeam-SS/tree/bbbae73f74fc14eafdf7abb93f9094eafccd21df), commit `bbbae73f74fc14eafdf7abb93f9094eafccd21df` | Repository MIT license, copyright 2025 Yasushi Sakita; saved in `upstream/LICENSE`. |
| TSE checkpoint | [best_model.pth](https://huggingface.co/datasets/helloidea/OpenSpeakerBeam-SS-dataset/blob/82e6d7646126d1fa33b30c0b44f1d076496cca71/checkpoints/best_model.pth), revision `82e6d7646126d1fa33b30c0b44f1d076496cca71` | No separate license declaration or model card was found in the inspected Hugging Face repository metadata/tree. The code's MIT license alone does not establish a separate weight license. |
| Speaker encoder | [Resemblyzer](https://github.com/resemble-ai/Resemblyzer), installed package `0.1.4`, bundled `pretrained.pt` | Separate dependency; repository advertises Apache-2.0. This check does not establish a separate grant for all pretrained assets or training data. |
| Demo audio | [Pinned repository samples](https://github.com/helloooideeeeea/OpenSpeakerBeam-SS/tree/bbbae73f74fc14eafdf7abb93f9094eafccd21df/data/sample) | No separate audio provenance/license established by this check. |
| Original research | [Sato et al., Interspeech 2024](https://www.isca-archive.org/interspeech_2024/sato24_interspeech.pdf) | Research reference; not the source of the tested weights. |

The maintainer identifies this as a trained checkpoint and explicitly disclaims affiliation with the original authors. Exact training history, data splits, and paper-equivalent performance are not independently verified. The file contains a state dictionary, not training logs or optimizer/epoch metadata.

Checkpoint size: 33,050,566 bytes. SHA-256, checked against Hugging Face LFS metadata:

```text
41c250587af2e947dc3e265cd6e7aa7de230048d408e93c53d8348d63cf75896
```

The separately loaded Resemblyzer checkpoint SHA-256 is `39373b86598fa3da9fcddee6142382efe09777e8d37dc9c0561f41f0070f134e`. Source metadata is saved in `hf_info.json` and `hf_tree.json`; input hashes are in the result JSON.

## What ran

- macOS arm64, Python 3.12.10, CPU with four PyTorch threads.
- Unmodified upstream model and enrollment helper; local verification harness uses SoundFile for WAV I/O.
- PyTorch 2.6.0, torchaudio 2.6.0, Asteroid 0.7.0, Resemblyzer 0.1.4.
- `torch.load(..., weights_only=True, map_location="cpu")`, then `load_state_dict(strict=True)`.
- All 136 state dictionary entries matched; zero missing or unexpected keys.
- TSE model: 8,249,609 parameters, excluding the separate speaker encoder. This differs from the README's 7.64M claim.
- Input: mono 16 kHz, 10-second mixture, plus 10.4-second enrollment. This path needs one mixture channel, not a microphone array.
- Output: `[1, 1, 160000]`, finite and nonzero. Full-clip model forward took approximately 0.113 seconds in the final run. This excludes enrollment computation, loading, and audio I/O and is not a streaming/mobile latency benchmark.
- A different enrollment changes the output (relative L2 difference 1.173). This shows enrollment dependence, not correct target selection.
- `pip check` passed. Initial installation failed because a dependency imports `pkg_resources`; pinning setuptools to 80.9.0 resolved it.

Detailed results: `outputs/verification_results.json`; execution log: `verification.log`.

## Differences from the original paper

The paper uses a jointly trained speaker encoder, channel-wise normalization for causality, X=2 convolution repetitions, and a 512-wide S4D feed-forward hidden layer. It describes recurrent SSM inference and sweeps frontend widths up to 4096. See Sections 3–5 of the [paper](https://www.isca-archive.org/interspeech_2024/sato24_interspeech.pdf).

The inspected implementation instead:

1. Supplies separately pretrained 256-dimensional Resemblyzer embeddings.
2. Uses Asteroid `gLN` in convolution blocks, even with `causal=True`. Global normalization includes the time dimension and allows future input to affect earlier output.
3. Builds three single Conv1DBlock/S4DBlock pairs before adaptation and one afterward, with a different dilation arrangement; its S4D feed-forward hidden width is 256. Its own source describes the block as an interpretation.
4. Processes a complete sequence using FFT convolution and returns no carried S4D state. No end-to-end incremental streaming interface is provided in the inspected inference path. FFT convolution itself is not inherently noncausal.
5. Defaults to a 4096-channel frontend. The tested checkpoint cannot be assumed equivalent to any paper table row or its reported scores/latency.

Source inspection: `upstream/model/__init__.py`, `upstream/model/s4d.py`, `upstream/tools/__init__.py`, and `upstream/train.py`.

## Concrete limitations and blockers

**Causality failed:** holding enrollment and the first five seconds of mixture fixed, zeroing the last five seconds changed the first second of output by relative L2 **0.2017** (20.17%). This rules out treating this full-clip implementation as a causal model. Replacing normalization or changing architecture requires retraining/fine-tuning and new evaluation; simply splitting audio into chunks does not restore the paper model.

**Output scaling needs investigation:** raw RMS is 4.0834 and peak magnitude is 29.0440. Saving directly as ordinary PCM would clip. `enhanced_demo.wav` preserves the raw float output; use `enhanced_demo_normalized_preview.wav` for listening (peak scaled to 0.9). Normalization fixes playback scale only, not extraction quality.

**Bundled result explained by clipping:** direct correlation with the repository's existing output is 0.7505 and raw RMSE is 3.6118 because the supplied result is limited to the audio range [-1, 1]. Clipping our raw output to that range yields correlation above 0.99999999999, RMSE approximately 0.00000122, and maximum absolute difference below one 16-bit PCM step. About 38.32% of raw samples lie outside that range. Thus the bundled file is consistent with a clipped version of this model's output; it is not a clean target and cannot establish extraction accuracy. The upstream save path does not explicitly document clipping, so this is an empirical comparison rather than a confirmed account of how that file was made.

**Quality unverified:** no isolated clean target was supplied in these downloaded demos, so no SI-SDR/SI-SNR improvement or intelligibility score is claimed. Only the aligned 10-second example was checked; arbitrary input lengths, absent-target behavior, noisy phone/TWS audio, and long-running inference remain untested.

**Deployment blockers:** weight licensing/provenance needs clarification, and this architecture/inference path is unsuitable for unmodified causal streaming. Mobile conversion and phone/TWS latency have not been tested.

An additional public candidate, [gabrielmbmb/voce-speakerbeam-ss-librimix-train360](https://huggingface.co/gabrielmbmb/voce-speakerbeam-ss-librimix-train360), was inspected at metadata level. Compatible source/configuration and license were not established in this check, so its weights were not downloaded or tested. Saved metadata does not imply endorsement or compatibility.

## Reproduce locally

From this directory, the prepared environment and files can be reused:

```sh
.venv/bin/python verify.py
```

To recreate the environment on a compatible system:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python verify.py
```

`requirements.lock.txt` records the full installed environment; `requirements.txt` records the principal pins. Upstream model files, demo inputs, and the downloaded checkpoint are retained locally. Rerunning overwrites verification outputs. No training or dataset-download scripts are run by `verify.py`.

## Suggested next step

Resolve the checkpoint's provenance/license and raw-output gain before relying on this as a baseline. The demo discrepancy is explained by clipping, but extraction quality remains unmeasured. In an offline evaluation notebook, label the model **OpenSpeakerBeam-SS independent variant** and keep paper reproduction and causal streaming as separate, unresolved milestones.
