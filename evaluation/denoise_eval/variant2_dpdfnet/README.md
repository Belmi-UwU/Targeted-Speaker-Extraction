# Variant 2 — DPDFNet live denoising experiment

Variant 1 is the existing parent-directory experiment: `live_models.py` for
GTCRN/RNNoise/DTLN and `live_denoise.py` for DeepFilterNet3. Variant 2 lives entirely
in this folder, with its own Python environment, preprocessing snapshot, audio
loop, model files, and output directory. Existing Variant 1 files were not moved
or edited; `variant1_snapshot.json` and `verification.json` record the check.

## Start a manual live test

Run from the project root. Stop the previous monitor with `q` first, so there is
one microphone-to-headphone playback route. Connect/select the paired Bluetooth
headset `BSD-APM260008 - Find My` in macOS Sound, and use the same headphone mode,
volume and microphone placement for both variants.

```sh
evaluation/denoise_eval/variant2_dpdfnet/.venv/bin/python -u \
  evaluation/denoise_eval/variant2_dpdfnet/live_dpdfnet.py \
  --model dpdfnet2 \
  --gate-open -50 --gate-close -55 \
  --gate-attack-ms 5 --gate-hold-ms 150 --gate-release-ms 100 \
  --limiter-dbfs -6 --limiter-release-ms 80 \
  --gain 0.35 --meter-interval 1
```

Wait for `LIVE READY` before speaking. Default device selection uses the exact
Bluetooth headset name for both input and output. Fresh device inventory:

```sh
evaluation/denoise_eval/variant2_dpdfnet/.venv/bin/python \
  evaluation/denoise_eval/variant2_dpdfnet/live_dpdfnet.py --devices
```

To use other devices explicitly, pass `--input 'Exact microphone name'` and
`--output 'Exact headphone name'`. IDs can change between connections.

Live controls: type a command then Enter.

| Command | Action |
| --- | --- |
| `1` | DPDFNet2 |
| `2` | DPDFNet4 |
| `b` / `d` | Bypass/restore denoiser; gate and limiter remain active |
| `gate -50 -55` | Change gate RMS open/close thresholds |
| `gate off` | Disable gate |
| `limit -6` | Change model-input sample-peak ceiling |
| `limit off` | Disable limiter |
| `pre off` / `pre on` | Bypass/restore preprocessing |
| `settings` | Show active thresholds |
| `q` | Stop and close microphone/output |

For raw microphone monitoring with matched timing, enter `pre off` then `b`.
Restore the full chain with `d` then `pre on`. Gate/limiter start disabled if
their threshold flags are omitted. Timing parameters can be changed at startup.
Model changes reset model, resampler and gate histories and briefly insert silence.
The model continues advancing during bypass. Live audio stays in memory.

## Implementation

Microphone → stateful soxr resampling → gate → limiter → DPDFNet → stateful soxr
resampling → headphones. All neural work runs in a worker thread. The audio
callback queues and plays samples through bounded buffers. No neural calls are
skipped during silence. Live mode is manual by default, with no automatic model
change or duration limit. `--seconds N` provides optional automatic stopping.

Models are official [Ceva-IP DPDFNet](https://github.com/ceva-ip/DPDFNet) pretrained
ONNX files. Both DPDFNet2 and DPDFNet4 are available locally, verified against
official Hugging Face LFS SHA256 hashes at the pinned revision in
`model_manifest.json`. Apache-2.0 upstream license. They are speech enhancement
models without a target-speaker enrollment. Noise removal, background talker
removal and nonverbal vocalization removal have not been rated on this microphone.

The pinned `dpdfnet==0.6.0` streaming API maintains recurrent state. It is called
only at the native model rate 16 kHz, avoiding its stateless per-call resampling
path. Native window 320 samples/20 ms, hop 160 samples/10 ms. One silent hop primes
overlap-add coverage. The official offline path compensates four additional
model hops; direct correlation checks on Mix.wav confirm an 800-sample/50 ms
shift in this streaming adapter for both models. Raw bypass uses that same shift.

The helper uses the full streaming enhanced output with no dry-signal blend or
attenuation-limit blend. Native streaming and the package's centered offline
`enhance()` API are different processing paths. Their outputs are not expected
to be identical. Variant 2 offline tests intentionally use the **live** path.

Playback prefill is 170 ms by default, matching Variant 1. This and the 50 ms sample
shift are configuration components; capture blocks, resampling and device/
Bluetooth scheduling also affect listener delay. Physical microphone-to-ear
latency is unmeasured. A delayed copy of one's own voice is expected when using
this live monitor. This experiment has no acoustic echo cancellation.

Logs show raw RMS, raw sample peak, near-fullscale input sample count, preprocessed
RMS/peak, gate state, limiter gain, playback clipping, underruns and drops.
Near-fullscale input count is a diagnostic hint, not proof of hardware clipping;
the headset's internal DSP may already have changed the signal. `clipped` counts
output samples at the final ±0.98 playback safety cap. Compute timings exclude
preprocessing and physical device latency.

A JSON session report is saved automatically under `outputs/` after stopping,
including thresholds/edits, device names, audio counters and PortAudio-reported
latencies. Filenames and timestamps use Asia/Jakarta. Backend-reported latency
is not a physical end-to-end measurement. `--log-file PATH` overrides its location.

## Offline verification and outputs

```sh
evaluation/denoise_eval/variant2_dpdfnet/.venv/bin/python -u \
  evaluation/denoise_eval/variant2_dpdfnet/live_dpdfnet.py \
  --all --test-file evaluation/data/user_recordings/mixture/Mix.wav \
  --output-dir evaluation/denoise_eval/variant2_dpdfnet/outputs/baseline

evaluation/denoise_eval/variant2_dpdfnet/.venv/bin/python \
  evaluation/denoise_eval/variant2_dpdfnet/verify_experiment.py
```

Completed on this MacBook: both model outputs are finite, with exact model-rate
duration after sample-rate rounding (23.1253125 s). Median model compute on the
full baseline was 0.769 ms per 10 ms hop for DPDFNet2, 1.225 ms for DPDFNet4. Both
also completed the user's gate -50/-55, limiter -6 configuration. Output FLOAT WAVs,
model-input WAVs and JSON measurements are in `outputs/baseline/` and
`outputs/preprocessed/`. These results establish execution and CPU headroom,
not subjective quality or superiority to Variant 1.

Verification checks exact raw-bypass alignment, neural output alignment,
STFT identity reconstruction including stream edges, and unchanged Variant 1
file hashes. Bluetooth was absent from the fresh device inventory during setup;
hardware capture/playback for Variant 2 and listening quality remain untested.

## Files and reproducibility

- `live_dpdfnet.py`: manual live/offline CLI and audio worker.
- `dpdfnet_model.py`: official streaming API adapter and model checksum checks.
- `audio_preprocessing.py`: independent snapshot of gate/limiter code.
- `audio_io.py`: independent device resolver and playback buffer.
- `setup_models.py`: official pinned downloads and SHA256 verification.
- `requirements.txt`, `environment_manifest.json`: dependency versions.
- `model_manifest.json`: downloaded model identities.
- `verification.json`: alignment and Variant 1 preservation results.

Recreate the environment using Python 3.11 or later:

```sh
uv venv --python python3.11 evaluation/denoise_eval/variant2_dpdfnet/.venv
uv pip install --python evaluation/denoise_eval/variant2_dpdfnet/.venv/bin/python \
  -r evaluation/denoise_eval/variant2_dpdfnet/requirements.txt
evaluation/denoise_eval/variant2_dpdfnet/.venv/bin/python \
  evaluation/denoise_eval/variant2_dpdfnet/setup_models.py
```

Models, environments and audio outputs are ignored by Git. Variant 1 keeps using
its original environment and commands.
