# DeepFilterNet3: discovery preprocessing and live listening

Environment: Python 3.11, DeepFilterNet/DeepFilterLib 0.5.6, Torch/torchaudio 2.6.0,
NumPy 1.26.4, sounddevice 0.5.6, soxr 1.1.0. Model is the official cached
DeepFilterNet3 `model_120.ckpt.best`, separate from the WeSep fine-tuning experiment.

From the repository root:

```sh
evaluation/denoise_eval/.venv/bin/python evaluation/denoise_eval/live_denoise.py --devices
evaluation/denoise_eval/.venv/bin/python -u evaluation/denoise_eval/live_denoise.py \
  --input 'BSD-APM260008 - Find My' --output 'BSD-APM260008 - Find My' --gain 0.35
```

Speak into the selected microphone and listen through headphones. Type `b` then
Enter for bypass, `d` then Enter for denoising, `q` then Enter or Ctrl+C to stop.
The script opens a new PortAudio session and resolves devices by name each run.
If the headset is absent, connect/select it in macOS Sound Output before launching.
The user selected the Bluetooth headset microphone, not the MacBook microphone.

## Implementation and limits

- Live route: microphone → resampler → DeepFilterNet3 → resampler → headphones.
- DF3 uses 48 kHz. The headset duplex route tested at 24 kHz; soxr carries its
  resampling state between chunks.
- Python listening prototype: 200 ms blocks, 800 ms past context recomputed for
  each block, 40 ms right-edge guard for model/STFT lookahead, 12 dB attenuation
  limit, post-filter disabled. It is not the upstream Rust frame-by-frame runtime.
- Bypass has the same context/guard alignment as denoised audio.
- Buffer target + guard is 390 ms before resampling and physical device/Bluetooth
  latency. Physical microphone-to-headphone delay has not been measured.
- Audio stays in memory in live mode; microphone recordings are not saved.
- The audio callback only queues/plays samples; model processing runs in a worker.
  Bounded queues report dropped input, playback drops, underruns, and audio status.

## Verification on 12 October 2026, Asia/Jakarta

- Offline block test on Mix.wav: finite output, exact 23.125 s aligned duration;
  compute median 21 ms, p90 24 ms per 200 ms block.
- Bluetooth duplex 24 kHz open/close succeeded.
- First launch could not find the headset after it disappeared from CoreAudio.
  Selecting the paired headset in macOS Sound Output made it available again.
- A 12 s live test then ran successfully: compute median about 22 ms per 200 ms,
  zero reported input drops, playback drops, underruns, or audio status events.
  Streams were closed automatically. The user confirmed hearing their voice
  through the headphones during this test. Noise reduction and speech quality
  versus matched-delay bypass have not yet been rated.

Offline reproduction (does not open audio devices):

```sh
evaluation/denoise_eval/.venv/bin/python evaluation/denoise_eval/live_denoise.py \
  --test-file evaluation/data/user_recordings/mixture/Mix.wav \
  --output-file /private/tmp/df3_live_block_test.wav
```

## Listening feedback, 13 October 2026

The user reported residual table taps, hand claps, and quiet/distant background
speech. No live process was running during the subsequent check, so the exact
arguments of that listening session were not recovered. The script default is
12 dB attenuation, post-filter off.

The upstream attenuation limiter mixes `10**(-limit/20)` of the original
spectrum with enhanced audio: about 25.1% at 12 dB, 6.3% at 24 dB. These are
signal mixture weights, not measured noise reduction or guarantees. The next
controlled listening trial increases only `--attenuation` to 24, retaining the
same microphone, gain, block/context and post-filter settings. Its subjective
result has not yet been rated.

Test tapping/clapping both during silence and while speaking, and check that
word beginnings/endings remain intact. Background speech is speech interference;
this model has no target identity/enrollment input, so quieter/distant speakers
may be retained. Their removal is not guaranteed by increasing attenuation.

## GTCRN / RNNoise / DTLN live comparison (13 October 2026)

Run from the repository root, using the same Bluetooth microphone and headphones:

```sh
evaluation/denoise_eval/.venv/bin/python -u evaluation/denoise_eval/live_models.py
```

The initial model is GTCRN. Enter a command followed by Enter:

| Command | Action |
| --- | --- |
| `1` | GTCRN |
| `2` | RNNoise v0.1.1 |
| `3` | DTLN |
| `b` | Raw microphone bypass with matching sample delay |
| `d` | Denoising |
| `q` | Stop and close audio devices |

Switching resets model/resampler history and briefly inserts silence. Bypass
keeps the model state advancing. Gain remains 0.35 across models; there is no
automatic loudness normalization or extra noise gate. Live audio is not recorded.
This monitor is separate from SepFormer and WeSep.

Automatic sequence: GTCRN → RNNoise → DTLN, 20 seconds each, then stop:

```sh
evaluation/denoise_eval/.venv/bin/python -u evaluation/denoise_eval/live_models.py \
  --suite-seconds 20 --log-file evaluation/denoise_eval/outputs/live_suite.json
```

Implementation uses the official models and streaming inference:

- [GTCRN](https://github.com/Xiaobin-Rong/gtcrn): official simplified streaming
  ONNX, 16 kHz, FFT512/hop256, periodic sqrt-Hann overlap-add, carried convolution,
  attention and recurrent caches. No block context recomputation.
- [RNNoise](https://github.com/xiph/rnnoise/tree/v0.1.1): **v0.1.1**, 48 kHz,
  480-sample frames, persistent C denoiser state. Compiled locally with clang;
  float audio is scaled to/from 16-bit PCM amplitude units as required by its API.
  The current upstream model archive is about 56 MB and downloaded at only a few
  kB/s, so that download was stopped. This trial does **not** represent the latest
  RNNoise weights or its newer transient-noise training data.
- [DTLN](https://github.com/breizhn/DTLN): official pretrained two-part ONNX,
  16 kHz, FFT512/hop128, persistent states in both stages; overlap-add follows
  `real_time_processing_onnx.py`, without adding an analysis Hann window.

Pinned source commits, file SHA256 hashes, dependency versions and RNNoise build
command are in `models_manifest.json`. Third-party repositories retain their
upstream licenses and are ignored by Git. The existing Python 3.11 environment
was extended with ONNX Runtime 1.31.0. To recreate dependencies after setting up
the DF3 environment: `uv pip install --python evaluation/denoise_eval/.venv/bin/python onnxruntime==1.31.0`.

Offline streaming check:

```sh
evaluation/denoise_eval/.venv/bin/python -u evaluation/denoise_eval/live_models.py \
  --all --test-file evaluation/data/user_recordings/mixture/Mix.wav
```

Measured on this MacBook, Mix.wav (about 23.125 s):

| Model | Hop | Median compute per hop | p90 |
| --- | --- | --- | --- |
| GTCRN | 16 ms | 0.409 ms | 0.464 ms |
| RNNoise v0.1.1 | 10 ms | 0.077 ms | 0.095 ms |
| DTLN | 8 ms | 0.094 ms | 0.098 ms |

All outputs were finite, preserved duration within sample-rate rounding, and were
saved as FLOAT WAV in ignored `outputs/`. This establishes execution and compute
headroom, not speech quality or measured noise suppression. GTCRN's streaming
STFT path reconstructed an identity spectrum with maximum error 1.49e-8,
including the delayed tail, which checks windowing and frame boundaries.

Raw bypass sample shifts: GTCRN16 ms, RNNoise10 ms, DTLN24 ms (window minus hop).
These are sample alignment shifts, not full algorithmic or physical latency.
Playback prefill is 170 ms; capture blocks are 20 ms, with additional resampler,
device and Bluetooth latency. Physical microphone-to-headphone delay is unmeasured.
DTLN needs a 32 ms window; its 8 ms hop is not an 8 ms total latency claim.

Listening protocol: same microphone, gain, headphone listening mode and room;
test taps/claps during both silence and speech, then word beginnings/endings,
volume stability, distant speech and perceived delay. Headphone Transparency can
let room sounds reach the ear outside the processed audio path, so record the
headphone mode and avoid changing it between models. None of these models takes
a target-speaker enrollment; suppression of distant/background speech is not
guaranteed.

### Completed live execution check

All three models ran sequentially for 20 seconds each through Bluetooth duplex
24 kHz, gain0.35, and exited successfully with microphone/output closed. Final
suite counters: zero underruns, input drops, playback drops, audio status events
and clipped samples. Median compute per hop was GTCRN1.706 ms, RNNoise0.328 ms,
DTLN0.550 ms; `outputs/live_suite.json` holds the detailed measurements. These
times are computation only, not microphone-to-ear latency.

Headphone availability initially disappeared; selecting the paired Bluetooth
output again in macOS Sound restored it. Sound Output then showed volume38%,
mute off. The headphone's Transparency listening mode was observed before the
test and was not changed. The user has not yet confirmed hearing the alternate
models or rated speech/noise quality; no best-model conclusion is available.

The user subsequently clarified they had not personally tried the sequence and
requested manual tests one model at a time. Treat the earlier sequence as an
execution check only. They authorized headphone listening mode Off; a fresh
System Settings inspection showed Off already selected, so no mode change was
needed. Subjective quality remains unrated.

Manual tests: run `live_models.py --model gtcrn`, wait for `LIVE READY`, and test
speech, taps and claps. Type `b` + Enter for bypass, `d` + Enter for denoise,
and `q` + Enter to stop. Repeat with `--model rnnoise`, then `--model dtln`.
Without `--suite-seconds` there is no automatic model change or time limit.

## Optional gate and limiter BEFORE the denoiser

Both `live_models.py` and the DF3 `live_denoise.py` now support the same optional
preprocessing. Route: microphone → stateful resampling to model rate → RMS gate
→ sample-peak limiter → denoiser → resampling → headphone. Thresholds are digital
dBFS, not acoustic dB SPL. No input gain boost, audio-frame deletion, inference
skipping or extra lookahead/buffering was added. This changes model input audio;
it does not reduce the number of neural-network operations.

Start with the original denoiser alone and update the input preprocessing live:

```sh
evaluation/denoise_eval/.venv/bin/python -u evaluation/denoise_eval/live_models.py \
  --model gtcrn --meter-interval 1
```

Wait for `LIVE READY`. First observe `in ... dBFS` while silent and speaking softly.
Choose thresholds above background noise and below the softest speech. If those
levels overlap, an amplitude gate cannot separate them reliably. Example trial
settings (not calibrated for this microphone):

```text
gate -50 -55
limit -6
```

Enter each line separately in the running Terminal. `gate` sets the RMS open and
close thresholds. Open must be >= close; e.g. -50 opens above -50, remains open
through the -50 to -55 band, and closes after the input drops below -55 for the
hold time. Detection uses a 10 ms exponential mean-square envelope. Gain fades
with attack/release time constants; closure approaches silence gradually. It
has no pre-roll, so soft word beginnings can be attenuated. Lower the opening
threshold if words disappear. Hold150 ms and release100 ms are initial settings,
not guarantees of preserving every word.

The limiter measures sample peak **after the gate**, attacks immediately and
releases with an 80 ms time constant. -6 dBFS limits the model input peak to
about0.501 amplitude. It attenuates loud audio; it does not raise quiet samples
to a minimum, remove claps, repair microphone clipping, or guarantee a bound on
the denoiser's output. Without lookahead, extreme transients may cause audible
gain changes. The existing final playback clip at±0.98 remains separate.

Live commands (then Enter):

| Command | Meaning |
| --- | --- |
| `gate -50 -55` | Set gate open/close and activate preprocessing |
| `gate -50` | Open -50, close defaults to -55 |
| `gate off` | Disable only the gate |
| `limit -6` | Set sample-peak ceiling and activate preprocessing |
| `limit off` | Disable only the limiter |
| `pre off` / `pre on` | Bypass/restore both using the stored settings |
| `p` | Toggle preprocessing bypass |
| `settings` | Print active thresholds |
| `b` / `d` | Bypass/restore **denoiser only**; preprocessing still applies |
| `q` | Stop |

With preprocessing enabled, `b` now lets you hear the preprocessed audio alone.
For the fully raw matched-delay microphone use `pre off` then `b`; use `d` then
`pre on` to return to the complete chain. Compare `pre off/on` with denoising
left on to isolate preprocessing's effect. Command edits are validated and
applied in the worker between blocks. Invalid edits preserve the active settings.
Threshold changes reset gate/limiter state and can briefly fade the signal.

Equivalent startup flags:

```sh
evaluation/denoise_eval/.venv/bin/python -u evaluation/denoise_eval/live_models.py \
  --model gtcrn --gate-open -50 --gate-close -55 --limiter-dbfs -6 \
  --gate-attack-ms 5 --gate-hold-ms 150 --gate-release-ms 100 \
  --limiter-release-ms 80 --meter-interval 1
```

Gate and limiter are independently off when their threshold flags are omitted.
Use `--model rnnoise` or `--model dtln` for the other models. DF3 accepts the same
flags on `live_denoise.py` (include `--gain 0.35` to match the three-model monitor).
Model switching resets gate envelope/gain while retaining the chosen settings.
Logs show preprocessed RMS/peak, gate state and limiter gain reduction. Model
compute timings exclude preprocessing. `--log-file` in the three-model monitor
records final settings and live edits. Offline mode also accepts these flags and
writes `{model}_input_pre.wav` to inspect exactly what is fed into each model.

Verification includes disabled sample identity, peak ceiling under positive and
negative transients, gate hold/release, chunk-boundary continuity and invalid
command rollback. Run `evaluation/denoise_eval/.venv/bin/python evaluation/denoise_eval/test_audio_preprocessing.py`.
Subjective live quality with these settings remains to be tested by the user.

Completed offline checks: all five DSP tests passed; with preprocessing inactive,
all three model WAVs reproduced the previous baseline exactly. With gate-50/-55
and limiter-12 dBFS, all three outputs remained finite and their model input peak
stayed at0.251188576 or lower (the -12 dBFS ceiling). Those trial outputs are in
`/private/tmp/tse_preprocessing_enabled`; disabled controls in
`/private/tmp/tse_preprocessing_disabled`. Peak limits apply to input only:
RNNoise/DTLN output peaks exceeded the input ceiling slightly in this check.
Preprocessing-only median runtime measured about0.08–0.14 ms per hop in an offline
microbenchmark; it adds work and does not skip denoiser inference.
