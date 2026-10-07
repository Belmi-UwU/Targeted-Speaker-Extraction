# Streaming WeSep: live extraction at about 0.17 s delay

**Date:** 7 October 2026

[streaming_wesep.py](streaming_wesep.py) runs the causal WeSep model **frame by frame** with carried state, instead of re-running it on block + past context. [live_demo_streaming.py](live_demo_streaming.py) is the live demo built on it. Both are new files; the earlier demos and results are unchanged.

## How it works

The causal `TSE_BSRNN_SPK` model has three parts that depend on time:

| Part | Streaming handling |
| --- | --- |
| STFT (512-sample Hann window, 128 hop) | Streaming framing, with reflect padding at the stream start exactly like `torch.stft(center=True)` |
| The time LSTM `band_rnn` in each of the six blocks | Hidden and cell state carried between chunks |
| iSTFT | Overlap-add with the window-square envelope; a sample is emitted once no later frame can touch it |

Everything else works on each frame on its own: channel-wise LayerNorm, the band-split convolutions, `band_comm` (an LSTM across frequency within one frame), TF-map and context attention to the enrollment frames, and the mask network.

The enrollment features are computed **once** per chosen speaker. Algorithmic latency is one window: 32 ms. The FFT runs in NumPy, so no complex tensors reach the Metal device and no CPU fallback is needed.

## Verified

Measured on a 12 s test conversation with the released `tfmap_context_causal_100` checkpoint:

- **Same output as the full model.** The streaming output equals the whole-clip forward pass to within 1–2e-6 of peak (float rounding) for 8, 16, and 32 ms chunks, on CPU and Metal.
- **Strictly causal.** Replacing all audio after 8 s changed the earlier output by exactly **0**.
- **Dither caused the earlier "leak".** The configs keep Kaldi fbank dither at 1.0 in the speaker encoder, which adds random noise at inference too: the same input twice differed by about 0.2% of peak. That randomness is what [WESEP_COMPARISON.md](../WESEP_COMPARISON.md) and [README.md](README.md) reported as a small causality leak (0.3% for `tfmap`, up to 13% for `spk_emb`). With dither off (`streaming_wesep.deterministic`), the model is causal and deterministic. Earlier WeSep scores include this small random noise; its effect on the medians is negligible. The training notebook now turns dither off for evaluation and in exported configs.

**Speed** is compute ÷ audio duration on an Apple M5 laptop with the 27 M-parameter released model; below 1 keeps up with live audio.

| Chunk | CPU, 2 threads | CPU, 4 threads | Metal (MPS) |
| ---: | ---: | ---: | ---: |
| 16 ms | 1.44 | 1.45 | 1.64 |
| 32 ms | **0.81** | 0.82 | 0.83 |
| 64 ms | **0.49** | 0.51 | 0.58 |

For chunks this small, per-call overhead dominates, so Metal does not help. For whole clips Metal is about 4× faster than CPU (0.08 vs 0.31 s per 1 s of audio). The small model from `training/` runs inference at about 0.036 s per 1 s on Metal and should allow smaller chunks.

## Live demo

```sh
cd evaluation/wesep_eval
.venv/bin/python live_demo_streaming.py --input-file ../data/user_recordings/mixture/Mix.wav   # simulated live
.venv/bin/python live_demo_streaming.py                                                     # microphone, headphones on
.venv/bin/python live_demo_streaming.py --chunk 0.032 --margin 0.07                         # lower delay, less headroom
.venv/bin/python live_demo_streaming.py --model retrained_finetune_research                 # a model from training/
```

Defaults are a 64 ms chunk, a 0.10 s safety margin, CPU, and 2 threads. With a simulated real-time sound card on Mix.wav:
- the delay from microphone input to headphone output was **167 ms**, before the real audio devices' own buffers;
- the output matched the offline streaming result (correlation 0.995);
- there was one underrun at start-up and none after.

Discovery, choosing, bypass (`b`), and switching speakers (`s N`) work as in the other demos. Switching updates the enrollment without resetting the LSTM state.

The margin must cover mic-block arrival (the input arrives in 20 ms blocks), about 35 ms of compute per 64 ms chunk, and the iSTFT's 24 ms start-up lag. A smaller margin produced underruns. With 32 ms chunks the theoretical floor is about 0.1 s, but the real-time factor of 0.81 leaves little headroom on this laptop.

## Quality

Streaming gives the same output as the whole-clip model, so quality is the released checkpoint's quality ([WESEP_COMPARISON.md](../WESEP_COMPARISON.md)):
- about +3 dB median in 4–6 speaker conversations;
- on the real recording, Sam and Mond come out right and Nicker does not. Voice match: Sam 0.57 vs 0.40, Mond 0.41 vs 0.21, Nicker 0.39 vs 0.46 (Sam).

The delay problem is solved. The remaining gap is separation quality, which the retraining in `training/` targets. Exported retrained models load directly with `--model`.

Relative to the PRD's ≤150 ms capture-to-playback target, 0.17 s plus device buffers is close but not yet there. Two ways down:
- smaller chunks on a faster machine, or the small retrained model;
- smaller microphone blocks than the 20 ms used by `live_demo.run_live`.
