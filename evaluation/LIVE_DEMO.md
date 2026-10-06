# Live demo: hear only the person you pick

[live_demo.py](live_demo.py) runs the discovery-enrollment flow on your laptop. In order, it:

1. finds the people talking (discovery);
2. plays each one so you can pick;
3. keeps only that person from the live audio and plays it to your headphones.

It uses the offline research models from the [discovery report](DISCOVERY_REPORT.md): windowed SepFormer plus ECAPA, and the competitive selector. Live processing runs in 1 s blocks with 2 s of past audio and a 20 ms crossfade, and uses no future audio.

**Expect about 2.2 s of delay** (1 s block + 1.2 s safety margin). This is a research demo, not the ≤150 ms conversational target. It uses about 80% of four CPU cores.

## Before you start

- **Wear headphones.** With laptop speakers, the output leaks back into the microphone.
- The first microphone run makes macOS ask for permission for your terminal app. If it never asks, or the recording is silent, enable your terminal (Terminal, iTerm or VS Code) in **System Settings → Privacy & Security → Microphone**, then restart it.
- Close other heavy apps. The status line shows `underruns`; if that number keeps growing, add `--margin 1.5`.
- Live audio is **not saved** unless you pass `--save`, in which case it goes to the Git-ignored `data/user_recordings/live_sessions/`. Ask people before recording them.

All commands run from the `evaluation/` folder:

```sh
cd evaluation
PY=../speakerbeam_ss_verification/.venv/bin/python
```

## A. Simulated live from a recording (easiest first test)

Streams `Mix.wav` in real time as if it came from the microphone. You hear the chosen person about 2.2 s later.

```sh
$PY live_demo.py --input-file data/user_recordings/mixture/Mix.wav
```

## B. Fully live from the microphone

Records 30 s for discovery; let everyone talk, ideally taking turns, with each person speaking alone for at least ~5 s. Then you pick, and live extraction starts.

```sh
$PY live_demo.py                       # default devices
$PY live_demo.py --discover-seconds 45 # longer discovery for more people
```

## C. Discover from a recording, extract live from the microphone

```sh
$PY live_demo.py --discover-file data/user_recordings/mixture/Mix.wav
```

## Controls

- **When choosing:**
  - type `2` to hear discovered speaker 2 (first 8 s);
  - type `pick 2` to choose them;
  - type `q` to quit.
- **While live:**
  - type `b` then Enter to toggle bypass: the unprocessed input, with the same delay, for comparison;
  - type `s 3` then Enter to switch to speaker 3;
  - press Enter on an empty line to stop.
- A status line every 5 s shows compute time per block (it must stay under 1 s), underruns, and which speaker is active.

## Devices

```sh
$PY live_demo.py --list-devices
$PY live_demo.py --input-device 0 --output-device "AirPods"
```

Bluetooth headsets may switch to a low-quality call mode when their microphone is used (PRD §3). Prefer the laptop microphone together with headphones.

## Offline check (no audio devices)

```sh
$PY live_demo.py --input-file data/user_recordings/mixture/Mix.wav --output-file /tmp/out.wav --choose 1
```

On the user's recording, speakers 1/2/3 came out as Nicker/Sam/Mond. Compute was about 0.85 s per 1 s block on an Apple M5. A real-time test against a simulated sound card gave the same output as this offline check, delayed by 2.200 s, with no underruns.
