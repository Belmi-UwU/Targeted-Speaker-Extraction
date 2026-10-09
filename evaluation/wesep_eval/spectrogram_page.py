"""Self-contained HTML page comparing spectrograms of audio files, with players and a moving cursor.

    .venv/bin/python spectrogram_page.py                         # default: your Mix + retrained outputs
    .venv/bin/python spectrogram_page.py --file "Mix=../data/user_recordings/mixture/Mix.wav" \\
        --file "Nicker (retrained)=../data/user_recordings/discovery/retrained/retrained_discovered_Nicker.wav" \\
        --out compare.html

Each file is converted to 16 kHz mono and peak-normalized, so colours are comparable across rows. Spectrograms use
a 32 ms window, 8 ms hop, 0-8 kHz, and a fixed -100..-20 dB range. Audio and images are embedded, so the page
opens anywhere with no server. Click a spectrogram to play from that point; every row's cursor follows the clip
that is playing.
"""

import argparse
import base64
import html
import io
import sys
from math import gcd
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf

HERE = Path(__file__).resolve().parent
USER = HERE.parent / "data/user_recordings"
RATE, N_FFT, HOP, FLOOR, CEIL, MAX_WIDTH = 16000, 512, 128, -100.0, -20.0, 2000


def load(path):
    wave, rate = sf.read(path, dtype="float32")
    if wave.ndim == 2:
        wave = wave.mean(axis=1)
    if rate != RATE:
        d = gcd(rate, RATE)
        wave = scipy.signal.resample_poly(wave, RATE // d, rate // d).astype("float32")
    peak = float(np.max(np.abs(wave))) if len(wave) else 0.0
    return wave * (0.89 / peak) if peak > 1e-9 else wave


def spectrogram_png(wave):
    from matplotlib import colormaps
    from matplotlib.image import imsave
    _, _, spec = scipy.signal.stft(wave, RATE, window="hann", nperseg=N_FFT, noverlap=N_FFT - HOP, boundary=None)
    db = 20 * np.log10(np.abs(spec) * 2 + 1e-9)  # hann-window magnitude, full-scale sine ~ 0 dB
    image = np.clip((db - FLOOR) / (CEIL - FLOOR), 0, 1)[::-1]  # low frequencies at the bottom
    if image.shape[1] > MAX_WIDTH:  # keep the page small: average neighbouring columns
        factor = int(np.ceil(image.shape[1] / MAX_WIDTH))
        cut = image.shape[1] // factor * factor
        image = image[:, :cut].reshape(image.shape[0], -1, factor).mean(axis=2)
    buffer = io.BytesIO()
    imsave(buffer, image, cmap=colormaps["magma"], format="png")
    return base64.b64encode(buffer.getvalue()).decode()


def wav_b64(wave):
    buffer = io.BytesIO()
    sf.write(buffer, wave, RATE, subtype="PCM_16", format="WAV")
    return base64.b64encode(buffer.getvalue()).decode()


def default_files():
    files = [("Input: Mix (3 people)", USER / "mixture/Mix.wav")]
    for name in ("Nicker", "Sam", "Mond"):
        files.append((f"Output: {name} (retrained WeSep, streaming)",
                      USER / f"discovery/retrained/retrained_discovered_{name}.wav"))
    return [(label, path) for label, path in files if path.exists()]


ROW = """<section class="panel">
  <div class="head"><h2>{label}</h2><span class="muted">{seconds:.1f} s · {name}</span></div>
  <audio controls preload="auto" src="data:audio/wav;base64,{audio}"></audio>
  <div class="plot"><div class="axis"><span style="top:0%">8 kHz</span><span style="top:25%">6</span><span style="top:50%">4</span><span style="top:75%">2</span><span style="top:100%">0</span></div>
  <div class="spec" data-seconds="{seconds}"><img alt="spectrogram of {label}" src="data:image/png;base64,{image}"><i class="cursor"></i></div></div>
</section>"""

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Spectrogram Comparison</title>
<style>
:root {{ --bg:#101216; --panel:#181b21; --ink:#e9e7e2; --muted:#9aa1ac; --line:#2c3038; --accent:#6cbfa6; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
main {{ max-width:1200px; margin:0 auto; padding:16px; }}
h1 {{ font-size:20px; margin:0 0 4px; }} h2 {{ font-size:15px; margin:0; }}
.muted {{ color:var(--muted); }}
.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:10px 12px 12px; margin-top:12px; }}
.head {{ display:flex; justify-content:space-between; gap:10px; flex-wrap:wrap; margin-bottom:6px; }}
audio {{ width:100%; height:32px; margin-bottom:6px; }}
.plot {{ display:grid; grid-template-columns:44px 1fr; gap:6px; }}
.axis {{ position:relative; font-size:11px; color:var(--muted); }}
.axis span {{ position:absolute; right:2px; transform:translateY(-50%); }}
.spec {{ position:relative; cursor:crosshair; }}
.spec img {{ display:block; width:100%; height:200px; border-radius:4px; image-rendering:pixelated; }}
.cursor {{ position:absolute; top:0; bottom:0; width:2px; left:0; background:var(--accent); box-shadow:0 0 4px var(--accent); pointer-events:none; }}
.legend {{ display:flex; align-items:center; gap:8px; font-size:12px; color:var(--muted); margin-top:8px; }}
.bar {{ width:180px; height:10px; border-radius:3px; background:linear-gradient(90deg,#000004,#3b0f70,#8c2981,#de4968,#fe9f6d,#fcfdbf); }}
@media (max-width:600px) {{ .spec img {{ height:140px; }} }}
</style></head>
<body><main>
<h1>Spectrogram comparison</h1>
<div class="muted">Time runs left to right, frequency bottom to top. Bright means louder. Click a spectrogram to play from there; all cursors follow the clip that is playing.</div>
<div class="legend">{floor:.0f} dB <span class="bar"></span> {ceil:.0f} dB · each clip peak-normalized · 32 ms window, 8 ms hop</div>
{rows}
</main>
<script>
const rows = [...document.querySelectorAll(".panel")].map(p => ({{ audio: p.querySelector("audio"), spec: p.querySelector(".spec") }}));
let active = null;
function moveCursors(t) {{
  for (const r of rows) {{
    const s = +r.spec.dataset.seconds;
    r.spec.querySelector(".cursor").style.left = `${{Math.min(100, t / s * 100)}}%`;
  }}
}}
for (const r of rows) {{
  r.audio.addEventListener("play", () => {{
    for (const o of rows) if (o !== r) o.audio.pause();
    active = r;
  }});
  r.audio.addEventListener("timeupdate", () => {{ if (active === r) moveCursors(r.audio.currentTime); }});
  r.spec.addEventListener("click", e => {{
    const box = r.spec.getBoundingClientRect();
    const t = (e.clientX - box.left) / box.width * +r.spec.dataset.seconds;
    r.audio.currentTime = t; r.audio.play(); active = r; moveCursors(t);
  }});
}}
(function tick() {{ if (active && !active.audio.paused) moveCursors(active.audio.currentTime); requestAnimationFrame(tick); }})();
</script></body></html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", action="append", default=[], metavar="LABEL=PATH",
                        help="audio file to include (repeatable); default: Mix + retrained outputs")
    parser.add_argument("--out", type=Path, default=USER / "discovery/spectrogram_compare.html")
    args = parser.parse_args()
    files = [(item.split("=", 1)[0], Path(item.split("=", 1)[1]).expanduser()) for item in args.file] \
        if args.file else default_files()
    if not files:
        sys.exit("no audio files found; pass --file LABEL=PATH")
    rows = []
    for label, path in files:
        wave = load(path)
        rows.append(ROW.format(label=html.escape(label), name=html.escape(path.name), seconds=len(wave) / RATE,
                               audio=wav_b64(wave), image=spectrogram_png(wave)))
        print(f"added {label} ({len(wave) / RATE:.1f} s)", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(PAGE.format(rows="\n".join(rows), floor=FLOOR, ceil=CEIL))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
