"""Inspect web-app discovery on Mix.wav using separate voices only for analysis.

Writes local listening clips and diagnostics without changing discovery settings
or using the enrollment identities to construct discovered profiles.
"""

import argparse
from collections import Counter
from datetime import datetime
import html
import json
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo

import numpy as np
import scipy.signal
import soundfile as sf
from sklearn.cluster import AgglomerativeClustering

import discovery_pipeline as dp
import live_demo as base


INPUTS = dp.ROOT / "data/user_recordings"
REFERENCES = {name: INPUTS / f"enrollments/{name.lower()}/{name}.wav"
              for name in ("Nicker", "Mond", "Sam")}


class CaptureSeparator:
    """Capture model output before the production pipeline drops quiet streams."""

    def __init__(self, separator):
        self.separator, self.calls = separator, []

    def separate_batch(self, segment):
        output = self.separator.separate_batch(segment)
        self.calls.append((segment[0].cpu().numpy().copy(),
                           output[0].cpu().numpy().copy()))
        return output


def match(vector, references):
    if vector is None:
        return {"scores": None, "best": None, "margin": None}
    scores = {name: float(vector @ ref) for name, ref in references.items()}
    ranked = sorted(scores, key=scores.get, reverse=True)
    return {"scores": scores, "best": ranked[0],
            "margin": scores[ranked[0]] - scores[ranked[1]]}


def write_clip(folder, stem, wave, rate=dp.RATE):
    wave = np.asarray(wave, dtype="float32")
    if not np.isfinite(wave).all():
        raise ValueError(f"Non-finite audio: {stem}")
    raw, listen = f"raw/{stem}.wav", f"listen/{stem}.wav"
    for relative in (raw, listen):
        (folder / relative).parent.mkdir(parents=True, exist_ok=True)
    sf.write(folder / raw, wave, rate, subtype="FLOAT")
    peak = float(np.max(np.abs(wave))) if len(wave) else 0.0
    gain = 0.9 / peak if peak > 1e-9 else 1.0
    sf.write(folder / listen, wave * gain, rate, subtype="PCM_16")
    return {"raw": raw, "listen": listen, "raw_peak": peak, "listening_gain": gain}


def describe(path):
    info = sf.info(path)
    return {"path": str(path), "sha256": dp.sha256(path),
            "sample_rate": info.samplerate, "channels": info.channels,
            "seconds": info.duration}


def score_text(result):
    if result["scores"] is None:
        return "Tidak cukup suara untuk embedding"
    values = " · ".join(f"{name}: {value:.2f}" for name, value in result["scores"].items())
    return f"{values} · paling dekat {result['best']} · margin {result['margin']:.2f}"


def audio(relative):
    return f'<audio controls preload="metadata" src="{html.escape(relative, quote=True)}"></audio>'


LISTEN_SCRIPT = """<script>
let context, active, timer, currentStatus;
document.querySelectorAll('audio').forEach(audio => {
  const button = document.createElement('button');
  button.textContent = 'Putar alternatif';
  const status = document.createElement('span');
  status.setAttribute('role', 'status');
  audio.after(button, status);
  button.addEventListener('click', async () => {
    try {
      context ||= new (window.AudioContext || window.webkitAudioContext)();
      await context.resume();
      document.querySelectorAll('audio').forEach(other => other.pause());
      if (active) active.stop();
      clearInterval(timer);
      if (currentStatus) currentStatus.textContent = ' Dihentikan';
      currentStatus = status;
      status.textContent = ' Memuat audio…';
      const response = await fetch(audio.src);
      if (!response.ok) throw new Error('HTTP ' + response.status);
      const buffer = await context.decodeAudioData(await response.arrayBuffer());
      const source = context.createBufferSource();
      source.buffer = buffer;
      source.connect(context.destination);
      active = source;
      const start = context.currentTime;
      const update = () => {
        status.textContent = ' Memutar ' + Math.min(context.currentTime - start, buffer.duration).toFixed(1)
          + ' / ' + buffer.duration.toFixed(1) + ' detik';
      };
      source.onended = () => {
        if (active !== source) return;
        clearInterval(timer);
        status.textContent = ' Selesai';
        active = null;
      };
      source.start();
      update();
      timer = setInterval(update, 250);
    } catch (error) {
      status.textContent = ' Gagal memutar: ' + error.message;
    }
  });
});
</script>"""


def write_reports(folder, record):
    profiles = record["profiles"]
    lines = ["# Diagnosis discovery: Mix.wav", "",
             f"Waktu: {record['created_at']}", "",
             f"Hasil: **{len(profiles)} profil**, dari {len(record['windows'])} potongan audio.", "",
             "Enrollment Nicker/Mond/Sam hanya dipakai untuk membandingkan identitas setelah discovery. "
             "Kemiripan ECAPA tidak membuktikan kemurnian suara; pemeriksaan mendengarkan tetap diperlukan.", "",
             "| Profil | Jendela | Kemiripan track penuh | Kemiripan preview web | Konflik dalam satu jendela |",
             "| --- | ---: | --- | --- | ---: |"]
    for p in profiles:
        lines.append(f"| {p['profile']} | {len(p['windows'])} | {score_text(p['match'])} | "
                     f"{score_text(p['preview_match'])} | {len(p['same_window_conflicts'])} |")
    lines += ["", "Pengaturan sama dengan web app: " + json.dumps(record["settings"]), "",
              "Lihat `index.html` untuk rekaman pembanding, profil penuh, preview web, dan setiap stream. "
              "Semua klip dengar dinormalisasi per klip; bandingkan identitas dan kebocoran, bukan volumenya.", "",
              "Tidak ada sumber bersih yang selaras dengan Mix.wav, sehingga tidak dihitung SI-SDR atau "
              "klaim jumlah suara berdasarkan ground truth."]
    lines += ["", "## Petunjuk yang perlu didengarkan", ""]
    for p in profiles:
        for index in p["same_window_conflicts"]:
            w = record["windows"][index - 1]
            combined = [s for s in w["streams"] if s["profile"] == p["profile"]]
            clues = "; ".join(f"stream {s['column'] + 1}: {score_text(s['match'])}" for s in combined)
            lines.append(f"- Profil {p['profile']}, {w['start_seconds']:.1f}–{w['end_seconds']:.1f} s: "
                         f"beberapa stream dijumlahkan. {clues}. Ini konflik potensial, bukan bukti dua identitas.")
    dropped = Counter(s["match"]["best"] for w in record["windows"] for s in w["streams"]
                      if s["decision"] == "Dibuang: energi rendah" and s["match"]["best"])
    lines.append(f"- Dugaan terdekat pada stream yang dibuang karena energi rendah: {dict(dropped)}. "
                 "Kemiripan ini hanya petunjuk untuk pemeriksaan audio.")
    (folder / "REPORT.md").write_text("\n".join(lines) + "\n")

    cards = [f'<article><h3>{html.escape(name)}</h3>{audio(clip["listen"])}</article>'
             for name, clip in record["reference_clips"].items()]
    chunks = ["<h2>1. Rekaman acuan</h2><div class='grid'>" + "".join(cards) + "</div>",
              "<h2>2. Campuran asli</h2>" + audio(record["mixture_clip"]["listen"]),
              f"<h2>3. Hasil discovery: {len(profiles)} profil</h2>"]
    for p in profiles:
        conflict = p["same_window_conflicts"]
        chunks.append(f"<article><h3>Profil {p['profile']}</h3>"
                      f"<p>{len(p['windows'])} jendela; dugaan terdekat per stream anggota: "
                      f"{html.escape(str(p['member_best_matches']))}. "
                      f"Jendela yang menjumlahkan lebih dari satu stream: {html.escape(str(conflict))}.</p>"
                      f"<p><b>Track penuh</b>: {html.escape(score_text(p['match']))}</p>{audio(p['clip']['listen'])}"
                      f"<p><b>Preview seperti web app</b>: {html.escape(score_text(p['preview_match']))}</p>"
                      f"{audio(p['preview_clip']['listen'])}</article>")
    chunks.append("<h2>4. Stream sebelum pengelompokan</h2><p>Nomor stream tidak merupakan identitas "
                  "tetap antarjendela. Kemiripan hanya petunjuk; dengarkan apakah satu stream berisi dua orang.</p>")
    for w in record["windows"]:
        rows = []
        for s in w["streams"]:
            rows.append(f"<tr><td>{s['column'] + 1}</td><td>{s['relative_db']:.1f} dB</td>"
                        f"<td>{html.escape(s['decision'])}</td><td>{html.escape(score_text(s['match']))}</td>"
                        f"<td>{audio(s['clip']['listen'])}</td></tr>")
        chunks.append(f"<details><summary>Jendela {w['window']}: {w['start_seconds']:.1f}–"
                      f"{w['end_seconds']:.1f} detik</summary><table><thead><tr><th>Stream</th>"
                      "<th>Energi relatif</th><th>Keputusan</th><th>Kemiripan acuan</th><th>Dengarkan</th>"
                      "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></details>")
    page = """<!doctype html><html lang="id"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Diagnosis discovery Mix.wav</title>
<style>body{font:16px/1.5 system-ui;background:#f5f6f8;color:#17212f;margin:0}main{max-width:1150px;margin:auto;padding:24px}
article,details{background:white;padding:16px;margin:12px 0;border:1px solid #dde2e8;border-radius:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:12px}audio{max-width:100%;width:300px}
table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:8px;border-bottom:1px solid #ddd}summary{cursor:pointer}
.note{background:#fff2cd;padding:14px;border-radius:10px}@media(max-width:800px){table{font-size:13px}td audio{width:150px}}</style>
<main><h1>Diagnosis discovery pada Mix.wav</h1><p class="note">Ini pengujian rekaman yang sudah ada, bukan kontrol percakapan
bergantian di ruangan tenang. Enrollment hanya menjadi acuan setelah discovery. Skor kemiripan tidak membuktikan kemurnian
atau kualitas suara. Klip dinormalisasi sendiri-sendiri untuk didengarkan; jangan membandingkan volumenya.</p>
""" + "".join(chunks) + "</main>" + LISTEN_SCRIPT + "</html>"
    (folder / "index.html").write_text(page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(ZoneInfo("Asia/Jakarta"))
    folder = args.output or INPUTS / "discovery_diagnostics" / stamp.strftime("%Y%m%d_%H%M%S")
    folder.mkdir(parents=True, exist_ok=False)
    mix_path = INPUTS / "mixture/Mix.wav"
    input_metadata = {name: describe(path) for name, path in {"Mix": mix_path, **REFERENCES}.items()}
    mix = base.read_any(mix_path)
    models = dp.Models()
    reference_vectors, reference_clips = {}, {}
    for name, path in REFERENCES.items():
        wave = base.read_any(path)
        reference_vectors[name] = models.embed("ecapa", wave)
        if reference_vectors[name] is None:
            raise ValueError(f"Insufficient enrollment speech: {name}")
        reference_clips[name] = write_clip(folder, f"reference_{name.lower()}", wave)
    params = base.settings()
    capture = CaptureSeparator(models.sepformer)
    models._sepformer = capture
    print(f"Discovery {len(mix)/dp.RATE:.2f}s; settings={params}", flush=True)
    start = time.perf_counter()
    found, separation = dp.discover(models, mix, "ecapa", params["distance_threshold"], params["min_windows"])
    elapsed = time.perf_counter() - start
    items = [(w, s) for w, window in enumerate(separation["windows"])
             for s, stream in enumerate(window["streams"]) if stream.get("ecapa") is not None]
    if len(items) > 1:
        vectors = np.stack([separation["windows"][w]["streams"][s]["ecapa"] for w, s in items])
        labels = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                        distance_threshold=params["distance_threshold"]).fit_predict(vectors)
    else:
        labels = np.zeros(len(items), dtype=int)
    groups, rejected_groups = [], []
    for label in sorted(set(labels)):
        members = [items[i] for i in np.flatnonzero(labels == label)]
        windows = sorted({w for w, _ in members})
        group = {"members": members, "windows": windows}
        (groups if len(windows) >= params["min_windows"] else rejected_groups).append(group)
    groups.sort(key=lambda g: g["windows"][0])
    if len(groups) != len(found):
        raise RuntimeError("Diagnostic groups differ from production discovery")
    profiles = []
    membership = {}
    for index, (speaker, group) in enumerate(zip(found, groups), 1):
        counts = Counter(w for w, _ in group["members"])
        bests = Counter()
        for w, s in group["members"]:
            stream = separation["windows"][w]["streams"][s]
            membership[(w, stream["column"])] = index
            bests[match(stream["ecapa"], reference_vectors)["best"]] += 1
        speech = dp.active_speech(speaker["track"])
        preview = (speech if len(speech) > dp.RATE else speaker["track"])[:12 * dp.RATE]
        profiles.append({"profile": index, "windows": [w + 1 for w in group["windows"]],
                         "member_best_matches": dict(bests),
                         "same_window_conflicts": [w + 1 for w, count in counts.items() if count > 1],
                         "match": match(models.embed("ecapa", speaker["track"]), reference_vectors),
                         "preview_match": match(models.embed("ecapa", preview), reference_vectors),
                         "clip": write_clip(folder, f"profile_{index}", speaker["track"]),
                         "preview_clip": write_clip(folder, f"web_preview_{index}", preview)})
    windows = []
    captured = iter(capture.calls)
    for w, window in enumerate(separation["windows"]):
        row = {"window": w + 1, "start_seconds": window["start"] / dp.RATE,
               "end_seconds": min((window["start"] + separation["window"]) / dp.RATE, len(mix) / dp.RATE),
               "fit_error": window["fit_error"], "streams": []}
        windows.append(row)
        if window["fit_error"] is None:
            continue
        seg8, raw = next(captured)
        gains = np.linalg.lstsq(raw.astype("float64"), seg8.astype("float64"), rcond=None)[0]
        fitted = raw * gains
        seg16 = np.pad(mix, (0, separation["padded_frames"] - len(mix)))[window["start"]:window["start"] + separation["window"]]
        energy = max(float(np.sum(seg16.astype("float64") ** 2)), 1e-20)
        kept = {s["column"]: s for s in window["streams"]}
        for column in range(raw.shape[1]):
            wave = scipy.signal.resample_poly(fitted[:, column], 2, 1)[:separation["window"]].astype("float32")
            relative_db = 10 * np.log10(max(float(np.sum(wave.astype("float64") ** 2)), 1e-20) / energy)
            vector = kept[column].get("ecapa") if column in kept else models.embed("ecapa", wave)
            profile = membership.get((w, column))
            if column not in kept:
                decision = "Dibuang: energi rendah"
            elif vector is None:
                decision = "Dibuang: suara terlalu pendek"
            elif profile is None:
                decision = "Dibuang: cluster terlalu sedikit jendela"
            else:
                decision = f"Profil {profile}"
            row["streams"].append({"column": column, "relative_db": float(relative_db),
                                   "mixture_fit_gain": float(gains[column]), "decision": decision,
                                   "profile": profile, "match": match(vector, reference_vectors),
                                   "clip": write_clip(folder, f"window_{w+1:02d}_stream_{column+1}", wave),
                                   "separator_raw": write_clip(folder, f"separator_raw_{w+1:02d}_{column+1}", raw[:, column], 8000)["raw"]})
    record = {"created_at": stamp.isoformat(), "git_commit": subprocess.check_output(
                  ["git", "rev-parse", "HEAD"], cwd=dp.ROOT, text=True).strip(),
              "scope": "Existing real recording; enrollment references are used only after discovery; no aligned clean sources",
              "inputs": input_metadata, "settings": {**params, "window_seconds": dp.WINDOW_SECONDS,
                  "hop_seconds": dp.HOP_SECONDS, "stream_floor_db": dp.STREAM_FLOOR_DB},
              "discovery_compute_seconds": elapsed,
              "enrollment_cross_similarity": {a: {b: float(v @ reference_vectors[b]) for b in reference_vectors}
                                               for a, v in reference_vectors.items()},
              "reference_clips": reference_clips, "mixture_clip": write_clip(folder, "mix", mix),
              "profiles": profiles, "rejected_cluster_window_counts": [len(g["windows"]) for g in rejected_groups],
              "windows": windows}
    record["code_sha256"] = {p.name: dp.sha256(p) for p in
                              (Path(__file__).resolve(), Path(dp.__file__), Path(base.__file__))}
    record["model_weight_sha256"] = {str(p.relative_to(dp.ROOT)): dp.sha256(p) for p in
                                    (*dp.SEPFORMER.glob("*.ckpt"), *dp.ECAPA.glob("*.ckpt"))}
    (folder / "diagnostics.json").write_text(json.dumps(record, indent=2) + "\n")
    write_reports(folder, record)
    # Verify the original recordings remain byte-for-byte identical.
    for description in input_metadata.values():
        if dp.sha256(Path(description["path"])) != description["sha256"]:
            raise RuntimeError("An input file changed during the diagnostic run")
    print(f"FOUND {len(profiles)} profiles in {elapsed:.1f}s", flush=True)
    for p in profiles:
        print(f"Profile {p['profile']}: {score_text(p['match'])}; conflicts={p['same_window_conflicts']}", flush=True)
    print(f"OUTPUT {folder}", flush=True)


if __name__ == "__main__":
    main()
