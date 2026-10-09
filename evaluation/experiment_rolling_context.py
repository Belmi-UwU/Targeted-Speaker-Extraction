"""Compare past-only SepFormer context lengths with a fixed 500 ms hop.

Uses the existing wall-clock-paced source and virtual playback sink. Each
processor retains only already delivered raw audio, separates past + new audio,
matches speakers over that complete window, and emits only the new region.
"""

import argparse
import html
import json
import math
import platform
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import soundfile as sf
import torch
from speechbrain.inference.classifiers import EncoderClassifier
from speechbrain.inference.separation import SepformerSeparation

from run_user_recordings import PROFILES, ROOT, embed, read_mono, resample, sha256
from simulate_realtime import WindowSeparator, paced_run, save_run, stats


HOP_SECONDS = .5
CONTEXT_MS = (2500, 1000, 500, 250)


class RollingSeparator:
    def __init__(self, processor, context_ms):
        self.processor = processor
        self.context_ms = context_ms
        self.history = None
        self.consumed_frames = 0
        self.rate = None

    def __call__(self, wave, rate, hop_seconds):
        if self.rate is not None and self.rate != rate:
            raise ValueError("Input rate changed during replay")
        self.rate = rate
        if self.history is None:
            self.history = wave[:0].copy()
        history_frames = len(self.history)
        joined = np.concatenate((self.history, wave), axis=0)
        # Available past only at startup. The base processor pads a short final
        # hop on the right; paced_run releases that final packet only at EOF.
        model_seconds = history_frames / rate + hop_seconds
        result, details = self.processor(joined, rate, model_seconds)
        offset16 = round(history_frames * 16000 / rate)
        count16 = math.ceil(len(wave) * 16000 / rate)
        result = {name: samples[offset16:offset16 + count16].copy()
                  for name, samples in result.items()}
        if any(len(samples) != count16 for samples in result.values()):
            raise AssertionError("Newest-region extraction lost audio samples")
        context_frames = round(self.context_ms * rate / 1000)
        self.history = joined[-context_frames:].copy()
        details.update({
            "configured_context_ms": self.context_ms,
            "available_history_frames": history_frames,
            "available_history_seconds": history_frames / rate,
            "model_window_seconds": model_seconds,
            "model_input_start_frame": self.consumed_frames - history_frames,
            "model_input_end_frame": self.consumed_frames + len(wave),
            "emitted_start_frame": self.consumed_frames,
            "emitted_end_frame": self.consumed_frames + len(wave),
            "emitted_samples_16k": count16,
            "steady_context": history_frames == context_frames,
        })
        self.consumed_frames += len(wave)
        return result, details


def context_diagnostics(summary, records, context_ms):
    steady = [r for r in records if r["steady_context"] and r["full_window"]]
    summary.update({
        "context_ms": context_ms,
        "total_window_seconds_with_full_history": context_ms / 1000 + HOP_SECONDS,
        "startup_policy": "use only available past; do not wait to fill history",
        "emission_policy": "newest hop only; no crossfade or future lookahead",
        "steady_full_window_count": len(steady),
        "steady_processing_seconds": stats([r["processing_seconds"] for r in steady]),
        "steady_separator_seconds": stats([r["separator_seconds"] for r in steady]),
        "steady_processing_budget_misses": sum(r["processing_seconds"] > HOP_SECONDS for r in steady),
        "steady_matched_context_cosine_p50": {
            name: float(np.median([r["enrollment_to_stream_cosine"][row]
                                   [r["matched_stream_by_speaker"][name]] for r in steady]))
            for row, name in enumerate(PROFILES)
        },
        "quality_status": "awaiting listening; cosine is a diagnostic, not a quality score",
    })


def export_listening_page(output, inputs, runs, previous_run_dirs=()):
    """Use one common playback gain and no loudness normalization across clips."""
    sources = []
    for label, context_ms in runs:
        sources.append((label, f"Konteks {context_ms} ms + audio baru 500 ms",
                        output / label, "{selection}_content_raw.wav"))
    for previous in previous_run_dirs:
        record = json.loads((previous / "experiment.json").read_text())
        for label, summary in record["runs"].items():
            if any(source[0] == label for source in sources):
                raise ValueError(f"Duplicate comparison condition: {label}")
            if summary["hop_seconds"] != HOP_SECONDS:
                raise ValueError("Previous comparison must use a 500 ms hop")
            sources.append((label, f"Konteks {summary['context_ms']} ms + audio baru 500 ms (run sebelumnya)",
                            previous / label, "{selection}_content_raw.wav"))
    baseline = inputs / "realtime_simulation/2026-10-07_cpu_windows/sepformer_500ms"
    if baseline.is_dir():
        sources.append(("previous_no_context", "Pembanding: 500 ms tanpa konteks (run sebelumnya)",
                        baseline, "{selection}_content_raw.wav"))
    offline = inputs / "results"
    if (offline / "run.json").exists():
        sources.append(("offline_full_recording", "Pembanding: SepFormer rekaman penuh",
                        offline, "{selection}.wav"))
    selections = list(json.loads((output / runs[0][0] / "summary.json").read_text())["files"])
    offline_record = json.loads((offline / "run.json").read_text()) if (offline / "run.json").exists() else {}
    arrays = {}
    for label, title, directory, pattern in sources:
        for selection in selections:
            path = directory / pattern.format(selection=selection)
            audio, rate = sf.read(path, dtype="float32")
            if rate != 16000:
                raise ValueError("Comparison audio must be 16 kHz")
            if label == "offline_full_recording":
                audio = audio / offline_record.get("preview_gain", 1.0)
            arrays[(label, selection)] = audio
    peak = max(float(np.max(np.abs(audio))) for audio in arrays.values())
    gain = min(1.0, .9 / max(peak, 1e-12))
    comparison = output / "comparison"
    comparison.mkdir()
    for (label, selection), audio in arrays.items():
        sf.write(comparison / f"{label}_{selection}.wav", audio * gain, 16000, subtype="PCM_16")
    sections = []
    for selection in selections:
        cards = []
        for label, title, _, _ in sources:
            cards.append(f'<article><h3>{html.escape(title)}</h3>'
                         f'<audio controls preload="none" src="comparison/{label}_{selection}.wav"></audio>'
                         f'<label>Penilaian <select data-key="{label}_{selection}">'
                         '<option value="">Belum dinilai</option><option>Bagus</option>'
                         '<option>Kurang bagus</option><option>Belum yakin</option></select></label></article>')
        sections.append(f'<section data-speaker="{selection}"><h2>{selection.replace("_", " ")}</h2>'
                        + "".join(cards) + '</section>')
    options = "".join(f'<option value="{name}">{name.replace("_", " ")}</option>' for name in selections)
    page = '''<!doctype html><html lang="id"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Perbandingan konteks SepFormer</title><style>
body{font:16px system-ui;max-width:900px;margin:32px auto;padding:0 20px;color:#20252b;background:#fafafa}
article{background:white;border:1px solid #ddd;border-radius:10px;padding:16px;margin:12px 0}
audio{display:block;width:100%;margin:12px 0}select,button{font:inherit;padding:6px}h3{margin:0;font-size:17px}
</style><h1>Perbandingan konteks SepFormer</h1>
<p>Audio baru tetap 500 ms. Dengarkan kejelasan speaker pilihan, kebocoran speaker lain,
suara terputus, dan sambungan. File pembanding memakai gain yang sama; tidak ada normalisasi loudness.
Ini audio content tanpa jeda playback. Delay tersedia di summary.json dan audio timeline.</p>
<p>Bandingkan bagian ucapan yang sama, terutama setelah detik 2,5 ketika semua konteks sudah tersedia.
Speaker yang sedang diam tidak cukup untuk menilai keberhasilan ekstraksi.</p>
<label>Pilihan speaker <select id="speaker">OPTIONS</select></label>
<button id="export">Simpan penilaian JSON</button>SECTIONS
<script>
const selector=document.querySelector('#speaker');
function show(){document.querySelectorAll('audio').forEach(a=>a.pause());
document.querySelectorAll('section').forEach(s=>s.hidden=s.dataset.speaker!==selector.value)}
selector.onchange=show;show();
const ratings={};document.querySelectorAll('[data-key]').forEach(s=>{s.onchange=()=>ratings[s.dataset.key]=s.value});
document.querySelector('#export').onclick=()=>{const a=document.createElement('a');
a.href=URL.createObjectURL(new Blob([JSON.stringify({ratings},null,2)],{type:'application/json'}));
a.download='listening_ratings.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
</script></html>'''
    (output / "listen.html").write_text(page.replace("OPTIONS", options).replace("SECTIONS", "".join(sections)))
    comparison_record = {"common_gain": gain, "selections": selections,
                         "conditions": [{"label": label, "title": title} for label, title, _, _ in sources],
                         "quality_status": "manual listening pending; no clean reference"}
    (output / "comparison.json").write_text(json.dumps(comparison_record, indent=2) + "\n")
    return comparison_record


def self_check(contexts=CONTEXT_MS):
    # Known sample identities expose accidentally replaying history or consuming
    # future frames. Exercise partial startup, bounded history, and final padding.
    def identity_processor(wave, rate, seconds):
        assert rate == 16000
        return {"mix": np.pad(wave, (0, round(seconds * rate) - len(wave)))}, {}
    for context_ms in contexts:
        processor = RollingSeparator(identity_processor, context_ms)
        wave = np.arange(58006, dtype=np.float32)
        parts, records = [], []
        for begin in range(0, len(wave), 8000):
            end = min(begin + 8000, len(wave))
            result, record = processor(wave[begin:end], 16000, .5)
            parts.append(result["mix"])
            records.append(record)
            assert record["model_input_end_frame"] == end
            assert record["model_input_start_frame"] == max(0, begin - context_ms * 16)
            assert len(processor.history) <= context_ms * 16
        assert np.array_equal(np.concatenate(parts), wave)
        assert records[0]["available_history_frames"] == 0
        assert records[-1]["emitted_samples_16k"] == 2006
    print("Rolling context startup, newest-only output, history bounds and tail checks passed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/user_recordings")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--context-ms", type=int, nargs="+", default=list(CONTEXT_MS),
                        help="Past context durations; default: 2500 1000 500 250")
    parser.add_argument("--comparison-run-dir", type=Path, action="append", default=[],
                        help="Include saved rolling-context runs in the listening page without rerunning them")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if any(value <= 0 for value in args.context_ms) or len(set(args.context_ms)) != len(args.context_ms):
        parser.error("Context durations must be positive and unique")
    if args.self_check:
        self_check(args.context_ms)
        return
    torch.set_num_threads(4)
    inputs = args.input_dir.resolve()
    stamp = datetime.now(ZoneInfo("Asia/Jakarta")).strftime("%Y%m%d_%H%M%S")
    output = args.output_dir or inputs / "rolling_context" / stamp
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    mix_path = inputs / "mixture/Mix.wav"
    wave, rate = sf.read(mix_path, dtype="float32")
    if wave.ndim not in (1, 2) or not len(wave) or not np.isfinite(wave).all():
        raise ValueError("Invalid Mix.wav")
    output.mkdir(parents=True)
    original_hashes = {str(path.relative_to(inputs)): sha256(path)
                       for path in [mix_path] + [inputs / p for p in PROFILES.values()]}
    previous_run_dirs = [directory.resolve() for directory in args.comparison_run_dir]
    for directory in previous_run_dirs:
        previous = json.loads((directory / "experiment.json").read_text())
        if previous["input_file_sha256"] != original_hashes:
            raise ValueError("Previous comparison uses different input recordings")
    experiment = {
        "scope": "past-only rolling context, paced 20 ms capture, virtual playback, CPU",
        "timestamp_jakarta": stamp, "platform": platform.platform(),
        "input_file_sha256": original_hashes, "input_rate_hz": rate,
        "input_seconds": len(wave) / rate, "hop_seconds": HOP_SECONDS,
        "context_ms_in_test_order": args.context_ms,
        "comparison_run_directories": [str(directory) for directory in previous_run_dirs],
        "torch_version": torch.__version__, "torch_threads": 4,
        "separator_revision": "1f2d824e2333aec9e60d0030fb144885c1afc426",
        "matcher_revision": "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286",
        "speaker_matching": "joint one-to-one ECAPA using the complete available past + new window",
        "startup": "available history only, no wait to fill context",
        "lookahead_seconds": 0, "crossfade": False,
        "warnings": ["no clean reference; no separation accuracy score",
                     "no target-absence rejection", "no physical device latency",
                     "history is reprocessed audio, not model state", "single short recording"],
        "runs": {},
    }
    print(f"Input {len(wave)/rate:.3f}s; output: {output}", flush=True)
    setup = time.perf_counter()
    separator = SepformerSeparation.from_hparams(
        source=str((ROOT / "sepformer_libr3mix_checkpoint").resolve()),
        savedir=str((ROOT / "sepformer_libr3mix_loaded").resolve()), run_opts={"device": "cpu"})
    matcher = EncoderClassifier.from_hparams(
        source=str((ROOT / "ecapa_checkpoint").resolve()),
        savedir=str((ROOT / "ecapa_loaded").resolve()),
        overrides={"pretrained_path": str((ROOT / "ecapa_checkpoint").resolve())},
        run_opts={"device": "cpu"})
    separator.eval()
    matcher.eval()
    enrollments = []
    for name, relative in PROFILES.items():
        enrollment, enrollment_rate = read_mono(inputs / relative)
        enrollments.append(embed(matcher, resample(enrollment, enrollment_rate, 16000)))
    base_processor = WindowSeparator(separator, matcher, np.stack(enrollments))
    experiment["setup_seconds_excluded_from_replay"] = time.perf_counter() - setup
    runs = []
    for context_ms in args.context_ms:
        label = f"context_{context_ms}ms_hop_500ms"
        # Warm model shapes encountered during both startup and steady operation.
        model_seconds = min(HOP_SECONDS, context_ms / 1000 + HOP_SECONDS)
        while model_seconds < context_ms / 1000 + HOP_SECONDS:
            base_processor(np.zeros(round(rate * model_seconds), dtype=np.float32), rate, model_seconds)
            model_seconds += HOP_SECONDS
        model_seconds = context_ms / 1000 + HOP_SECONDS
        base_processor(np.zeros(round(rate * model_seconds), dtype=np.float32), rate, model_seconds)
        processor = RollingSeparator(base_processor, context_ms)
        print(f"Starting {label}", flush=True)
        summary, records, outputs = paced_run(wave, rate, HOP_SECONDS, processor, label)
        context_diagnostics(summary, records, context_ms)
        save_run(output / label, summary, records, outputs)
        experiment["runs"][label] = summary
        runs.append((label, context_ms))
        (output / "experiment.json").write_text(json.dumps(experiment, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"run": label, "steady_processing_seconds": summary["steady_processing_seconds"],
                          "steady_budget_misses": summary["steady_processing_budget_misses"],
                          "virtual_play_delay_seconds": summary["virtual_play_delay_seconds"]}), flush=True)
    experiment["listening_comparison"] = export_listening_page(output, inputs, runs, previous_run_dirs)
    experiment["original_inputs_unchanged"] = all(sha256(inputs / p) == digest
                                                  for p, digest in original_hashes.items())
    if not experiment["original_inputs_unchanged"]:
        raise AssertionError("Input recording changed")
    (output / "experiment.json").write_text(json.dumps(experiment, indent=2, allow_nan=False) + "\n")
    print(f"Complete: {output / 'listen.html'}", flush=True)


if __name__ == "__main__":
    main()
