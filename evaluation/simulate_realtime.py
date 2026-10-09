"""Paced live-input simulation of the local Mix.wav with windowed SepFormer.

Input is released in 20 ms frames on a wall clock. A separate producer keeps
capture moving while one worker processes complete, non-overlapping windows.
The playback sink is virtual: timestamps determine when each window could
play, preserving gaps and initial delay in saved timeline previews.
"""

import argparse
import json
import math
import platform
import queue
import threading
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import soundfile as sf
import torch
from scipy.optimize import linear_sum_assignment
from speechbrain.inference.classifiers import EncoderClassifier
from speechbrain.inference.separation import SepformerSeparation

from run_user_recordings import PROFILES, ROOT, embed, read_mono, resample, sha256


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {"p50": float(np.median(values)), "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values))}


def paced_run(wave, rate, window_seconds, processor, label, frame_ms=20):
    """Never hand the processor samples whose nominal capture time is future."""
    window_frames = round(window_seconds * rate)
    frame_frames = round(frame_ms * rate / 1000)
    if window_frames % frame_frames:
        raise ValueError("Window must be a multiple of the capture frame")
    packets = queue.Queue()  # unbounded: a slow worker cannot stall capture
    stop = threading.Event()
    source = {"released_windows": 0, "max_pending_windows": 0, "frame_lateness": []}
    epoch = time.perf_counter()

    def capture():
        pending = []
        window_start = 0
        for begin in range(0, len(wave), frame_frames):
            end = min(begin + frame_frames, len(wave))
            deadline = epoch + end / rate
            if stop.wait(max(0.0, deadline - time.perf_counter())):
                return
            arrived = time.perf_counter() - epoch
            source["frame_lateness"].append(arrived - end / rate)
            pending.append(wave[begin:end].copy())
            if end - window_start == window_frames or end == len(wave):
                captured = np.concatenate(pending, axis=0)
                packets.put({"wave": captured, "start_frame": window_start,
                             "end_frame": end, "released_t": time.perf_counter() - epoch})
                source["released_windows"] += 1
                source["max_pending_windows"] = max(source["max_pending_windows"], packets.qsize())
                window_start = end
                pending.clear()
        packets.put(None)

    producer = threading.Thread(target=capture, name="paced-audio-source", daemon=True)
    producer.start()
    records, outputs = [], {}
    previous_play_end = None
    next_progress = 5.0
    try:
        while True:
            packet = packets.get()
            if packet is None:
                break
            start_t = packet["start_frame"] / rate
            end_t = packet["end_frame"] / rate
            started = time.perf_counter() - epoch
            if started + 1e-6 < end_t:
                raise AssertionError("Processor received future audio")
            result, details = processor(packet["wave"], rate, window_seconds)
            ready = time.perf_counter() - epoch
            for name, samples in result.items():
                if samples.ndim != 1 or not np.isfinite(samples).all():
                    raise ValueError(f"Invalid output for {name}")
                outputs.setdefault(name, []).append(samples)
            play_start = ready if previous_play_end is None else max(ready, previous_play_end)
            gap = 0.0 if previous_play_end is None else max(0.0, ready - previous_play_end)
            previous_play_end = play_start + (end_t - start_t)
            records.append({
                "index": len(records), "capture_start_seconds": start_t,
                "capture_end_seconds": end_t, "released_seconds": packet["released_t"],
                "processing_start_seconds": started, "output_ready_seconds": ready,
                "processing_seconds": ready - started,
                "queue_wait_seconds": started - packet["released_t"],
                "pending_windows_after_processing": max(0, source["released_windows"] - len(records) - 1),
                "first_sample_ready_delay_seconds": ready - start_t,
                "last_sample_ready_delay_seconds": ready - end_t,
                "virtual_play_start_seconds": play_start,
                "virtual_play_end_seconds": previous_play_end,
                "virtual_play_delay_seconds": play_start - start_t,
                "virtual_play_gap_seconds": gap,
                "full_window": packet["end_frame"] - packet["start_frame"] == window_frames,
                **details,
            })
            if end_t >= next_progress:
                print(f"{label}: input {end_t:.1f}s; ready delay {ready-start_t:.3f}s; "
                      f"pending {records[-1]['pending_windows_after_processing']}", flush=True)
                next_progress += 5.0
    finally:
        stop.set()
        producer.join(timeout=1)

    if not records or records[-1]["capture_end_seconds"] != len(wave) / rate:
        raise AssertionError("Capture ended before the complete input was consumed")
    if any(len(chunks) != len(records) for chunks in outputs.values()):
        raise AssertionError("Output windows were lost")
    full = [record for record in records if record["full_window"]]
    compute_sum = sum(record["processing_seconds"] for record in records)
    summary = {
        "label": label, "input_seconds": len(wave) / rate,
        "window_seconds": window_seconds, "hop_seconds": window_seconds,
        "capture_frame_ms": frame_ms, "windows": len(records),
        "processing_seconds": stats([r["processing_seconds"] for r in full]),
        "queue_wait_seconds": stats([r["queue_wait_seconds"] for r in records]),
        "first_sample_ready_delay_seconds": stats([r["first_sample_ready_delay_seconds"] for r in records]),
        "last_sample_ready_delay_seconds": stats([r["last_sample_ready_delay_seconds"] for r in records]),
        "virtual_play_delay_seconds": stats([r["virtual_play_delay_seconds"] for r in records]),
        "capture_frame_lateness_seconds": stats(source["frame_lateness"]),
        "max_pending_windows": source["max_pending_windows"],
        "processing_budget_misses_full_windows": sum(r["processing_seconds"] > window_seconds for r in full),
        "full_window_count": len(full), "compute_sum_seconds": compute_sum,
        "processing_real_time_factor": compute_sum / (len(wave) / rate),
        "first_output_ready_seconds": records[0]["output_ready_seconds"],
        "last_output_ready_seconds": records[-1]["output_ready_seconds"],
        "virtual_playback_finish_seconds": records[-1]["virtual_play_end_seconds"],
        "virtual_playback_gap_sum_seconds": sum(r["virtual_play_gap_seconds"] for r in records),
        "virtual_playback_gaps_over_1ms": sum(r["virtual_play_gap_seconds"] > .001 for r in records),
        "virtual_play_delay_growth_seconds": records[-1]["virtual_play_delay_seconds"] - records[0]["virtual_play_delay_seconds"],
    }
    return summary, records, outputs


def passthrough(wave, rate, window_seconds):
    if wave.ndim == 2:
        wave = wave.mean(axis=1)
    return {"mix": resample(wave, rate, 16000)}, {}


class WindowSeparator:
    def __init__(self, separator, matcher, enrollments):
        self.separator, self.matcher = separator, matcher
        self.names = list(PROFILES)
        self.enrollments = enrollments
        self.selections = [(name,) for name in self.names] + list(combinations(self.names, 2))

    def __call__(self, wave, rate, window_seconds):
        stage_start = time.perf_counter()
        mono = wave.mean(axis=1) if wave.ndim == 2 else wave
        mix8 = resample(mono, rate, 8000)
        valid8 = len(mix8)
        padded8 = np.pad(mix8, (0, max(0, round(window_seconds * 8000) - valid8)))
        pre_done = time.perf_counter()
        with torch.inference_mode():
            estimates = self.separator.separate_batch(torch.from_numpy(padded8).unsqueeze(0))
        streams8 = estimates[0].cpu().numpy().astype("float32")
        sep_done = time.perf_counter()
        if streams8.shape != (len(padded8), 3) or not np.isfinite(streams8).all():
            raise ValueError("Unexpected separator output")
        gains, _, rank, singular = np.linalg.lstsq(
            streams8.astype("float64"), padded8.astype("float64"), rcond=1e-5)
        if not np.isfinite(gains).all():
            raise ValueError("Invalid per-window mixture-fit gains")
        calibrated8 = streams8 * gains.astype("float32")
        fit_error = float(np.linalg.norm(calibrated8.sum(axis=1) - padded8) /
                          max(np.linalg.norm(padded8), 1e-12))
        streams16 = np.stack([resample(calibrated8[:, i], 8000, 16000) for i in range(3)])
        with torch.inference_mode():
            encoded = self.matcher.encode_batch(torch.from_numpy(streams16)).cpu().numpy().reshape(3, -1)
        vectors = encoded / np.maximum(np.linalg.norm(encoded, axis=1, keepdims=True), 1e-12)
        similarity = self.enrollments @ vectors.T
        rows, cols = linear_sum_assignment(-similarity)
        mapping = {self.names[row]: int(col) for row, col in zip(rows, cols)}
        valid16 = math.ceil(len(wave) * 16000 / rate)
        outputs = {}
        for selected in self.selections:
            label = "only_" + selected[0] if len(selected) == 1 else "_and_".join(selected)
            outputs[label] = streams16[[mapping[name] for name in selected]].sum(axis=0)[:valid16]
        finish = time.perf_counter()
        return outputs, {
            "resampling_before_separator_seconds": pre_done - stage_start,
            "separator_seconds": sep_done - pre_done,
            "gain_matching_selection_seconds": finish - sep_done,
            "tail_zero_padding_samples_8k": len(padded8) - valid8,
            "mixture_fit_rank": int(rank), "mixture_fit_gains": gains.tolist(),
            "mixture_fit_relative_error": fit_error,
            "enrollment_to_stream_cosine": similarity.tolist(),
            "matched_stream_by_speaker": mapping,
        }


def save_run(directory, summary, records, outputs):
    directory.mkdir(parents=True, exist_ok=True)
    peak = max(float(np.max(np.abs(x))) for chunks in outputs.values() for x in chunks)
    gain = min(1.0, 0.9 / max(peak, 1e-12))
    files = {}
    for label, chunks in outputs.items():
        content = np.concatenate(chunks)
        count = math.ceil(summary["virtual_playback_finish_seconds"] * 16000) + 2
        timeline = np.zeros(count, dtype=np.float32)
        written_end = 0
        for record, chunk in zip(records, chunks):
            begin = max(written_end, round(record["virtual_play_start_seconds"] * 16000))
            end = begin + len(chunk)
            if end > len(timeline):
                timeline = np.pad(timeline, (0, end - len(timeline)))
            timeline[begin:end] = chunk
            written_end = end
        timeline = timeline[:written_end]
        paths = {"raw_content": directory / f"{label}_content_raw.wav",
                 "content_preview": directory / f"{label}_content.wav",
                 "timeline_preview": directory / f"{label}_timeline.wav"}
        sf.write(paths["raw_content"], content, 16000, subtype="FLOAT")
        sf.write(paths["content_preview"], content * gain, 16000, subtype="PCM_16")
        sf.write(paths["timeline_preview"], timeline * gain, 16000, subtype="PCM_16")
        files[label] = {key: path.name for key, path in paths.items()}
    summary["preview_gain_applied_after_timing"] = gain
    summary["files"] = files
    (directory / "trace.jsonl").write_text("\n".join(json.dumps(r, allow_nan=False) for r in records) + "\n")
    (directory / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


def self_check():
    # A producer must continue capturing while its worker is overloaded, and
    # virtual playback must expose the resulting gaps rather than hide them.
    wave = np.ones(160, dtype=np.float32) * .1
    def slow_processor(segment, rate, seconds):
        time.sleep(.05)
        return {"mix": segment}, {}
    summary, trace, outputs = paced_run(wave, 1000, .02, slow_processor, "self-check")
    assert sum(len(x) for x in outputs["mix"]) == len(wave)
    assert all(r["processing_start_seconds"] >= r["capture_end_seconds"] for r in trace)
    assert summary["max_pending_windows"] >= 2
    assert summary["virtual_play_delay_growth_seconds"] > .1
    assert summary["virtual_playback_gap_sum_seconds"] > .1
    assert summary["processing_budget_misses_full_windows"] == 8
    print("Paced source and overloaded-worker self-check passed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/user_recordings")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    torch.set_num_threads(4)
    inputs = args.input_dir.resolve()
    stamp = datetime.now(ZoneInfo("Asia/Jakarta")).strftime("%Y%m%d_%H%M%S")
    output = args.output_dir or inputs / "realtime_simulation" / stamp
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    mix_path = inputs / "mixture/Mix.wav"
    wave, rate = sf.read(mix_path, dtype="float32")
    if wave.ndim not in (1, 2) or not len(wave) or not np.isfinite(wave).all():
        raise ValueError("Invalid Mix.wav")
    output.mkdir(parents=True)
    experiment = {
        "scope": "wall-clock-paced input, CPU windowed offline models, virtual playback; no physical audio route",
        "input_path": str(mix_path.relative_to(inputs)), "input_sha256": sha256(mix_path),
        "input_rate_hz": rate, "input_seconds": len(wave) / rate,
        "timestamp_jakarta": stamp, "platform": platform.platform(),
        "torch_version": torch.__version__, "torch_threads": 4,
        "separator_revision": "1f2d824e2333aec9e60d0030fb144885c1afc426",
        "matcher_revision": "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286",
        "speaker_matching": "joint one-to-one ECAPA assignment of all three enrollment profiles per window",
        "warnings": ["short-window speaker identity unverified", "no target-absence rejection",
                     "no clean references or SI-SDR score", "no microphone/TWS/playback-device delay",
                     "tail padded only after EOF", "preview safety gain calculated after timing"],
        "runs": {},
    }
    print(f"Input {len(wave)/rate:.3f}s, {rate} Hz; output: {output}", flush=True)
    summary, records, outputs = paced_run(wave, rate, .02, passthrough, "passthrough_20ms")
    save_run(output / "passthrough_20ms", summary, records, outputs)
    experiment["runs"]["passthrough_20ms"] = summary
    (output / "experiment.json").write_text(json.dumps(experiment, indent=2, allow_nan=False) + "\n")
    print("Passthrough complete; loading models and preparing enrollments", flush=True)
    setup_start = time.perf_counter()
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
    enrolled = []
    for name, relative in PROFILES.items():
        enrollment, enrollment_rate = read_mono(inputs / relative)
        enrolled.append(embed(matcher, resample(enrollment, enrollment_rate, 16000)))
    processor = WindowSeparator(separator, matcher, np.stack(enrolled))
    experiment["model_load_and_enrollment_seconds"] = time.perf_counter() - setup_start
    for seconds, label in ((.5, "sepformer_500ms"), (1.0, "sepformer_1000ms")):
        processor(np.zeros(round(rate * seconds), dtype=np.float32), rate, seconds)
        print(f"Starting {label} with live-paced input", flush=True)
        summary, records, outputs = paced_run(wave, rate, seconds, processor, label)
        save_run(output / label, summary, records, outputs)
        experiment["runs"][label] = summary
        (output / "experiment.json").write_text(json.dumps(experiment, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"run": label, "summary": summary}, indent=2), flush=True)
    (output / "experiment.json").write_text(json.dumps(experiment, indent=2, allow_nan=False) + "\n")
    print(f"Complete: {output / 'experiment.json'}", flush=True)


if __name__ == "__main__":
    main()
