"""Create anonymous enrollment profiles from an initial mixture, then replay new audio.

Default calibration is 30 s. The fixed Libri3Mix separator supplies three
candidate streams, not a speaker-count detector. No separate enrollment files
are read unless the optional manual-control comparison is requested.
This is file-based, wall-clock-paced replay with virtual playback, not mic I/O.
"""

import argparse
import html
import json
import math
import os
import time
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import numpy as np
import soundfile as sf
import torch
from scipy.optimize import linear_sum_assignment
from speechbrain.inference.classifiers import EncoderClassifier

from experiment_mps_sepformer import HybridMatcher, SynchronizedProcessor, load_separator
from experiment_rolling_context import RollingSeparator, context_diagnostics
from run_user_recordings import PROFILES, ROOT, embed, read_mono, resample, sha256
from simulate_realtime import WindowSeparator, paced_run, save_run

SEP_REV = "1f2d824e2333aec9e60d0030fb144885c1afc426"
ENC_REV = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def calibration_boundary(wave, rate, seconds=30.):
    boundary = round(seconds * rate)
    if boundary > len(wave):
        raise ValueError(f"Input has {len(wave)/rate:.3f} s, shorter than requested {seconds:g} s; no repetition/padding")
    return boundary


def activity_crop(wave, rate=16000):
    """Energy-only crop; does not certify speech, purity, or speaker identity."""
    frame = round(.04 * rate)
    padded = np.pad(wave, (0, (-len(wave)) % frame)).reshape(-1, frame)
    rms = np.sqrt(np.mean(padded.astype("float64") ** 2, axis=1))
    threshold = max(1e-4, .1 * float(np.percentile(rms, 95)))
    active = rms > threshold
    speech_like_seconds = min(float(active.sum() * frame / rate), len(wave) / rate)
    # 80 ms on either side helps avoid cutting consonants at energy boundaries.
    keep = np.convolve(active.astype(int), np.ones(5, dtype=int), mode="full")[2:2 + len(active)] > 0
    sample_mask = np.repeat(keep, frame)[:len(wave)]
    starts = np.flatnonzero(np.diff(np.r_[False, sample_mask].astype(int)) == 1)
    ends = np.flatnonzero(np.diff(np.r_[sample_mask, False].astype(int)) == -1) + 1
    spans = [[int(a), int(b)] for a, b in zip(starts, ends)]
    return wave[sample_mask].copy(), {
        "energy_active_seconds": speech_like_seconds,
        "retained_seconds": int(sample_mask.sum()) / rate,
        "energy_threshold_rms": threshold,
        "retained_source_spans_samples_16k": spans,
        "method": "40 ms RMS > max(0.0001, 0.1 * p95 RMS), with 80 ms padding; not a VAD",
    }


def build_profiles(wave, rate, separator, matcher, directory, source):
    if not len(wave) or not np.isfinite(wave).all() or float(np.sqrt(np.mean(wave.astype("float64") ** 2))) < 1e-4:
        raise ValueError("Calibration is silent or too quiet; record audible speech")
    directory.mkdir(parents=True)
    begin = time.perf_counter()
    mix8 = resample(wave, rate, 8000)
    with torch.inference_mode():
        streams = separator.separate_batch(torch.from_numpy(mix8).unsqueeze(0))[0].cpu().numpy()
    separated_at = time.perf_counter()
    if streams.shape != (len(mix8), 3) or not np.isfinite(streams).all():
        raise ValueError("Calibration separator did not return three finite streams")
    gains, _, rank, _ = np.linalg.lstsq(streams.astype("float64"), mix8.astype("float64"), rcond=1e-5)
    if not np.isfinite(gains).all() or rank < 3:
        raise ValueError("Calibration streams are degenerate; record clearer speech")
    scaled = streams * gains.astype("float32")
    mix16 = resample(wave, rate, 16000)
    sources = [resample(scaled[:, i], 8000, 16000)[:len(mix16)] for i in range(3)]
    preview_gain = min(1., .9 / max(max(float(np.max(np.abs(s))) for s in sources), 1e-12))
    sf.write(directory / "calibration_mix.wav", mix16 * min(1., .9 / max(float(np.max(np.abs(mix16))), 1e-12)),
             16000, subtype="PCM_16")
    profiles, vectors = [], []
    for index, signal in enumerate(sources):
        speaker = f"speaker_{index + 1:02d}"
        sf.write(directory / f"{speaker}_separated_raw.wav", signal, 16000, subtype="FLOAT")
        sf.write(directory / f"{speaker}_separated.wav", signal * preview_gain, 16000, subtype="PCM_16")
        cropped, activity = activity_crop(signal)
        if activity["energy_active_seconds"] < 2 or len(cropped) < 32000:
            raise ValueError(f"{speaker}: less than 2 s of energy-active audio; profiles not ready")
        # Embedding is made from the generated signal, never the original enrollments.
        cropped_gain = min(1., .9 / max(float(np.max(np.abs(cropped))), 1e-12))
        enrollment = cropped * cropped_gain
        enrollment_path = directory / f"{speaker}_enrollment.wav"
        sf.write(enrollment_path, enrollment, 16000, subtype="FLOAT")
        vector = embed(matcher, enrollment)
        if not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1, atol=1e-4):
            raise ValueError("Invalid generated enrollment embedding")
        # Chunk agreement is diagnostic only; thresholds have not been calibrated.
        chunk_vectors = [embed(matcher, enrollment[start:start + 48000])
                         for start in range(0, len(enrollment) - 32000 + 1, 48000)]
        agreement = [float(v @ vector) for v in chunk_vectors]
        vectors.append(vector)
        profiles.append({"speaker_id": speaker, "display_name": f"Speaker {index + 1:02d}",
                         "calibration_stream_index": index, "enrollment_file": enrollment_path.name,
                         "enrollment_sha256": sha256(enrollment_path), **activity,
                         "enrollment_gain": cropped_gain,
                         "chunk_to_profile_cosine": agreement})
    vectors = np.stack(vectors).astype("float32")
    np.save(directory / "embeddings.npy", vectors, allow_pickle=False)
    pairwise = vectors @ vectors.T
    warnings = ["three candidate streams are not an automatic speaker count",
                "generated enrollment can contain another speaker or separation artifacts",
                "energy crop is not speech detection; listen before choosing a profile",
                "no target-absence rejection; a new speaker requires recalibration",
                "identities are anonymous until the user labels them"]
    # Warn on conspicuously similar candidates rather than silently merging voices.
    suspicious_pairs = [[profiles[i]["speaker_id"], profiles[j]["speaker_id"]]
                        for i in range(3) for j in range(i + 1, 3) if pairwise[i, j] > .75]
    if suspicious_pairs:
        warnings.append("possible duplicate/mixed profiles; >0.75 cosine heuristic is unvalidated")
    record = {"schema_version": 1, "profile_status": "candidate_profiles_require_listening",
              "creation_method": "blind full-calibration separation + energy crop + ECAPA",
              "source": source, "calibration_seconds": len(wave) / rate,
              "input_clipping_fraction": float(np.mean(np.abs(wave) >= .999)),
              "calibration_only_no_post_boundary_audio": True,
              "separator_revision": SEP_REV, "matcher_revision": ENC_REV,
              "embedding_sample_rate_hz": 16000, "embedding_dimension": int(vectors.shape[1]),
              "embedding_file": "embeddings.npy", "embedding_sha256": sha256(directory / "embeddings.npy"),
              "speaker_order": [p["speaker_id"] for p in profiles], "profiles": profiles,
              "pairwise_profile_cosine": pairwise.tolist(), "possible_duplicate_pairs": suspicious_pairs,
              "mixture_fit_gains": gains.tolist(),
              "mixture_fit_relative_error": float(np.linalg.norm(scaled.sum(axis=1) - mix8)
                                                   / max(np.linalg.norm(mix8), 1e-12)),
              "separation_seconds": separated_at - begin,
              "profile_creation_seconds": time.perf_counter() - begin,
              "preview_gain": preview_gain, "warnings": warnings}
    write_json(directory / "profiles.json", record)
    return record, vectors


def load_profiles(directory):
    """Reload persisted profiles with checkpoint, ordering, and integrity checks."""
    record = json.loads((directory / "profiles.json").read_text())
    if (record["schema_version"] != 1 or record["matcher_revision"] != ENC_REV
            or record["separator_revision"] != SEP_REV or record["embedding_sample_rate_hz"] != 16000
            or record["embedding_file"] != "embeddings.npy"):
        raise ValueError("Incompatible generated profile bundle")
    if sha256(directory / "embeddings.npy") != record["embedding_sha256"]:
        raise ValueError("Embedding file checksum mismatch")
    vectors = np.load(directory / "embeddings.npy", allow_pickle=False)
    names = record["speaker_order"]
    if (vectors.shape != (len(names), 192) or not 1 <= len(names) <= 3
            or not np.isfinite(vectors).all()
            or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4)
            or names != [p["speaker_id"] for p in record["profiles"]]):
        raise ValueError("Invalid profile ordering or embedding shape/norm")
    for profile in record["profiles"]:
        path = directory / profile["enrollment_file"]
        if path.parent != directory or sha256(path) != profile["enrollment_sha256"]:
            raise ValueError("Enrollment file path/checksum mismatch")
    return record, vectors


def listening_page(output, profile_record, runs, manual_mapping):
    def audio(path):
        return f'<audio controls preload="none" src="{html.escape(path, quote=True)}"></audio>'
    cards = []
    for profile in profile_record["profiles"]:
        speaker = profile["speaker_id"]
        body = [f'<h2>{html.escape(profile["display_name"])} — {speaker}</h2>',
                '<p>Cuplikan enrollment otomatis: dengarkan dan tentukan siapa pemilik suara.</p>',
                audio(f"profiles/{speaker}_enrollment.wav"),
                '<p>Jalur kalibrasi utuh:</p>', audio(f"profiles/{speaker}_separated.wav")]
        for run in runs:
            control_name = manual_mapping.get(speaker) if run == "manual_control" else speaker
            if control_name:
                label = "only_" + control_name
                if label not in runs[run]["files"]:
                    continue
                description = "Profil otomatis" if run == "auto_profiles" else "Enrollment lama (pembanding)"
                body += [f'<p>{description} — audio setelah kalibrasi</p>',
                         audio(f"{run}/{label}_content.wav"),
                         '<p>Timeline dengan delay dan celah:</p>', audio(f"{run}/{label}_timeline.wav")]
        cards.append("<section>" + "\n".join(body) + "</section>")
    pairs = []
    for label in runs.get("auto_profiles", {}).get("files", {}):
        if "_and_" in label:
            pairs.append(f'<p>{html.escape(label)}</p>' + audio(f"auto_profiles/{label}_content.wav"))
            selected = label.split("_and_")
            if manual_mapping and all(s in manual_mapping for s in selected):
                controls = sorted([manual_mapping[s] for s in selected], key=list(PROFILES).index)
                control_label = "_and_".join(controls)
                if control_label in runs.get("manual_control", {}).get("files", {}):
                    pairs.append('<p>Enrollment lama (pembanding)</p>' + audio(f"manual_control/{control_label}_content.wav"))
    page = '''<!doctype html><html lang="id"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Enrollment otomatis</title>
<style>body{font:17px system-ui;max-width:960px;margin:32px auto;padding:0 20px;background:#f5f7fa;color:#182333}
section{background:white;padding:20px;margin:20px 0;border-radius:12px}audio{width:100%}p{line-height:1.5}</style>
<h1>Kalibrasi campuran → profil speaker → input lanjutan</h1>
<p>Speaker 01–03 adalah kandidat anonim, bukan hitungan speaker yang terverifikasi.
Audio enrollment berasal hanya dari bagian kalibrasi. Audio hasil memakai bagian setelahnya.
Kontrol manual, jika tersedia, memakai rekaman enrollment lama hanya sebagai pembanding.</p>
<p>Dengarkan kejernihan, speaker lain yang bocor, identitas yang bertukar, dan speaker pilihan yang hilang.
Content untuk kualitas; timeline untuk delay playback virtual. Rute mic/AirPods belum diukur.</p>'''
    page += "\n".join(cards) + '<section><h2>Dua speaker dipilih</h2>' + "\n".join(pairs) + "</section></html>"
    (output / "listen.html").write_text(page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-wav", type=Path, default=ROOT / "data/user_recordings/mixture/Mix.wav")
    parser.add_argument("--calibration-seconds", type=float, default=30.)
    parser.add_argument("--live-wav", type=Path, help="Separate subsequent recording; otherwise use input after calibration")
    parser.add_argument("--profiles-dir", type=Path, help="Reuse generated bundle; requires --live-wav")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--context-ms", type=int, default=1500)
    parser.add_argument("--hop-ms", type=int, default=200)
    parser.add_argument("--skip-replay", action="store_true", help="Only create profiles")
    parser.add_argument("--selected-speakers", nargs="+", help="Only emit this selected set, e.g. speaker_01 speaker_03")
    parser.add_argument("--compare-manual-enrollments", action="store_true", help="Optional old enrollment control, not used to create profiles")
    args = parser.parse_args()
    if not math.isfinite(args.calibration_seconds) or args.calibration_seconds < 2:
        parser.error("Calibration must be at least 2 seconds")
    if args.hop_ms <= 0 or args.hop_ms % 20 or args.context_ms <= 0:
        parser.error("Hop must be a positive multiple of 20 ms; context must be positive")
    if args.profiles_dir and (not args.live_wav or args.skip_replay):
        parser.error("--profiles-dir requires --live-wav and replay")
    if args.compare_manual_enrollments and args.selected_speakers:
        parser.error("Manual comparison exports all selections; omit --selected-speakers")
    if args.selected_speakers and (len(set(args.selected_speakers)) != len(args.selected_speakers)
                                  or not set(args.selected_speakers) <= {"speaker_01", "speaker_02", "speaker_03"}):
        parser.error("Selected speakers must be distinct speaker_01 / speaker_02 / speaker_03 IDs")
    if args.output_dir.exists():
        raise FileExistsError("Use a new output folder")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; run with Metal GPU access, or explicitly choose --device cpu")
    torch.set_num_threads(4)
    setup_begin = time.perf_counter()
    source_hashes = {}
    wave = rate = boundary = None
    if not args.profiles_dir:
        wave, rate = read_mono(args.input_wav)
        boundary = calibration_boundary(wave, rate, args.calibration_seconds)
        source_hashes[str(args.input_wav.resolve())] = sha256(args.input_wav)
    live = live_rate = None
    live_source = None
    if not args.skip_replay:
        if args.live_wav:
            if not args.profiles_dir and args.live_wav.resolve() == args.input_wav.resolve():
                parser.error("Use a different --live-wav or omit it to use the disjoint suffix")
            live, live_rate = read_mono(args.live_wav)
            source_hashes[str(args.live_wav.resolve())] = sha256(args.live_wav)
            if args.profiles_dir:
                reused, _ = load_profiles(args.profiles_dir.resolve())
                if source_hashes[str(args.live_wav.resolve())] == reused["source"]["sha256"]:
                    parser.error("Replay must differ from the calibration recording")
            live_source = {"path": str(args.live_wav.resolve()), "start_frame": 0, "end_frame": len(live)}
        else:
            live, live_rate = wave[boundary:].copy(), rate
            live_source = {"path": str(args.input_wav.resolve()), "start_frame": boundary, "end_frame": len(wave)}
        if len(live) / live_rate < args.context_ms / 1000 + 2 * args.hop_ms / 1000:
            raise ValueError("Need enough subsequent audio for full-context replay; use --skip-replay for calibration only")
    args.output_dir.mkdir(parents=True)
    separator = load_separator(args.device)
    matcher = EncoderClassifier.from_hparams(
        source=str((ROOT / "ecapa_checkpoint").resolve()), savedir=str((ROOT / "ecapa_loaded").resolve()),
        overrides={"pretrained_path": str((ROOT / "ecapa_checkpoint").resolve())}, run_opts={"device": "cpu"}).eval()
    model_load_seconds = time.perf_counter() - setup_begin
    profiles_dir = args.output_dir / "profiles"
    if args.profiles_dir:
        import shutil
        shutil.copytree(args.profiles_dir, profiles_dir)
        profile_record, enrollments = load_profiles(profiles_dir)
    else:
        print(f"Creating profiles from first {boundary/rate:.3f}s only", flush=True)
        source = {"path": str(args.input_wav.resolve()), "sha256": sha256(args.input_wav),
                  "sample_rate_hz": rate, "start_frame": 0, "end_frame": boundary}
        profile_record, enrollments = build_profiles(wave[:boundary], rate, separator, matcher, profiles_dir, source)
        loaded_record, loaded_vectors = load_profiles(profiles_dir)
        if not np.array_equal(enrollments, loaded_vectors):
            raise AssertionError("Stored profile embeddings differ after reload")
        profile_record, enrollments = loaded_record, loaded_vectors
    names = profile_record["speaker_order"]
    print(json.dumps({"speaker_ids": names, "profile_creation_seconds": profile_record["profile_creation_seconds"],
                      "pairwise_profile_cosine": profile_record["pairwise_profile_cosine"]}), flush=True)
    experiment = {"scope": "generated candidate enrollments + paced post-calibration file input; virtual playback",
                  "profile_bundle": "profiles/profiles.json", "calibration_seconds": profile_record["calibration_seconds"],
                  "device": args.device, "context_ms": args.context_ms, "hop_ms": args.hop_ms,
                  "model_load_seconds": model_load_seconds, "reused_profile_bundle": bool(args.profiles_dir),
                  "source_sha256": source_hashes, "live_source": live_source,
                  "profile_creation_uses_manual_enrollments": False,
                  "subsequent_audio_used_to_create_profiles": False,
                  "startup_policy": "profiles and warmup finish before paced source starts; no concurrent mic capture during calibration processing",
                  "runs": {}, "manual_control": {}, "warnings": profile_record["warnings"] + [
                      "calibration duration plus profile processing/warmup is startup, not per-block latency",
                      "no physical microphone or AirPods route", "no clean references or separation quality score"]}
    write_json(args.output_dir / "experiment.json", experiment)
    if args.skip_replay:
        experiment["original_inputs_unchanged"] = all(sha256(Path(p)) == digest for p, digest in source_hashes.items())
        if not experiment["original_inputs_unchanged"]:
            raise AssertionError("Original input changed")
        write_json(args.output_dir / "experiment.json", experiment)
        write_json(args.output_dir / "verification.json", {
            "persisted_profiles_reloaded": True, "source_files_unchanged": True,
            "profile_audio_boundary": profile_record["source"], "replay_performed": False,
            "subsequent_audio_used_for_enrollment": False, "listening_required": True})
        listening_page(args.output_dir, profile_record, {}, {})
        print(f"Profiles ready for listening: {args.output_dir / 'listen.html'}", flush=True)
        return
    live_matcher = HybridMatcher(matcher) if args.device == "mps" else matcher
    base = WindowSeparator(separator, live_matcher, enrollments, names=names)
    if args.selected_speakers:
        if len(set(args.selected_speakers)) != len(args.selected_speakers) or not set(args.selected_speakers) <= set(names):
            parser.error("Selected speakers must be distinct IDs from this profile bundle")
        base.selections = [tuple(args.selected_speakers)]
    # Compile every startup shape before measurement, as in the accepted MPS run.
    warm_begin = time.perf_counter()
    hop = args.hop_ms / 1000
    total = args.context_ms / 1000 + hop
    warm_seconds = hop
    while warm_seconds < total:
        base(np.zeros(round(live_rate * warm_seconds), dtype="float32"), live_rate, warm_seconds)
        warm_seconds += hop
    base(np.zeros(round(live_rate * total), dtype="float32"), live_rate, total)
    if args.device == "mps":
        torch.mps.synchronize()
    experiment["warmup_seconds"] = time.perf_counter() - warm_begin
    manual_mapping = {}
    processors = {"auto_profiles": base}
    if args.compare_manual_enrollments:
        manual_vectors = []
        manual_hashes = {}
        for name, relative in PROFILES.items():
            path = ROOT / "data/user_recordings" / relative
            sample, sr = read_mono(path)
            manual_vectors.append(embed(matcher, resample(sample, sr, 16000)))
            manual_hashes[str(path.resolve())] = sha256(path)
        manual_vectors = np.stack(manual_vectors)
        cosine = enrollments @ manual_vectors.T
        rows, cols = linear_sum_assignment(-cosine)
        manual_mapping = {names[row]: list(PROFILES)[col] for row, col in zip(rows, cols)}
        experiment["manual_control"] = {"source_sha256": manual_hashes,
            "anonymous_to_manual_profile_assignment": manual_mapping,
            "auto_to_manual_profile_cosine": cosine.tolist(),
            "scope": "post-hoc diagnostic/control only; this assignment is not proof of identity"}
        source_hashes.update(manual_hashes)
        processors["manual_control"] = WindowSeparator(separator, live_matcher, manual_vectors)
    for label, processor in processors.items():
        rolling = RollingSeparator(processor, args.context_ms)
        synchronized = SynchronizedProcessor(rolling) if args.device == "mps" else rolling
        print(f"Starting {label}, {len(live)/live_rate:.3f}s disjoint subsequent input", flush=True)
        summary, records, outputs = paced_run(live, live_rate, hop, synchronized, label)
        context_diagnostics(summary, records, args.context_ms)
        summary["speaker_order"] = processor.names
        summary["separator_device"] = args.device
        summary["live_source"] = live_source
        save_run(args.output_dir / label, summary, records, outputs)
        expected = math.ceil(len(live) * 16000 / live_rate)
        for selection, files in summary["files"].items():
            info = sf.info(args.output_dir / label / files["raw_content"])
            if info.frames != expected or info.samplerate != 16000:
                raise AssertionError(f"Lost input frames: {selection}")
        for record in records:
            if (record["model_input_end_frame"] != record["emitted_end_frame"]
                    or record["available_history_seconds"] > args.context_ms / 1000 + 1e-6):
                raise AssertionError("Model exceeded captured input/history boundary")
        experiment["runs"][label] = summary
        write_json(args.output_dir / "experiment.json", experiment)
        print(json.dumps({"run": label, "steady_processing_seconds": summary["steady_processing_seconds"],
                          "virtual_play_delay_seconds": summary["virtual_play_delay_seconds"]}), flush=True)
    if "manual_control" in experiment["runs"]:
        auto_trace = [json.loads(line) for line in (args.output_dir / "auto_profiles/trace.jsonl").read_text().splitlines()]
        manual_trace = [json.loads(line) for line in (args.output_dir / "manual_control/trace.jsonl").read_text().splitlines()]
        steady = [i for i, r in enumerate(auto_trace) if r["steady_context"] and r["full_window"]]
        experiment["manual_control"]["steady_windows_compared"] = len(steady)
        experiment["manual_control"]["stream_assignment_disagreements_by_profile"] = {
            speaker: sum(auto_trace[i]["matched_stream_by_speaker"][speaker]
                         != manual_trace[i]["matched_stream_by_speaker"][manual_name] for i in steady)
            for speaker, manual_name in manual_mapping.items()}
        experiment["manual_control"]["disagreement_interpretation"] = "neither assignment is ground truth; listening needed"
    experiment["original_inputs_unchanged"] = all(sha256(Path(p)) == digest for p, digest in source_hashes.items())
    if not experiment["original_inputs_unchanged"]:
        raise AssertionError("Original input changed")
    write_json(args.output_dir / "experiment.json", experiment)
    listening_page(args.output_dir, profile_record, experiment["runs"], manual_mapping)
    write_json(args.output_dir / "verification.json", {
        "persisted_profiles_reloaded": True, "source_files_unchanged": True,
        "profile_audio_boundary": profile_record["source"], "live_audio_boundary": live_source,
        "calibration_and_replay_disjoint_by_source_frames": not bool(args.live_wav),
        "subsequent_audio_used_for_enrollment": False, "output_frame_accounting_passed": True,
        "past_only_model_window_checks_passed": True, "listening_required": True})
    print(f"Complete: {args.output_dir / 'listen.html'}", flush=True)


if __name__ == "__main__":
    main()
