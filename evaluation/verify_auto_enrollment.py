"""Verify calibration provenance, persisted profiles, replay artifacts, and reuse."""

import argparse
import json
import math
import re
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from experiment_auto_enrollment import build_profiles, calibration_boundary, listening_page, load_profiles, write_json
from run_user_recordings import read_mono, sha256


def verify(directory, reuse_directory=None):
    directory = directory.resolve()
    experiment = json.loads((directory / "experiment.json").read_text())
    bundle, embeddings = load_profiles(directory / "profiles")
    source = bundle["source"]
    assert sha256(Path(source["path"])) == source["sha256"]
    assert source["start_frame"] == 0
    assert source["end_frame"] == round(bundle["calibration_seconds"] * source["sample_rate_hz"])
    assert experiment["profile_creation_uses_manual_enrollments"] is False
    assert experiment["subsequent_audio_used_to_create_profiles"] is False
    live = experiment["live_source"]
    if live["path"] == source["path"]:
        assert live["start_frame"] == source["end_frame"]
    for profile in bundle["profiles"]:
        name = profile["speaker_id"]
        raw, rate = sf.read(directory / "profiles" / f"{name}_separated_raw.wav", dtype="float32")
        cropped, enrollment_rate = sf.read(directory / "profiles" / profile["enrollment_file"], dtype="float32")
        assert rate == enrollment_rate == 16000
        assert len(raw) == math.ceil(bundle["calibration_seconds"] * 16000)
        spans = profile["retained_source_spans_samples_16k"]
        assert all(0 <= a < b <= len(raw) for a, b in spans)
        assert all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1))
        expected = np.concatenate([raw[a:b] for a, b in spans]) * profile["enrollment_gain"]
        np.testing.assert_array_equal(cropped, expected)
        assert np.isfinite(cropped).all() and np.max(np.abs(cropped)) <= .900001
    artifact_checks = {}
    for run, summary in experiment["runs"].items():
        trace = [json.loads(line) for line in (directory / run / "trace.jsonl").read_text().splitlines()]
        assert len(trace) == summary["windows"]
        assert trace[0]["emitted_start_frame"] == 0
        assert trace[-1]["emitted_end_frame"] == live["end_frame"] - live["start_frame"]
        assert all(a["emitted_end_frame"] == b["emitted_start_frame"] for a, b in zip(trace, trace[1:]))
        assert all(r["model_input_end_frame"] == r["emitted_end_frame"]
                   and r["model_input_start_frame"] >= 0
                   and r["processing_start_seconds"] >= r["capture_end_seconds"]
                   and r["available_history_seconds"] <= experiment["context_ms"] / 1000 + 1e-6
                   for r in trace)
        total_frames = sum(r["emitted_samples_16k"] for r in trace)
        wav_count = 0
        for selection, files in summary["files"].items():
            raw, sr = sf.read(directory / run / files["raw_content"], dtype="float32")
            preview, psr = sf.read(directory / run / files["content_preview"], dtype="float32")
            timeline, tsr = sf.read(directory / run / files["timeline_preview"], dtype="float32")
            assert sr == psr == tsr == 16000
            assert len(raw) == len(preview) == total_frames
            assert np.isfinite(raw).all() and np.isfinite(timeline).all()
            assert np.max(np.abs(preview)) < 1 and np.max(np.abs(timeline)) < 1
            gain = summary["preview_gain_applied_after_timing"]
            assert np.max(np.abs(preview - raw * gain)) <= 1 / 32768 + 1e-6
            content_offset = written_end = 0
            for r in trace:
                begin = max(written_end, round(r["virtual_play_start_seconds"] * 16000))
                size = r["emitted_samples_16k"]
                assert not np.any(timeline[written_end:begin])
                np.testing.assert_array_equal(timeline[begin:begin + size], preview[content_offset:content_offset + size])
                written_end = begin + size
                content_offset += size
            assert len(timeline) == written_end
            wav_count += 3
        artifact_checks[run] = {"wav_count": wav_count, "content_frames": total_frames,
                                "trace_boundaries_and_timeline_passed": True}
    mapping = experiment["manual_control"].get("anonymous_to_manual_profile_assignment", {})
    if "manual_control" in experiment["runs"]:
        a = [json.loads(line) for line in (directory / "auto_profiles/trace.jsonl").read_text().splitlines()]
        b = [json.loads(line) for line in (directory / "manual_control/trace.jsonl").read_text().splitlines()]
        steady = [i for i, r in enumerate(a) if r["steady_context"] and r["full_window"]]
        experiment["manual_control"]["steady_windows_compared"] = len(steady)
        experiment["manual_control"]["stream_assignment_disagreements_by_profile"] = {
            s: sum(a[i]["matched_stream_by_speaker"][s] != b[i]["matched_stream_by_speaker"][n]
                   for i in steady) for s, n in mapping.items()}
        experiment["manual_control"]["disagreement_interpretation"] = "neither assignment is ground truth; listening needed"
        write_json(directory / "experiment.json", experiment)
    listening_page(directory, bundle, experiment["runs"], mapping)
    links = re.findall(r'<audio[^>]*src="([^"]+)"', (directory / "listen.html").read_text())
    assert all((directory / path).is_file() for path in links)
    reuse = None
    if reuse_directory:
        other, other_embeddings = load_profiles(reuse_directory.resolve() / "profiles")
        np.testing.assert_array_equal(embeddings, other_embeddings)
        assert other["source"] == source
        reused_experiment = json.loads((reuse_directory / "experiment.json").read_text())
        assert reused_experiment["reused_profile_bundle"]
        assert list(reused_experiment["runs"]["auto_profiles"]["files"]) == ["speaker_01_and_speaker_03"]
        left, _ = sf.read(directory / "auto_profiles/speaker_01_and_speaker_03_content_raw.wav", dtype="float32")
        right, _ = sf.read(reuse_directory / "auto_profiles/speaker_01_and_speaker_03_content_raw.wav", dtype="float32")
        error = float(np.linalg.norm(left[:len(right)] - right) / max(np.linalg.norm(right), 1e-12))
        assert error < .005, f"Reloaded profiles changed the selected-pair output: {error}"
        reuse = {"unchanged_embeddings": True, "selected_pair_only": True,
                 "relative_rms_error_vs_original_pair": error}
    # Corrupt embeddings must never be accepted as a compatible profile bundle.
    with tempfile.TemporaryDirectory(prefix="tse-profile-check-") as temp:
        temp = Path(temp)
        (temp / "profiles.json").write_text((directory / "profiles/profiles.json").read_text())
        changed = embeddings.copy()
        changed[0, 0] += .1
        np.save(temp / "embeddings.npy", changed, allow_pickle=False)
        try:
            load_profiles(temp)
        except ValueError as error:
            assert "checksum" in str(error)
        else:
            raise AssertionError("Corrupt embeddings accepted")
        try:
            build_profiles(np.zeros(160000, dtype="float32"), 16000, None, None, temp / "silent", {})
        except ValueError as error:
            assert "silent" in str(error)
        else:
            raise AssertionError("Silent calibration accepted")
        # Existing 23.125 s file must fail the 30 s default, not repeat/pad it.
        short, short_rate = read_mono(Path(source["path"]))
        try:
            calibration_boundary(short, short_rate)
        except ValueError as error:
            assert "shorter than requested 30 s" in str(error)
        else:
            raise AssertionError("Short recording accepted as 30 s calibration")
    assert all(sha256(Path(path)) == digest for path, digest in experiment["source_sha256"].items())
    report = {"persisted_profiles_reloaded": True, "source_files_unchanged": True,
              "profile_audio_boundary": source, "live_audio_boundary": live,
              "enrollment_samples_reconstructed_from_calibration_only": True,
              "subsequent_audio_used_for_enrollment": False, "artifact_checks": artifact_checks,
              "listening_player_links_verified": len(links), "profile_reuse_selected_pair": reuse,
              "corrupt_embeddings_rejected": True, "silent_calibration_rejected": True,
              "default_30s_rejects_short_input_without_padding": True, "listening_required": True}
    write_json(directory / "verification.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reuse-run-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.run_dir, args.reuse_run_dir), indent=2))
