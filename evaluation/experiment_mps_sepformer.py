"""MPS acceleration and 100/200 ms live-paced SepFormer replay.

Runs the original pretrained separator without changing its architecture.
MPS work is synchronized before/after timing; CPU resampling and mixture-fit
are included in the pipeline. CPU fallback for unsupported MPS ops is disabled.
"""

import argparse
import copy
import json
import os
import platform
import time
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import numpy as np
import soundfile as sf
import torch
from speechbrain.inference.classifiers import EncoderClassifier
from speechbrain.inference.separation import SepformerSeparation

from experiment_rolling_context import RollingSeparator, context_diagnostics, export_listening_page
from run_user_recordings import PROFILES, ROOT, embed, read_mono, resample, sha256
from simulate_realtime import WindowSeparator, paced_run, save_run, stats


class HybridMatcher:
    """Keep FFT/features on CPU, run pretrained ECAPA embedding layers on MPS."""
    def __init__(self, cpu_matcher):
        self.features = cpu_matcher.mods.compute_features
        self.normalizer = cpu_matcher.mods.mean_var_norm
        self.embedding = copy.deepcopy(cpu_matcher.mods.embedding_model).to("mps").eval()

    def encode_batch(self, wavs):
        wavs = wavs.float().cpu()
        if wavs.ndim == 1:
            wavs = wavs.unsqueeze(0)
        lengths = torch.ones(wavs.shape[0])
        feats = self.normalizer(self.features(wavs), lengths)
        return self.embedding(feats.to("mps"), lengths.to("mps"))


def sync():
    torch.mps.synchronize()


def load_separator(device):
    return SepformerSeparation.from_hparams(
        source=str((ROOT / "sepformer_libr3mix_checkpoint").resolve()),
        savedir=str((ROOT / "sepformer_libr3mix_loaded").resolve()),
        run_opts={"device": device}).eval()


def difference(reference, actual):
    ref = reference.astype("float64")
    delta = actual.astype("float64") - ref
    return {"relative_rms_error": float(np.linalg.norm(delta) / max(np.linalg.norm(ref), 1e-12)),
            "max_abs_error": float(np.max(np.abs(delta)))}


def probe(wave, rate, seconds, cpu_separator, mps_separator, cpu_matcher, enrollments):
    sample = wave[:round(rate * seconds)]
    mono = sample.mean(axis=1) if sample.ndim == 2 else sample
    mix8 = resample(mono, rate, 8000)
    with torch.inference_mode():
        cpu_streams = cpu_separator.separate_batch(torch.from_numpy(mix8).unsqueeze(0))[0].cpu().numpy()
        mps_streams = mps_separator.separate_batch(torch.from_numpy(mix8).unsqueeze(0))[0].cpu().numpy()
    sync()
    if not np.isfinite(mps_streams).all():
        raise ValueError("Nonfinite MPS separator output")
    parity = difference(cpu_streams, mps_streams)
    # This is a numerical guard, not a source-separation quality claim.
    if parity["relative_rms_error"] > .01:
        raise ValueError(f"MPS differs from CPU by more than 1% RMS: {parity}")
    cpu_processor = WindowSeparator(cpu_separator, cpu_matcher, enrollments)
    cpu_output, cpu_details = cpu_processor(sample, rate, seconds)
    candidates = {"cpu": cpu_matcher}
    errors = {}
    try:
        candidates["cpu_features_mps_embedding"] = HybridMatcher(cpu_matcher)
    except (RuntimeError, NotImplementedError) as error:
        errors["cpu_features_mps_embedding"] = str(error)
    measurements = {}
    for label, matcher in candidates.items():
        processor = WindowSeparator(mps_separator, matcher, enrollments)
        try:
            processor(sample, rate, seconds)
            elapsed, separator_elapsed, output_parity = [], [], {}
            for _ in range(3):
                sync()
                start = time.perf_counter()
                result, details = processor(sample, rate, seconds)
                sync()
                elapsed.append(time.perf_counter() - start)
                separator_elapsed.append(details["separator_seconds"])
            for selected in cpu_output:
                output_parity[selected] = difference(cpu_output[selected], result[selected])
            same_mapping = details["matched_stream_by_speaker"] == cpu_details["matched_stream_by_speaker"]
            if not same_mapping or max(v["relative_rms_error"] for v in output_parity.values()) > .01:
                raise ValueError("MPS pipeline did not preserve CPU mapping/output within 1% on probe")
            measurements[label] = {"processing_seconds": stats(elapsed),
                                   "separator_seconds": stats(separator_elapsed),
                                   "same_speaker_mapping_as_cpu": same_mapping,
                                   "output_parity_vs_cpu": output_parity}
        except (RuntimeError, NotImplementedError, ValueError) as error:
            errors[label] = str(error)
    if not measurements:
        raise RuntimeError(f"No verified MPS pipeline: {errors}")
    selected = min(measurements, key=lambda label: measurements[label]["processing_seconds"]["p50"])
    return {"total_window_seconds": seconds, "separator_parity_vs_cpu": parity,
            "measurements": measurements, "unsupported_or_rejected_variants": errors,
            "selected_matcher": selected}, candidates[selected]


class SynchronizedProcessor:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, wave, rate, seconds):
        sync()
        result, details = self.processor(wave, rate, seconds)
        sync()
        return result, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-ms", type=int, default=1500)
    parser.add_argument("--hop-ms", type=int, nargs="+", default=[100, 200])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()
    if args.context_ms <= 0 or any(hop <= 0 or hop % 20 for hop in args.hop_ms):
        parser.error("Context must be positive; hops must be positive multiples of 20 ms")
    if args.output_dir.exists():
        raise FileExistsError(f"Use a new output folder: {args.output_dir}")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; no CPU substitution. Run with access to Metal GPU.")
    torch.set_num_threads(4)
    inputs = ROOT / "data/user_recordings"
    mix_path = inputs / "mixture/Mix.wav"
    wave, rate = sf.read(mix_path, dtype="float32")
    if not len(wave) or not np.isfinite(wave).all():
        raise ValueError("Invalid input mixture")
    hashes = {str(path.relative_to(inputs)): sha256(path)
              for path in [mix_path] + [inputs / p for p in PROFILES.values()]}
    output = args.output_dir
    output.mkdir(parents=True)
    setup = time.perf_counter()
    cpu_separator = load_separator("cpu")
    mps_separator = load_separator("mps")
    matcher = EncoderClassifier.from_hparams(
        source=str((ROOT / "ecapa_checkpoint").resolve()),
        savedir=str((ROOT / "ecapa_loaded").resolve()),
        overrides={"pretrained_path": str((ROOT / "ecapa_checkpoint").resolve())},
        run_opts={"device": "cpu"}).eval()
    enrolled = []
    for relative in PROFILES.values():
        audio, original_rate = read_mono(inputs / relative)
        enrolled.append(embed(matcher, resample(audio, original_rate, 16000)))
    enrollments = np.stack(enrolled)
    experiment = {
        "date_jakarta": "2026-10-09",
        "scope": "MPS accelerated original SepFormer; live-paced input and virtual playback",
        "platform": platform.platform(), "torch_version": torch.__version__,
        "torch_cpu_threads": 4, "mps_available": True, "mps_cpu_fallback_enabled": False,
        "input_file_sha256": hashes, "input_rate_hz": rate, "input_seconds": len(wave)/rate,
        "separator_revision": "1f2d824e2333aec9e60d0030fb144885c1afc426",
        "matcher_revision": "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286",
        "context_ms": args.context_ms, "hop_ms": args.hop_ms,
        "setup_seconds_excluded_from_replay": time.perf_counter() - setup,
        "warnings": ["no physical mic or playback device", "no clean reference", "listening pending",
                     "no target-absence rejection", "past audio reprocessed, no separator state cache",
                     "no crossfade, no additional future lookahead", "single short input"],
        "probes": {}, "runs": {},
    }
    def write_record():
        (output / "experiment.json").write_text(json.dumps(experiment, indent=2, allow_nan=False) + "\n")
    runs = []
    for hop_ms in args.hop_ms:
        hop_seconds = hop_ms / 1000
        seconds = args.context_ms / 1000 + hop_seconds
        print(f"Probing MPS total window {seconds:.3f}s, hop {hop_ms}ms", flush=True)
        report, chosen_matcher = probe(wave, rate, seconds, cpu_separator, mps_separator, matcher, enrollments)
        experiment["probes"][str(hop_ms)] = report
        write_record()
        print(json.dumps({"hop_ms": hop_ms, "selected_matcher": report["selected_matcher"],
                          "processing_seconds": report["measurements"][report["selected_matcher"]]["processing_seconds"],
                          "rejected": report["unsupported_or_rejected_variants"]}), flush=True)
        if args.probe_only:
            continue
        base = WindowSeparator(mps_separator, chosen_matcher, enrollments)
        # Warm every startup length so first-use MPS graph compilation is outside
        # the measured replay, as CPU warm-up was in previous experiments.
        model_seconds = hop_seconds
        while model_seconds < seconds:
            base(np.zeros(round(rate * model_seconds), dtype="float32"), rate, model_seconds)
            model_seconds += hop_seconds
        base(np.zeros(round(rate * seconds), dtype="float32"), rate, seconds)
        sync()
        processor = SynchronizedProcessor(RollingSeparator(base, args.context_ms))
        label = f"mps_context_{args.context_ms}ms_hop_{hop_ms}ms"
        print(f"Starting paced replay {label}", flush=True)
        summary, records, outputs = paced_run(wave, rate, hop_seconds, processor, label)
        context_diagnostics(summary, records, args.context_ms)
        summary["separator_device"] = "mps"
        summary["matcher_execution"] = report["selected_matcher"]
        summary["virtual_p95_below_500ms"] = summary["virtual_play_delay_seconds"]["p95"] < .5
        save_run(output / label, summary, records, outputs)
        experiment["runs"][label] = summary
        runs.append((label, args.context_ms))
        write_record()
        print(json.dumps({"run": label, "steady_processing_seconds": summary["steady_processing_seconds"],
                          "steady_budget_misses": summary["steady_processing_budget_misses"],
                          "virtual_play_delay_seconds": summary["virtual_play_delay_seconds"]}), flush=True)
    if runs:
        previous = inputs / "rolling_context/2026-10-07_context_1500ms"
        previous_dirs = [previous] if previous.is_dir() else []
        experiment["listening_comparison"] = export_listening_page(output, inputs, runs, previous_dirs)
    experiment["original_inputs_unchanged"] = all(sha256(inputs / p) == digest for p, digest in hashes.items())
    if not experiment["original_inputs_unchanged"]:
        raise AssertionError("Original recording changed")
    write_record()
    print(f"Complete: {output}", flush=True)


if __name__ == "__main__":
    main()
