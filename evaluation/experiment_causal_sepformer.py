"""Diagnostic causal SepFormer conversion; never overwrites the pretrained model.

This is an inference-time experiment, not a trained streaming checkpoint. It
compares the original model, attention masking alone, and attention masking
plus per-time-step normalization on one reproducible Libri3Mix mixture.
"""

import json
import time
import types
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf
import torch
from scipy.optimize import linear_sum_assignment
from speechbrain.inference.separation import SepformerSeparation
from speechbrain.lobes.models.dual_path import CumulativeLayerNorm, SBTransformerBlock

from score_outputs import si_sdr


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/m2_three_speakers"
OUTPUT = ROOT / "data/causal_sepformer_experiment"
MODEL = ROOT / "sepformer_libr3mix_checkpoint"
LOADED = ROOT / "sepformer_libr3mix_loaded"
MODEL_REVISION = "1f2d824e2333aec9e60d0030fb144885c1afc426"
SAMPLE_RATE = 8000


def masked_forward(self, x):
    # True entries prohibit attention to a later position in the same sequence.
    length = x.size(1)
    future_mask = torch.ones(
        (length, length), dtype=torch.bool, device=x.device
    ).triu_(diagonal=1)
    if self.use_positional_encoding:
        x = x + self.pos_enc(x)
    return self.mdl(x, src_mask=future_mask)[0]


def add_attention_masks(masknet):
    blocks = [m for m in masknet.modules() if isinstance(m, SBTransformerBlock)]
    if len(blocks) != 4:
        raise RuntimeError(f"Expected 4 transformer blocks; found {len(blocks)}")
    for block in blocks:
        block.forward = types.MethodType(masked_forward, block)
    return len(blocks)


def replace_global_norms(masknet):
    count = 0

    def visit(parent):
        nonlocal count
        for name, child in list(parent.named_children()):
            if isinstance(child, torch.nn.GroupNorm):
                if child.num_groups != 1:
                    raise RuntimeError(f"Unexpected GroupNorm group count: {child.num_groups}")
                local = CumulativeLayerNorm(child.num_channels, eps=child.eps)
                with torch.no_grad():
                    local.weight.copy_(child.weight)
                    local.bias.copy_(child.bias)
                setattr(parent, name, local)
                count += 1
            else:
                visit(child)

    visit(masknet)
    if count != 5:
        raise RuntimeError(f"Expected 5 global norms; found {count}")
    return count


def separate(model, wave):
    with torch.inference_mode():
        return model.separate_batch(
            torch.from_numpy(np.asarray(wave, dtype=np.float32)).unsqueeze(0)
        )[0].cpu().numpy()


def prefix_test(model, wave):
    short = separate(model, wave[:int(1.5 * SAMPLE_RATE)])
    long = separate(model, wave[:int(2.5 * SAMPLE_RATE)])
    # The encoder has a 16-sample kernel and 8-sample stride. Keep the final
    # 8 samples separate to expose its observed ~1 ms boundary lookahead.
    boundary_samples = 8
    stable = short[:-boundary_samples] - long[:len(short) - boundary_samples]
    all_error = short - long[:len(short)]
    rms_short = float(np.sqrt(np.mean(short[:-boundary_samples] ** 2)))
    return {
        "short_seconds": 1.5,
        "long_seconds": 2.5,
        "boundary_samples_excluded": boundary_samples,
        "before_boundary_error_rms": float(np.sqrt(np.mean(stable ** 2))),
        "before_boundary_error_rms_relative_to_short_output": float(
            np.sqrt(np.mean(stable ** 2)) / max(rms_short, 1e-12)
        ),
        "before_boundary_max_absolute_error": float(np.max(np.abs(stable))),
        "including_boundary_max_absolute_error": float(np.max(np.abs(all_error))),
        "samples_with_error_over_1e_minus_4": int(
            np.count_nonzero(np.max(np.abs(all_error), axis=1) > 1e-4)
        ),
    }


def benchmark(model, wave):
    records = {}
    for seconds in (0.25, 0.5, 1.0):
        segment = np.asarray(wave[:int(seconds * SAMPLE_RATE)], dtype=np.float32)
        separate(model, segment)  # warmup for this shape
        durations = []
        for _ in range(3):
            start = time.perf_counter()
            separate(model, segment)
            durations.append(time.perf_counter() - start)
        records[str(seconds)] = {
            "runs_seconds": [round(t, 4) for t in durations],
            "median_seconds": round(float(np.median(durations)), 4),
        }
    return records


def quality(estimate, sources, speaker_ids):
    scores = np.array([
        [si_sdr(ref, estimate[:, j]) for j in range(3)] for ref in sources
    ])
    rows, cols = linear_sum_assignment(-scores)
    assignment = {speaker_ids[i]: int(j) for i, j in zip(rows, cols)}
    selected_scores = {speaker_ids[i]: float(scores[i, j]) for i, j in zip(rows, cols)}
    return assignment, selected_scores


def main():
    torch.set_num_threads(4)
    trial = json.loads((DATA / "single_trials.jsonl").read_text().splitlines()[0])
    mix16, mix_rate = sf.read(DATA / trial["mixture_path"], dtype="float32")
    if mix_rate != 16000 or mix16.ndim != 1:
        raise RuntimeError("Expected mono 16 kHz Libri3Mix example")
    mix8 = scipy.signal.resample_poly(mix16, 1, 2).astype("float32")
    speaker_ids = [source["speaker_id"] for source in trial["sources"]]
    sources = []
    for source in trial["sources"]:
        audio, rate = sf.read(DATA / source["source_path"], dtype="float32")
        if rate != 16000 or len(audio) != len(mix16):
            raise RuntimeError("Clean source mismatch")
        sources.append(scipy.signal.resample_poly(audio, 1, 2).astype("float32"))

    model = SepformerSeparation.from_hparams(
        source=str(MODEL), savedir=str(LOADED), run_opts={"device": "cpu"}
    )
    model.eval()
    if model.hparams.sample_rate != 8000 or model.hparams.num_spks != 3:
        raise RuntimeError("Unexpected model configuration")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": "One clean Libri3Mix mixture; diagnostic inference-time edits; no retraining",
        "model_revision": MODEL_REVISION,
        "trial_id": trial["trial_id"],
        "mixture_seconds": len(mix8) / SAMPLE_RATE,
        "hardware": "local CPU, torch 4 threads",
        "variants": {},
    }

    for variant in ("original", "masked_only", "masked_local_norm"):
        if variant == "masked_only":
            report["transformer_blocks_masked"] = add_attention_masks(model.mods.masknet)
        if variant == "masked_local_norm":
            report["global_norms_replaced"] = replace_global_norms(model.mods.masknet)
        estimate = separate(model, mix8)
        if estimate.shape != (len(mix8), 3) or not np.isfinite(estimate).all():
            raise RuntimeError(f"Bad separation output: {estimate.shape}")
        assignment, scores = quality(estimate, sources, speaker_ids)
        # Single gain per variant keeps all three streams playable without
        # changing their relative levels. Scoring uses unmodified float output.
        preview_gain = min(1.0, 0.9 / max(float(np.max(np.abs(estimate))), 1e-12))
        for speaker_id, column in assignment.items():
            preview = scipy.signal.resample_poly(
                estimate[:, column] * preview_gain, 2, 1
            ).astype("float32")
            sf.write(OUTPUT / f"{variant}_{speaker_id}.wav", preview, 16000,
                     subtype="PCM_16")
        report["variants"][variant] = {
            "prefix_test": prefix_test(model, mix8),
            "si_sdr_db_by_speaker_oracle_assignment": scores,
            "mean_si_sdr_db": float(np.mean(list(scores.values()))),
            "oracle_stream_by_speaker": assignment,
            "preview_gain": preview_gain,
            "benchmark": benchmark(model, mix8),
        }
        print(variant, json.dumps(report["variants"][variant], indent=2), flush=True)

    (OUTPUT / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {OUTPUT / 'results.json'}")


if __name__ == "__main__":
    main()
