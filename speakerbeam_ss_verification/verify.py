"""Offline compatibility check using only upstream demo clips (no evaluation corpus)."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "upstream"))

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    import resemblyzer
    from resemblyzer import VoiceEncoder
    from model import SpeakerBeamSS
    from tools import get_speaker_embeddings_batch

    torch.set_num_threads(4)
    torch.manual_seed(0)
    outdir = ROOT / "outputs"
    outdir.mkdir(exist_ok=True)
    report = {
        "status": "started", "scope": "Repository demo only; not a quality benchmark",
        "platform": platform.platform(), "python": sys.version,
        "device": "cpu", "threads": torch.get_num_threads(),
        "code_commit": "bbbae73f74fc14eafdf7abb93f9094eafccd21df",
        "weights_revision": "82e6d7646126d1fa33b30c0b44f1d076496cca71",
        "packages": {p: importlib.metadata.version(p) for p in
                     ["torch", "torchaudio", "asteroid", "resemblyzer", "numpy", "soundfile", "setuptools"]},
    }
    try:
        checkpoint = ROOT / "checkpoints/best_model.pth"
        report["checkpoint_sha256"] = sha256(checkpoint)
        assert report["checkpoint_sha256"] == "41c250587af2e947dc3e265cd6e7aa7de230048d408e93c53d8348d63cf75896"
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = SpeakerBeamSS().eval()
        load_result = model.load_state_dict(state, strict=True)
        report["strict_load"] = {"missing_keys": load_result.missing_keys,
                                 "unexpected_keys": load_result.unexpected_keys}
        report["state_dict_entries"] = len(state)
        report["parameter_count"] = sum(p.numel() for p in model.parameters())

        def load_audio(name):
            path = ROOT / "samples" / name
            data, sr = sf.read(path, dtype="float32", always_2d=True)
            wav = torch.from_numpy(data.mean(axis=1)).reshape(1, 1, -1)
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            report.setdefault("samples", {})[name] = {
                "sha256": sha256(path), "original_sample_rate": sr,
                "samples_16k": wav.shape[-1], "duration_seconds": wav.shape[-1] / 16000,
            }
            return wav

        mixture = load_audio("mixture_000001.wav")
        enroll = load_audio("enrollment_000001.wav")
        enroll_other = load_audio("enrollment_000002.wav")
        speaker_encoder = VoiceEncoder(device="cpu")
        encoder_weights = Path(resemblyzer.__file__).parent / "pretrained.pt"
        report["speaker_encoder_weights_sha256"] = sha256(encoder_weights)
        with torch.inference_mode():
            embedding = get_speaker_embeddings_batch(speaker_encoder, enroll)
            report["embedding_shape"] = list(embedding.shape)
            t0 = time.perf_counter()
            enhanced = model(mixture, embedding)
            report["single_full_clip_inference_seconds"] = time.perf_counter() - t0
            assert enhanced.shape == mixture.shape, (enhanced.shape, mixture.shape)
            assert torch.isfinite(enhanced).all()
            assert enhanced.square().mean() > 1e-12
            output = enhanced[0, 0].numpy()
            sf.write(outdir / "enhanced_demo.wav", output, 16000, subtype="FLOAT")
            preview_gain = 0.9 / max(float(np.max(np.abs(output))), 1e-12)
            sf.write(outdir / "enhanced_demo_normalized_preview.wav", output * preview_gain,
                     16000, subtype="PCM_16")
            report["preview"] = {"gain": preview_gain,
                                 "note": "Peak normalized for listening; raw output retained separately."}
            report["output"] = {"shape": list(enhanced.shape), "finite": True,
                                "rms": float(np.sqrt(np.mean(output**2))),
                                "peak": float(np.max(np.abs(output)))}
            bundled = load_audio("upstream_result_000001.wav")[0, 0].numpy()
            if bundled.shape == output.shape:
                clipped = np.clip(output, -1.0, 1.0)
                report["bundled_output_comparison"] = {
                    "rmse": float(np.sqrt(np.mean((bundled-output)**2))),
                    "correlation": float(np.corrcoef(bundled, output)[0, 1]),
                    "clipped_raw_rmse": float(np.sqrt(np.mean((bundled-clipped)**2))),
                    "clipped_raw_correlation": float(np.corrcoef(bundled, clipped)[0, 1]),
                    "clipped_raw_max_abs_difference": float(np.max(np.abs(bundled-clipped))),
                    "raw_fraction_outside_unit_range": float(np.mean(np.abs(output) > 1.0)),
                    "note": "Bundled output is another model output, not clean target ground truth.",
                }
            other_embedding = get_speaker_embeddings_batch(speaker_encoder, enroll_other)
            other_output = model(mixture, other_embedding)
            report["enrollment_sensitivity_relative_l2"] = float(
                torch.linalg.vector_norm(other_output-enhanced) / torch.linalg.vector_norm(enhanced))

            # A concrete causal check: change only the final half of the mixture.
            changed = mixture.clone()
            midpoint = mixture.shape[-1] // 2
            changed[..., midpoint:] = 0
            changed_output = model(changed, embedding)
            early = 16000  # First second, far from the altered future samples.
            report["future_dependence_check"] = {
                "future_changed_from_seconds": midpoint / 16000,
                "earlier_window_seconds": early / 16000,
                "early_output_relative_l2": float(torch.linalg.vector_norm(
                    changed_output[..., :early]-enhanced[..., :early]) /
                    torch.linalg.vector_norm(enhanced[..., :early])),
                "early_output_max_abs_change": float((
                    changed_output[..., :early]-enhanced[..., :early]).abs().max()),
            }
        report["status"] = "passed_checkpoint_and_offline_inference"
        report["quality_metrics"] = "Not computed: repository demo has no isolated clean target here."
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        raise
    finally:
        (outdir / "verification_results.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
