"""Create clearly labeled, level-comparable WAVs for the first two trials."""

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from validate_manifest import audio_path, read_manifest

ROOT = Path(__file__).resolve().parent
TRIALS = ROOT / "data/trials"
# Chosen once from the first trial's least-squares match to clean target,
# rounded and held fixed for both trial outputs. This is evaluation-only.
FIXED_MODEL_GAIN = 0.014


def main():
    out_dir = TRIALS / "listening_comparisons"
    out_dir.mkdir(exist_ok=True)
    summaries = []
    manifest = TRIALS / "trials.jsonl"
    for trial in read_manifest(manifest):
        trial_id = trial["trial_id"]
        mix, _ = sf.read(audio_path(manifest, trial["mixture_path"]), dtype="float32")
        selected = set(trial["selected_speaker_ids"])
        clean = np.zeros_like(mix)
        for source in trial["sources"]:
            if source["speaker_id"] in selected:
                signal, _ = sf.read(audio_path(manifest, source["source_path"]), dtype="float32")
                clean += signal
        estimate, _ = sf.read(TRIALS / "model_outputs" / f"{trial_id}.wav", dtype="float32")
        scaled = estimate * FIXED_MODEL_GAIN
        if max(float(np.max(np.abs(x))) for x in (mix, clean, scaled)) >= 1.0:
            raise ValueError(f"comparison would clip: {trial_id}")
        for name, signal in (("mix", mix), ("selected_clean", clean), ("model_fixed_gain", scaled)):
            sf.write(out_dir / f"{trial_id}__{name}.wav", signal, 16000, subtype="PCM_16")
        summaries.append({"trial_id": trial_id, "mix_peak": float(np.max(np.abs(mix))),
                          "clean_peak": float(np.max(np.abs(clean))),
                          "scaled_model_peak": float(np.max(np.abs(scaled)))})
    record = {"fixed_model_gain": FIXED_MODEL_GAIN,
              "note": "Fixed gain estimated using the first clean trial; research listening only, not a deployable gain policy or quality metric.",
              "trials": summaries}
    (out_dir / "comparison_metadata.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
