"""Check manifest and SI-SDR scoring mechanics with generated signals only."""

import json
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import soundfile as sf

from score_outputs import score_trial
from validate_manifest import read_manifest, validate_trial


def main():
    sr = 16000
    t = np.arange(sr, dtype=np.float64) / sr
    signals = [0.12 * np.sin(2 * np.pi * hz * t) for hz in (197, 313, 463)]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "outputs").mkdir()

        def write(name, data):
            sf.write(root / name, data, sr, subtype="FLOAT")
            return name

        sources = []
        for index, signal in enumerate(signals):
            speaker = f"speaker_{index + 1}"
            source_path = write(f"{speaker}_source.wav", signal)
            # Distinct file and utterance ID; this verifies metadata mechanics, not identity.
            enrollment_path = write(f"{speaker}_enrollment.wav", signal[: sr // 2])
            sources.append({
                "speaker_id": speaker, "source_path": source_path,
                "source_utterance_id": f"{speaker}_source_utt",
                "enrollment_path": enrollment_path,
                "enrollment_utterance_id": f"{speaker}_enrollment_utt",
            })

        trials = []
        for count, selected in ((2, ["speaker_1"]), (3, ["speaker_1", "speaker_3"])):
            trial_id = f"synthetic_{count}mix"
            mixture_path = write(f"{trial_id}_mix.wav", sum(signals[:count]))
            trial = {
                "trial_id": trial_id,
                "dataset": f"Libri{count}Mix", "dataset_revision": "synthetic-self-check",
                "speech_license": "synthetic-generated", "split": "test", "condition": "mix_clean",
                "sample_rate_hz": sr, "mixture_path": mixture_path,
                "selected_speaker_ids": selected, "sources": sources[:count],
            }
            write(f"outputs/{trial_id}.wav", sum(
                signal for signal, item in zip(signals, sources) if item["speaker_id"] in selected
            ))
            trials.append(trial)

        manifest = root / "trials.jsonl"
        manifest.write_text("\n".join(json.dumps(t) for t in trials) + "\n")
        assert len(read_manifest(manifest)) == 2
        results = []
        for trial in trials:
            validate_trial(manifest, trial, check_mix_sum=True)
            scored = score_trial(manifest, trial, root / "outputs")
            assert scored["si_sdri_db"] > 20, scored
            results.append(scored)

        broken_cases = []
        broken = deepcopy(trials[0])
        broken["selected_speaker_ids"] = ["unknown_speaker"]
        broken_cases.append((broken, "not in sources"))
        broken = deepcopy(trials[0])
        broken["sources"][0]["enrollment_utterance_id"] = broken["sources"][0]["source_utterance_id"]
        broken_cases.append((broken, "different utterance IDs"))
        broken = deepcopy(trials[0])
        broken["sources"][0]["source_path"] = "missing.wav"
        broken_cases.append((broken, "audio file missing"))
        write("wrong_rate.wav", signals[0])
        sf.write(root / "wrong_rate.wav", signals[0], 8000, subtype="FLOAT")
        broken = deepcopy(trials[0])
        broken["sources"][0]["enrollment_path"] = "wrong_rate.wav"
        broken_cases.append((broken, "expected nonempty mono"))
        for broken, expected_message in broken_cases:
            try:
                validate_trial(manifest, broken)
            except ValueError as error:
                assert expected_message in str(error), error
            else:
                raise AssertionError(f"validator accepted broken trial: {expected_message}")

        print(json.dumps({"status": "passed_synthetic_contract_check", "results": results}, indent=2))


if __name__ == "__main__":
    main()
