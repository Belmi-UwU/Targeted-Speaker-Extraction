# Listening comparison: selected speakers 4077 + 5142

Play `01`–`04` to compare the original three-speaker overlap. Play `05`–`08` for the controlled turns clip: 4077 alone, 5683 alone, all three, then 5142 alone. In both cases, the intended output contains **4077 and 5142 only**.

| File | Meaning |
| --- | --- |
| `01_three_voice_mix.wav` / `05_turns_mix.wav` | Original mixture, including 5683. |
| `02_ideal_4077_5142.wav` / `06_turns_ideal_4077_5142.wav` | Clean sum of the two desired source recordings; reference only. |
| `03_speakerbeam_ss_4077_5142.wav` / `07_turns_speakerbeam_ss_4077_5142.wav` | Current independent SpeakerBeam-SS checkpoint, one extraction per chosen speaker, raw outputs summed and uniformly reduced for playback. |
| `04_sepformer_mixfit_4077_5142.wav` / `08_turns_sepformer_mixfit_4077_5142.wav` | SpeechBrain Libri3Mix SepFormer, three anonymous streams matched using enrollment voice embeddings, then combined using gains fitted to the original mixture. |
| `09_sepformer_only_5142.wav` | SepFormer output containing **only selected speaker 5142** from the same three-voice mixture as `01`. This is the model result, exported at its fitted level with no extra gain. |
| `10_ideal_only_5142.wav` | The original clean 5142 source from that mixture; reference only. |

Listen for whether **both chosen voices remain clear** and whether **5683 leaks through**. For `05`–`08`, focus on the second 0.9-second interval: neither chosen voice speaks there, so the ideal reference is silent. For the single-speaker request, compare `09` against `01` and `10`: `09` should keep 5142 while reducing 4077 and 5683. Playback levels are not identical across methods; absolute loudness is not a quality score. [Methods and measured results](../../M2_MODEL_COMPARISON.md).

Speech recordings derive from [LibriSpeech test-clean](https://www.openslr.org/12), CC BY 4.0. The model comparison is exploratory; these WAVs are not product output.
