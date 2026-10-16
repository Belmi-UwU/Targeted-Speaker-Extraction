from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from df.enhance import init_df, enhance

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "evaluation/data/user_recordings/mixture"
INPUT = FOLDER / "Mix.wav"
OUTPUT = FOLDER / "Mix_denoised.wav"

torch.set_num_threads(2)

# Gabungkan stereo menjadi mono, seperti input discovery aplikasi.
wave, rate = sf.read(INPUT, dtype="float32", always_2d=True)
audio = torch.from_numpy(wave.mean(axis=1)).unsqueeze(0)

# Versi DeepFilterNet yang terpasang mengembalikan tiga nilai.
model, state, _ = init_df(
    model_base_dir="DeepFilterNet3",
    post_filter=False,
    log_file=None,
)

model_rate = state.sr()
if rate != model_rate:
    audio = torchaudio.functional.resample(audio, rate, model_rate)

# Mulai dengan batas pengurangan noise 12 dB.
result = enhance(
    model,
    state,
    audio,
    pad=True,
    atten_lim_db=12,
)

cleaned = result.squeeze(0).cpu().numpy()
if cleaned.shape != (audio.shape[-1],) or not np.isfinite(cleaned).all():
    raise RuntimeError("Output audio tidak valid")

sf.write(OUTPUT, cleaned, model_rate, subtype="FLOAT")
print(f"Disimpan: {OUTPUT}")
print(f"Durasi: {len(cleaned) / model_rate:.2f} detik")