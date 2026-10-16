# %% [markdown]
# # Retrain causal WeSep for live target-speaker extraction
#
# Implements R0–R2 of [`evaluation/WESEP_RETRAINING_PLAN.md`](../evaluation/WESEP_RETRAINING_PLAN.md) on an
# Apple Silicon Mac (Metal / MPS):
#
# - **R0, data.** Builds LibriSpeech conversations on the fly:
#   - 2–6 speakers, with turn-taking and overlap;
#   - stretches where the target is silent;
#   - noise and reverb;
#   - enrollments that look like discovered tracks (8 kHz band-limited, leakage, gaps, random level).
# - **R1, research pilot (`MODE = "finetune_research"`).** Fine-tunes the released REAL-TSE causal checkpoint.
#   - It is quick, and it shows whether the data alone fixes the weaknesses with groups and with an absent target.
#   - The checkpoint has **no weights licence**, so the result is evaluation-only and must never ship.
# - **R2, shippable path (`MODE = "scratch_small"` or `"scratch_base"`).** Same recipe from scratch.
#   - All training data and code are licence-clean: LibriSpeech CC BY 4.0, WeSep Apache-2.0, WeSpeaker ECAPA.
#
# **How to use**
# 1. Upload this notebook. For R1, also upload the `tfmap_context_causal_100/` checkpoint folder.
#    For evaluation, upload `eval_pack.zip` (built by `training/make_eval_pack.py`).
# 2. Edit the **Config** cell. Run with `SMOKE = True` once (a few minutes), then set `SMOKE = False`.
# 3. Run all cells. If anything stops it (kernel crash, reboot, power loss), just run all cells again: finished
#    steps are skipped and training resumes from the last checkpoint (at most ~10 minutes lost).
#    For unattended runs, `run_until_done.sh` runs this same code as a script and restarts it automatically.
#
# Metal note: PyTorch does not implement `torch.angle` for complex tensors on MPS yet. The first code cell enables
# `PYTORCH_ENABLE_MPS_FALLBACK`, which runs only that op on the CPU.
#
# Speed measured on an Apple M5 laptop, one train step with batch 4 × 4 s:
#
# | Model | MPS | CPU |
# | --- | --- | --- |
# | small (11.5 M parameters) | 0.7 s | 2.5 s |
# | base (27 M parameters) | 2.1 s | 8.0 s |
#
# A Mac Studio Max should be faster.

# %%
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # must be set before torch is imported

import json
import math
import random
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from hashlib import md5, sha256
from pathlib import Path


def atomic(path, write):
    """Write `path` via a temporary file and an atomic rename, so a crash never leaves it half-written."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)

# %% [markdown]
# ## Config

# %%
MODE = os.environ.get("TSE_MODE", "finetune_research")  # "finetune_research" | "scratch_small" | "scratch_base"
PROFILE = os.environ.get("TSE_PROFILE", "baseline")
if PROFILE not in ("baseline", "noise_robust"):
    raise ValueError("TSE_PROFILE must be baseline or noise_robust")
ROBUST = PROFILE == "noise_robust"
if ROBUST and MODE != "finetune_research":
    raise ValueError("noise_robust currently continues a finetune_research checkpoint")
SMOKE = os.environ.get("TSE_SMOKE", "0") == "1"  # tiny end-to-end check; set False for real training

WORK = Path(os.environ.get("TSE_WORK", "~/tse_training")).expanduser()
LIBRISPEECH = Path(os.environ.get("TSE_LIBRISPEECH", WORK / "data/LibriSpeech")).expanduser()
TRAIN_SETS = ["train-clean-100", "train-clean-360"]  # add "train-other-500" for more speakers (+30 GB)
VAL_SET = "dev-clean"  # speakers disjoint from training and from the test-clean evaluation pack
REALTSE_CKPT = Path(os.environ.get("TSE_INIT_CHECKPOINT", WORK / "pretrained/tfmap_context_causal_100")).expanduser()
EVAL_PACK = WORK / "eval_pack"  # unzip eval_pack.zip so that EVAL_PACK/index.json exists
RUN_NAME = os.environ.get("TSE_RUN_NAME", "finetune_noise_robust_research" if ROBUST else MODE)
RUN = WORK / "runs" / RUN_NAME

RATE = 16000
CHUNK_SECONDS = 4.0  # training mixture length
ENROLL_SECONDS = 5.0  # training enrollment length (inference uses up to 8 s of speech)
BATCH = 8
STEPS = {"finetune_research": 20_000, "scratch_small": 150_000, "scratch_base": 200_000}[MODE]
LR = {"finetune_research": 2e-4, "scratch_small": 1e-3, "scratch_base": 1e-3}[MODE]
WARMUP = 1_000
LOG_EVERY, VAL_EVERY, CKPT_EVERY = 50, 2_000, 1_000
CKPT_MINUTES = 10  # also checkpoint at least this often, so a crash loses at most ~10 minutes
MAX_BAD_STEPS = 20  # stop (keeping the last good checkpoint) after this many non-finite losses in a row
DATA_THREADS = 6  # background threads simulating batches (no worker processes: macOS "spawn" re-runs this script)
VAL_EXAMPLES = 200
FREEZE_SPEAKER_ENCODER = MODE != "finetune_research"  # cheaper from-scratch training; ECAPA stays pretrained
SEED = 2026

# Simulator knobs (see the plan, section 3).
SPEAKER_COUNTS, SPEAKER_WEIGHTS = [2, 3, 4, 5, 6], [0.15, 0.25, 0.25, 0.2, 0.15]
ACTIVE_COUNTS, ACTIVE_WEIGHTS = [0, 1, 2, 3, 4], [0.05, 0.4, 0.3, 0.2, 0.05]
TARGET_NEVER_PRESENT = 0.05  # whole chunk without the target
REVERB_PROB, NOISE_PROB = 0.5, 0.5
DISCOVERED_STYLE_PROB = 0.6  # enrollment looks like a SepFormer-discovered track

if ROBUST:
    TRAIN_SETS = ["train-clean-100"]
    BATCH, STEPS, LR, WARMUP = 2, 5_000, 5e-5, 200
    DATA_THREADS, VAL_EXAMPLES = 2, 48
    LOG_EVERY, VAL_EVERY, CKPT_EVERY = 25, 500, 250
    FREEZE_SPEAKER_ENCODER = True
    NOISE_PROB, DISCOVERED_STYLE_PROB, TARGET_NEVER_PRESENT = 0.75, 0.5, 0.10
BATCH = int(os.environ.get("TSE_BATCH", BATCH))
STEPS = int(os.environ.get("TSE_STEPS", STEPS))
LR = float(os.environ.get("TSE_LR", LR))

if SMOKE:
    TRAIN_SETS, VAL_SET = [os.environ.get("TSE_SMOKE_SET", "dev-clean")], os.environ.get("TSE_SMOKE_SET", "dev-clean")
    STEPS, BATCH, DATA_THREADS, VAL_EXAMPLES = 20, 2, 2, 6  # threads stay on: smoke runs test the same path
    LOG_EVERY, VAL_EVERY, CKPT_EVERY, WARMUP = 5, 10, 5, 5
    RUN = WORK / "runs" / f"smoke_{RUN_NAME}"
for folder in (WORK, RUN):
    folder.mkdir(parents=True, exist_ok=True)
print(f"MODE={MODE} PROFILE={PROFILE} SMOKE={SMOKE} STEPS={STEPS} BATCH={BATCH} RUN={RUN}")

# %% [markdown]
# ## Environment
#
# The code is pinned to known commits:
# - WeSep (Apache-2.0) is cloned into `WORK/wesep`.
# - WeSpeaker, which WeSep's speaker modules import, is pip-installed.
#
# Set `INSTALL = False` once the environment is ready.

# %%
INSTALL = os.environ.get("TSE_INSTALL", "1") == "1"
WESEP_COMMIT = "dc11cb18ee8636064a7851f38b9348143c343cdc"
WESPEAKER = "git+https://github.com/wenet-e2e/wespeaker.git@8f53b6485d9f88a207bd17e7f8dba899495ec794"
PACKAGES = ["torch>=2.6", "torchaudio>=2.6", "numpy<2", "scipy", "soundfile", "pyyaml", "tqdm", "matplotlib",
            "pyroomacoustics", "huggingface_hub", "silero-vad==5.1.2", "torchmetrics", "thop", "tableprint",
            WESPEAKER]


def run(cmd):
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


if INSTALL:
    run([sys.executable, "-m", "pip", "install", "-q", *PACKAGES])
WESEP_DIR = WORK / "wesep"
if not (WESEP_DIR / ".git").exists():
    run(["git", "clone", "-q", "https://github.com/wenet-e2e/wesep.git", str(WESEP_DIR)])
run(["git", "-C", str(WESEP_DIR), "checkout", "-q", WESEP_COMMIT])
sys.path.insert(0, str(WESEP_DIR))

import numpy as np  # noqa: E402
import scipy.signal  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

DEVICE = torch.device(os.environ.get("TSE_DEVICE", "cuda" if torch.cuda.is_available() else
                                     "mps" if torch.backends.mps.is_available() else "cpu"))
torch.set_num_threads(int(os.environ.get("TSE_THREADS", "4")))
torch.manual_seed(SEED)
print("torch", torch.__version__, "device", DEVICE)
RUN_CONFIG = {"mode": MODE, "profile": PROFILE, "smoke": SMOKE, "seed": SEED,
              "batch": BATCH, "steps": STEPS, "lr": LR, "train_sets": TRAIN_SETS,
              "validation_set": VAL_SET, "device": str(DEVICE),
              "noise_probability": NOISE_PROB, "discovered_style_probability": DISCOVERED_STYLE_PROB,
              "target_never_present": TARGET_NEVER_PRESENT,
              "initial_checkpoint": str(REALTSE_CKPT),
              "initial_checkpoint_sha256": None, "wesep_commit": WESEP_COMMIT}
if MODE == "finetune_research":
    digest = sha256()
    with (REALTSE_CKPT / "avg_model.pt").open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    RUN_CONFIG["initial_checkpoint_sha256"] = digest.hexdigest()
if ROBUST:
    config_path = RUN / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != RUN_CONFIG:
        raise ValueError("Run configuration changed; use a separate TSE_RUN_NAME")
    atomic(config_path, lambda tmp: tmp.write_text(json.dumps(RUN_CONFIG, indent=2) + "\n"))

# %% [markdown]
# ## Data: LibriSpeech
#
# Downloads from OpenSLR with MD5 checks and skips anything already extracted. Approximate sizes:
#
# | Set | Size | Speakers |
# | --- | --- | --- |
# | `train-clean-100` | 6.3 GB | 251 |
# | `train-clean-360` | 23 GB | 921 |
# | `dev-clean` | 0.3 GB | 40 |
#
# All are CC BY 4.0. `test-clean` is never used for training; the evaluation pack comes from it.

# %%
MD5 = {"train-clean-100": "2a93770f6d5c6c964bc36631d331a522", "train-clean-360": "c0e676e450a7ff2f54aeade5171606fa",
       "train-other-500": "d1a0fd59409feb2c614ce4d30c387708", "dev-clean": "42e2234ba48799c1f50f24a7926300a1",
       "test-clean": "32fa31d27d2e1cad72775fee3f4849a9"}


def file_md5(path):
    digest = md5()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_extract(tar, destination):
    """tar.extractall with the "data" filter, or an equivalent check on Pythons without extraction filters
    (< 3.9.17 / 3.10.12, e.g. Apple's system Python 3.9): only regular files and folders inside `destination`."""
    try:
        tar.extractall(destination, filter="data")
        return
    except TypeError:  # extraction filters not available on this Python
        pass
    root = Path(destination).resolve()
    members = tar.getmembers()
    for member in members:
        target = (root / member.name).resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"unsafe path in archive: {member.name}")
        if not (member.isfile() or member.isdir()):
            raise ValueError(f"unexpected link or special file in archive: {member.name}")
    tar.extractall(destination, members=members)


def ensure_librispeech(name):
    if (LIBRISPEECH / name).is_dir():
        return
    archive = WORK / "data/archives" / f"{name}.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        print(f"downloading {name} ...", flush=True)
        atomic(archive, lambda tmp: urllib.request.urlretrieve(
            f"https://www.openslr.org/resources/12/{name}.tar.gz", tmp))
    if file_md5(archive) != MD5[name]:
        archive.unlink()
        raise ValueError(f"MD5 mismatch for {name}; the bad archive was deleted, re-run this cell to download again")
    staging = LIBRISPEECH.parent / f".extract_{name}"
    shutil.rmtree(staging, ignore_errors=True)  # leftovers of an interrupted extraction
    with tarfile.open(archive) as tar:
        safe_extract(tar, staging)
    LIBRISPEECH.mkdir(parents=True, exist_ok=True)
    os.replace(staging / "LibriSpeech" / name, LIBRISPEECH / name)  # appears only once complete
    shutil.rmtree(staging, ignore_errors=True)
    print(f"extracted {name}", flush=True)


for name in sorted(set(TRAIN_SETS + [VAL_SET])):
    ensure_librispeech(name)


def build_index(sets):
    """{speaker: [(path, frames), ...]} with a JSON cache next to the data."""
    cache = LIBRISPEECH / f"index_{'_'.join(sets)}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    index = {}
    for name in sets:
        for path in sorted((LIBRISPEECH / name).rglob("*.flac")):
            speaker = f"{name}/{path.parts[-3]}"
            index.setdefault(speaker, []).append((str(path), sf.info(path).frames))
    atomic(cache, lambda tmp: tmp.write_text(json.dumps(index)))
    return index


TRAIN_INDEX = build_index(TRAIN_SETS)
VAL_INDEX = build_index([VAL_SET])
if SMOKE:  # same set for both: split speakers so validation still uses unseen voices
    speakers = sorted(TRAIN_INDEX)
    VAL_INDEX = {s: TRAIN_INDEX[s] for s in speakers[::4]}
    TRAIN_INDEX = {s: TRAIN_INDEX[s] for s in speakers if s not in VAL_INDEX}
print(f"train speakers {len(TRAIN_INDEX)}, utterances {sum(len(v) for v in TRAIN_INDEX.values())}; "
      f"validation speakers {len(VAL_INDEX)}")

# %% [markdown]
# ## Conversation simulator
#
# Each example:
# - picks 2–6 speakers and a target among them;
# - lays out 0.4–2 s segments with 0–4 people talking at once;
# - draws each speaker's speech from their own utterances;
# - adds reverb (early reflections are kept in the target reference) and babble or coloured noise;
# - returns the target's own enrollment from **different** utterances, either clean or "discovered-style".
#
# In 5% of chunks the target never talks. The loss then rewards silence.

# %%
def read_audio(path, start=0, frames=-1):
    wave, rate = sf.read(path, start=start, frames=frames, dtype="float32")
    if rate != RATE:
        raise ValueError(f"expected 16 kHz: {path}")
    return wave


def trim_silence(wave, frame=320, floor_db=-40.0):
    n = len(wave) // frame
    if n == 0:
        return wave
    energy = np.square(wave[:n * frame]).reshape(n, frame).mean(axis=1)
    keep = np.flatnonzero(energy > energy.max() * 10 ** (floor_db / 10))
    return wave[keep[0] * frame:(keep[-1] + 1) * frame] if len(keep) else wave


def speech_from(rng, utterances, frames, exclude=()):
    """Contiguous speech of `frames` samples from random utterances, skipping `exclude` paths."""
    out, used = [], []
    total = 0
    pool = [u for u in utterances if u[0] not in exclude] or utterances
    while total < frames:
        path, length = pool[rng.integers(len(pool))]
        start = int(rng.integers(0, max(1, length - frames))) if length > frames else 0
        chunk = trim_silence(read_audio(path, start, min(length - start, frames - total + RATE)))
        out.append(chunk)
        used.append(path)
        total += len(chunk)
    return np.concatenate(out)[:frames], used


def fade(wave, n=160):
    wave = wave.copy()
    if len(wave) > 2 * n:
        wave[:n] *= np.linspace(0, 1, n, dtype="float32")
        wave[-n:] *= np.linspace(1, 0, n, dtype="float32")
    return wave


def rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype="float64")))) + 1e-12


class RirBank:
    """Pre-computed shoebox rooms (pyroomacoustics, MIT): several source positions per room."""

    def __init__(self, rooms=200 if not SMOKE else 4, sources=6, seed=SEED):
        import pyroomacoustics as pra
        cache = WORK / f"rir_bank_{rooms}.npz"
        if cache.exists():
            data = np.load(cache, allow_pickle=True)
            self.rooms = list(data["rooms"])
            return
        rng = np.random.default_rng(seed)
        self.rooms = []
        for _ in range(rooms):
            size = rng.uniform([3, 3, 2.4], [9, 8, 3.5])
            rt60 = rng.uniform(0.2, 0.8)
            absorption, order = pra.inverse_sabine(rt60, size)
            room = pra.ShoeBox(size, fs=RATE, materials=pra.Material(absorption), max_order=min(order, 20))
            mic = rng.uniform([0.5, 0.5, 0.8], size - [0.5, 0.5, 1.0])
            room.add_microphone(mic)
            for _ in range(sources):
                room.add_source(rng.uniform([0.3, 0.3, 1.0], size - [0.3, 0.3, 0.8]))
            room.compute_rir()
            self.rooms.append([np.asarray(room.rir[0][s][:RATE], dtype="float32") for s in range(sources)])
        rooms_array = np.array(self.rooms, dtype=object)

        def write(tmp):
            with open(tmp, "wb") as handle:
                np.savez(handle, rooms=rooms_array, allow_pickle=True)

        atomic(cache, write)

    def sample(self, rng, count):
        room = self.rooms[rng.integers(len(self.rooms))]
        return [room[i] for i in rng.choice(len(room), size=min(count, len(room)), replace=False)]


def reverberate(wave, rir):
    """Return (reverberant mixture contribution, target reference with direct path + 50 ms reflections)."""
    peak = int(np.argmax(np.abs(rir)))
    early = rir[:peak + int(0.05 * RATE)]
    full = scipy.signal.fftconvolve(wave, rir)[:len(wave)].astype("float32")
    target = scipy.signal.fftconvolve(wave, early)[:len(wave)].astype("float32")
    return full, target


def coloured_noise(rng, frames):
    white = rng.standard_normal(frames + 1).astype("float32")
    kind = rng.integers(3)
    if kind == 0:
        return white[:frames]
    spectrum = np.fft.rfft(white)
    freqs = np.maximum(np.fft.rfftfreq(len(white)), 1e-3)
    spectrum /= freqs ** (0.5 if kind == 1 else 1.0)  # pink or brown
    return np.fft.irfft(spectrum)[:frames].astype("float32")


def environmental_noise(rng, frames):
    """Synthetic electrical buzz and clap-like bursts, not recordings of actual rooms."""
    t = np.arange(frames) / RATE
    fundamental = rng.choice([50.0, 60.0]) * rng.uniform(0.98, 1.02)
    hum = sum(np.sin(2 * np.pi * fundamental * k * t + rng.uniform(0, 2 * np.pi)) / k
              for k in range(1, int(rng.integers(4, 13))))
    hum *= 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.2, 2.0) * t) ** 2
    bursts = np.zeros(frames)
    for _ in range(int(rng.integers(1, 9))):
        start = int(rng.integers(frames))
        count = min(frames - start, int(rng.uniform(0.03, 0.25) * RATE))
        burst = rng.standard_normal(count) * np.exp(-np.arange(count) / (RATE * rng.uniform(0.008, 0.05)))
        bursts[start:start + count] += burst * rng.uniform(0.3, 2.0)
    kind = int(rng.integers(3))
    return (hum if kind == 0 else bursts if kind == 1 else hum + bursts).astype("float32")


def channel_augment(rng, wave):
    """Vary the enrollment channel while preserving speaker identity and duration."""
    sos = scipy.signal.butter(2, [rng.uniform(60, 200), rng.uniform(2800, 7500)],
                              btype="bandpass", fs=RATE, output="sos")
    return scipy.signal.sosfilt(sos, wave).astype("float32")


def discovered_style(rng, enroll, leak_source):
    """Make a clean enrollment look like a SepFormer-discovered track."""
    x = scipy.signal.resample_poly(scipy.signal.resample_poly(enroll, 1, 2), 2, 1)[:len(enroll)]
    x = np.pad(x, (0, len(enroll) - len(x))).astype("float32")
    if leak_source is not None and len(leak_source) and (ROBUST or len(leak_source) >= len(x)):
        # R1's 4 s source with a 5 s enrollment skipped leakage. Preserve its baseline recipe.
        source = np.pad(leak_source, (0, max(0, len(x) - len(leak_source))))[:len(x)]
        low, high = (-20, -6) if ROBUST else (-25, -12)
        leak = source * (rms(x) / rms(source)) * 10 ** (rng.uniform(low, high) / 20)
        x = x + leak.astype("float32")
    for _ in range(rng.integers(0, 4)):
        start, length = rng.integers(0, len(x)), int(rng.uniform(0.1, 0.5) * RATE)
        x[start:start + length] = 0
    return x


def simulate(rng, index, speakers, rirs=None):
    """One training example: mixture, target reference, target enrollment, target activity."""
    frames, enroll_frames = int(CHUNK_SECONDS * RATE), int(ENROLL_SECONDS * RATE)
    n = int(rng.choice(SPEAKER_COUNTS, p=SPEAKER_WEIGHTS))
    chosen = [speakers[i] for i in rng.choice(len(speakers), size=n, replace=False)]
    target_never = rng.random() < TARGET_NEVER_PRESENT
    activity = np.zeros((n, frames), dtype=bool)
    t = 0
    while t < frames:
        length = int(rng.uniform(0.4, 2.0) * RATE)
        k = min(int(rng.choice(ACTIVE_COUNTS, p=ACTIVE_WEIGHTS)), n)
        candidates = np.arange(1, n) if target_never else np.arange(n)
        active = rng.choice(candidates, size=min(k, len(candidates)), replace=False)
        activity[active, t:t + length] = True
        t += length
    if not target_never and not activity[0].any():  # make sure the target talks at least once
        start = int(rng.integers(0, frames - RATE))
        activity[0, start:start + RATE] = True
    sources, references, used = [], [], set()
    rir_list = rirs.sample(rng, n) if rirs is not None and rng.random() < REVERB_PROB else None
    for i, speaker in enumerate(chosen):
        track = np.zeros(frames, dtype="float32")
        if activity[i].any():
            speech, paths = speech_from(rng, index[speaker], int(activity[i].sum()))
            used.update(paths)
            edges = np.flatnonzero(np.diff(np.concatenate([[0], activity[i].astype(int), [0]])))
            cursor = 0
            for a, b in zip(edges[::2], edges[1::2]):
                track[a:b] = fade(speech[cursor:cursor + (b - a)])
                cursor += b - a
            track *= 10 ** (rng.uniform(-10, 10) / 20) * 0.05 / rms(track[activity[i]])
            if ROBUST and i == 0 and rng.random() < 0.5:
                track *= 10 ** (-rng.uniform(3, 12) / 20)
        reference = track
        if rir_list is not None and i < len(rir_list):
            track, reference = reverberate(track, rir_list[i])
        sources.append(track)
        references.append(reference)
    mix = np.sum(sources, axis=0)
    if rng.random() < NOISE_PROB and rms(mix) > 1e-6:
        kind = rng.random()
        if kind < (0.25 if ROBUST else 0.5):  # babble from speakers outside this mixture
            others = [s for s in speakers if s not in chosen]
            babble = sum(speech_from(rng, index[others[j]], frames)[0]
                         for j in rng.choice(len(others), size=min(int(rng.integers(3, 9)), len(others)),
                                             replace=False))
        elif ROBUST and kind >= 0.5:
            babble = environmental_noise(rng, frames)
        else:
            babble = coloured_noise(rng, frames)
        mix = mix + babble * (rms(mix) / rms(babble)) * 10 ** (-rng.uniform(0 if ROBUST else 5, 30) / 20)
    level = 10 ** (rng.uniform(-35, -15) / 20) / rms(mix)
    level = min(level, 0.95 / (np.max(np.abs(mix)) + 1e-9))
    mix, target = (mix * level).astype("float32"), (references[0] * level).astype("float32")
    enroll, _ = speech_from(rng, index[chosen[0]], enroll_frames, exclude=used)
    if rng.random() < DISCOVERED_STYLE_PROB:
        leak = sources[1] if n > 1 and rms(sources[1]) > 1e-6 else None
        enroll = discovered_style(rng, enroll, leak)
    if ROBUST and rng.random() < 0.5:
        enroll = channel_augment(rng, enroll)
    enroll = enroll * 10 ** (rng.uniform(-40, -10) / 20) / rms(enroll)  # level augmentation
    return {"mix": mix, "target": target, "enroll": enroll.astype("float32"), "active": activity[0]}


RIRS = RirBank()
_example = simulate(np.random.default_rng(0), TRAIN_INDEX, sorted(TRAIN_INDEX), RIRS)
print({k: (v.shape, v.dtype) for k, v in _example.items()})
try:  # listen in Jupyter
    from IPython.display import Audio, display
    display(Audio(_example["mix"], rate=RATE), Audio(_example["target"], rate=RATE), Audio(_example["enroll"], rate=RATE))
except Exception:
    pass

# %% [markdown]
# ## Datasets
#
# - **Training:** infinite, simulated on the fly by background **threads**.
#   - DataLoader worker processes are not used: on macOS they start with "spawn", which re-runs this whole script
#     (or cannot import classes defined in a notebook).
#   - The simulation is mostly NumPy/FFT/audio decoding, which release Python's GIL, so threads run in parallel.
# - **Validation:** a fixed set from `dev-clean` speakers, which never appear in training.

# %%
import queue  # noqa: E402
import threading  # noqa: E402


class ThreadedBatches:
    """Infinite iterator of training batches, each simulated whole by one of `threads` daemon threads."""

    def __init__(self, index, seed, batch, threads, prefetch=2):
        self.queue = queue.Queue(maxsize=max(2, threads * prefetch))
        speakers = sorted(index)
        for worker in range(threads):
            rng = np.random.default_rng([seed, worker, int(time.time())])
            threading.Thread(target=self._produce, args=(rng, index, speakers, batch), daemon=True).start()

    def _produce(self, rng, index, speakers, batch):
        try:
            while True:
                items = [simulate(rng, index, speakers, RIRS) for _ in range(batch)]
                self.queue.put({k: torch.from_numpy(np.stack([it[k] for it in items])) for k in items[0]})
        except Exception as error:  # surface data errors in the training loop instead of hanging
            self.queue.put(error)

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()
        if isinstance(item, Exception):
            raise item
        return item


def validation_set():
    recipe = {"profile": PROFILE, "smoke": SMOKE, "seed": SEED, "chunk": CHUNK_SECONDS,
              "enroll": ENROLL_SECONDS, "speakers": sorted(VAL_INDEX),
              "code": sha256(Path(__file__).read_bytes()).hexdigest() if "__file__" in globals() else
                      "notebook-noise-robust-v1"}
    signature = sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:12]
    cache = RUN.parent / f"validation_{VAL_SET}_{VAL_EXAMPLES}_{signature}.pt"
    if cache.exists():
        return torch.load(cache, map_location="cpu", weights_only=True)
    rng, speakers = np.random.default_rng(SEED + 1), sorted(VAL_INDEX)
    items = [simulate(rng, VAL_INDEX, speakers, RIRS) for _ in range(VAL_EXAMPLES)]
    data = {k: torch.from_numpy(np.stack([it[k] for it in items])) for k in items[0]}
    atomic(cache, lambda tmp: torch.save(data, tmp))
    return data


VALIDATION = validation_set()
print("validation", {k: tuple(v.shape) for k, v in VALIDATION.items()})

# %% [markdown]
# ## Model
#
# BSRNN in causal mode (32 ms window, 8 ms hop), conditioned on TF-map + frame-level context from the enrollment
# audio. This is the configuration of the released `tfmap_context_causal_100`.
#
# | Model | Feature dim | Repeats | Parameters |
# | --- | --- | --- | --- |
# | base | 128 | 6 | 27 M |
# | small | 64 | 4 | 11.5 M |
#
# For from-scratch modes, the context speaker encoder starts from WeSpeaker's VoxCeleb ECAPA512 (Hugging Face
# `Wespeaker/wespeaker-voxceleb-ecapa-tdnn512` @ `89426e1`). The model card says Apache-2.0; WeSpeaker's docs say
# CC BY 4.0, following VoxCeleb. Keep the attribution.

# %%
from wesep.models import get_model  # noqa: E402

MODEL_ARGS = {
    "joint_training": True, "multi_task": False,
    "separator": {"causal": True, "feature_dim": 128, "nspk": 1, "num_repeat": 6, "sr": RATE,
                  "stride": 128, "win": 512},
    "speaker": {
        "features": {
            "context": {"embed_dim": 512, "enabled": True, "fusion": "multiply"},
            "listen": {"enabled": False}, "spkemb": {"enabled": False}, "tfmap": {"enabled": True},
            "usef": {"embed_dim": 64, "enabled": False}},
        "speaker_model": {
            "fbank": {"dither": 1.0, "frame_length": 25, "frame_shift": 10, "num_mel_bins": 80, "sample_rate": RATE},
            "speaker_encoder": {"model": "ECAPA_TDNN_GLOB_c512", "pretrained": None,
                                "spk_args": {"embed_dim": 192, "feat_dim": 80, "pooling_func": "ASTP"}}}},
}
if MODE == "scratch_small":
    MODEL_ARGS["separator"].update(feature_dim=64, num_repeat=4)
MODEL_ARGS["speaker"]["speaker_model"]["speaker_encoder"]["freeze"] = FREEZE_SPEAKER_ENCODER
if MODE != "finetune_research":
    from huggingface_hub import hf_hub_download
    ecapa = hf_hub_download("Wespeaker/wespeaker-voxceleb-ecapa-tdnn512", "avg_model.pt",
                            revision="89426e111abc9ffcc40279baf808e087693c6e9b", local_dir=WORK / "pretrained/ecapa512")
    encoder = MODEL_ARGS["speaker"]["speaker_model"]["speaker_encoder"]
    encoder.update(pretrained=str(ecapa), freeze=FREEZE_SPEAKER_ENCODER)

model = get_model("TSE_BSRNN_SPK")(MODEL_ARGS)
if MODE == "finetune_research":
    state = torch.load(REALTSE_CKPT / "avg_model.pt", map_location="cpu", weights_only=True)["models"][0]
    model.load_state_dict(state, strict=True)
    print(f"initialised from {REALTSE_CKPT} (research fine-tune)")
model = model.to(DEVICE)
trainable = [p for p in model.parameters() if p.requires_grad]
print(f"parameters {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M, "
      f"trainable {sum(p.numel() for p in trainable) / 1e6:.1f} M")


def set_dither(value):
    """Kaldi fbank dither in the speaker encoder: 1.0 while training (augmentation), 0.0 for evaluation/inference.

    With dither on, inference is random: identical inputs differed by ~0.2% of peak, which earlier tests misread
    as a causality leak.
    """
    for module in model.modules():
        if hasattr(module, "dither"):
            module.dither = value


def extract(mix, enroll):
    """mix: (B, T), enroll: (B, T_e) -> (B, T)."""
    out = model({"wav_mix": mix.unsqueeze(1), "audio_aux": enroll.unsqueeze(1)})["speech"][:, 0]
    return out[..., :mix.shape[-1]]

# %% [markdown]
# ## Loss
#
# - **Target present:** negative SNR, scale-dependent and soft-capped at 30 dB. The reference is silent wherever
#   the target is silent, so leaking other voices there is penalised.
# - **Target absent for the whole chunk:** output energy relative to the mixture, floored at −30 dB, so the model
#   learns to go quiet.
#
# Scale-invariant SI-SDR, which the released model used, cannot express "be silent".

# %%
TAU = 10 ** (-30 / 10)


def tse_loss(est, target, mix):
    t_e, m_e = target.pow(2).sum(-1), mix.pow(2).sum(-1)
    present = t_e > 1e-5 * m_e
    snr = 10 * torch.log10(t_e / ((target - est).pow(2).sum(-1) + TAU * t_e + 1e-10) + 1e-10)
    quiet = 10 * torch.log10((est.pow(2).sum(-1) + TAU * m_e + 1e-10) / (m_e + 1e-10))
    primary = torch.where(present, -snr, quiet).mean()
    return primary + 0.2 * frame_energy_loss(est, target) if ROBUST else primary


def frame_energy_loss(est, target, frame=1600):
    """Local energy fidelity to the target, including its natural loudness changes.

    Silent reference frames remain covered by waveform and absence penalties.
    """
    count = est.shape[-1] // frame
    if count == 0:
        return est.sum() * 0
    e = est[..., :count * frame].reshape(est.shape[0], count, frame).pow(2).mean(-1)
    r = target[..., :count * frame].reshape(target.shape[0], count, frame).pow(2).mean(-1)
    mask = (r > r.amax(-1, keepdim=True) * 1e-4) & (r > 1e-10)
    error = (10 * torch.log10(e + 1e-10) - 10 * torch.log10(r + 1e-10)).abs()
    return (error * mask).sum() / mask.sum().clamp(min=1)


def si_sdr(ref, est):
    ref, est = ref - ref.mean(-1, keepdim=True), est - est.mean(-1, keepdim=True)
    proj = (est * ref).sum(-1, keepdim=True) / (ref.pow(2).sum(-1, keepdim=True) + 1e-10) * ref
    return 10 * torch.log10(proj.pow(2).sum(-1) / ((est - proj).pow(2).sum(-1) + 1e-10) + 1e-10)


@torch.no_grad()
def validate():
    model.eval()
    set_dither(0.0)
    losses, sdri, absent, energy_errors = [], [], [], []
    for i in range(0, VAL_EXAMPLES, BATCH):
        b = {k: v[i:i + BATCH].to(DEVICE) for k, v in VALIDATION.items()}
        est = extract(b["mix"], b["enroll"])
        losses.append(tse_loss(est, b["target"], b["mix"]).item())
        if ROBUST:
            energy_errors.append(frame_energy_loss(est, b["target"]).item())
        for j in range(est.shape[0]):
            act = b["active"][j]
            if act.sum() > RATE // 2:
                sdri.append((si_sdr(b["target"][j][act], est[j][act]) - si_sdr(b["target"][j][act], b["mix"][j][act])).item())
            if (~act).sum() > RATE // 2:  # how much more the output drops than the input while the target is silent
                out_ratio = est[j][~act].pow(2).mean() / (est[j].pow(2).mean() + 1e-10)
                in_ratio = b["mix"][j][~act].pow(2).mean() / (b["mix"][j].pow(2).mean() + 1e-10)
                absent.append((10 * torch.log10(in_ratio + 1e-10) - 10 * torch.log10(out_ratio + 1e-10)).item())
    set_dither(1.0)
    model.train()
    return {"val_loss": float(np.mean(losses)), "val_si_sdri_active": float(np.median(sdri)) if sdri else None,
            "val_absent_suppression": float(np.median(absent)) if absent else None,
            **({"val_frame_energy_error_db": float(np.mean(energy_errors))} if ROBUST else {})}

# %% [markdown]
# ## Train
#
# - Resumes from `RUN/last.pt` if present, or from `RUN/prev.pt` if `last.pt` is damaged.
# - Saves atomically every `CKPT_EVERY` steps and at least every `CKPT_MINUTES`, so a crash loses at most
#   about 10 minutes of training.
# - Skips batches with a non-finite loss or gradient. It stops after `MAX_BAD_STEPS` in a row and never saves
#   diverged weights.
# - Logs to `RUN/log.jsonl`.
# - Keeps `RUN/best.pt`, chosen by validation loss.
# - Re-run this cell to continue after an interruption.

# %%
optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=1e-4)


def lr_at(step):
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    progress = (step - WARMUP) / max(1, STEPS - WARMUP)
    return LR * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))


start_step, best = 0, float("inf")


def load_checkpoint():
    """Newest readable checkpoint: last.pt, else prev.pt (kept in case last.pt is damaged)."""
    for candidate in ("last.pt", "prev.pt"):
        path = RUN / candidate
        if path.exists():
            try:
                return torch.load(path, map_location="cpu", weights_only=False), candidate
            except Exception as error:  # damaged file: fall back to the older one
                print(f"{candidate} unreadable ({type(error).__name__}); trying an older checkpoint")
    return None, None


ckpt, source = load_checkpoint()
if ckpt is not None:
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_step, best = ckpt["step"], ckpt["best"]
    print(f"resumed at step {start_step} from {source}")
    if (RUN / "log.jsonl").exists():  # drop log lines from steps that were lost in the crash
        kept = [line for line in (RUN / "log.jsonl").read_text().splitlines()
                if line and json.loads(line)["step"] <= start_step]
        atomic(RUN / "log.jsonl", lambda tmp: tmp.write_text("".join(f"{line}\n" for line in kept)))


def save(path, step):
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "best": best,
             "mode": MODE, "profile": PROFILE, "model_args": MODEL_ARGS}
    atomic(path, lambda tmp: torch.save(state, tmp))


def save_last(step):
    """Rotate last.pt -> prev.pt, then write the new last.pt atomically."""
    if (RUN / "last.pt").exists():
        os.replace(RUN / "last.pt", RUN / "prev.pt")
    save(RUN / "last.pt", step)


if ckpt is not None and ckpt.get("profile", "baseline") != PROFILE:
    raise ValueError("Resume checkpoint profile differs; use a separate TSE_RUN_NAME")
if ROBUST and ckpt is None:
    initial = validate()
    best = initial["val_loss"]
    save(RUN / "best.pt", 0)  # retain R1 if fine-tuning never beats it on the new validation set
    (RUN / "initial_validation.json").write_text(json.dumps(initial, indent=2) + "\n")
    print({"step": 0, **initial}, flush=True)

log = open(RUN / "log.jsonl", "a")
batches = ThreadedBatches(TRAIN_INDEX, SEED, BATCH, DATA_THREADS)
model.train()
began, window, bad_steps, last_save = time.time(), [], 0, time.time()
for step in range(start_step, STEPS):
    for group in optimizer.param_groups:
        group["lr"] = lr_at(step)
    batch = {k: v.to(DEVICE) for k, v in next(batches).items()}
    loss = tse_loss(extract(batch["mix"], batch["enroll"]), batch["target"], batch["mix"])
    optimizer.zero_grad(set_to_none=True)
    if not torch.isfinite(loss):
        bad_steps += 1
        print(f"step {step + 1}: non-finite loss, batch skipped ({bad_steps} in a row)", flush=True)
        if bad_steps >= MAX_BAD_STEPS:
            (RUN / "DIVERGED").write_text(f"step {step + 1}\n")  # stops run_until_done.sh from retrying
            raise RuntimeError("training diverged; last.pt still holds the last good weights. Lower LR in the "
                               "Config cell, delete RUN/DIVERGED, and re-run to resume.")
        continue
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 5.0)
    if not torch.isfinite(grad_norm):
        bad_steps += 1
        optimizer.zero_grad(set_to_none=True)
        print(f"step {step + 1}: non-finite gradient, update skipped ({bad_steps} in a row)", flush=True)
        if bad_steps >= MAX_BAD_STEPS:
            (RUN / "DIVERGED").write_text(f"step {step + 1}\n")
            raise RuntimeError("training diverged; last.pt still holds the last good weights. Lower LR in the "
                               "Config cell, delete RUN/DIVERGED, and re-run to resume.")
        continue
    bad_steps = 0
    optimizer.step()
    window.append(loss.item())
    if (step + 1) % LOG_EVERY == 0:
        rate = (time.time() - began) / (step + 1 - start_step)
        record = {"step": step + 1, "loss": float(np.mean(window)), "lr": lr_at(step), "sec_per_step": rate,
                  "eta_hours": rate * (STEPS - step - 1) / 3600}
        window = []
        if (step + 1) % VAL_EVERY == 0:
            record.update(validate())
            if record["val_loss"] < best:
                best = record["val_loss"]
                save(RUN / "best.pt", step + 1)
        log.write(json.dumps(record) + "\n")
        log.flush()
        print(record, flush=True)
    if (step + 1) % CKPT_EVERY == 0 or time.time() - last_save > CKPT_MINUTES * 60:
        save_last(step + 1)
        last_save = time.time()
if not (RUN / "last.pt").exists() or torch.load(RUN / "last.pt", map_location="cpu",
                                                  weights_only=False)["step"] < STEPS:
    save_last(STEPS)
if not (RUN / "best.pt").exists():
    save(RUN / "best.pt", STEPS)
print("training finished")

# %% [markdown]
# ## Training curves

# %%
records = [json.loads(line) for line in (RUN / "log.jsonl").read_text().splitlines() if line]
try:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))
    axes[0].plot([r["step"] for r in records], [r["loss"] for r in records])
    axes[0].set_title("train loss")
    val = [r for r in records if "val_loss" in r]
    for ax, key in zip(axes[1:], ("val_si_sdri_active", "val_absent_suppression")):
        ax.plot([r["step"] for r in val], [r[key] for r in val], marker="o")
        ax.set_title(key)
    plt.show()
except Exception as error:
    print("plot skipped:", error)

# %% [markdown]
# ## Evaluate on the held-out pack
#
# The pack holds the six controlled 4–6 speaker conversations from `test-clean` (built by
# `training/make_eval_pack.py`). They are scored exactly like `evaluation/run_discovery_experiment.py`:
# - whole-section SI-SDRi;
# - target-alone SI-SDR;
# - absent suppression.
#
# Enrollments are clean utterances and SepFormer-discovered tracks, normalised to −25 dBFS speech. Gates from the
# plan, for discovered enrollment: SI-SDRi ≥ +10 dB, absent suppression ≥ 10 dB, target-alone ≥ 20 dB, and a
# drop vs clean of ≤ 1 dB.

# %%
def np_si_sdr(ref, est):
    ref, est = ref.astype("float64") - ref.mean(), est.astype("float64") - est.mean()
    proj = np.dot(est, ref) / max(np.dot(ref, ref), 1e-12) * ref
    return float(np.clip(10 * np.log10(max(np.dot(proj, proj), 1e-20) / max(np.dot(est - proj, est - proj), 1e-20)),
                         -120, 120))


def active_speech(wave, frame=1600, floor_db=-35.0):
    n = len(wave) // frame
    if n == 0:
        return wave
    blocks = wave[:n * frame].reshape(n, frame)
    energy = np.square(blocks).mean(axis=1)
    return blocks[energy > energy.max() * 10 ** (floor_db / 10)].reshape(-1) if energy.max() > 0 else wave[:0]


def prepare_enrollment(wave):
    speech = active_speech(wave)[:8 * RATE]
    return speech * (10 ** (-25 / 20) / rms(speech)) if len(speech) else speech


def score_live(output, mix, reference, segments, target):
    def gather(x, spans):
        return np.concatenate([x[a:b] for a, b in spans]) if spans else np.zeros(0)

    def level(x):
        return 10 * np.log10(max(float(np.mean(np.square(x, dtype="float64"))), 1e-20))

    alone = [(s["start_frame"], s["end_frame"]) for s in segments if s["active_speaker_ids"] == [target]]
    active = [(s["start_frame"], s["end_frame"]) for s in segments if target in s["active_speaker_ids"]]
    absent = [(s["start_frame"], s["end_frame"]) for s in segments if target not in s["active_speaker_ids"]]
    return {"si_sdri": np_si_sdr(reference, output) - np_si_sdr(reference, mix),
            "alone_si_sdr": np_si_sdr(gather(reference, alone), gather(output, alone)) if alone else None,
            "absent_suppression": (level(gather(mix, absent)) - level(gather(mix, active)))
                                  - (level(gather(output, absent)) - level(gather(output, active)))}


@torch.no_grad()
def evaluate_pack(split="test"):
    model.eval()
    set_dither(0.0)
    index = json.loads((EVAL_PACK / "index.json").read_text())
    rows = []
    for conv in [c for c in index["conversations"] if c["split"] == split]:
        folder = EVAL_PACK / split / conv["id"]
        mix = read_audio(folder / "live_mix.wav")
        for target in conv["speakers"]:
            reference = read_audio(folder / f"sources/{target}.wav")
            for kind in ("clean", "discovered"):
                path = folder / f"enroll_{kind}/{target}.wav"
                if not path.exists():
                    continue
                enroll = prepare_enrollment(read_audio(path))
                out = extract(torch.from_numpy(mix)[None].to(DEVICE),
                              torch.from_numpy(enroll.astype("float32"))[None].to(DEVICE))[0].cpu().numpy()
                rows.append({"conversation": conv["id"], "target": target, "enrollment": kind,
                             **score_live(out, mix, reference, conv["segments"], target)})
    set_dither(1.0)
    model.train()
    return rows


if (EVAL_PACK / "index.json").exists():
    model.load_state_dict(torch.load(RUN / "best.pt", map_location="cpu", weights_only=False)["model"])
    rows = evaluate_pack("test")
    (RUN / "eval_pack_scores.json").write_text(json.dumps(rows, indent=2))
    print(f"{'enrollment':12s} {'median SI-SDRi':>15s} {'p10':>6s} {'alone SI-SDR':>13s} {'absent supp.':>13s}")
    for kind in ("clean", "discovered"):
        sel = [r for r in rows if r["enrollment"] == kind]
        if sel:
            print(f"{kind:12s} {np.median([r['si_sdri'] for r in sel]):15.1f} "
                  f"{np.percentile([r['si_sdri'] for r in sel], 10):6.1f} "
                  f"{np.median([r['alone_si_sdr'] for r in sel if r['alone_si_sdr'] is not None]):13.1f} "
                  f"{np.median([r['absent_suppression'] for r in sel]):13.1f}")
    print("\nBaselines (same pack):")
    for name, value in json.loads((EVAL_PACK / "baselines.json").read_text()).items():
        print(f"  {name:70s} SI-SDRi {value['median_si_sdri_db']:5.1f}  absent {value['median_absent_suppression_db']:5.1f}")
else:
    print(f"no evaluation pack at {EVAL_PACK}; upload and unzip eval_pack.zip there")

# %% [markdown]
# ## Causality and speed checks
#
# - **Causality:** replacing future audio must not change earlier output. With dither off, a causal model gives
#   exactly 0 (float noise at most).
# - **Speed:** one forward pass on 1 s of audio on this machine. The live demo needs compute below the block length.

# %%
@torch.no_grad()
def causality_check():
    model.eval()
    set_dither(0.0)
    g = torch.Generator().manual_seed(0)
    mix = torch.randn(1, 8 * RATE, generator=g) * 0.05
    enroll = torch.randn(1, 3 * RATE, generator=g) * 0.05
    base = extract(mix.to(DEVICE), enroll.to(DEVICE)).cpu()
    worst = 0.0
    for cut in (3, 5):
        altered = mix.clone()
        altered[:, cut * RATE:] = torch.rand(1, altered.shape[1] - cut * RATE, generator=g) - 0.5
        out = extract(altered.to(DEVICE), enroll.to(DEVICE)).cpu()
        before = slice(0, cut * RATE - 512)
        worst = max(worst, float((out[0, before] - base[0, before]).abs().max() / base[0, before].abs().max()))
    set_dither(1.0)
    model.train()
    return worst


@torch.no_grad()
def speed_check(seconds=1.0, repeats=5):
    model.eval()
    set_dither(0.0)
    mix, enroll = torch.randn(1, int(seconds * RATE), device=DEVICE) * 0.05, torch.randn(1, 5 * RATE, device=DEVICE) * 0.05
    extract(mix, enroll)
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        extract(mix, enroll)
        if DEVICE.type == "mps":
            torch.mps.synchronize()
        times.append(time.perf_counter() - t)
    set_dither(1.0)
    model.train()
    return float(np.median(times))


print(f"causality: largest relative change before the cut {causality_check():.2e}")
print(f"speed: {speed_check():.3f} s compute for 1 s of audio on {DEVICE}")

# %% [markdown]
# ## Export
#
# Writes `RUN/export/<name>/config.yaml` and `avg_model.pt` in WeSep's checkpoint format. To try the model live:
# 1. Copy the folder to `evaluation/wesep_eval/checkpoints/pretrained/` on the laptop.
# 2. Run `live_demo_wesep.py --model <name>`.
#
# R1 exports derive from unlicensed weights and are marked research-only.

# %%
name = f"retrained_{RUN_NAME}"
export = RUN / "export" / name
export.mkdir(parents=True, exist_ok=True)
best_state = torch.load(RUN / "best.pt", map_location="cpu", weights_only=False)["model"]
torch.save({"models": [best_state]}, export / "avg_model.pt")
config = {"model": {"tse_model": "TSE_BSRNN_SPK"}, "model_args": {"tse_model": MODEL_ARGS},
          "dataset_args": {"resample_rate": RATE}}
config["model_args"]["tse_model"]["speaker"]["speaker_model"]["speaker_encoder"]["pretrained"] = None
config["model_args"]["tse_model"]["speaker"]["speaker_model"]["fbank"]["dither"] = 0.0  # deterministic inference
(export / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
(export / "NOTICE.txt").write_text(
    ("RESEARCH ONLY: fine-tuned from the REAL-TSE checkpoint, which has no stated weights licence. Do not ship.\n"
     if MODE == "finetune_research" else
     "Trained from scratch with WeSep (Apache-2.0) on LibriSpeech (CC BY 4.0); context encoder initialised from "
     "WeSpeaker VoxCeleb ECAPA512 (attribution required).\n")
    + f"mode={MODE} profile={PROFILE} smoke={SMOKE} steps={STEPS} date={time.strftime('%Y-%m-%d')}\n")
print("exported to", export)
(RUN / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
