"""Discovery enrollment and live extraction building blocks (offline research code).

Discovery: windowed blind separation with the 3-stream Libri3Mix SepFormer,
mixture-fit gains per window, near-silent stream removal, speaker embeddings
per stream, agglomerative clustering into speakers, and overlap-add stitching
into one track per discovered speaker. The total speaker count is open as long
as at most three people talk inside one window.

Live models:
* SpeakerBeam-SS: the pinned OpenSpeakerBeam-SS checkpoint conditioned on a
  Resemblyzer embedding; one forward pass over the whole input.
* SepFormer selector: the same windowed separation, keeping in each window
  the streams whose ECAPA embedding matches the target above a threshold.
Neither is causal; both are offline references for the live stage.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent
SEPFORMER = ROOT / "sepformer_libr3mix_checkpoint"
ECAPA = ROOT / "ecapa_checkpoint"
SPEAKERBEAM_ROOT = ROOT.parent / "speakerbeam_ss_verification"
SPEAKERBEAM_HASH = "41c250587af2e947dc3e265cd6e7aa7de230048d408e93c53d8348d63cf75896"
RATE = 16000
WINDOW_SECONDS = 4.0
HOP_SECONDS = 2.0
# A stream below this energy relative to its window mixture is treated as empty.
STREAM_FLOOR_DB = -15.0
SILENT_WINDOW_RMS = 1e-4


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_wave(path):
    wave, rate = sf.read(path, dtype="float32")
    if rate != RATE or wave.ndim != 1 or not np.isfinite(wave).all():
        raise ValueError(f"expected finite mono 16 kHz audio: {path}")
    return wave


def unit(vector):
    vector = np.asarray(vector, dtype="float64").reshape(-1)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def active_speech(wave, frame=1600, floor_db=-35.0):
    """Concatenate 100 ms frames within floor_db of the loudest frame."""
    frames = len(wave) // frame
    if frames == 0:
        return wave
    blocks = wave[:frames * frame].reshape(frames, frame)
    energy = np.square(blocks).mean(axis=1)
    if energy.max() <= 0:
        return np.zeros(0, dtype="float32")
    keep = energy > energy.max() * 10 ** (floor_db / 10)
    return blocks[keep].reshape(-1)


class Models:
    """Lazily loaded pinned models shared by all runs."""

    def __init__(self):
        torch.set_num_threads(4)
        self._sepformer = self._ecapa = self._resemblyzer = self._speakerbeam = None

    @property
    def sepformer(self):
        if self._sepformer is None:
            from speechbrain.inference.separation import SepformerSeparation
            self._sepformer = SepformerSeparation.from_hparams(
                source=str(SEPFORMER), savedir=str(ROOT / "sepformer_libr3mix_loaded"),
                run_opts={"device": "cpu"})
            if self._sepformer.hparams.sample_rate != 8000 or self._sepformer.hparams.num_spks != 3:
                raise ValueError("unexpected SepFormer checkpoint configuration")
        return self._sepformer

    @property
    def ecapa(self):
        if self._ecapa is None:
            from speechbrain.inference.classifiers import EncoderClassifier
            self._ecapa = EncoderClassifier.from_hparams(
                source=str(ECAPA), savedir=str(ROOT / "ecapa_loaded"),
                overrides={"pretrained_path": str(ECAPA)}, run_opts={"device": "cpu"})
        return self._ecapa

    @property
    def resemblyzer(self):
        if self._resemblyzer is None:
            from resemblyzer import VoiceEncoder
            self._resemblyzer = VoiceEncoder(device="cpu", verbose=False)
        return self._resemblyzer

    @property
    def speakerbeam(self):
        if self._speakerbeam is None:
            checkpoint = SPEAKERBEAM_ROOT / "checkpoints/best_model.pth"
            if sha256(checkpoint) != SPEAKERBEAM_HASH:
                raise ValueError("SpeakerBeam-SS checkpoint SHA-256 differs from verified value")
            sys.path.insert(0, str(SPEAKERBEAM_ROOT / "upstream"))
            from model import SpeakerBeamSS
            model = SpeakerBeamSS().eval()
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True),
                                  strict=True)
            self._speakerbeam = model
        return self._speakerbeam

    def embed(self, encoder, wave):
        """Unit speaker embedding of 16 kHz audio, or None if there is no usable speech."""
        speech = active_speech(wave)
        if len(speech) < int(0.5 * RATE):
            return None
        if encoder == "ecapa":
            with torch.inference_mode():
                encoded = self.ecapa.encode_batch(torch.from_numpy(speech.astype("float32")).unsqueeze(0))
            return unit(encoded.cpu().numpy())
        if encoder == "resemblyzer":
            from resemblyzer import preprocess_wav
            prepared = preprocess_wav(speech.astype("float32"), source_sr=RATE)
            if len(prepared) < int(0.5 * RATE):
                return None
            return unit(self.resemblyzer.embed_utterance(prepared))
        raise ValueError(encoder)


def separate_windows(models, mix16):
    """3-stream SepFormer per overlapping window, with mixture-fit gains.

    Returns window starts, the padded length, and per window a list of kept
    16 kHz streams with their energy relative to the window mixture.
    """
    window = int(WINDOW_SECONDS * RATE)
    hop = int(HOP_SECONDS * RATE)
    count = max(1, int(np.ceil(max(len(mix16) - window, 0) / hop)) + 1)
    padded = np.zeros((count - 1) * hop + window, dtype="float32")
    padded[:len(mix16)] = mix16
    windows = []
    for index in range(count):
        start = index * hop
        segment = padded[start:start + window]
        record = {"start": start, "streams": [], "fit_error": None}
        windows.append(record)
        if np.sqrt(np.mean(segment ** 2)) < SILENT_WINDOW_RMS:
            continue
        seg8 = scipy.signal.resample_poly(segment, 1, 2).astype("float32")
        with torch.inference_mode():
            estimate = models.sepformer.separate_batch(torch.from_numpy(seg8).unsqueeze(0))
        streams8 = estimate[0].cpu().numpy().astype("float64")[:len(seg8)]
        gains, _, _, _ = np.linalg.lstsq(streams8, seg8.astype("float64"), rcond=None)
        fitted8 = streams8 * gains
        record["fit_error"] = float(np.linalg.norm(fitted8.sum(axis=1) - seg8) /
                                    max(np.linalg.norm(seg8), 1e-12))
        mix_energy = float(np.sum(segment.astype("float64") ** 2))
        for column in range(fitted8.shape[1]):
            stream = scipy.signal.resample_poly(fitted8[:, column], 2, 1)[:window]
            stream = np.pad(stream, (0, window - len(stream))).astype("float32")
            relative_db = 10 * np.log10(max(float(np.sum(stream.astype("float64") ** 2)), 1e-20) /
                                        max(mix_energy, 1e-20))
            if relative_db >= STREAM_FLOOR_DB:
                record["streams"].append({"column": column, "wave": stream,
                                          "relative_db": float(relative_db)})
    return {"windows": windows, "window": window, "hop": hop,
            "padded_frames": len(padded), "frames": len(mix16)}


def embed_streams(models, separation, encoder):
    """Attach an embedding per kept stream (key = encoder name); drops streams without speech."""
    for record in separation["windows"]:
        for stream in record["streams"]:
            stream[encoder] = models.embed(encoder, stream["wave"])
    return separation


def overlap_add(separation, pieces):
    """pieces: {window_index: waveform}; weighted average over overlapping windows."""
    window, frames = separation["window"], separation["padded_frames"]
    taper = scipy.signal.windows.hann(window, sym=False).astype("float64") + 1e-3
    total = np.zeros(frames)
    weight = np.zeros(frames)
    for index, record in enumerate(separation["windows"]):
        start = record["start"]
        weight[start:start + window] += taper
        if index in pieces:
            total[start:start + window] += pieces[index] * taper
    out = total / np.maximum(weight, 1e-6)
    return out[:separation["frames"]].astype("float32")


def cluster_streams(separation, encoder, distance_threshold, min_windows):
    """Agglomerative clustering of stream embeddings; returns discovered speakers."""
    from sklearn.cluster import AgglomerativeClustering
    items = [(w, s) for w, record in enumerate(separation["windows"])
             for s, stream in enumerate(record["streams"]) if stream.get(encoder) is not None]
    if not items:
        return []
    vectors = np.stack([separation["windows"][w]["streams"][s][encoder] for w, s in items])
    if len(items) == 1:
        labels = np.zeros(1, dtype=int)
    else:
        labels = AgglomerativeClustering(
            n_clusters=None, metric="cosine", linkage="average",
            distance_threshold=distance_threshold).fit_predict(vectors)
    speakers = []
    for label in sorted(set(labels)):
        members = [items[i] for i in np.flatnonzero(labels == label)]
        windows = sorted({w for w, _ in members})
        if len(windows) < min_windows:
            continue
        pieces = {}
        for w, s in members:
            pieces[w] = pieces.get(w, 0) + separation["windows"][w]["streams"][s]["wave"].astype("float64")
        speakers.append({"windows": windows, "members": len(members),
                         "track": overlap_add(separation, pieces)})
    # Order by first appearance, like a listener would meet them.
    speakers.sort(key=lambda sp: sp["windows"][0])
    return speakers


def discover(models, mix16, encoder="ecapa", distance_threshold=0.6, min_windows=2,
             separation=None):
    if separation is None:
        separation = separate_windows(models, mix16)
    embed_streams(models, separation, encoder)
    return cluster_streams(separation, encoder, distance_threshold, min_windows), separation


def select_streams(separation, target, threshold):
    """SepFormer selector: per window, keep streams matching the target ECAPA embedding."""
    pieces = {}
    for index, record in enumerate(separation["windows"]):
        chosen = [stream["wave"].astype("float64") for stream in record["streams"]
                  if stream.get("ecapa") is not None and
                  float(np.dot(stream["ecapa"], target)) >= threshold]
        if chosen:
            pieces[index] = np.sum(chosen, axis=0)
    return overlap_add(separation, pieces)


def select_streams_competitive(separation, candidates, target_index, floor):
    """Keep a stream only if the target is its closest known speaker (and above floor).

    candidates: unit ECAPA embeddings of every known speaker (all discovered
    speakers, or all enrolled ones); target_index points at the chosen one.
    """
    known = np.stack(candidates)
    pieces = {}
    for index, record in enumerate(separation["windows"]):
        chosen = []
        for stream in record["streams"]:
            if stream.get("ecapa") is None:
                continue
            similarity = known @ stream["ecapa"]
            if int(np.argmax(similarity)) == target_index and similarity[target_index] >= floor:
                chosen.append(stream["wave"].astype("float64"))
        if chosen:
            pieces[index] = np.sum(chosen, axis=0)
    return overlap_add(separation, pieces)


SPEAKERBEAM_STRIDE = 160  # encoder Conv1d stride; output drops a partial last frame


def speakerbeam_extract(models, mix16, resemblyzer_embedding):
    padded = np.pad(mix16, (0, -len(mix16) % SPEAKERBEAM_STRIDE))
    mixture = torch.from_numpy(padded.astype("float32")).reshape(1, 1, -1)
    embedding = torch.from_numpy(np.asarray(resemblyzer_embedding, dtype="float32")).reshape(1, -1)
    with torch.inference_mode():
        result = models.speakerbeam(mixture, embedding)
    out = result[0, 0].cpu().numpy().astype("float32")
    if out.shape != padded.shape or not np.isfinite(out).all():
        raise ValueError(f"invalid SpeakerBeam-SS output {out.shape}")
    return out[:len(mix16)]
