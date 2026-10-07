"""Stateful streaming inference for WeSep TSE_BSRNN_SPK (causal checkpoints).

Instead of re-running the model on block + past context, this runs it frame by
frame (hop 128 samples = 8 ms at 16 kHz) and carries what depends on the past:

- streaming STFT: centered 512-sample Hann frames, reflect padding at the
  stream start exactly like torch.stft(center=True);
- the time LSTM in every BSNet block (`band_rnn`): hidden/cell state carried;
- streaming iSTFT: overlap-add with the window-square envelope, emitting
  samples as soon as no later frame can touch them.

Everything else in the causal model works per frame:
- channel-wise LayerNorm and band-split convolutions;
- the band-to-band LSTM `band_comm`, which runs across frequency within one frame;
- TF-map attention and context cross-attention, from the current frame to the enrollment frames;
- the mask network.

Enrollment features (TF-map spectrum and the context encoder's frame-level
features) are computed once. Algorithmic latency is one window (512 samples =
32 ms); compute and audio buffers come on top.

Complex tensors never go to the device, so it runs on Metal (MPS) without the
CPU fallback.

    .venv/bin/python streaming_wesep.py          # equivalence, causality, and speed checks
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "upstream/wesep-real-tse"))

RATE = 16000


def deterministic(model):
    """Turn off Kaldi fbank dither in the speaker encoders.

    The checkpoint configs train with dither=1.0, which keeps adding random noise at inference too: two runs on
    the same input differed by ~0.2% of peak. That noise was misread as a small causality leak in earlier tests.
    """
    for module in model.modules():
        if hasattr(module, "dither"):
            module.dither = 0.0
    return model


class StreamingWesep:
    """Feed any number of 16 kHz samples with push(); get back the samples that became final."""

    def __init__(self, model, enrollment, device="cpu"):
        self.model, self.device = deterministic(model.eval()).to(device), torch.device(device)
        sep = model.sep_model
        self.win, self.hop = sep.win, sep.stride
        self.window = torch.hann_window(self.win).numpy().astype("float64")  # periodic, as torch.stft
        self.nband, self.feat = sep.nband, sep.feature_dim
        self.spk = model.spk_configs["features"]
        if self.spk["listen"]["enabled"] or self.spk["usef"]["enabled"]:
            raise ValueError("streaming supports tfmap/context/spkemb checkpoints only")
        if any(b.band_rnn.bidirectional for b in sep.separator.separation):
            raise ValueError("checkpoint is not causal (bidirectional time LSTM)")
        self.set_enrollment(enrollment)
        self.reset()

    @torch.no_grad()
    def set_enrollment(self, enrollment):
        enroll = torch.from_numpy(np.asarray(enrollment, dtype="float32"))[None].to(self.device)
        cpu_enroll = enroll.cpu()
        self.enroll_mag = None
        if self.spk["tfmap"]["enabled"]:
            spec = torch.stft(cpu_enroll, self.win, self.hop, self.win, window=torch.hann_window(self.win),
                              return_complex=True)
            self.enroll_mag = spec.abs().to(self.device)  # (1, F, T_e)
        self.enroll_ctx = self.model.spk_ft.context.compute(enroll) if self.spk["context"]["enabled"] else None
        self.enroll_emb = None
        if self.spk["spkemb"]["enabled"]:
            self.enroll_emb = self.model.spk_ft.spkemb.compute(enroll).unsqueeze(1).unsqueeze(3)

    def reset(self):
        self.pending = np.zeros(0)        # input samples not yet framed (padded coordinates)
        self.started = False
        self.head = np.zeros(0)           # first samples, until the start reflection can be built
        self.frames = 0                   # frames processed so far
        self.states = [None] * len(self.model.sep_model.separator.separation)
        self.ola = np.zeros(self.win)     # overlap-add buffer, padded coordinate `ola_start`
        self.env = np.zeros(self.win)
        self.ola_start = 0
        self.dropped = 0                  # padded samples discarded before original time 0
        self.compute = 0.0
        self.processed = 0

    def push(self, samples):
        samples = np.asarray(samples, dtype="float64")
        self.processed += len(samples)
        if not self.started:
            self.head = np.concatenate([self.head, samples])
            if len(self.head) <= self.win // 2:
                return np.zeros(0, dtype="float32")
            pad = self.head[1:self.win // 2 + 1][::-1]  # torch.stft reflect padding at the start
            self.pending = np.concatenate([pad, self.head])
            self.started, self.head = True, None
        else:
            self.pending = np.concatenate([self.pending, samples])
        count = (len(self.pending) - self.win) // self.hop + 1 if len(self.pending) >= self.win else 0
        if count <= 0:
            return np.zeros(0, dtype="float32")
        frames = np.stack([self.pending[i * self.hop:i * self.hop + self.win] for i in range(count)])
        self.pending = self.pending[count * self.hop:]
        began = time.perf_counter()
        spec = np.fft.rfft(frames * self.window, axis=1).T  # (F, K) complex, like torch.stft
        est = self._frames(spec)
        out = self._overlap_add(est)
        self.compute += time.perf_counter() - began
        return out

    @torch.no_grad()
    def _frames(self, spec):
        m, sep, dev = self.model, self.model.sep_model, self.device
        real = torch.from_numpy(spec.real.astype("float32"))[None].to(dev)  # (1, F, K)
        imag = torch.from_numpy(spec.imag.astype("float32"))[None].to(dev)
        spec_ri = torch.stack([real, imag], 1)  # (1, 2, F, K)
        if self.enroll_mag is not None:
            tfmap = m.spk_ft.tfmap.compute(self.enroll_mag, torch.sqrt(real ** 2 + imag ** 2))
            spec_ri = m.spk_ft.tfmap.post(spec_ri, tfmap.unsqueeze(1))
        feature = sep.subband_norm(sep.band_split(spec_ri))  # (1, nband, feat, K)
        if self.enroll_ctx is not None:
            feature = m.spk_ft.context.post(feature, self.enroll_ctx)
        if self.enroll_emb is not None:
            feature = m.spk_ft.spkemb.post(feature, self.enroll_emb)
        x = feature.reshape(1, self.nband * self.feat, -1)
        frames = x.shape[-1]
        for i, block in enumerate(sep.separator.separation):  # BSNet with carried time-LSTM state
            band_in = x.view(self.nband, self.feat, frames)
            rnn = block.band_rnn
            out, self.states[i] = rnn.rnn(rnn.norm(band_in).transpose(1, 2).contiguous(), self.states[i])
            out = rnn.proj(out.reshape(-1, out.shape[2])).view(self.nband, frames, self.feat)
            band_out = (band_in + out.transpose(1, 2)).view(1, self.nband, self.feat, frames)
            band_out = band_out.permute(0, 3, 2, 1).contiguous().view(frames, self.feat, self.nband)
            x = block.band_comm(band_out).view(1, frames, self.feat, self.nband).permute(0, 3, 2, 1)
            x = x.contiguous().view(1, self.nband * self.feat, frames)
        sep_out = x.view(1, self.nband, self.feat, frames)
        est_r, est_i, start = [], [], 0
        for i, mask_fn in enumerate(sep.band_masker.mask):  # BandMasker with real arithmetic
            bw = sep.band_masker.band_width[i]
            o = mask_fn(sep_out[:, i]).view(1, 1, 2, 2, bw, frames)
            mask = o[:, :, 0] * torch.sigmoid(o[:, :, 1])
            mr, mi = mask[:, 0, 0], mask[:, 0, 1]
            xr, xi = real[:, start:start + bw], imag[:, start:start + bw]
            est_r.append(xr * mr - xi * mi)
            est_i.append(xr * mi + xi * mr)
            start += bw
        est_r = torch.cat(est_r, 1)[0].cpu().numpy().astype("float64")
        est_i = torch.cat(est_i, 1)[0].cpu().numpy().astype("float64")
        return est_r + 1j * est_i  # (F, K)

    def _overlap_add(self, est):
        frames = np.fft.irfft(est.T, n=self.win, axis=1) * self.window  # (K, win)
        out = []
        for frame in frames:
            offset = self.frames * self.hop - self.ola_start
            need = offset + self.win
            if need > len(self.ola):
                self.ola = np.concatenate([self.ola, np.zeros(need - len(self.ola))])
                self.env = np.concatenate([self.env, np.zeros(need - len(self.env))])
            self.ola[offset:offset + self.win] += frame
            self.env[offset:offset + self.win] += self.window ** 2
            self.frames += 1
            final = self.frames * self.hop - self.ola_start  # no later frame starts before this
            ready = self.ola[:final] / np.maximum(self.env[:final], 1e-11)
            self.ola, self.env = self.ola[final:], self.env[final:]
            self.ola_start += final
            skip = min(len(ready), max(0, self.win // 2 - self.dropped))  # centered: drop padding
            self.dropped += skip
            out.append(ready[skip:])
        return np.concatenate(out).astype("float32") if out else np.zeros(0, dtype="float32")

    def stats(self):
        seconds = self.processed / RATE
        return {"audio_seconds": seconds, "compute_seconds": self.compute,
                "real_time_factor": self.compute / max(seconds, 1e-9)}


def stream(model, mix, enrollment, device="cpu", chunk=128):
    streamer = StreamingWesep(model, enrollment, device)
    out = [streamer.push(mix[i:i + chunk]) for i in range(0, len(mix), chunk)]
    return np.concatenate(out), streamer.stats()


def main():
    import json
    sys.path.insert(0, str(HERE.parent))
    from run_wesep_experiment import enrollment, extract, load
    from run_discovery_experiment import DATA, load_mixture
    torch.set_num_threads(4)
    plan = json.loads((DATA / "mixtures.json").read_text())
    mixture = load_mixture("test", plan["test"][0])
    mix = mixture["sections"]["live"]["mix"][:12 * RATE]
    enroll = enrollment(mixture["enrollments"][mixture["speakers"][0]])
    extractor = load("tfmap_context_causal_100")
    deterministic(extractor.model)
    full = extract(extractor, mix, enroll)
    again = extract(extractor, mix, enroll)
    print(f"full pass twice, dither off: max diff {np.max(np.abs(full - again)):.1e}")
    for device in ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []):
        for chunk in (128, 256, 512):
            out, stats = stream(extractor.model, mix, enroll, device, chunk)
            n = min(len(out), len(full)) - 512  # the full pass reflects at the end; skip its last window
            diff = np.max(np.abs(out[:n] - full[:n])) / np.max(np.abs(full[:n]))
            print(f"{device} chunk {chunk / RATE * 1000:4.0f} ms: max |stream - full| = {diff:.2e} of peak, "
                  f"real-time factor {stats['real_time_factor']:.2f}, output {len(out)} of {len(mix)} samples")
        extractor.model.to("cpu")
    out, _ = stream(extractor.model, mix, enroll, "cpu", 128)
    altered = mix.copy()
    altered[8 * RATE:] = np.random.default_rng(0).uniform(-0.5, 0.5, len(altered) - 8 * RATE)
    out2, _ = stream(extractor.model, altered, enroll, "cpu", 128)
    cut = 8 * RATE - 512  # output up to here only needs input before 8 s
    print(f"strict causality: max change before {cut / RATE:.3f} s = {np.max(np.abs(out[:cut] - out2[:cut])):.1e}")


if __name__ == "__main__":
    main()
