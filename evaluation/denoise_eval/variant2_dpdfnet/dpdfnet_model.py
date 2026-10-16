"""Variant 2: verified official DPDFNet models, stateful native-rate streaming."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
MODELS = ('dpdfnet2','dpdfnet4')


class DPDFModel:
    def __init__(self, name='dpdfnet2'):
        from dpdfnet import StreamEnhancer
        from dpdfnet.audio import ATTN_LIMIT_NOISY_FRAME_OFFSET
        import onnxruntime as ort
        ort.disable_telemetry_events()
        manifest = json.loads((ROOT/'model_manifest.json').read_text())
        record = manifest['models'][name]
        self.path = ROOT/record['file']
        if hashlib.sha256(self.path.read_bytes()).hexdigest() != record['sha256']:
            raise RuntimeError(f'{name} checksum mismatch; run setup_models.py again')
        self.name = name
        self.enhancer = StreamEnhancer(model=name,onnx_path=self.path)
        self.rate = self.enhancer._model_sr
        self.block = self.enhancer._hop_size
        self.window = self.enhancer._win_len
        # Pinned official package advances offline output by four hops, while
        # streaming preserves that model delay. Add one hop for causal OLA.
        # Confirmed using both models on Mix.wav: peak-correlation lag800=50ms.
        self.delay = self.window-self.block+ATTN_LIMIT_NOISY_FRAME_OFFSET*self.block
        if (self.rate,self.block,self.window) != (16000,160,320):
            raise RuntimeError('Unexpected model geometry; alignment needs revalidation')
        self.reset()
        for _ in range(8):
            self.process(np.zeros(self.block,dtype=np.float32))
        self.reset()

    def reset(self):
        self.enhancer.reset()
        # Prime one silent hop for full overlap-add coverage at stream start.
        initial = self.enhancer.process(np.zeros(self.window-self.block,dtype=np.float32),sample_rate=self.rate)
        if len(initial):
            raise RuntimeError('Unexpected priming output')
        self.history = np.zeros(self.delay,dtype=np.float32)
        self.times = []

    def process(self, chunk, bypass=False):
        if chunk.shape != (self.block,) or not np.isfinite(chunk).all():
            raise ValueError('Expected one finite mono model hop')
        raw = np.concatenate((self.history,chunk))
        reference = raw[:self.block].copy()
        self.history = raw[self.block:].copy()
        began = time.perf_counter()
        # Native model rate ONLY; external soxr streams carry resampling state.
        out = self.enhancer.process(chunk,sample_rate=self.rate)
        self.times.append(time.perf_counter()-began)
        if out.shape != chunk.shape or not np.isfinite(out).all():
            raise RuntimeError('DPDFNet returned invalid audio')
        return reference if bypass else out

    def statistics(self):
        t = np.asarray(self.times)
        return {'model':self.name,'rate':self.rate,'hop_ms':1000*self.block/self.rate,
                'sample_alignment_ms':1000*self.delay/self.rate,
                'median_compute_ms':float(np.median(t)*1000) if len(t) else None,
                'p90_compute_ms':float(np.percentile(t,90)*1000) if len(t) else None}

    def close(self):
        # No hardware or file handles; ONNX Runtime owns the in-memory session.
        pass
