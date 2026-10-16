"""Official pretrained streaming models; caches and overlap survive each hop."""
import ctypes
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parent / 'third_party'


def onnx_session(path):
    import onnxruntime as ort
    ort.disable_telemetry_events()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])


class StreamingModel:
    def reset(self):
        self.history = np.zeros(self.delay, dtype=np.float32)
        self.times = []

    def process(self, chunk, bypass=False):
        if chunk.shape != (self.block,):
            raise ValueError('Incorrect model hop size')
        # Keep recurrent state advancing even during bypass. Match the raw
        # signal's sample delay to the denoised output's overlap-add delay.
        raw = np.concatenate((self.history, chunk))
        delayed = raw[:self.block].copy()
        self.history = raw[self.block:].copy()
        began = time.perf_counter()
        out = np.asarray(self.enhance(chunk), dtype=np.float32)
        self.times.append(time.perf_counter() - began)
        if out.shape != chunk.shape or not np.isfinite(out).all():
            raise RuntimeError('Invalid model output')
        return delayed if bypass else out

    def close(self):
        pass


class GTCRN(StreamingModel):
    name, rate, block, delay = 'GTCRN (DNS3, ONNX)', 16000, 256, 256

    def __init__(self):
        self.session = onnx_session(ROOT / 'gtcrn/stream/onnx_models/gtcrn_simple.onnx')
        # Periodic sqrt-Hann, as torch.hann_window(512).pow(0.5) upstream.
        self.window = np.sqrt(0.5 - 0.5 * np.cos(2*np.pi*np.arange(512)/512)).astype(np.float32)
        self.reset()

    def reset(self):
        super().reset()
        self.input_buffer = np.zeros(512, dtype=np.float32)
        self.output_buffer = np.zeros(512, dtype=np.float32)
        self.caches = [np.zeros(s, dtype=np.float32) for s in
                       ((2,1,16,16,33), (2,3,1,1,16), (2,1,33,16))]

    def enhance(self, chunk):
        self.input_buffer[:256] = self.input_buffer[256:]
        self.input_buffer[256:] = chunk
        spectrum = np.fft.rfft(self.input_buffer * self.window)
        mix = np.stack((spectrum.real, spectrum.imag), axis=-1)[None,:,None,:].astype(np.float32)
        result = self.session.run(None, dict(zip(
            ('mix','conv_cache','tra_cache','inter_cache'), [mix,*self.caches])))
        self.caches = result[1:]
        z = result[0][0,:,0,0] + 1j*result[0][0,:,0,1]
        frame = np.fft.irfft(z, n=512).astype(np.float32)*self.window
        self.output_buffer[:256] = self.output_buffer[256:]
        self.output_buffer[256:] = 0
        self.output_buffer += frame
        # 50% overlap of periodic Hann squared sums to one.
        return self.output_buffer[:256].copy()


class DTLN(StreamingModel):
    name, rate, block, delay = 'DTLN (pretrained, ONNX)', 16000, 128, 384

    def __init__(self):
        self.sessions = [onnx_session(ROOT / f'DTLN/pretrained_model/model_{n}.onnx') for n in (1,2)]
        self.reset()

    def reset(self):
        super().reset()
        self.inputs = [{i.name:np.zeros(i.shape, dtype=np.float32) for i in s.get_inputs()}
                       for s in self.sessions]
        self.input_buffer = np.zeros(512, dtype=np.float32)
        self.output_buffer = np.zeros(512, dtype=np.float32)

    def stage(self, index, signal):
        values = self.inputs[index]
        names = list(values)
        values[names[0]] = signal.astype(np.float32)
        output, state = self.sessions[index].run(None, values)
        values[names[1]] = state
        return output

    def enhance(self, chunk):
        self.input_buffer[:-128] = self.input_buffer[128:]
        self.input_buffer[-128:] = chunk
        spectrum = np.fft.rfft(self.input_buffer)
        magnitude = np.abs(spectrum)[None,None,:].astype(np.float32)
        mask = self.stage(0, magnitude)
        frame = np.fft.irfft(magnitude*mask*np.exp(1j*np.angle(spectrum)), n=512)
        out = self.stage(1, frame).reshape(512)
        # Match the upstream two-stage ONNX implementation; no added Hann.
        self.output_buffer[:-128] = self.output_buffer[128:]
        self.output_buffer[-128:] = 0
        self.output_buffer += out
        return self.output_buffer[:128].copy()


class RNNoise(StreamingModel):
    name, rate, block, delay = 'RNNoise (official v0.1.1, C)', 48000, 480, 480

    def __init__(self):
        self.lib = ctypes.CDLL(str(ROOT / 'rnnoise-v0.1.1/librnnoise.dylib'))
        self.lib.rnnoise_create.argtypes = [ctypes.c_void_p]
        self.lib.rnnoise_create.restype = ctypes.c_void_p
        self.lib.rnnoise_destroy.argtypes = [ctypes.c_void_p]
        ptr = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS')
        self.lib.rnnoise_process_frame.argtypes = [ctypes.c_void_p, ptr, ptr]
        self.lib.rnnoise_process_frame.restype = ctypes.c_float
        self.lib.rnnoise_get_frame_size.restype = ctypes.c_int
        assert self.lib.rnnoise_get_frame_size() == self.block
        self.state = None
        self.reset()

    def reset(self):
        super().reset()
        self.close()
        self.state = self.lib.rnnoise_create(None)
        if not self.state:
            raise RuntimeError('RNNoise state allocation failed')

    def enhance(self, chunk):
        # RNNoise expects float samples in signed 16-bit PCM amplitude units.
        x = np.ascontiguousarray(chunk*32768, dtype=np.float32)
        out = np.empty(self.block, dtype=np.float32)
        self.lib.rnnoise_process_frame(self.state, out, x)
        return out/32768

    def close(self):
        if self.state:
            self.lib.rnnoise_destroy(self.state)
            self.state = None


MODELS = {'gtcrn':GTCRN, 'rnnoise':RNNoise, 'dtln':DTLN}
