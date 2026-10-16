"""Live DeepFilterNet3 monitor using short blocks and recomputed past context.

This is a listening prototype, not the upstream frame-by-frame Rust runtime.
Microphone audio stays in memory. Use headphones; type b + Enter for matched-delay
bypass, d + Enter for denoising, q + Enter (or Ctrl+C) to stop.
"""
import argparse
from collections import deque
from contextlib import ExitStack
import queue
import threading
import time

import numpy as np
import soundfile as sf
import torch
import torchaudio
from df.enhance import init_df, enhance
from audio_preprocessing import Preprocessor, add_arguments, settings_from_args


class BlockDenoiser:
    def __init__(self, block_seconds=0.2, context_seconds=0.8, attenuation=12):
        torch.set_num_threads(2)
        self.model, self.state, _ = init_df(model_base_dir='DeepFilterNet3', post_filter=False, log_file=None)
        self.rate = self.state.sr()
        self.block = round(block_seconds * self.rate)
        self.history = np.zeros(round(context_seconds * self.rate), dtype='float32')
        # DF3 config: 20 ms neural lookahead + 10 ms STFT delay. Keep 40 ms
        # behind the captured right edge rather than emitting its zero-padded tail.
        self.guard = round(0.04 * self.rate)
        if self.block % self.state.hop_size() or self.block <= self.guard or len(self.history) < self.guard:
            raise ValueError('Block must be a multiple of 10 ms, >40 ms; context must be >=40 ms')
        self.attenuation = attenuation
        self.bypass = False
        self.times = []
        self.process(np.zeros(self.block, dtype='float32'))  # warm up before opening audio
        self.history.fill(0)
        self.times.clear()

    def process(self, chunk):
        if chunk.shape != (self.block,):
            raise ValueError('Incorrect block size')
        window = np.concatenate([self.history, chunk])
        start = len(window) - self.guard - self.block
        began = time.perf_counter()
        if self.bypass:
            output = window[start:start + self.block].copy()
        else:
            result = enhance(self.model, self.state, torch.from_numpy(window)[None],
                             pad=True, atten_lim_db=self.attenuation)
            output = result[0, start:start + self.block].cpu().numpy().copy()
            self.times.append(time.perf_counter() - began)
        self.history[:] = window[-len(self.history):]
        if output.shape != chunk.shape or not np.isfinite(output).all():
            raise RuntimeError('Invalid denoiser output')
        return output


class PlaybackBuffer:
    def __init__(self, prefill, maximum):
        self.parts = deque([np.zeros(prefill, dtype='float32')])
        self.frames = prefill
        self.maximum = maximum
        self.lock = threading.Lock()
        self.underruns = 0
        self.drops = 0

    def write(self, wave):
        if not len(wave):
            return
        with self.lock:
            if self.frames + len(wave) > self.maximum:
                self.parts.clear()
                self.frames = 0
                self.drops += 1
            self.parts.append(wave)
            self.frames += len(wave)

    def read(self, frames):
        out = np.zeros(frames, dtype='float32')
        used = 0
        with self.lock:
            while self.parts and used < frames:
                part = self.parts.popleft()
                count = min(len(part), frames - used)
                out[used:used + count] = part[:count]
                if count < len(part):
                    self.parts.appendleft(part[count:])
                self.frames -= count
                used += count
            if used < frames:
                self.underruns += 1
        return out


def resolve_device(sd, value, direction):
    devices = sd.query_devices()
    key = f'max_{direction}_channels'
    if value is None:
        index = sd.default.device[0 if direction == 'input' else 1]
    elif value.isdigit():
        index = int(value)
    else:
        matches = [i for i, d in enumerate(devices) if d['name'] == value and d[key] > 0]
        if len(matches) != 1:
            raise ValueError(f'{direction} device not found/unambiguous: {value}; use --devices')
        index = matches[0]
    if not 0 <= index < len(devices) or devices[index][key] < 1:
        raise ValueError(f'Invalid {direction} device; use --devices')
    return index, devices[index]


def offline(args):
    denoiser = BlockDenoiser(args.block, args.context, args.attenuation)
    pre = Preprocessor(denoiser.rate, settings_from_args(args))
    wave, rate = sf.read(args.test_file, dtype='float32', always_2d=True)
    wave = wave.mean(axis=1)
    if rate != denoiser.rate:
        wave = torchaudio.functional.resample(torch.from_numpy(wave), rate, denoiser.rate).numpy()
    frames = len(wave)
    # Flush the 40 ms guard and retain exact original duration after alignment.
    padded = np.pad(wave, (0, denoiser.block + denoiser.guard))
    padded = np.pad(padded, (0, (-len(padded)) % denoiser.block))
    output = np.concatenate([denoiser.process(pre.process(padded[i:i + denoiser.block]))
                             for i in range(0, len(padded), denoiser.block)])
    output = output[denoiser.guard:denoiser.guard + frames]
    if args.output_file:
        sf.write(args.output_file, output, denoiser.rate, subtype='FLOAT')
    times = np.asarray(denoiser.times)
    print(f'Finite output: {np.isfinite(output).all()}; duration {len(output)/denoiser.rate:.3f}s', flush=True)
    print(f'Compute median {np.median(times)*1000:.1f} ms, p90 {np.percentile(times,90)*1000:.1f} ms '
          f'per {args.block*1000:.0f} ms block', flush=True)


def live(args):
    import sounddevice as sd
    import soxr
    if args.devices:
        print(sd.query_devices())
        return
    inp, in_info = resolve_device(sd, args.input, 'input')
    out, out_info = resolve_device(sd, args.output, 'output')
    rate = int(in_info['default_samplerate'])
    sd.check_input_settings(device=inp, channels=1, dtype='float32', samplerate=rate)
    sd.check_output_settings(device=out, channels=1, dtype='float32', samplerate=rate)
    print(f"Input: {in_info['name']} | Output: {out_info['name']} | duplex {rate} Hz", flush=True)
    denoiser = BlockDenoiser(args.block, args.context, args.attenuation)
    pre = Preprocessor(denoiser.rate, settings_from_args(args))
    pre_commands = queue.Queue()
    limit = f'{args.attenuation:g} dB' if args.attenuation else 'unlimited'
    print(f'DeepFilterNet3 | attenuation limit {limit} | post-filter off | monitor gain {args.gain:g}', flush=True)
    print(pre.describe(), flush=True)
    print('gate OPEN [CLOSE] | gate off | limit DBFS | limit off | pre on/off | p toggle | settings',flush=True)
    to_model = soxr.ResampleStream(rate, denoiser.rate, 1, dtype='float32', quality='HQ') if rate != denoiser.rate else None
    to_device = soxr.ResampleStream(denoiser.rate, rate, 1, dtype='float32', quality='HQ') if rate != denoiser.rate else None
    playback = PlaybackBuffer(round((args.block + args.margin) * rate), round(1.5 * rate))
    incoming = queue.Queue(maxsize=50)  # bounded to 1 s of 20 ms capture chunks
    stopping = threading.Event()
    errors = queue.Queue()
    counters = {'input_drops': 0, 'audio_status': 0, 'input_dbfs': -120.0, 'output_dbfs': -120.0}

    def callback(indata, outdata, frames, timing, status):
        if status:
            counters['audio_status'] += 1
        counters['input_dbfs'] = float(20*np.log10(np.sqrt(np.mean(indata[:, 0]**2))+1e-9))
        try:
            incoming.put_nowait(indata[:, 0].copy())
        except queue.Full:
            counters['input_drops'] += 1
        outdata[:, 0] = np.clip(playback.read(frames) * args.gain, -0.98, 0.98)
        counters['output_dbfs'] = float(20*np.log10(np.sqrt(np.mean(outdata[:, 0]**2))+1e-9))

    def worker():
        pending = np.zeros(0, dtype='float32')
        try:
            while not stopping.is_set():
                while not pre_commands.empty():
                    try:
                        if not pre.control(pre_commands.get_nowait()):
                            print('Unknown preprocessing command',flush=True)
                    except ValueError as error:
                        print(f'Settings unchanged: {error}',flush=True)
                try:
                    x = incoming.get(timeout=0.1)
                except queue.Empty:
                    continue
                if to_model:
                    x = to_model.resample_chunk(x)
                pending = np.concatenate([pending, x])
                while len(pending) >= denoiser.block:
                    chunk, pending = pending[:denoiser.block], pending[denoiser.block:]
                    y = denoiser.process(pre.process(chunk))
                    playback.write(to_device.resample_chunk(y) if to_device else y)
        except Exception as error:
            errors.put(error)
            stopping.set()

    def controls():
        try:
            while not stopping.is_set():
                command = input().strip().lower()
                if command == 'q':
                    stopping.set()
                elif command in ('b', 'd'):
                    denoiser.bypass = command == 'b'
                    print('DENOISER BYPASS (preprocessing still applies)' if denoiser.bypass else 'DENOISE', flush=True)
                elif command:
                    pre_commands.put(command)
        except EOFError:
            pass

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    print(f'App buffering + guard: {(args.block+args.margin+0.04)*1000:.0f} ms '
          '(excludes resampling/device/Bluetooth delay)', flush=True)
    print('b + Enter: bypass | d + Enter: denoise | q + Enter / Ctrl+C: stop', flush=True)
    try:
        with ExitStack() as stack:
            stream = stack.enter_context(sd.Stream(device=(inp, out), samplerate=rate, channels=1,
                                                   dtype='float32', blocksize=round(rate*0.02),
                                                   callback=callback, latency='high'))
            threading.Thread(target=controls, daemon=True).start()
            print('LIVE READY — speak into the selected microphone.', flush=True)
            last_log = time.monotonic()
            began = time.monotonic()
            while not stopping.wait(0.1):
                if not stream.active:
                    raise RuntimeError('Audio stream stopped')
                if args.seconds and time.monotonic()-began >= args.seconds:
                    break
                if time.monotonic()-last_log >= args.meter_interval:
                    recent = np.asarray(denoiser.times[-25:])
                    compute = f'{np.median(recent)*1000:.0f}ms' if len(recent) else 'bypass'
                    print(f'Compute {compute} / {args.block*1000:.0f}ms | underruns {playback.underruns} '
                          f'| input drops {counters["input_drops"]} | playback drops {playback.drops} '
                          f'| audio status {counters["audio_status"]} '
                          f'| in {counters["input_dbfs"]:.0f} dBFS, out {counters["output_dbfs"]:.0f} dBFS '
                          f'| {pre.meter()}', flush=True)
                    last_log = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        stopping.set()
        thread.join(timeout=3)
    if not errors.empty():
        raise errors.get()
    print('Stopped; microphone/output streams closed.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', help='Exact input device name or ID; default is the macOS default')
    parser.add_argument('--output', help='Exact output device name or ID; default is the macOS default')
    parser.add_argument('--devices', action='store_true')
    parser.add_argument('--block', type=float, default=0.2)
    parser.add_argument('--context', type=float, default=0.8)
    parser.add_argument('--margin', type=float, default=0.15)
    parser.add_argument('--attenuation', type=float, default=12,
                        help='Noise attenuation limit in dB (default 12); 0 removes the limit')
    parser.add_argument('--gain', type=float, default=0.7)
    parser.add_argument('--seconds', type=float, default=0, help='Optional automatic stop after N seconds')
    parser.add_argument('--test-file', help='Offline block-processing test; does not open microphone/output')
    parser.add_argument('--output-file', help='WAV output for --test-file')
    parser.add_argument('--meter-interval',type=float,default=5)
    add_arguments(parser)
    args = parser.parse_args()
    if args.block <= 0.04 or args.context < 0.04 or args.margin < 0 or not 0 < args.gain <= 1:
        parser.error('block >0.04, context >=0.04, margin >=0, gain in (0,1] required')
    if not np.isfinite(args.meter_interval) or args.meter_interval <= 0:
        parser.error('--meter-interval must be finite and >0')
    try:
        settings_from_args(args)
    except ValueError as error:
        parser.error(str(error))
    if args.test_file:
        offline(args)
    else:
        live(args)


if __name__ == '__main__':
    main()
