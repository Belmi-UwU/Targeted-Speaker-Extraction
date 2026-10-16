"""Standalone Variant 2 device selection and bounded playback buffer."""
from collections import deque
import threading

import numpy as np


def resolve_device(sd, value, direction):
    devices = sd.query_devices()
    key = f'max_{direction}_channels'
    if value is None:
        index = sd.default.device[0 if direction == 'input' else 1]
    elif value.isdigit():
        index = int(value)
    else:
        matches = [i for i,d in enumerate(devices) if d['name'] == value and d[key]>0]
        if len(matches) != 1:
            raise ValueError(f'{direction} device unavailable: {value}. Use --devices; reconnect/select the headset in macOS Sound.')
        index = matches[0]
    if not 0 <= index < len(devices) or devices[index][key] < 1:
        raise ValueError(f'Invalid {direction} device; use --devices')
    return index,devices[index]


class PlaybackBuffer:
    def __init__(self, prefill, maximum):
        self.lock = threading.Lock()
        self.maximum = maximum
        self.underruns = self.drops = 0
        self.refill(prefill)

    def refill(self, frames):
        with self.lock:
            self.parts = deque([np.zeros(frames,dtype=np.float32)])
            self.frames = frames

    def write(self, wave):
        if not len(wave):
            return
        with self.lock:
            if self.frames+len(wave)>self.maximum:
                self.parts.clear()
                self.frames = 0
                self.drops += 1
            self.parts.append(wave)
            self.frames += len(wave)

    def read(self, frames):
        out = np.zeros(frames,dtype=np.float32)
        used = 0
        with self.lock:
            while self.parts and used<frames:
                part = self.parts.popleft()
                count = min(len(part),frames-used)
                out[used:used+count] = part[:count]
                if count<len(part):
                    self.parts.appendleft(part[count:])
                self.frames -= count
                used += count
            if used<frames:
                self.underruns += 1
        return out
