"""Optional, causal RMS gate and sample-peak limiter before model inference.

No lookahead, added buffering, sample deletion, or inference skipping. All
controls are applied by the audio worker between blocks, never by the callback.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np


def dbfs(amplitude):
    return 20*math.log10(max(float(amplitude), 1e-9))


def threshold(value):
    value = float(value)
    if not math.isfinite(value) or not -120 <= value <= 0:
        raise ValueError('Threshold must be finite and between -120 and 0 dBFS')
    return value


@dataclass
class Settings:
    gate_open: float | None = None
    gate_close: float | None = None
    gate_attack_ms: float = 5
    gate_hold_ms: float = 150
    gate_release_ms: float = 100
    limiter_dbfs: float | None = None
    limiter_release_ms: float = 80

    def validate(self):
        if self.gate_open is None and self.gate_close is not None:
            raise ValueError('--gate-close requires --gate-open')
        if self.gate_open is not None:
            self.gate_open = threshold(self.gate_open)
            self.gate_close = threshold(self.gate_close if self.gate_close is not None
                                        else max(-120, self.gate_open-5))
            if self.gate_close > self.gate_open:
                raise ValueError('Gate close threshold must be <= open threshold')
        if self.limiter_dbfs is not None:
            self.limiter_dbfs = threshold(self.limiter_dbfs)
        for key in ('gate_attack_ms','gate_hold_ms','gate_release_ms','limiter_release_ms'):
            value = getattr(self,key)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{key} must be finite and nonnegative')


def add_arguments(parser):
    parser.add_argument('--gate-open',type=float,help='RMS opening threshold in dBFS; omitted = gate off')
    parser.add_argument('--gate-close',type=float,help='RMS closing threshold; default open minus 5 dB')
    parser.add_argument('--gate-attack-ms',type=float,default=5)
    parser.add_argument('--gate-hold-ms',type=float,default=150)
    parser.add_argument('--gate-release-ms',type=float,default=100)
    parser.add_argument('--limiter-dbfs',type=float,help='Sample-peak ceiling in dBFS; omitted = limiter off')
    parser.add_argument('--limiter-release-ms',type=float,default=80)


def settings_from_args(args):
    settings = Settings(**{key:getattr(args,key) for key in Settings.__dataclass_fields__})
    settings.validate()
    return settings


class Preprocessor:
    def __init__(self, rate, settings):
        self.rate = rate
        self.settings = Settings(**asdict(settings))
        self.settings.validate()
        self.enabled = True
        self.events = []
        self.reset()

    def coefficient(self, milliseconds):
        return math.exp(-1/(self.rate*milliseconds/1000)) if milliseconds else 0.

    def reset(self):
        self.power = 0.
        self.gate_opened = False
        self.gate_gain = 0.
        self.hold_left = 0
        self.limiter_gain = 1.
        self.input_rms_dbfs = self.output_rms_dbfs = self.output_peak_dbfs = -180.

    def describe(self):
        s = self.settings
        gate = f'{s.gate_open:g}/{s.gate_close:g} dBFS (open/close)' if s.gate_open is not None else 'off'
        limiter = f'{s.limiter_dbfs:g} dBFS peak' if s.limiter_dbfs is not None else 'off'
        return f'PRE {"on" if self.enabled else "bypassed"} | gate {gate} | limiter {limiter}'

    def control(self, command):
        words = command.split()
        if not words or words[0] not in ('gate','limit','pre','p','settings'):
            return False
        if words == ['settings']:
            print(self.describe(),flush=True)
            return True
        candidate = Settings(**asdict(self.settings))
        enabled = self.enabled
        if words == ['p']:
            enabled = not enabled
        elif len(words) == 2 and words[0] == 'pre' and words[1] in ('on','off'):
            enabled = words[1] == 'on'
        elif words == ['gate','off']:
            candidate.gate_open = candidate.gate_close = None
        elif words[0] == 'gate' and len(words) in (2,3):
            candidate.gate_open = threshold(words[1])
            candidate.gate_close = threshold(words[2]) if len(words) == 3 else None
            enabled = True
        elif words == ['limit','off']:
            candidate.limiter_dbfs = None
        elif words[0] == 'limit' and len(words) == 2:
            candidate.limiter_dbfs = threshold(words[1])
            enabled = True
        else:
            raise ValueError('Use: gate OPEN [CLOSE] | gate off | limit DBFS | limit off | pre on/off | p | settings')
        candidate.validate()  # reject invalid edits without changing active settings
        self.settings, self.enabled = candidate, enabled
        self.reset()
        self.events.append({'command':command,'settings':asdict(candidate),'enabled':enabled})
        print(self.describe(),flush=True)
        return True

    def process(self, wave):
        x = np.asarray(wave,dtype=np.float32)
        if x.ndim != 1 or not np.isfinite(x).all():
            raise ValueError('Preprocessor requires finite mono float audio')
        if not len(x):
            return x.copy()
        self.input_rms_dbfs = dbfs(np.sqrt(np.mean(x.astype(np.float64)**2)))
        s = self.settings
        gated = self.enabled and s.gate_open is not None
        limited = self.enabled and s.limiter_dbfs is not None
        if not gated and not limited:
            y = x.copy()  # default route is bit-for-bit unchanged
        else:
            y = np.empty_like(x)
            detector = self.coefficient(10)  # 10 ms exponential RMS time constant
            attack, release = self.coefficient(s.gate_attack_ms), self.coefficient(s.gate_release_ms)
            limit_release = self.coefficient(s.limiter_release_ms)
            open_power = 10**(s.gate_open/10) if gated else 0
            close_power = 10**(s.gate_close/10) if gated else 0
            hold = round(s.gate_hold_ms*self.rate/1000)
            # Tiny margin accounts for float32 rounding of the requested ceiling.
            ceiling = 10**(s.limiter_dbfs/20)*(1-2*np.finfo(np.float32).eps) if limited else 1
            for i, sample in enumerate(x):
                value = float(sample)
                if gated:
                    self.power = detector*self.power+(1-detector)*value*value
                    if self.power > open_power:
                        self.gate_opened = True
                        self.hold_left = hold
                    elif self.gate_opened:
                        if self.power >= close_power:
                            self.hold_left = hold
                        elif self.hold_left > 0:
                            self.hold_left -= 1
                        else:
                            self.gate_opened = False
                    target = float(self.gate_opened)
                    coefficient = attack if target > self.gate_gain else release
                    self.gate_gain = coefficient*self.gate_gain+(1-coefficient)*target
                    value *= self.gate_gain
                if limited:
                    required = min(1.,ceiling/max(abs(value),1e-20))
                    self.limiter_gain = (required if required < self.limiter_gain else
                                         limit_release*self.limiter_gain+(1-limit_release)*required)
                    value *= self.limiter_gain
                y[i] = value
        self.output_rms_dbfs = dbfs(np.sqrt(np.mean(y.astype(np.float64)**2)))
        self.output_peak_dbfs = dbfs(np.max(np.abs(y)))
        return y

    def meter(self):
        gate = ('open' if self.gate_opened else 'closed') if self.enabled and self.settings.gate_open is not None else 'off'
        return (f'pre RMS {self.output_rms_dbfs:.0f}, peak {self.output_peak_dbfs:.0f} dBFS '
                f'| gate {gate} | limiter gain {dbfs(self.limiter_gain):.1f} dB')

    def report(self):
        return {'enabled':self.enabled,'settings':asdict(self.settings),'changes':self.events}
