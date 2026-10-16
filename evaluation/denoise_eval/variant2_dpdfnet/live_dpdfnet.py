"""Variant 2: standalone DPDFNet live microphone denoising experiment.

1/2 + Enter selects DPDFNet2/DPDFNet4, b bypass, d denoise, q stop.
Live samples stay in memory. --test-file performs an offline streaming check.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import queue
import threading
import time
from zoneinfo import ZoneInfo

import numpy as np
import soundfile as sf
import soxr

from dpdfnet_model import DPDFModel
MODELS = {name:(lambda name=name:DPDFModel(name)) for name in ('dpdfnet2','dpdfnet4')}
from audio_preprocessing import Preprocessor, add_arguments, settings_from_args
# This variant uses its own preprocessing and audio modules.
from audio_io import PlaybackBuffer, resolve_device

ORDER = list(MODELS)


def statistics(model):
    return model.statistics()


def offline(args):
    wave, rate = sf.read(args.test_file, dtype='float32', always_2d=True)
    wave = wave.mean(axis=1)
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    results = []
    for name in ORDER if args.all else [args.model]:
        model = MODELS[name]()
        try:
            pre = Preprocessor(model.rate, settings_from_args(args))
            audio = soxr.resample(wave, rate, model.rate).astype(np.float32) if rate != model.rate else wave
            length = len(audio)
            padded = np.pad(audio, (0, model.delay + model.block))
            padded = np.pad(padded, (0, -len(padded)%model.block))
            prepared, parts = [], []
            for i in range(0,len(padded),model.block):
                chunk = pre.process(padded[i:i+model.block])
                prepared.append(chunk)
                parts.append(model.process(chunk))
            output = np.concatenate(parts)
            output = output[model.delay:model.delay+length]
            model_input = np.concatenate(prepared)[:length]
            assert len(output) == length and np.isfinite(output).all()
            sf.write(destination / f'{name}.wav', output, model.rate, subtype='FLOAT')
            sf.write(destination / f'{name}_input_pre.wav', model_input, model.rate, subtype='FLOAT')
            report = statistics(model)
            report.update(duration_seconds=len(output)/model.rate, finite=True,
                          input_rms=float(np.sqrt(np.mean(audio**2))),
                          output_rms=float(np.sqrt(np.mean(output**2))),
                          peak=float(np.max(np.abs(output))), preprocessing=pre.report(),
                          model_input_peak=float(np.max(np.abs(model_input))))
            results.append(report)
            print(json.dumps(report), flush=True)
        finally:
            model.close()
    (destination/'offline_results.json').write_text(json.dumps(results,indent=2)+'\n')


def live(args):
    import sounddevice as sd
    if args.devices:
        print(sd.query_devices())
        return
    inp, info = resolve_device(sd, args.input, 'input')
    out, out_info = resolve_device(sd, args.output, 'output')
    rate = int(info['default_samplerate'])
    sd.check_input_settings(device=inp, channels=1, dtype='float32', samplerate=rate)
    sd.check_output_settings(device=out, channels=1, dtype='float32', samplerate=rate)
    # Load and warm up all models before opening hardware. Switching resets
    # recurrent state and resamplers so different model histories do not mix.
    models = {}
    try:
        for name, constructor in MODELS.items():
            models[name] = model = constructor()
            for _ in range(4):
                model.process(np.zeros(model.block, dtype=np.float32))
            model.reset()
        run_monitor(args, sd, inp, out, info, out_info, rate, models)
    finally:
        for model in models.values():
            model.close()


def run_monitor(args, sd, inp, out, info, out_info, rate, models):
    started_at = datetime.now(ZoneInfo('Asia/Jakarta')).isoformat()
    playback = PlaybackBuffer(round((args.margin+0.02)*rate), round(1.5*rate))
    incoming = queue.Queue(maxsize=50)
    commands = queue.Queue()
    errors = queue.Queue()
    stopping = threading.Event()
    state = {'name':args.model, 'bypass':False, 'input_drops':0, 'audio_status':0,
             'clip_frames':0, 'input_dbfs':-120., 'output_dbfs':-120.,
             'input_peak_dbfs':-120., 'input_near_fullscale_samples':0}
    reports = []
    pre = Preprocessor(models[args.model].rate, settings_from_args(args))

    def callback(indata, outdata, frames, timing, status):
        if status:
            state['audio_status'] += 1
        state['input_dbfs'] = float(20*np.log10(np.sqrt(np.mean(indata[:,0]**2))+1e-9))
        state['input_peak_dbfs'] = float(20*np.log10(np.max(np.abs(indata[:,0]))+1e-9))
        state['input_near_fullscale_samples'] += int(np.count_nonzero(np.abs(indata[:,0])>=.999))
        try:
            incoming.put_nowait(indata[:,0].copy())
        except queue.Full:
            state['input_drops'] += 1
        y = playback.read(frames)*args.gain
        state['clip_frames'] += int(np.count_nonzero(np.abs(y)>.98))
        outdata[:,0] = np.clip(y,-.98,.98)
        state['output_dbfs'] = float(20*np.log10(np.sqrt(np.mean(outdata[:,0]**2))+1e-9))

    def worker():
        name = None
        try:
            while not stopping.is_set():
                requested = state['name']
                while not commands.empty():
                    command = commands.get_nowait()
                    if command in MODELS:
                        requested = command
                    else:
                        try:
                            if not pre.control(command):
                                print('Unknown command; use 1/2, b/d, gate, limit, pre, settings, q',flush=True)
                        except ValueError as error:
                            print(f'Settings unchanged: {error}',flush=True)
                if requested != name:
                    if name is not None and models[name].times:
                        reports.append(statistics(models[name]))
                    name = requested
                    model = models[name]
                    model.reset()
                    pre.rate = model.rate
                    pre.reset()
                    pending = np.zeros(0, dtype=np.float32)
                    to_model = soxr.ResampleStream(rate,model.rate,1,dtype='float32',quality='HQ') if rate != model.rate else None
                    to_device = soxr.ResampleStream(model.rate,rate,1,dtype='float32',quality='HQ') if rate != model.rate else None
                    # Intentional silence while a new model's resamplers prime.
                    with playback.lock:
                        playback.parts.clear()
                        playback.parts.append(np.zeros(round((args.margin+0.02)*rate),dtype=np.float32))
                        playback.frames = len(playback.parts[0])
                    while not incoming.empty():
                        incoming.get_nowait()
                    state['name'] = name
                    print(f'ACTIVE: {model.name} | {model.rate} Hz | hop {1000*model.block/model.rate:g} ms '
                          f'| sample delay {1000*model.delay/model.rate:g} ms',flush=True)
                try:
                    x = incoming.get(timeout=.1)
                except queue.Empty:
                    continue
                x = to_model.resample_chunk(x) if to_model else x
                pending = np.concatenate((pending,x))
                while len(pending) >= model.block:
                    chunk, pending = pending[:model.block],pending[model.block:]
                    y = model.process(pre.process(chunk),bypass=state['bypass'])
                    playback.write(to_device.resample_chunk(y) if to_device else y)
        except Exception as error:
            errors.put(error)
            stopping.set()

    def controls():
        try:
            while not stopping.is_set():
                command = input().strip().lower()
                if command in ('1','2'):
                    commands.put(ORDER[int(command)-1])
                elif command in ('b','d'):
                    state['bypass'] = command == 'b'
                    print('DENOISER BYPASS (preprocessing still applies)' if state['bypass'] else 'DENOISE',flush=True)
                elif command == 'q':
                    stopping.set()
                elif command:
                    commands.put(command)
        except EOFError:
            pass

    print(f"Input: {info['name']} | Output: {out_info['name']} | duplex {rate} Hz | gain {args.gain:g}",flush=True)
    print('VARIANT 2 | official DPDFNet ONNX | full enhancement, no dry-signal blend',flush=True)
    print(f'Playback prefill: {(args.margin+.02)*1000:g} ms; sample delay listed separately. '
          'Resampler/device/Bluetooth latency is additional and unmeasured.',flush=True)
    print('1 DPDFNet2 | 2 DPDFNet4 | b bypass | d denoise | q stop (then Enter)',flush=True)
    print(pre.describe(),flush=True)
    print('gate OPEN [CLOSE] | gate off | limit DBFS | limit off | pre on/off | p toggle | settings',flush=True)
    thread = threading.Thread(target=worker,daemon=True)
    thread.start()
    try:
        with sd.Stream(device=(inp,out),samplerate=rate,channels=1,dtype='float32',
                       blocksize=round(rate*.02),latency='high',callback=callback) as stream:
            state['portaudio_input_latency_ms'] = stream.latency[0]*1000
            state['portaudio_output_latency_ms'] = stream.latency[1]*1000
            threading.Thread(target=controls,daemon=True).start()
            print('LIVE READY — speak into the Bluetooth microphone.',flush=True)
            began = last_log = time.monotonic()
            suite_index = 0
            while not stopping.wait(.1):
                if not stream.active:
                    raise RuntimeError('Audio stream stopped')
                elapsed = time.monotonic()-began
                if args.suite_seconds and elapsed >= (suite_index+1)*args.suite_seconds:
                    suite_index += 1
                    if suite_index == len(ORDER):
                        break
                    commands.put(ORDER[suite_index])
                if args.seconds and elapsed >= args.seconds:
                    break
                if time.monotonic()-last_log >= args.meter_interval:
                    model = models[state['name']]
                    recent = model.times[-500:]
                    compute = np.median(recent)*1000 if recent else 0
                    print(f'{state["name"]} {"BYPASS" if state["bypass"] else "DENOISE"} '
                          f'compute {compute:.2f}ms / {1000*model.block/model.rate:g}ms '
                          f'| underruns {playback.underruns} | input drops {state["input_drops"]} '
                          f'| playback drops {playback.drops} | audio status {state["audio_status"]} '
                          f'| clipped {state["clip_frames"]} | in {state["input_dbfs"]:.0f}, out {state["output_dbfs"]:.0f} dBFS '
                          f'| input peak {state["input_peak_dbfs"]:.0f} dBFS | input near-fullscale {state["input_near_fullscale_samples"]} '
                          f'| {pre.meter()}',flush=True)
                    last_log = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        stopping.set()
        thread.join(timeout=3)
    model = models[state['name']]
    if model.times:
        reports.append(statistics(model))
    if args.log_file:
        Path(args.log_file).write_text(json.dumps({'variant':2,'models':reports,'audio':state,
            'underruns':playback.underruns,'playback_drops':playback.drops,
            'device_rate':rate,'gain':args.gain,'preprocessing':pre.report(),
            'input_device':info['name'],'output_device':out_info['name'],
            'started_at':started_at},indent=2)+'\n')
        print(f'Session log: {args.log_file}',flush=True)
    if not errors.empty():
        raise errors.get()
    print('Stopped; microphone/output streams closed.',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=ORDER,default='dpdfnet2')
    parser.add_argument('--input',default='BSD-APM260008 - Find My')
    parser.add_argument('--output',default='BSD-APM260008 - Find My')
    parser.add_argument('--gain',type=float,default=.35)
    parser.add_argument('--margin',type=float,default=.15)
    parser.add_argument('--seconds',type=float,default=0)
    parser.add_argument('--suite-seconds',type=float,default=0,help='Run each DPDFNet model for N seconds, then stop')
    parser.add_argument('--devices',action='store_true')
    parser.add_argument('--test-file')
    parser.add_argument('--all',action='store_true',help='Offline: test DPDFNet2 and DPDFNet4')
    parser.add_argument('--output-dir',default=str(Path(__file__).resolve().parent/'outputs'))
    parser.add_argument('--log-file')
    parser.add_argument('--meter-interval',type=float,default=5,help='Level log interval in seconds')
    add_arguments(parser)
    args = parser.parse_args()
    if (not all(np.isfinite(v) for v in (args.gain,args.margin,args.seconds,args.suite_seconds)) or
        not 0 < args.gain <= 1 or args.margin < 0 or args.seconds < 0 or args.suite_seconds < 0):
        parser.error('gain in (0,1], nonnegative margin and durations required')
    if args.suite_seconds:
        args.model = ORDER[0]
    if not np.isfinite(args.meter_interval) or args.meter_interval <= 0:
        parser.error('--meter-interval must be finite and >0')
    try:
        settings_from_args(args)
    except ValueError as error:
        parser.error(str(error))
    if not args.test_file and not args.devices:
        if not args.log_file:
            stamp = datetime.now(ZoneInfo('Asia/Jakarta')).strftime('%Y%m%d_%H%M%S_%f')
            args.log_file = str(Path(__file__).resolve().parent/'outputs'/f'live_{stamp}_WIB.json')
        Path(args.log_file).parent.mkdir(parents=True,exist_ok=True)
    offline(args) if args.test_file else live(args)


if __name__ == '__main__':
    main()
