"""Check Variant 2 streaming alignment and confirm Variant 1 files are untouched."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
from scipy.signal import correlate, correlation_lags

from dpdfnet_model import DPDFModel, ROOT, MODELS


def process_file(model, wave, bypass=False):
    padded = np.pad(wave,(0,model.delay+model.block))
    padded = np.pad(padded,(0,-len(padded)%model.block))
    return np.concatenate([model.process(padded[i:i+model.block],bypass=bypass)
                           for i in range(0,len(padded),model.block)])


def main():
    variant1 = ROOT.parent
    snapshot = json.loads((ROOT/'variant1_snapshot.json').read_text())
    changed = [name for name,digest in snapshot.items()
               if hashlib.sha256((variant1/name).read_bytes()).hexdigest() != digest]
    if changed:
        raise RuntimeError(f'Variant 1 files changed since experiment setup: {changed}')
    mixture = variant1.parent/'data/user_recordings/mixture/Mix.wav'
    wave, rate = sf.read(mixture,dtype='float32',always_2d=True)
    wave = soxr.resample(wave.mean(axis=1)[:rate*4],rate,16000).astype(np.float32)
    result = {'variant1_unchanged':True,'variant1_files_checked':len(snapshot),'models':{}}
    for name in MODELS:
        model = DPDFModel(name)
        raw = process_file(model,wave,bypass=True)
        assert np.array_equal(raw[model.delay:model.delay+len(wave)],wave)
        model.reset()
        output = process_file(model,wave)
        assert np.isfinite(output).all()
        corr = correlate(output,wave,method='fft')
        lags = correlation_lags(len(output),len(wave))
        keep = (lags>=0)&(lags<=1600)
        lag = int(lags[keep][np.argmax(np.abs(corr[keep]))])
        assert lag == model.delay,(name,lag,model.delay)
        report = model.statistics()
        report.update(measured_correlation_lag_samples=lag,matched_bypass_exact=True)
        result['models'][name] = report
        print(json.dumps(report),flush=True)

    # An identity spectrum isolates STFT/overlap-add reconstruction from the
    # neural model. This checks beginning/end coverage after priming.
    class IdentitySession:
        def run(self,names,values):
            return [values[model.enhancer._runtime.in_spec_name],
                    values[model.enhancer._runtime.in_state_name]]
    model.enhancer._runtime = replace(model.enhancer._runtime,session=IdentitySession())
    model.reset()
    output = process_file(model,wave)
    error = float(np.max(np.abs(output[model.block:model.block+len(wave)]-wave)))
    assert error < 1e-6,error
    result['stft_identity_max_error'] = error
    (ROOT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Variant 1 unchanged; streaming, bypass alignment and STFT checks passed.',flush=True)


if __name__ == '__main__':
    main()
