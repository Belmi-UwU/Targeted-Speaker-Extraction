"""Controlled R1/new-checkpoint streaming listening comparison on the user's Mix.wav.

No aligned clean sources are available: this produces listening controls, not SI-SDR scores.
Uses the saved discovery tracks and the same prepared enrollment for both models.
"""
import argparse
from datetime import datetime
import hashlib
import html
import json
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from live_demo_streaming import StreamingAdapter, base, enrollment
from diagnose_user_discovery import audio, describe, LISTEN_SCRIPT, write_clip
from streaming_wesep import deterministic
from run_wesep_experiment import extract
from wesep.cli.extractor import load_model_local


def array_hash(wave):
    return hashlib.sha256(np.asarray(wave, dtype='<f4').tobytes()).hexdigest()


def process(model, mix, enroll):
    adapter = StreamingAdapter(model, [enroll], 0, 0.064, 'cpu')
    started = time.perf_counter()
    pieces = []
    for offset in range(0, len(mix), adapter.chunk):
        pieces.extend(adapter.push(mix[offset:offset + adapter.chunk]))
    out = np.concatenate(pieces)
    if not len(out) or not np.isfinite(out).all():
        raise ValueError('Empty or non-finite streaming output')
    times = np.asarray(adapter.compute)
    return out, {'compute_seconds': time.perf_counter() - started,
                 'median_chunk_ms': float(np.median(times) * 1000),
                 'p90_chunk_ms': float(np.percentile(times, 90) * 1000),
                 'real_time_factor_median': float(np.median(times) / 0.064),
                 'output_frames': len(out), 'unemitted_tail_frames': len(mix) - len(out),
                 'output_rms': float(np.sqrt(np.mean(out.astype('float64') ** 2)))}


def render(folder, record):
    cards = []
    for pair in record['pairs']:
        name = html.escape(pair['person'])
        variant = 'profil discovery' if pair['enrollment_kind'] == 'discovered' else 'rekaman asli'
        parts = [f'<section><h2>{name} · enrollment {variant}</h2>',
                 '<p>A dan B memakai penguatan yang sama. Dengarkan kejernihan, suara bocor, dan perubahan volume.</p>',
                 '<div class="grid">']
        for key, label in [('r1', 'A · R1 sebelumnya'), ('new', 'B · hasil fine-tuning')]:
            run = pair['runs'][key]
            parts.append(f'<article><h3>{label}</h3>{audio(run["clip"]["common_listen"])}'
                         '<details><summary>Versi diperkeras sendiri untuk memeriksa suara pelan</summary>'
                         f'{audio(run["clip"]["listen"])}</details></article>')
        parts.append('</div><details><summary>Enrollment yang sama untuk A dan B</summary>'
                     f'{audio(pair["enrollment_clip"]["listen"])}</details></section>')
        cards.extend(parts)
    page = '''<!doctype html><html lang="id"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>R1 vs fine-tuning · Mix.wav</title>
<style>body{font:16px/1.5 system-ui;margin:0;background:#f5f6f8;color:#17212f}
main{max-width:1000px;margin:auto;padding:24px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}
section{margin:28px 0}article{background:white;padding:18px;border:1px solid #dde2e8;border-radius:12px}
audio{width:100%;display:block;margin:12px 0}button{padding:12px;margin:0 8px 12px 0;cursor:pointer}
details{margin:12px 0}summary{cursor:pointer}.note{background:#fff2cd;padding:16px;border-radius:12px}
</style><main><h1>R1 vs hasil fine-tuning</h1>
<p class="note">Mix.wav yang sama, streaming CPU 64 ms, enrollment identik untuk A dan B.
B adalah checkpoint terbaik menurut loss validasi (langkah ''' + str(record['selected_training_step']) + ''').
Mulai dari Nicker dengan enrollment discovery. Tidak ada skor kualitas pada Mix.wav karena suara bersih selaras tidak tersedia.</p>
<p>Gunakan <b>Putar alternatif</b>. Bandingkan potongan waktu yang sama: apakah kata Nicker lebih utuh,
volume lebih stabil, dan Sam/Mond lebih sedikit? Lanjutkan Sam dan Mond untuk memeriksa penurunan kualitas.</p>
<h2>Campuran asli</h2>''' + audio(record['mixture_clip']['listen']) + ''.join(cards)
    page += '<p><a href="REPORT.md">Ringkasan teknis</a> · <a href="results.json">Metadata pengujian</a></p></main>' + LISTEN_SCRIPT + '</html>'
    (folder / 'index.html').write_text(page)
    lines = ['# Perbandingan R1 vs fine-tuning pada Mix.wav', '',
             f"Dibuat: {record['created_at']}",
             f"Checkpoint baru dipilih pada langkah {record['selected_training_step']} menurut loss validasi.", '',
             'Streaming CPU, 2 threads, chunk 64 ms. Enrollment diproses sekali dan dipakai identik oleh kedua model.',
             'Penguatan audio utama sama untuk A/B dalam setiap pasangan; versi individual tersedia terpisah.',
             'Identitas profil discovery mengikuti hipotesis kemiripan ECAPA dari diagnosis sebelumnya.',
             'Tidak ada sumber bersih selaras: RMS keluaran dan perbedaan waveform bukan skor kualitas.', '',
             '| Target | Enrollment | Model | Median chunk (ms) | P90 (ms) | RTF median |',
             '| --- | --- | --- | ---: | ---: | ---: |']
    for pair in record['pairs']:
        for key, run in pair['runs'].items():
            lines.append(f"| {pair['person']} | {pair['enrollment_kind']} | {key} | {run['median_chunk_ms']:.1f} | "
                         f"{run['p90_chunk_ms']:.1f} | {run['real_time_factor_median']:.2f} |")
    lines += ['', '## Pemeriksaan kompatibilitas', '']
    for key, model in record['models'].items():
        lines.append(f"- {key}: strict weight load berhasil; stream/full relative peak difference "
                     f"{model['stream_full_relative_peak_error']:.3g} pada kontrol 3 detik Nicker discovery.")
    lines += ['', '## Catatan dengar', '',
              'Belum dinilai pengguna. Periksa kejernihan kata, volume, kebocoran, dan noise pada A/B.',
              'Checkpoint aktif web app tidak diganti oleh pengujian ini.']
    (folder / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('diagnostic', type=Path)
    parser.add_argument('--new-checkpoint', required=True, type=Path)
    parser.add_argument('--output-name', default='checkpoint_comparison')
    args = parser.parse_args()
    if Path(args.output_name).name != args.output_name or args.output_name in ('.', '..'):
        parser.error('output-name must be a folder name')
    diagnostic = args.diagnostic.resolve()
    previous = json.loads((diagnostic / 'diagnostics.json').read_text())
    folder = diagnostic / args.output_name
    folder.mkdir(exist_ok=False)
    new_dir = args.new_checkpoint.resolve()
    run_dir = new_dir.parent.parent
    config = json.loads((run_dir / 'run_config.json').read_text())
    initial = json.loads((run_dir / 'initial_validation.json').read_text())
    val_records = [json.loads(line) for line in (run_dir / 'log.jsonl').read_text().splitlines() if line]
    best = min([{'step': 0, **initial}] + [r for r in val_records if 'val_loss' in r], key=lambda r: r['val_loss'])
    mix_path = Path(previous['inputs']['Mix']['path'])
    if describe(mix_path)['sha256'] != previous['inputs']['Mix']['sha256']:
        raise ValueError('Mix.wav changed since discovery')
    mix = base.read_any(mix_path)
    torch.set_num_threads(2)
    record = {'created_at': datetime.now(ZoneInfo('Asia/Jakarta')).isoformat(),
              'scope': 'Same-recording listening comparison; no aligned sources or objective quality score',
              'mixture': describe(mix_path), 'chunk_seconds': 0.064, 'device': 'cpu', 'threads': 2,
              'enrollment_cap_seconds': 8, 'enrollment_rms_dbfs': -25,
              'selected_training_step': best['step'], 'initial_validation': initial,
              'selected_validation': best, 'final_validation': val_records[-1], 'models': {}, 'pairs': []}
    waveforms = {}
    # Discovery is unchanged; prepare each enrollment exactly once for both models.
    for name in ('Nicker', 'Sam', 'Mond'):
        profile = next(p for p in previous['profiles'] if p['match']['best'] == name)
        for kind in ('discovered', 'original'):
            path = diagnostic / profile['clip']['raw'] if kind == 'discovered' else Path(previous['inputs'][name]['path'])
            if kind == 'original' and describe(path)['sha256'] != previous['inputs'][name]['sha256']:
                raise ValueError(f'{name} enrollment changed')
            prepared = enrollment(base.read_any(path)).astype('float32')
            if len(prepared) < base.RATE:
                raise ValueError('Enrollment too short')
            pair = {'person': name, 'enrollment_kind': kind, 'discovery_profile': profile['profile'],
                    'source': describe(path), 'enrollment_frames': len(prepared),
                    'prepared_enrollment_sha256': array_hash(prepared),
                    'enrollment_clip': write_clip(folder, f'enrollment_{name.lower()}_{kind}', prepared), 'runs': {}}
            record['pairs'].append(pair)
            waveforms[(name, kind)] = {'enrollment': prepared}
    for key, path in [('r1', HERE / 'checkpoints/pretrained/retrained_finetune_research'), ('new', new_dir)]:
        print(f'Loading {key}: {path}', flush=True)
        extractor = load_model_local(str(path))
        weights = torch.load(path / 'avg_model.pt', map_location='cpu', weights_only=True)['models'][0]
        extractor.model.load_state_dict(weights, strict=True)
        del weights
        extractor.set_output_norm(False)
        extractor.set_device('cpu')
        deterministic(extractor.model)
        hashes = {n: describe(path / n)['sha256'] if n.endswith('.wav') else base.dp.sha256(path / n)
                  for n in ('avg_model.pt', 'config.yaml')}
        if key == 'r1' and hashes['avg_model.pt'] != config['initial_checkpoint_sha256']:
            raise ValueError('R1 checkpoint differs from training initialization')
        control_mix = mix[:3 * base.RATE]
        control_enroll = waveforms[('Nicker', 'discovered')]['enrollment']
        control_stream, _ = process(extractor.model, control_mix, control_enroll)
        control_full = extract(extractor, control_mix, control_enroll)
        n = min(len(control_stream), len(control_full)) - 512
        error = float(np.max(np.abs(control_stream[:n] - control_full[:n])) /
                      max(float(np.max(np.abs(control_full[:n]))), 1e-12))
        if error > 1e-3:
            raise ValueError(f'{key} streaming/full mismatch: {error}')
        record['models'][key] = {'directory': str(path), 'hashes': hashes, 'strict_load': True,
                                 'stream_full_relative_peak_error': error}
        print(f'  stream/full relative peak difference: {error:.3g}', flush=True)
        for pair in record['pairs']:
            name, kind = pair['person'], pair['enrollment_kind']
            print(f'Processing {key}: {name}/{kind}', flush=True)
            prepared = waveforms[(name, kind)]['enrollment']
            before = array_hash(prepared)
            out, stats = process(extractor.model, mix, prepared)
            if array_hash(prepared) != before:
                raise ValueError('Prepared enrollment mutated')
            stats['clip'] = write_clip(folder, f'{name.lower()}_{kind}_{key}', out)
            pair['runs'][key] = stats
            waveforms[(name, kind)][key] = out
            print(f"  median {stats['median_chunk_ms']:.1f}ms, p90 {stats['p90_chunk_ms']:.1f}ms", flush=True)
        del extractor
    for pair in record['pairs']:
        waves = waveforms[(pair['person'], pair['enrollment_kind'])]
        if len(waves['r1']) != len(waves['new']):
            raise ValueError('Different output lengths')
        gain = 0.9 / max(float(np.max(np.abs(waves[k]))) for k in ('r1', 'new'))
        pair['common_listening_gain'] = gain
        for key in ('r1', 'new'):
            relative = f"listen/{pair['person'].lower()}_{pair['enrollment_kind']}_{key}_common.wav"
            sf.write(folder / relative, waves[key] * gain, base.RATE, subtype='PCM_16')
            pair['runs'][key]['clip']['common_listen'] = relative
    record['mixture_clip'] = write_clip(folder, 'mix', mix)
    (folder / 'results.json').write_text(json.dumps(record, indent=2) + '\n')
    render(folder, record)
    print(f"Ready: {folder / 'index.html'}", flush=True)


if __name__ == '__main__':
    main()
