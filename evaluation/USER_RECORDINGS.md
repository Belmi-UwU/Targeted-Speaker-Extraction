# Where to place your recordings

Put the **original files** in `data/user_recordings/` using this layout:

```text
data/user_recordings/
├── mixture/
│   └── three_voice_mix.wav
├── enrollments/
│   ├── nicker/
│   │   ├── 01.wav
│   │   ├── 02.wav
│   │   └── 03.wav
│   ├── mond/
│   │   ├── 01.wav
│   │   ├── 02.wav
│   │   └── 03.wav
│   └── sam/
│       ├── 01.wav
│       ├── 02.wav
│       └── 03.wav
└── clean_references/          # optional; see below
```

The three separate recordings for each person are **enrollment examples**: they tell the speaker matcher what Nicker, Mond, or Sam sounds like. They do not need to say the same words as the mixture. If a file is `.m4a`, `.flac`, or another format, keep its real extension; do not rename it to `.wav` without converting it. Keep the original sample rate and channels for now; the evaluation pipeline will make working copies at the rates needed by SepFormer and the speaker matcher.

Use `clean_references/` only if you have an isolated, time-aligned track of one person's voice **from the exact same mixed recording**. Separate example recordings are not a clean reference for the mixture. We can still run separation and listen without these reference tracks; objective SI-SDR scoring requires them.

The LibriMix dataset scripts do **not** automatically pick up this folder. Use the dedicated [run_user_recordings.py](run_user_recordings.py) script for the pinned SepFormer + ECAPA offline pipeline after checking the input files. The local audio folder is ignored by Git so personal recordings are not added to source control by accident.

## First local run

The uploaded `Mix.wav`, `Nicker.wav`, `Mond.wav`, and `Sam.wav` were processed with [run_user_recordings.py](run_user_recordings.py). The original files remain untouched. The script downmixes stereo 48 kHz input to mono, feeds 8 kHz audio to SepFormer, uses 16 kHz ECAPA enrollment matching, and exports 16 kHz PCM16 listening WAVs in `data/user_recordings/results/`:

- `original_mix_mono_16k.wav` — downmixed original mixture.
- `only_nicker.wav`, `only_mond.wav`, `only_sam.wav` — one selected speaker each.
- `nicker_and_mond.wav`, `nicker_and_sam.wav`, `mond_and_sam.wav` — two selected speakers each.

The run's `results/run.json` records input hashes, model revisions, speaker-to-stream matches, gain fitting, and output hashes. There are no clean references from this mixed recording, so objective source-separation scores are unavailable. The model's selected-speaker outputs require listening checks, especially for unwanted-speaker leakage. This is offline processing, not a live-app latency test.

## Simulasi waktu nyata

[simulate_realtime.py](simulate_realtime.py) melepas `Mix.wav` dalam frame 20 ms
mengikuti waktu nyata dan memproses jendela 500 ms serta 1 detik. Hasil
pengujian pertama berada di
`data/user_recordings/realtime_simulation/2026-10-07_cpu_windows/`.
Dengarkan `*_timeline.wav` untuk jeda dan celah playback virtual, serta
`*_content.wav` untuk kualitas hasil tanpa jeda penjadwalan. Lihat
[laporan simulasi](REALTIME_SIMULATION.md) untuk hasil waktu dan cara mengulang.

## Percobaan konteks bergulir

[experiment_rolling_context.py](experiment_rolling_context.py) mempertahankan
konteks 2.500, 1.000, 500, atau 250 ms, dengan audio baru tetap 500 ms.
Hanya audio baru yang dikeluarkan, sehingga konteks tidak diputar ulang.
Hasil berada di `data/user_recordings/rolling_context/2026-10-07_context_sweep/`.
Buka `listen.html` untuk membandingkan enam pilihan speaker dengan hasil
rekaman penuh dan jendela 500 ms sebelumnya. Mulai dari konteks 250 ms,
kemudian naikkan konteks jika kualitasnya belum memadai.

Konteks 250 ms memiliki proses median 480 ms, tetapi 8 dari 45 putaran
dengan riwayat penuh masih melampaui 500 ms. Konteks yang lebih panjang
menumpuk antrean pada CPU ini. Kualitas dengar dan konteks terpendek yang
bagus **belum ditetapkan**. Lihat [laporan konteks](ROLLING_CONTEXT_EXPERIMENT.md).

Konteks **1.500 ms + audio baru 500 ms** juga sudah diuji dalam replay
tambahan. Hasil dan halaman pembanding terbaru berada di
`data/user_recordings/rolling_context/2026-10-07_context_1500ms/`.
Halaman `listen.html` di sana memuat kelima konteks. Waktu proses median
846 ms dan p95 939 ms; semua 43 putaran dengan konteks penuh melewati
anggaran 500 ms. Hasil suara masih perlu didengarkan.

## SepFormer GPU/MPS, hop 100 dan 200 ms

Pengujian 9 Oktober 2026 memakai konteks 1.500 ms yang sama, dengan
SepFormer pada GPU Metal dan ECAPA hybrid CPU/MPS. Hasil berada di
`data/user_recordings/mps_sepformer/2026-10-09_context_1500ms_hops_100_200/`.
Halaman `listen.html` di sana membandingkan hop 100 dan 200 ms dengan
hasil CPU sebelumnya dan hasil rekaman penuh.

**Hop 200 ms mencapai delay playback virtual p95 sekitar 424 ms** pada
rekaman ini, dengan waktu proses p95 sekitar 197 ms; 3 dari 107 blok
konteks penuh melewati anggaran 200 ms. Hop 100 ms menumpuk antrean.
Pengguna menerima kualitas konfigurasi konteks 1.500 ms / hop 200 ms pada
rekaman ini. Delay hardware dan operasi panjang belum diukur.
Lihat [laporan MPS](MPS_SEPFORMER_EXPERIMENT.md).

## Enrollment otomatis dari campuran

[experiment_auto_enrollment.py](experiment_auto_enrollment.py) membuat
profil anonim dari interval awal, lalu memakai profil itu pada audio
lanjutan. Default kalibrasi 30 detik; uji awal memakai 10 detik pertama
Mix.wav karena file yang ada hanya 23,125 detik. Hasil berada di
`data/user_recordings/auto_enrollment/2026-10-09_mix_first10s/listen.html`.

Pembuatan profil memerlukan tambahan 1,906 detik pada contoh 10 detik ini.
Replay sisa audio mencapai delay playback virtual p95 403 ms. Kualitas
profil otomatis masih perlu kamu dengarkan; ada perbedaan pencocokan
dengan enrollment lama. Sistem saat ini menghasilkan tiga kandidat,
belum menghitung jumlah speaker secara otomatis.

Untuk percobaan berikutnya, letakkan rekaman kalibrasi sebenarnya di
`data/user_recordings/calibration/Calibration30.wav`, lalu rekaman lanjutan
di `data/user_recordings/mixture/Following.wav`. Semua orang yang ingin
dipilih perlu berbicara selama kalibrasi. Lihat [laporan enrollment otomatis](AUTO_ENROLLMENT_EXPERIMENT.md)
untuk membuat profil, mendengarkan, memilih satu/beberapa ID, dan memakai
ulang profil tersebut. File hasil pemisahan bukan clean reference.
