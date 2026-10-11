# Enrollment otomatis dari rekaman campuran — 9 Oktober 2026

## Tujuan dan hasil awal

Pengguna menerima kualitas **SepFormer MPS, konteks 1.500 ms, hop 200 ms**
pada rekaman lokal, lalu meminta alur baru: rekam campuran 30 detik,
buat profil suara dari campuran itu, pilih speaker, dan lanjutkan filtering.

[experiment_auto_enrollment.py](experiment_auto_enrollment.py) menyediakan
kalibrasi dari WAV, profil yang bisa disimpan/dimuat ulang, serta simulasi
input lanjutan mengikuti jam nyata. Pengujian menggunakan **10 detik pertama
Mix.wav** karena file yang tersedia hanya 23,125 detik. **13,125 detik sisanya**
menjadi input lanjutan. Bagian akhir tidak masuk model saat membuat profil.
Rekaman tidak diperpanjang, diulang, atau diberi padding menjadi 30 detik.

| Tahap / konfigurasi | Hasil |
| --- | ---: |
| Kalibrasi yang benar-benar diuji | 10 detik |
| Pemisahan + pembuatan profil otomatis | **1,906 detik** setelah audio tersedia |
| Model loading, di luar replay | 0,376 detik |
| Warm-up bentuk jendela startup, di luar replay | 2,693 detik |
| Otomatis: proses steady p50 / p95 | **184 / 187 ms** |
| Otomatis: delay playback virtual p95 | **403 ms** |
| Otomatis: blok steady melewati 200 ms | 1 / 57 |
| Enrollment lama, input lanjutan identik: proses p50 / p95 | 185 / 188 ms |
| Enrollment lama: delay playback virtual p95 | 405 ms |

Angka baru tidak menunjukkan percepatan terhadap run 23 detik sebelumnya:
panjang dan bagian input berbeda. Kualitas profil otomatis belum disetujui
melalui listening. Latency mic/AirPods dan kestabilan 30 menit belum diukur.
Pembuatan profil dari **30 detik sebenarnya belum diuji**; CLI default sudah
meminta 30 detik dan menolak input yang terlalu pendek.

## Cara kerjanya

1. Ambil hanya interval kalibrasi awal, downmix ke mono, resample ke 8 kHz.
2. Jalankan SepFormer sekali pada interval tersebut, dengan checkpoint asli.
3. Fit gain setiap jalur terhadap campuran seperti pipeline sebelumnya.
4. Resample jalur ke 16 kHz. Potong bagian berenergi rendah dan simpan
   cuplikan enrollment hasil pemisahan. Pemotongan memakai RMS 40 ms,
   ambang `max(0,0001; 0,1 × p95 RMS)`, dan padding 80 ms. **Ini bukan VAD**
   dan tidak membuktikan bahwa cuplikan berisi satu speaker saja.
5. Hitung embedding ECAPA sekali untuk setiap cuplikan; simpan bersama
   ID anonim `speaker_01`, `speaker_02`, `speaker_03`.
6. Dengarkan cuplikan, beri label jika diinginkan, lalu pilih satu atau
   beberapa ID. Pemisahan live tetap memproses konteks 1.500 ms + hop
   200 ms; ECAPA mencocokkan jalur ke profil yang sudah disimpan.

`WindowSeparator` kini menerima daftar ID profil, dengan default Nicker,
Mond, Sam tetap digunakan oleh runner lama. Diagnostik konteks membaca
urutan ID dari mapping run. Tidak ada perubahan weights, attention,
normalisasi, atau pemisahan pada pipeline MPS yang sudah diterima.

## Batas yang harus diperiksa

- Checkpoint [SepFormer Libri3Mix](https://huggingface.co/speechbrain/sepformer-libri3mix)
  menghasilkan **tiga jalur**, bukan jumlah orang yang terdeteksi otomatis.
  Versi ini hanya diuji pada contoh tiga orang. Satu/dua orang, lebih dari
  tiga orang, noise, dan perubahan identitas di dalam jalur belum divalidasi.
- [ECAPA](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) membuat
  embedding identitas dan perbandingan cosine. Enrollment hasil pemisahan
  dapat mengandung kebocoran/artifact. File tersebut **bukan clean reference**.
- Minimal dua detik audio berenergi dibutuhkan per kandidat, dan input
  senyap/embedding rusak ditolak. Ini hanya pemeriksaan teknis awal,
  bukan acceptance gate untuk speech, speaker purity, atau clipping input.
  Kemiripan antarprofil di atas 0,75 memicu warning kandidat duplikat;
  ambang ini belum dikalibrasi. Identitas tidak digabungkan diam-diam.
- Orang yang tidak berbicara saat kalibrasi tidak memiliki profil yang
  dapat diandalkan. Speaker baru memerlukan kalibrasi ulang. Sistem masih
  memaksakan pencocokan dan belum menolak target yang sedang tidak aktif.
- **30 detik adalah waktu pengambilan audio.** Pemisahan, embedding, dan
  warm-up menambah waktu sebelum filtering siap. Durasi tambahan untuk
  input 30 detik belum diukur; jangan mengasumsikan sama dengan 1,906 detik.
- Replay mulai setelah profil dan warm-up selesai. Belum ada capture mic
  berjalan terus selama fase pembuatan profil, buffer transisi, atau output
  fisik ke AirPods. Delay replay dihitung dari awal input lanjutan virtual.

## Pembanding identitas

Run `auto_profiles` menggunakan profil hasil campuran saja. Run
`manual_control` terpisah memakai enrollment lama, hanya untuk diagnosis.
Pencocokan setelah profil otomatis dibuat memberi dugaan:

| Profil anonim | Dugaan dari enrollment lama | Cosine pasangan | Window steady memilih jalur berbeda dari kontrol |
| --- | --- | ---: | ---: |
| Speaker 01 | Mond | 0,387 | 9 / 57 |
| Speaker 02 | Sam | 0,539 | 18 / 57 |
| Speaker 03 | Nicker | 0,422 | 17 / 57 |

Ini **bukan ground truth atau persentase akurasi**. Enrollment lama juga
bisa salah pada window singkat. Perbedaan cukup sering untuk memerlukan
listening sebelum memilih alur otomatis sebagai pengganti. Tidak ada
clean reference dari Mix.wav untuk menentukan SI-SDR atau jalur yang benar.

## File hasil dan langkah berikutnya

```text
data/user_recordings/auto_enrollment/2026-10-09_mix_first10s/
├── listen.html                 # enrollment, single/pair outputs, kontrol lama
├── experiment.json             # startup, waktu replay, mapping pembanding
├── verification.json
├── profiles/
│   ├── profiles.json
│   ├── embeddings.npy
│   ├── calibration_mix.wav
│   ├── speaker_01_enrollment.wav
│   ├── speaker_01_separated.wav
│   ├── speaker_01_separated_raw.wav
│   └── ...                     # Speaker 02 dan 03
├── auto_profiles/              # enam pilihan × raw/content/timeline
└── manual_control/             # enam pilihan yang sama, enrollment lama
```

**Berikutnya:** buka `listen.html`, tentukan identitas Speaker 01–03,
bandingkan output otomatis dengan kontrol untuk speaker tunggal dan pasangan.
Catat identitas bertukar, kebocoran suara lain, suara hilang, dan kejernihan.
Jika hasil diterima, ambil rekaman kalibrasi **30 detik sebenarnya** dan
rekaman lanjutan baru sebelum integrasi mic/AirPods.

Profil, embedding, dan audio personal berada di folder lokal yang diabaikan
Git. Tidak ada dataset, model, atau dependency tambahan yang diunduh.
Rekaman/enrollment lama tidak diubah. `profiles.json` menyimpan revision
checkpoint, urutan ID, span kalibrasi, gain, diagnostics dan hash file.

## Menjalankan

Dari folder `evaluation`, untuk mengulang eksperimen 10 detik:

```sh
../speakerbeam_ss_verification/.venv/bin/python experiment_auto_enrollment.py --calibration-seconds 10 --compare-manual-enrollments --output-dir data/user_recordings/auto_enrollment/run_baru
```

Tanpa `--compare-manual-enrollments`, kode tidak membaca enrollment lama.
Gunakan terminal yang mendapat akses GPU Metal. `--device cpu` tersedia
secara eksplisit, tetapi bukan klaim performa real-time pada CPU.

Simpan rekaman 30 detik di
`data/user_recordings/calibration/Calibration30.wav`. Buat profil saja:

```sh
../speakerbeam_ss_verification/.venv/bin/python experiment_auto_enrollment.py --input-wav data/user_recordings/calibration/Calibration30.wav --skip-replay --output-dir data/user_recordings/auto_enrollment/calibration30_baru
```

Dengarkan halaman hasil dan pilih ID. Untuk memakai profil itu pada rekaman
berikutnya, misalnya Speaker 01 dan 03:

```sh
../speakerbeam_ss_verification/.venv/bin/python experiment_auto_enrollment.py --profiles-dir data/user_recordings/auto_enrollment/calibration30_baru/profiles --live-wav data/user_recordings/mixture/Following.wav --selected-speakers speaker_01 speaker_03 --output-dir data/user_recordings/auto_enrollment/filter_baru
```

`--live-wav` adalah simulasi dari file lanjutan, belum mikrofon langsung.
Tanpa argumen pilihan speaker, runner mengekspor semua tiga pilihan tunggal
dan tiga pasangan untuk listening. Selalu gunakan folder output baru.
`display_name` di `profiles.json` bisa diberi nama orang; pertahankan
`speaker_id` dan urutan embedding agar profil tetap kompatibel.

## Verifikasi

[verify_auto_enrollment.py](verify_auto_enrollment.py) memeriksa span
kalibrasi, merekonstruksi enrollment dari jalur kalibrasi saja, integrity
embedding/audio, seluruh WAV output, jumlah frame, timeline/gap, dan tautan
player. Pengujian kedua memuat ulang profil yang sama pada input lanjutan
2,4 detik dan mengeluarkan **hanya pasangan Speaker 01 + 03**; hasilnya
dibandingkan secara numerik dengan pasangan pada run awal.
Pemeriksaan negatif mencakup embedding rusak, input senyap, dan default
30 detik pada file yang terlalu pendek. Hasil ada di `verification.json`.

```sh
../speakerbeam_ss_verification/.venv/bin/python verify_auto_enrollment.py --run-dir data/user_recordings/auto_enrollment/2026-10-09_mix_first10s --reuse-run-dir data/user_recordings/auto_enrollment/2026-10-09_reuse_two_speakers
```
