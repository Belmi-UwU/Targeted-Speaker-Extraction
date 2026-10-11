# SepFormer di MPS, hop 100 dan 200 ms — 9 Oktober 2026

Pengguna memilih percobaan akselerasi SepFormer yang sekarang sebelum
beralih ke checkpoint TSE streaming lain. Tujuan percobaan ini adalah
mengeksplorasi **delay di bawah 500 ms**, memakai Mix.wav yang sama dan
konteks masa lalu 1.500 ms. Target sementara PRD untuk rute kabel sebesar
150 ms belum diganti oleh target eksplorasi ini.

## Kesimpulan

**Hop 200 ms mencapai delay playback virtual p95 424 ms** pada satu
rekaman 23,125 detik di Mac ini. Hop 100 ms tidak mampu mengikuti input;
antrean terus bertambah. Ini bukti awal untuk konfigurasi 200 ms di
komputer ini, belum hasil latency mikrofon sampai earphone fisik atau
bukti stabil selama 30 menit.

Pada tindak lanjut, pengguna menyatakan kualitas **konteks 1.500 ms / hop
200 ms** sudah bagus pada contoh ini. Percobaan berikutnya adalah
[kalibrasi campuran untuk enrollment otomatis](AUTO_ENROLLMENT_EXPERIMENT.md).
Penerimaan subjektif itu belum memvalidasi profil otomatis, rute fisik,
atau operasi panjang.

## Perangkat dan metode

- Mac Apple M5, GPU 10 core, Metal didukung; Python 3.12.10 dan PyTorch
  2.6.0. PyTorch dibatasi ke empat thread CPU.
- Sandbox awal melaporkan MPS tidak tersedia. Pemeriksaan dengan akses
  GPU di luar sandbox memastikan MPS tersedia dan tensor benar-benar bisa
  dijalankan pada GPU. Kedua replay juga dijalankan dengan akses tersebut.
- SepFormer memakai checkpoint Libri3Mix asli, revision
  `1f2d824e2333aec9e60d0030fb144885c1afc426`, di device `mps`.
  Tidak ada perubahan attention, normalisasi, weights, atau fine-tuning.
- ECAPA memakai checkpoint yang sama, revision
  `0f99f2d0ebe89ac095bcc5903c4dd8f72b367286`. Enrollment dihitung sekali
  pada CPU sebelum sesi, lalu dipakai untuk pencocokan.
- Probe mengukur SepFormer MPS + matcher CPU dan SepFormer MPS + matcher
  hybrid. Matcher hybrid menjalankan fitur/FFT dan normalisasi pada CPU,
  lalu lapisan embedding ECAPA pada MPS dengan weights identik. Kedua
  varian lolos pemeriksaan numerik; hybrid lebih cepat pada probe dan
  dipilih untuk kedua replay. Tidak memakai embedding model berbeda.
- `PYTORCH_ENABLE_MPS_FALLBACK=0`: operator MPS yang tidak didukung tidak
  boleh diam-diam beralih ke CPU. Resampling, gain fitting, dan assignment
  tetap merupakan tahap CPU yang disengaja.
- GPU disinkronkan sebelum dan sesudah pengukuran. Transfer hasil kembali
  ke CPU, resampling, matching, dan ekstraksi bagian baru masuk waktu
  proses. Angka bukan waktu mengirim pekerjaan ke GPU saja.
- Input stereo 48 kHz dilepas setiap 20 ms mengikuti jam nyata. Worker
  memakai 1.500 ms audio masa lalu ditambah 100 atau 200 ms audio baru,
  sehingga total jendela penuh adalah 1.600 atau 1.700 ms.
- Hanya audio baru yang dikeluarkan. Audio masa lalu dihitung ulang,
  tidak diputar ulang dan tidak disimpan sebagai cache state SepFormer.
  Tidak ada crossfade atau lookahead tambahan di luar blok yang sudah masuk.
- Startup memakai riwayat yang sudah tersedia. Model dipanaskan pada
  panjang startup dan panjang penuh sebelum replay. Ekor pendek hanya
  diberi zero padding setelah EOF diketahui.
- Capture tetap berjalan saat worker lambat. Sink playback virtual
  mempertahankan urutan audio dan mencatat delay/celah tanpa membuang input.
  Enam pilihan satu/dua speaker dibuat dari satu separation per putaran.
- Setup, warm-up, probe numerik, ekspor WAV, dan halaman pembanding berada
  di luar waktu replay. Tidak ada dataset, checkpoint baru, atau paket
  Python tambahan yang diunduh.

## Hasil

Statistik proses memakai blok baru penuh dengan riwayat 1.500 ms penuh,
mengecualikan startup dan ekor pendek. Delay playback dihitung atas seluruh
replay, termasuk startup dan antrean. P50 adalah median; p95 berarti 95%
pengukuran berada pada atau di bawah angka tersebut.

| Audio baru / hop | Total jendela | Proses lengkap p50 / p95 | Blok konteks penuh melewati anggaran | Delay playback virtual p50 / p95 |
| --- | ---: | ---: | ---: | ---: |
| 100 ms | 1.600 ms | 173 / 176 ms | 216 / 216 | 7,790 / 15,526 detik |
| **200 ms** | **1.700 ms** | **190 / 197 ms** | **3 / 107** | **413 / 424 ms** |

Pada hop 200 ms:

- Waktu proses maksimum sekitar 221 ms; RTF keseluruhan 0,944.
- Hasil pertama siap pada 295 ms. Delay playback maksimum sekitar 424 ms.
- Total celah playback sekitar 129 ms. Delay bertambah dari 295 menjadi
  424 ms sepanjang contoh. Queue maksimum satu blok; input tidak hilang.
- Waktu SepFormer saja p50/p95 sekitar 163/168 ms. Sisanya berasal dari
  penyusunan konteks, transfer, resampling, fitting gain dan speaker matching.
- Playback virtual selesai pada 23,550 detik, sekitar 424 ms setelah input
  selesai. **Masih hanya ada sekitar 76 ms ruang sebelum batas 500 ms**
  untuk tambahan delay hardware yang belum diukur.

Pada hop 100 ms, proses median sekitar 173 ms harus dilakukan setiap 100 ms.
RTF keseluruhan 1,700, queue mencapai 93 blok, total celah sekitar 16,207
detik, dan delay maksimum mencapai 16,385 detik. Hop lebih kecil tidak
menolong ketika komputasi tiap putaran melebihi interval masuknya audio.

Pembanding CPU sebelumnya dengan konteks 1.500 ms dan hop 500 ms memiliki
proses median 846 ms. Percobaan baru mengubah **device dan hop**, sehingga
rasio angka itu bukan pengukuran speedup dengan input jendela identik.

## Pemeriksaan kompatibilitas CPU dan MPS

Pada prefix Mix.wav sepanjang 1,6 dan 1,7 detik, stream keluaran asli
SepFormer di CPU dan MPS memiliki relative RMS error masing-masing
`5,46e-6` dan `6,61e-6` (kurang dari 0,001%). Pemeriksaan pipeline lengkap
juga mempertahankan mapping speaker pada kedua prefix dan memiliki
relative RMS error output di bawah batas pemeriksaan 1%. Detail setiap
pilihan speaker dan tiga pengulangan timing disimpan dalam metadata probe.

Ini pemeriksaan kesetaraan implementasi pada dua input, **bukan skor kualitas
separation** atau bukti identitas speaker benar. Mix.wav tidak memiliki
clean reference terpisah. Pencocokan masih memaksakan assignment ketika
speaker pilihan sedang diam. Penilaian kualitas membutuhkan mendengarkan.

## Hasil lokal dan cara mendengarkan

```text
data/user_recordings/mps_sepformer/2026-10-09_context_1500ms_hops_100_200/
├── experiment.json
├── verification.json
├── hardware.txt
├── listen.html
├── comparison.json
├── comparison/
├── mps_context_1500ms_hop_100ms/
└── mps_context_1500ms_hop_200ms/
```

Buka `listen.html` di browser. Pilih speaker tunggal atau pasangan, kemudian
bandingkan **MPS hop 200 ms** dengan hasil rekaman penuh dan CPU hop 500 ms
dengan konteks 1.500 ms. Ada juga MPS hop 100 ms dan jendela 500 ms tanpa
konteks sebagai pembanding. Halaman ini menggunakan audio content tanpa
jeda dan satu gain bersama, tanpa normalisasi loudness. Semua preview
memiliki 16 kHz; nilai gain bersama pada run ini adalah 1,0.

Untuk mendengar delay awal serta celah, buka `*_timeline.wav` dalam folder
konfigurasi. `*_content.wav` menghapus jeda penjadwalan untuk menilai suara;
`*_content_raw.wav` menyimpan hasil float. Setiap folder juga menyimpan
`summary.json` dan `trace.jsonl` untuk batas input, gain, assignment dan waktu.

Seluruh **66 WAV** telah diperiksa panjangnya, nilai finite, preview tanpa
clipping, serta kesesuaian potongan dan silence timeline. Kedua replay
menyelesaikan semua input: 232 blok pada hop 100 ms dan 116 blok pada hop
200 ms. Content setiap pilihan memiliki 370.006 sampel pada 16 kHz.
Seluruh 30 tautan pemutar tersedia. Hash Mix.wav dan ketiga enrollment tetap
sama. Hasil dan rekaman tetap lokal dan diabaikan Git; run lama tidak ditimpa.
Folder tambahan memakai sekitar 59 MiB. Verifikasi disimpan dalam
`verification.json`.

## Mengulang

Dari folder `evaluation`, gunakan terminal yang memiliki akses GPU Metal:

```sh
../speakerbeam_ss_verification/.venv/bin/python experiment_mps_sepformer.py --output-dir data/user_recordings/mps_sepformer/run_baru
```

Default adalah konteks 1.500 ms dan hop 100 serta 200 ms. Untuk hop 200 ms
saja tambahkan `--hop-ms 200`. `--context-ms` memilih panjang konteks;
`--probe-only` hanya menjalankan perbandingan CPU/MPS dan timing singkat.
Gunakan folder hasil baru. Runner berhenti jika MPS tidak tersedia atau
tidak ada varian MPS yang lolos pemeriksaan, tanpa substitusi CPU otomatis.

## Langkah berikutnya

1. Nilai kualitas MPS hop 200 ms, terutama kejernihan, speaker lain yang
   bocor, speaker pilihan yang hilang, dan sambungan; periksa pilihan pasangan.
2. Jika kualitas diterima, uji operasi panjang untuk melihat apakah delay
   menetap di bawah 500 ms ketika suhu/beban berubah. Ruang pemrosesan
   sekitar 10 ms terhadap hop 200 ms masih kecil.
3. Ukur mikrofon sampai output fisik di komputer pada rute kabel terlebih
   dahulu. Rute ponsel/TWS tetap membutuhkan pengujian sendiri. Hasil GPU
   M5 tidak boleh dianggap sebagai performa ponsel.

Model tetap SepFormer noncausal yang dijalankan pada jendela terbatas.
Percobaan ini tidak mengubahnya menjadi trained causal TSE atau memenuhi
gate streaming aplikasi dalam PRD.
