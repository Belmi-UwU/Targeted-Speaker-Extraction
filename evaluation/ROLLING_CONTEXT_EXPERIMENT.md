# Pengurangan konteks SepFormer — 7 Oktober 2026

Eksperimen ini menguji usulan mempertahankan audio sebelumnya untuk membantu
SepFormer memproses audio baru. Input adalah `Mix.wav` yang sama, sepanjang
23,125 detik, dengan enrollment Nicker, Mond, dan Sam yang sama.

## Metode

- Audio input dilepas dalam frame 20 ms mengikuti jam nyata. Audio baru
  dikumpulkan selama 500 ms pada setiap putaran.
- Konteks diuji berurutan: 2.500, 1.000, 500, lalu 250 ms. Total jendela
  saat riwayat penuh adalah 3.000, 1.500, 1.000, dan 750 ms.
- Worker menyimpan audio mentah yang sudah diterima. Model menghitung ulang
  seluruh konteks plus audio baru; tidak ada cache state SepFormer.
- Saat awal, hanya riwayat yang sudah tersedia dipakai. Tidak menunggu
  2,5 detik untuk mulai. Pada ekor rekaman, padding kanan hanya dilakukan
  setelah EOF diketahui, seperti simulasi sebelumnya.
- SepFormer asli dan ECAPA menggunakan checkpoint lokal yang sama. Tidak
  ada perubahan attention, normalisasi, fine-tuning, atau download dataset.
- Fitting gain dan pencocokan satu-ke-satu dengan ketiga enrollment memakai
  seluruh jendela yang tersedia. Hanya hasil untuk audio baru yang dikeluarkan.
  Konteks lama tidak diputar ulang.
- Tidak ada lookahead di luar blok baru atau crossfade. Ini memisahkan
  pengaruh panjang konteks dari perubahan metode penyambungan.
- Enam pilihan satu/dua speaker dibuat dari separation yang sama. CPU
  menggunakan empat thread PyTorch. Model dipanaskan sebelum tiap replay.
  Loading, enrollment, warm-up, ekspor audio, dan pembuatan halaman dengar
  berada di luar waktu pemrosesan replay.
- Capture terus berjalan ketika worker tertinggal. Queue dan sink playback
  virtual memperlihatkan penumpukan delay; tidak membuang potongan audio.

## Hasil

Angka proses berikut memakai putaran 500 ms dengan riwayat konteks penuh;
startup dan ekor pendek tidak masuk statistik proses tersebut. Waktu proses
mencakup penyusunan konteks, resampling, SepFormer, gain fitting, ECAPA,
pencocokan, dan ekstraksi audio baru. Delay playback dihitung untuk seluruh
rekaman pada sink virtual, termasuk startup dan penumpukan antrean.

| Konteks lama | Total jendela | Proses p50 / p95 | Putaran konteks penuh >500 ms | Delay playback p50 / p95 |
| --- | ---: | ---: | ---: | ---: |
| 2.500 ms | 3.000 ms | 1.131 / 1.315 ms | 41 / 41 | 13,651 / 27,588 detik |
| 1.500 ms (tambahan) | 2.000 ms | 846 / 939 ms | 43 / 43 | 8,499 / 16,092 detik |
| 1.000 ms | 1.500 ms | 715 / 838 ms | 44 / 44 | 6,845 / 10,831 detik |
| 500 ms | 1.000 ms | 560 / 615 ms | 45 / 45 | 2,660 / 3,966 detik |
| 250 ms | 750 ms | 480 / 525 ms | 8 / 45 | 1,065 / 1,065 detik |

P50 adalah median; p95 berarti 95% pengukuran berada pada atau di bawah
nilai tersebut. Jumlah putaran konteks penuh berbeda karena riwayat panjang
membutuhkan lebih banyak putaran startup.

Konteks 1.500 ms diuji pada replay tambahan dengan input, checkpoint,
hop, dan metode yang sama; empat konfigurasi sebelumnya tidak dijalankan
ulang. Total jendela adalah 2 detik. Semua 43 putaran dengan riwayat penuh
melampaui anggaran 500 ms, dengan proses maksimum sekitar 1.001 ms.
RTF keseluruhan adalah 1,701 dan total celah playback sekitar 16,028 detik.
Delay playback p95 16,092 detik adalah hasil penumpukan antrean sepanjang
rekaman, bukan waktu proses satu blok. Kualitas suara hasil ini belum dinilai.

Konteks 250 ms menjadi **kandidat tercepat**, tetapi masih memiliki lonjakan
waktu proses: maksimum sekitar 561 ms. RTF pemrosesan keseluruhan adalah
0,983, sehingga ruang komputasinya tipis. Celah playback virtual berjumlah
sekitar 130 ms, dengan kenaikan delay dari awal ke akhir sebesar nilai yang
sama. Ini bukan bukti konfigurasi tersebut stabil untuk penggunaan panjang.

Konteks 500, 1.000, dan 2.500 ms tidak mampu mengikuti hop 500 ms pada CPU
ini. Setiap putaran dengan riwayat penuh melampaui anggaran. Delay yang
besar di tabel adalah akibat antrean yang terus bertambah, bukan waktu
proses satu jendela. Total celah playback masing-masing sekitar 3,169,
10,277, dan 28,113 detik. Konteks 250 ms pun masih melewati target sementara
PRD sebesar 150 ms dan belum mengukur delay audio hardware.

**Kualitas suara belum dinilai dengan mendengarkan**, sehingga konteks
terpendek yang masih bagus belum ditetapkan. Rekaman ini tidak memiliki
clean reference terpisah; tidak ada klaim SI-SDR atau akurasi separation.

Median kemiripan assignment ECAPA pada jendela dengan konteks penuh:

| Konteks lama | Nicker | Mond | Sam |
| --- | ---: | ---: | ---: |
| 2.500 ms | 0,303 | 0,285 | 0,390 |
| 1.500 ms (tambahan) | 0,241 | 0,230 | 0,333 |
| 1.000 ms | 0,219 | 0,158 | 0,256 |
| 500 ms | 0,212 | 0,179 | 0,224 |
| 250 ms | 0,161 | 0,113 | 0,184 |

Angka ini memakai **seluruh jendela konteks**, bukan hanya 500 ms yang
dikeluarkan. Panjang audio ikut memengaruhi embedding, dan bagian konteks
lama dapat membantu assignment meskipun bagian terbaru masih kurang baik.
Angka ini tidak menentukan konfigurasi yang lolos kualitas. Nomor stream
juga dapat berubah tanpa identitas speaker berubah; pergantian nomor saja
tidak dinilai sebagai kesalahan.

## File dan cara mendengarkan

**Halaman terbaru**, termasuk konteks 1.500 ms dan empat konteks sebelumnya:

```text
data/user_recordings/rolling_context/2026-10-07_context_1500ms/listen.html
```

Folder ini menyimpan `context_1500ms_hop_500ms/`, metadata dan verifikasi
run tambahan, serta salinan preview untuk pembandingan dengan lima panjang
konteks, hasil tanpa konteks, dan hasil rekaman penuh. Ada enam pilihan
speaker dan 42 pemutar. Semua 60 WAV dalam folder ini diperiksa seperti
run pertama: panjang, nilai finite, preview tanpa clipping, dan timeline
sesuai jadwal. Hash rekaman asli tetap sama. Folder tambahan memakai sekitar
50 MiB. File dan metadata run pertama tetap tersedia di lokasi berikut.

Hasil lokal berada di:

```text
data/user_recordings/rolling_context/2026-10-07_context_sweep/
├── experiment.json
├── listen.html
├── comparison.json
├── verification.json
├── comparison/
├── context_2500ms_hop_500ms/
├── context_1000ms_hop_500ms/
├── context_500ms_hop_500ms/
└── context_250ms_hop_500ms/
```

Buka `listen.html` di browser. Pilih speaker tunggal atau pasangan. Halaman
memuat empat konteks, hasil 500 ms tanpa konteks dari simulasi sebelumnya,
dan hasil rekaman penuh yang sebelumnya dinilai bagus oleh pengguna.
Semua audio pembanding memakai satu safety gain bersama dan tidak diberi
normalisasi loudness. Preview offline yang tersimpan sebelumnya telah
melewati kuantisasi PCM16; halaman ini tidak meregenerasi inferensi offline.

Bandingkan bagian ucapan yang sama, terutama setelah detik 2,5 ketika
riwayat semua konfigurasi sudah penuh. Mulai dari konteks 250 ms, kemudian
500, 1.000, 1.500, dan 2.500 ms jika hasil sebelumnya masih kurang bagus.

Nilai apakah suara pilihan jelas, suara lain cukup ditekan, suara pilihan
tidak hilang/berpindah, dan sambungan tidak mengganggu. Periksa juga bagian
speaker pilihan diam serta keluaran dua speaker. Kemiripan ECAPA pada
jendela konteks bukan probabilitas, akurasi, atau pengganti penilaian dengar.

Pilihan penilaian di halaman dapat diekspor dengan tombol **Simpan penilaian
JSON**. Penilaian di halaman belum disimpan otomatis; ekspor sebelum menutup.

Setiap folder konfigurasi menyimpan:

- `*_content_raw.wav`: hasil float tanpa jeda playback, untuk analisis.
- `*_content.wav`: preview PCM16 tanpa jeda, untuk kualitas suara.
- `*_timeline.wav`: preview dengan delay awal dan celah playback virtual.
- `summary.json`: statistik waktu, anggaran 500 ms, serta diagnostik konteks.
- `trace.jsonl`: waktu, batas input/keluaran, gain, similarity, dan mapping
  speaker setiap putaran.

## Mengulang

Dari folder `evaluation`:

```sh
../speakerbeam_ss_verification/.venv/bin/python experiment_rolling_context.py --self-check
../speakerbeam_ss_verification/.venv/bin/python experiment_rolling_context.py
```

Perintah kedua membuat folder hasil baru berdasarkan waktu Jakarta.
Self-check menguji riwayat yang terbatas, startup, ekstraksi bagian baru
tanpa mengulang konteks, batas input tanpa audio masa depan, serta ekor pendek.

Untuk konteks 1.500 ms saja, dengan pembanding dari sweep yang sudah tersimpan:

```sh
../speakerbeam_ss_verification/.venv/bin/python experiment_rolling_context.py --context-ms 1500 --comparison-run-dir data/user_recordings/rolling_context/2026-10-07_context_sweep
```

`--context-ms` dapat menentukan satu atau beberapa konteks; tanpa argumen
ini default tetap empat konteks awal. `--comparison-run-dir` hanya membaca
hasil sebelumnya dan membuat salinan preview dalam folder baru, tanpa
mengulangi inference sebelumnya. Self-check dengan `--context-ms 1500`
juga lulus sebelum replay tambahan.

## Verifikasi hasil

- Keempat konfigurasi masing-masing menyelesaikan 47 putaran dan menyimpan
  enam pilihan speaker, tanpa audio input yang hilang atau konteks terulang.
- Content setiap pilihan memiliki 370.006 sampel pada 16 kHz, sekitar
  23,125 detik; selisih pecahan sampel berasal dari pembulatan resampling.
- Semua **108 WAV** hasil dan pembanding memiliki nilai finite. Preview
  tidak clipping. Setiap potongan timeline cocok dengan content-nya dan
  celah virtual berisi silence sesuai jadwal yang dicatat.
- Batas model input dan audio baru cocok dengan waktu sumber. Worker tidak
  menerima blok sebelum waktu capture selesai. Hash Mix.wav dan ketiga
  enrollment sama setelah eksperimen.
- Seluruh 36 tautan pemutar di halaman dengar mengarah ke WAV yang tersedia.
  Gain bersama halaman pembanding adalah 1,0. Folder percobaan memakai
  sekitar **102 MiB**, termasuk salinan preview pembanding.
- Hasil pemeriksaan disimpan dalam `verification.json`. File rekaman dan
  hasil audio tetap lokal, diabaikan Git.

## Keputusan berikutnya

Pilih konteks terpendek yang lolos penilaian dengar. Setelah itu bandingkan
anggaran pemrosesan dan delay dengan kebutuhan PRD. Konfigurasi yang terdengar
bagus tetap belum menjadi solusi real-time bila pemrosesan tidak mampu
mengikuti interval 500 ms. Satu rekaman pendek ini juga belum membuktikan
ketahanan 30 menit, performa ponsel, atau delay mikrofon/earphone fisik.
