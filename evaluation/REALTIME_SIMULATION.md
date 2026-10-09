# Simulasi waktu nyata — Mix.wav

Pengujian 7 Oktober 2026 memakai rekaman `Mix.wav` selama **23,125 detik**
di komputer ini. Audio stereo 48 kHz dilepas dalam frame 20 ms mengikuti jam
nyata. SepFormer asli dijalankan pada jendela tanpa overlap, lalu ECAPA
memilih stream berdasarkan enrollment Nicker, Mond, dan Sam.

**Jendela 500 ms menghasilkan jeda playback median 1,018 detik dan p95
1,222 detik, dengan total celah playback 289 ms. Jendela 1 detik lebih
stabil, tetapi jeda playback mencapai sekitar 1,581 detik.** Keduanya
melewati target sementara PRD sebesar 150 ms untuk rute kabel.

## Metode

- Producer melepas frame input sesuai waktu sampelnya. Worker terpisah
  melakukan resampling, pemisahan SepFormer, fitting gain dari campuran
  saat ini, embedding ECAPA, pencocokan, dan penggabungan pilihan speaker.
  Queue tidak membatasi capture; jika worker lambat, antrean tetap terlihat.
- Kontrol tanpa model memakai frame 20 ms. Dua pengujian model memakai
  jendela/hop 500/500 ms dan 1000/1000 ms.
- Worker hanya menerima audio setelah seluruh jendelanya masuk. Resampling
  dilakukan per jendela, sehingga tidak mengambil sampel dari jendela masa
  depan. Sisa audio pendek di akhir diberi zero padding setelah EOF diketahui.
- Model CPU menggunakan empat thread PyTorch dan dipanaskan sebelum replay.
  Loading model dan persiapan enrollment memerlukan **0,837 detik**, di luar
  pengukuran delay. Penulisan WAV ke disk juga dilakukan setelah replay.
- Ketiga enrollment dicocokkan sekaligus dengan penugasan satu-ke-satu
  pada setiap jendela. Ini berbeda dari pencocokan pilihan speaker atas
  satu rekaman penuh pada runner offline sebelumnya.
- Enam hasil pilihan speaker dibuat dari satu proses separation dan matching
  yang sama. Tidak ada enam panggilan model terpisah untuk setiap pilihan.
- Playback bersifat **virtual**. Setiap potongan mulai pada waktu paling
  awal ketika hasil tersedia dan potongan sebelumnya selesai dimainkan.
  Jika hasil berikutnya belum tersedia, celah disimpan dalam WAV timeline.
  Mikrofon, speaker, earphone, Bluetooth, dan driver audio belum dilibatkan.

## Hasil waktu

P50 adalah median, dan p95 berarti 95% nilai yang diukur berada pada atau
di bawah nilai tersebut. Angka delay playback dihitung dari waktu asal
sampel sampai jadwal sampel dimainkan di sink virtual.

| Kondisi | Waktu proses p50 / p95 | Delay playback p50 / p95 | Jendela penuh melewati anggaran proses | Celah playback >1 ms |
| --- | ---: | ---: | ---: | ---: |
| Tanpa model, frame 20 ms | 0,60 / 0,88 ms | 26,37 / 26,79 ms | 0 / 1156 | 0 |
| SepFormer + ECAPA, 500 ms | 361,58 / 569,51 ms | 1018,48 / 1222,34 ms | 4 / 46 | 3; total semua celah 288,77 ms |
| SepFormer + ECAPA, 1 detik | 506,37 / 569,36 ms | 1580,81 / 1580,81 ms | 0 / 23 | 1; total semua celah 7,81 ms |

Hasil pertama tersedia pada **0,934 detik** untuk jendela 500 ms dan
**1,573 detik** untuk jendela 1 detik. Median waktu hasil tersedia relatif
terhadap sampel pertama tiap jendela adalah 0,867 dan 1,511 detik.
Playback kontinu harus mempertahankan urutan audio; celah yang sudah
terjadi menambah delay potongan berikutnya, sehingga delay playback bisa
lebih besar daripada waktu hasil tersedia.

Seluruh playback virtual selesai pada **24,348 detik** (jendela 500 ms)
dan **24,706 detik** (jendela 1 detik), dihitung sejak input mulai mengalir.
Artinya, ekor rekaman selesai dimainkan sekitar 1,22 dan 1,58 detik setelah
input 23,125 detik berakhir.

Pada jendela 500 ms, jumlah waktu komputasi adalah **18,558 detik** untuk
23,125 detik audio (RTF 0,803). Pada jendela 1 detik, nilainya **12,383
detik** (RTF 0,535). Rata-rata komputasi lebih cepat daripada input, tetapi
lonjakan waktu proses pada 500 ms tetap menyebabkan celah. Antrean tidak
tumbuh terus sepanjang contoh ini; kenaikan delay playback dari awal ke
akhir adalah 288,77 ms dan 7,81 ms.

Producer juga memiliki keterlambatan penjadwalan: p95 sekitar **4,8 ms**
pada 500 ms dan **4,7 ms** pada 1 detik. Ini tercatat dalam hasil, sehingga
delay tidak dianggap berasal dari model saja. Satu replay 23 detik belum
menunjukkan kemampuan operasi berkelanjutan selama 30 menit atau performa
ponsel.

## File hasil

Hasil lengkap berada di folder lokal yang diabaikan Git:

```text
data/user_recordings/realtime_simulation/2026-10-07_cpu_windows/
├── experiment.json
├── passthrough_20ms/
├── sepformer_500ms/
└── sepformer_1000ms/
```

Setiap folder memiliki `summary.json` dan `trace.jsonl` dengan waktu dan
mapping speaker per jendela. Folder SepFormer memuat enam pilihan:
`only_nicker`, `only_mond`, `only_sam`, `nicker_and_mond`, `nicker_and_sam`,
dan `mond_and_sam`. Setiap pilihan memiliki:

- `*_timeline.wav`: dengarkan ini untuk mendengar jeda awal dan celah
  playback yang disimulasikan.
- `*_content.wav`: potongan hasil digabung tanpa jeda penjadwalan, untuk
  memeriksa suara, identitas speaker, kebocoran, dan sambungan antarjendela.
- `*_content_raw.wav`: hasil float untuk analisis; safety gain preview
  dihitung setelah pengukuran. Pada run ini gain preview adalah 1,0.

WAV memakai 16 kHz. Content memiliki panjang 370006 sampel, sekitar 23,125
detik; timeline lebih panjang sesuai penundaan playback. Semua 39 WAV
diverifikasi panjangnya, memiliki nilai finite, dan preview tidak clipping.
Hash `Mix.wav` setelah pengujian sama dengan sebelum pengujian.

## Mengulang

Dari folder `evaluation`:

```sh
../speakerbeam_ss_verification/.venv/bin/python simulate_realtime.py --self-check
../speakerbeam_ss_verification/.venv/bin/python simulate_realtime.py
```

Perintah kedua membuat folder hasil baru dengan waktu Jakarta. Self-check
memastikan input tidak diberikan lebih awal daripada waktu capture,
capture tetap berjalan saat worker overload, dan celah playback serta
pertumbuhan delay tidak disembunyikan.

## Implikasi untuk proyek

Pengguna mendengarkan hasil dan menilai kualitas pada jendela waktu nyata
lebih buruk daripada hasil rekaman penuh. Pada rekaman ini, konfigurasi
windowing yang diuji **belum memenuhi hasil dengar yang diinginkan**.
Penilaian ini berlaku pada contoh lokal ini, bukan skor benchmark umum.

Log juga menunjukkan penurunan kemiripan ECAPA. Pada rekaman penuh,
kemiripan stream yang cocok dengan enrollment adalah sekitar 0,49 untuk
Nicker, 0,46 untuk Mond, dan 0,55 untuk Sam. Median kemiripan assignment
per jendela penuh pada simulasi adalah:

| Speaker | Jendela 500 ms | Jendela 1 detik |
| --- | ---: | ---: |
| Nicker | 0,123 | 0,206 |
| Mond | 0,097 | 0,158 |
| Sam | 0,167 | 0,209 |

Cosine similarity ini bukan probabilitas atau ukuran akurasi. Audio pendek
dan hasil separation yang berubah dapat sama-sama menurunkan nilainya;
angka ini belum menentukan apakah masalah utamanya separation atau
assignment speaker. Sambungan tanpa overlap dan gain yang dipasang per
jendela juga dapat mengubah suara. SepFormer rekaman penuh tetap menjadi
referensi kualitas offline.

Windowing memungkinkan SepFormer menghasilkan audio sebelum rekaman
selesai, tetapi hasil ini belum cocok untuk percakapan dengan target delay
150 ms. Jendela 1 detik menjadi pembanding yang lebih stabil pada komputer
ini; jendela 500 ms memiliki delay lebih pendek dengan beberapa celah.

Belum ada clean reference dari rekaman ini, sehingga tidak ada klaim
SI-SDR atau akurasi identitas speaker. ECAPA pada potongan pendek dan
sambungan tanpa overlap perlu didengarkan. Sistem masih memaksakan
assignment saat speaker pilihan tidak hadir. Langkah berikutnya adalah
meninjau audio hasil, lalu menjalankan model yang dilatih untuk streaming
melalui harness waktu yang sama sebelum pengujian rute perangkat.
