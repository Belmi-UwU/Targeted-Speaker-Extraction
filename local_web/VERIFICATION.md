# Website lokal — verifikasi 9 Oktober 2026

Persyaratan: [PRD tunggal](../PRD.md), bagian kontrak website lokal.
Server aktif pada `http://127.0.0.1:8765`, SepFormer MPS dan ECAPA hybrid.
Tidak ada checkpoint/dependency baru yang diunduh. aiohttp 3.14.3 sudah
tersedia dalam virtualenv. Original Mix.wav dan enrollment tidak ditulis ulang.

## Hasil penting

**Website berjalan dan melakukan inference bertahap dengan checkpoint asli.**
Audio hasil dikirim kembali sebelum input selesai, kemudian dijadwalkan
melalui Web Audio. Mikrofon hanya diakses melalui tindakan pengguna.

**Hop 200 ms tidak stabil pada pengujian website ini.** Antrean mencapai
batas dan sistem berhenti secara terlihat. Ini temuan timing yang gagal,
bukan fitur yang dianggap sudah mencapai <500 ms. Hop 300 ms disediakan
sebagai pilihan pembanding dengan konteks tetap 1.500 ms; kualitas subjektif
konfigurasi tersebut belum diterima oleh pengguna.

| HTTP/WebSocket, PCM file paced | Blok hasil | Proses p50 / p95 | Hasil pertama | Status |
| --- | ---: | ---: | ---: | --- |
| Satu speaker, hop 200 ms | 15 / 20 yang direncanakan | 235 / 247 ms | 360 ms | Overload; batas antrean menghentikan sesi |
| Satu speaker, hop 300 ms | 20 / 20 | 241 / 245 ms | 439 ms | Selesai, pending maksimum 0 |
| Dua speaker, hop 300 ms | 20 / 20 | 242 / 245 ms | 419 ms | Selesai, pending maksimum 0 |
| Bypass, hop 20 ms | 50 / 50 | 0,36 / 0,74 ms | 21,8 ms | Selesai, pending maksimum 0 |

Tabel ini mengukur server/file-transport; hasil pertama bukan delay
playback browser maupun delay fisik mikrofon-ke-telinga. Untuk 200 ms,
client berhenti mengirim ketika mendapat overload. Tidak ada klaim bahwa
seluruh 20 blok selesai; output yang masih tertunda dibuang ketika sesi berhenti.
`verification_initial_200ms.json` menyimpan run awal yang juga gagal
(14/20 output, p95 proses 262 ms).

## Browser Safari yang benar-benar diuji

- Halaman lokal dibuka pada tab baru; model **MPS ready** terlihat.
- Profil contoh dari kalibrasi 10 detik diimpor dan tiga kartu tampil.
- Demo **hop 200 ms** mengeluarkan audio secara incremental, lalu overload.
  UI mencatat estimasi delay aplikasi p95 sekitar **707 ms** sebelum stop.
- Demo **hop 300 ms** pada **12,9 detik audio lanjutan** selesai dengan
  status “Demo selesai”. UI akhir: **601 ms p95 estimasi delay aplikasi**,
  **253 ms p95 proses**, pending 0, dan **0 ms celah playback tercatat**.
- Screenshot desktop memverifikasi kartu, pilihan/mode/volume/transport,
  serta metrik tanpa elemen yang saling menutupi. CSS adaptif tersedia;
  browser/viewport ponsel belum diuji secara visual.
- Safari yang tersedia tidak menawarkan `AudioContext.setSinkId`; UI
  menunjukkan penggunaan output default macOS secara eksplisit.
- Tidak mengaktifkan/mengizinkan mikrofon, memilih rute AirPods di macOS,
  atau menilai identitas/kualitas secara subjektif selama verifikasi.
- Angka 601 ms berasal dari jam capture graph dan playback terjadwal di
  browser. **Ini bukan pengukuran akustik dan belum memenuhi target <500 ms.**

## Pemeriksaan yang lulus

[verify_local_web.py](verify_local_web.py) memakai HTTP dan WebSocket sungguhan
pada server model, dan menyimpan [verification.json](verification.json):

- Static assets tersedia; origin luar ditolak; path tidak valid ditolak.
- Preview profil tersimpan bisa diambil sebagai WAV finite 16 kHz.
- Upload tidak valid dan WAV 10 detik ditolak oleh batas kalibrasi 30 detik.
- Paket output memiliki jumlah sampel yang benar, finite, tidak melebihi
  peak limiter, dan urutan contiguous sebelum Stop/overload.
- Satu/dua speaker pada 300 ms dan Bypass selesai; Stop kemudian Start bekerja.
- Pilihan kosong, gap sequence dan tab audio kedua ditolak secara terlihat.
- Burst input memicu batas antrean tanpa pertumbuhan memori/delay tak terbatas.
- Kalibrasi menerima tepat **150 × 200 ms = 30 s** dan membuat tiga profil
  melalui model asli. **Fixture mengulang 10 detik awal Mix.wav tiga kali dan
  dikirim dipercepat**; ini bukan rekaman alami atau uji mikrofon browser.
- Rename/delete dilakukan hanya pada profil disposable yang dibuat oleh test.
  Profil tersebut dibersihkan; profil asli/history tidak dihapus.

[verify_worklet.cjs](verify_worklet.cjs) menguji callback audio dengan input
terkendali pada 16/44,1/48/96 kHz dan hop 20/200/300 ms. Dua belas konfigurasi
lulus jumlah sampel, mono downmix, urutan, start boundary, capture berhenti
pada limit, graf capture tidak memainkan audio mentah, dan tidak ada capture
setelah Stop. [worklet_verification.json](worklet_verification.json) adalah
uji logika callback, bukan bukti pemrosesan mikrofon fisik.

Node syntax checks, Python AST checks, dan `git diff --check` lulus.
Tidak ada live PCM yang disimpan oleh jalur listening. Kalibrasi/profil
tersimpan di folder user-recordings yang diabaikan Git; metrics fixture
tidak berisi audio/embedding.

## Yang pengguna lakukan berikutnya

1. Buka website, pilih output AirPods Max USB-C di macOS, izinkan mic Mac,
   dan uji **Bypass** untuk memastikan suara terdengar dan level sesuai.
2. Stop; rekam percakapan **30 detik sebenarnya**. Semua speaker yang
   ingin dipilih harus berbicara. Dengarkan/beri nama kandidat profil.
3. Pilih satu speaker, lalu pasangan. Coba 200 ms dan 300 ms terpisah;
   laporkan kejernihan, kebocoran, suara hilang, pertukaran identitas dan delay.
4. Unduh laporan JSON. Jika kualitas diterima, lakukan run lebih panjang
   dan physical loopback; jangan menganggap 6–13 detik membuktikan 30 menit.

Penyebab pasti perbedaan performa dibanding replay 184–190 ms sebelumnya
belum diisolasi; beban/performa host dan jalur browser berbeda. Jangan
mengatribusikan perlambatan ke satu faktor tanpa pengukuran tambahan.
