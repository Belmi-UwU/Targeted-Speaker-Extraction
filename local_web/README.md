# Suara — website TSE lokal

Acuan persyaratan tunggal: [PRD.md](../PRD.md), bagian **Current implementation
contract — local website**. Status pekerjaan dan langkah berikutnya:
[PROGRESS.md](../PROGRESS.md). Tidak ada build Node atau layanan cloud.

## Menyalakan

```sh
cd "/Users/mumu/Apple/Final Challenge/TSE Trial1"
./run_local_web.sh
```

Buka **http://127.0.0.1:8765**. Tunggu status **Model siap · MPS**.
Tekan `Ctrl+C` di terminal server untuk menutupnya. Port alternatif:
`./run_local_web.sh --port 8766`. CPU tersedia secara eksplisit melalui
`--device cpu` untuk debugging, dengan peringatan performa pada UI.
Server terikat ke 127.0.0.1; ini belum situs yang bisa dibuka dari ponsel
di jaringan yang sama.

Gunakan virtualenv proyek yang sudah ada. Dependency transport tambahan
`aiohttp==3.14.3` sudah tersedia pada komputer ini. Jika menyiapkan ulang
environment, setelah memasang dependency model yang sudah didokumentasikan:

```sh
speakerbeam_ss_verification/.venv/bin/python -m pip install -r local_web/requirements.txt
```

## Percobaan pertama: demo file

1. Klik **Pakai profil uji Mix.wav**. Profil ini berasal dari 10 detik pertama
   Mix.wav pada eksperimen sebelumnya; ini bukan kalibrasi 30 detik baru.
2. Pilih sumber **Demo Mix.wav**. Hanya audio setelah detik ke-10 dikirim
   bertahap melalui AudioWorklet dan WebSocket. Hasil dibuat oleh checkpoint
   aktif; tidak memutar WAV hasil pemisahan yang sudah tersimpan.
3. Pilih satu atau beberapa speaker. Dengarkan cuplikannya untuk mengenali
   suara. Nama bisa diubah lewat kolom di kartu.
4. Coba hop **200 ms**. Pada pengujian website saat ini, konfigurasi ini
   overload; aplikasi berhenti dan menampilkan alasannya. Pilih **300 ms**
   untuk pembanding dengan waktu proses lebih longgar. Konteks tetap 1.500 ms.
5. Tekan **Mulai mendengar**, kemudian **Stop** atau tunggu demo selesai.
   Demo dibatasi ke hop penuh: sekitar 13,0 s pada 200 ms, 12,9 s pada 300 ms,
   13,12 s pada Bypass 20 ms. Ekor pendek file tidak digunakan dalam demo.
6. Unduh laporan JSON untuk timing sesi. Estimasi delay aplikasi belum
   mencakup delay akustik/input/output perangkat.

**Hop 300 ms belum memiliki penerimaan kualitas dari pengguna** dan bisa
memiliki delay lebih tinggi. Tidak ada jaminan total delay di bawah 500 ms.

## Percobaan mikrofon + AirPods

1. Sambungkan AirPods Max melalui USB-C untuk uji awal. Pilih output AirPods
   di macOS **System Settings → Sound → Output**. Jika browser mendukung
   `AudioContext.setSinkId`, output juga bisa dipilih dalam halaman.
2. Pilih **Mikrofon langsung**, klik **Izinkan & cari perangkat**, lalu pilih
   mic Mac. Izin hanya diminta melalui tindakan pengguna. Track untuk pencarian
   perangkat langsung dihentikan setelah perangkat terdaftar.
3. Pilih **Bypass** dan Mulai. Ini meneruskan suara asli pada blok 20 ms tanpa
   pemisahan model. Periksa suara masuk, output, dan volume headphone.
4. Stop. Klik **Rekam kalibrasi 30 detik** dan minta tiga peserta berbicara.
   Mikrofon berhenti setelah tepat 30 s berdasarkan jumlah sampel. Audio baru
   tidak ditampung selama profil sedang dibuat. Pembuatan profil menambah
   waktu startup; pemanasan model juga terjadi saat server pertama dimuat.
5. Dengarkan/beri nama tiga kandidat, pilih speaker, pilih Dengan filter,
   lalu mulai sesi baru. Jangan memakai profil Mix.wav untuk mengharapkan
   penyaringan orang lain yang tidak berbicara di rekaman kalibrasinya.
6. Periksa suara bocor, speaker hilang, identitas tertukar, noise, sambungan
   antarblok dan kenyamanan delay. Bandingkan Bypass pada volume yang sama;
   loudness sumber campuran dan hasil filter tidak dijamin identik.

Pengaturan echo cancellation, noise suppression dan auto gain diminta off.
UI menampilkan **pengaturan efektif**, karena browser bisa mengabaikan permintaan.
Safari yang diuji memakai output default macOS; pemilihan sink belum tersedia
di browser tersebut. Dukungan Chrome/earbud mic/rute Bluetooth perlu diuji
di perangkatnya masing-masing. Jangan menyimpulkan akses ke semua mic ANC AirPods.

## Profil, penyimpanan, dan penggantian pilihan

- Alternatif kalibrasi: impor WAV minimal 30 detik, maksimal 30 MiB. Hanya
  30 detik pertama digunakan. Input senyap/terlalu pendek/clipping ditolak.
- Profil dan kalibrasi disimpan di
  `evaluation/data/user_recordings/local_web/profiles/<id>/`, diabaikan Git.
  Batas 20 grup profil tersimpan; hapus melalui UI jika penuh.
- Rekaman **live** tidak disimpan. Laporan JSON berisi metrik/configuration,
  tanpa PCM atau embedding. Data lokal tidak dikirim ke internet.
- Hapus hanya menghapus profil yang dibuat/disalin oleh website, bukan
  enrollment asli atau hasil eksperimen sebelumnya. Tombol Hapus adalah
  penghapusan permanen profil website yang dipilih.
- Mengganti mode atau pilihan speaker saat live menghentikan sesi; tekan
  Mulai lagi. Antrean/context/playback lama dibersihkan, bukan diteruskan
  dengan speaker yang sudah tidak dipilih.
- Stop langsung menghentikan input/playback. Jika model sudah membuat
  profil ketika Stop ditekan, pekerjaan yang berjalan bisa selesai dan
  profilnya tersimpan; ini tidak otomatis mengaktifkan mikrofon lagi.
- Satu koneksi audio aktif per server. Tab lain memperoleh pesan busy.
- Batas antrean dua blok input menunggu; backlog browser juga dibatasi.
  Overload berhenti dengan pesan, tidak diam-diam menumpuk delay tanpa batas.

## Metrik dan batas model

UI menampilkan p95 proses dan estimasi delay pada hingga 300 blok terbaru,
pending server, celah playback, dan buffer browser. Report menyimpan hingga
9.000 blok terbaru. Bypass memiliki lebih banyak blok per detik, sehingga
9.000 blok bukan otomatis laporan penuh 30 menit.

Timestamp capture/playback memakai jam browser; jam Python dipakai hanya
untuk durasi proses/queue. Output dijadwalkan 20 ms ke depan minimal. Belum
ada loopback fisik atau bukti 30 menit stabil. SepFormer tetap noncausal pada
jendela terbatas, tiga output, 8 kHz internal/16 kHz output. Target absence,
speaker baru, jumlah orang selain tiga, dan purity profil belum tervalidasi.
Limiter menggunakan peak blok yang sudah diproses dan menghitung event gain
di bawah 1; belum ada crossfade/limiter dengan lookahead di luar audio masuk.

## Verifikasi developer

Dengan server MPS aktif:

```sh
node local_web/verify_worklet.cjs
speakerbeam_ss_verification/.venv/bin/python local_web/verify_local_web.py
```

Lihat [VERIFICATION.md](VERIFICATION.md) serta `verification.json`,
`verification_initial_200ms.json` dan `worklet_verification.json`.
Kalibrasi 30 detik pada integration test memakai fixture yang mengulang
10 detik Mix.wav tiga kali dan dikirim dipercepat; **bukan capture mic**.
Profil fixture test dihapus setelah pemeriksaan. Uji mic/AirPods dilakukan
pengguna melalui website, dengan permission dan rute yang sebenarnya.

Referensi API: [AudioWorklet](https://developer.mozilla.org/en-US/docs/Web/API/AudioWorklet),
[getUserMedia](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia),
[pemilihan output](https://developer.mozilla.org/en-US/docs/Web/API/AudioContext/setSinkId).
