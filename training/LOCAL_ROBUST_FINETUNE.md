# Fine-tuning WeSep pada MacBook

Disiapkan pada 12 Oktober 2026. Ini eksperimen lanjutan dari checkpoint R1 yang saat ini dipakai web app.

## Alasan dan perubahan

Pemeriksaan dengar pada Mix.wav menunjukkan Sam tetap jelas dengan enrollment discovery, sedangkan Nicker mengalami kebocoran Sam/Mond dan volume yang naik-turun. Mengganti enrollment Nicker dengan sumber 8–16 detik tidak memberi perubahan yang terasa menurut pengguna.

Profil `noise_robust` pada `wesep_retrain.py` mengubah training sebagai berikut:

- Memperbaiki augmentasi suara bocor: training lama menyediakan sumber 4 detik untuk enrollment 5 detik, sehingga kondisi panjang menyebabkan augmentasi ini terlewati. Profil baru memakai sumber yang tersedia dengan padding; profil baseline mempertahankan resep historisnya.
- Menambahkan dengung/buzz dan burst yang menyerupai tepukan **sintetis**. Ini belum menggantikan kebutuhan evaluasi pada rekaman noise lingkungan nyata.
- Menambah contoh target yang lebih pelan dan kondisi SNR lebih rendah.
- Menambah variasi filter kanal pada enrollment, dengan 50% clean dan 50% discovered-style enrollment.
- Menambah loss energi target per 100 ms. Tujuannya mengikuti energi referensi, termasuk perubahan volume alaminya; bukan membuat semua suara bervolume konstan.
- Membekukan speaker encoder dan memakai learning rate lebih kecil.

## Run utama

| Pengaturan | Nilai |
| --- | --- |
| Model awal | `evaluation/wesep_eval/checkpoints/pretrained/retrained_finetune_research` |
| Akselerasi | Metal/MPS pada MacBook ini |
| Langkah | 5.000 |
| Batch | 2 |
| Learning rate | 5e-5, warmup 200 langkah |
| Training | LibriSpeech train-clean-100 |
| Validation | dev-clean, 48 contoh tetap dengan speaker berbeda dari training |
| Run | `training/local_work/runs/finetune_noise_robust_research` |

Launcher `start_local_robust.py` melanjutkan unduhan, memverifikasi MD5, lalu menjalankan training. Arsip train-clean-100 sekitar 6,4 GB dan dev-clean sekitar 0,34 GB; data hasil ekstraksi membutuhkan ruang tambahan. Metal terdeteksi di luar sandbox. Job memakai `caffeinate` agar Mac tidak tidur selama proses aktif; biarkan Mac tersambung daya dan tetap terbuka.

Checkpoint awal dan checkpoint aktif web app tidak diganti oleh job ini. Output baru:

`training/local_work/runs/finetune_noise_robust_research/export/retrained_finetune_noise_robust_research/`

## Status dan log

- `training/local_work/local_status.json`: tahap unduhan/training, PID, atau pesan kegagalan.
- `training/local_work/local_training.log`: log launcher dan trainer.
- `training/local_work/runs/finetune_noise_robust_research/run_config.json`: konfigurasi dan hash bobot awal.
- `training/local_work/runs/finetune_noise_robust_research/log.jsonl`: loss, validasi, dan ETA.
- `last.pt` / `prev.pt`: checkpoint untuk resume; `best.pt`: bobot terbaik menurut loss validasi.

Perintah menjalankan atau melanjutkan job setelah job sebelumnya berhenti:

```sh
caffeinate -dims evaluation/wesep_eval/.venv/bin/python -u training/start_local_robust.py > training/local_work/local_training.log 2>&1
```

## Yang sudah diperiksa

- Empat regression checks pada `test_robust_recipe.py` lulus: kebocoran 4→5 detik benar-benar ditambahkan, augmentasi menghasilkan sinyal valid, loss mengikuti volume referensi, dan loss/gradient tetap finite untuk target hadir maupun absen.
- Smoke check 20 langkah selesai pada MPS: training, validasi, checkpoint, export, dan uji kausalitas berjalan. Perubahan pada prefix test adalah 0. Kecepatan smoke sekitar 1,8 detik/langkah untuk batch 2; waktu run utama dapat berbeda.
- Smoke menggunakan sebagian `test-clean` yang tersedia, dengan pemisahan speaker train/validation, **hanya untuk memeriksa pipeline**. Bobot dan skor smoke bukan bukti peningkatan kualitas dan tidak dipasang ke web app. Run utama memakai train-clean-100 dan dev-clean.
- Bobot awal dicatat sebagai kandidat step 0 pada validasi baru. Jika fine-tuning tidak mengalahkannya menurut loss validasi, export mempertahankan bobot awal tersebut.

## Cara menilai hasil

Pantau `val_si_sdri_active`, `val_absent_suppression`, dan `val_frame_energy_error_db` terhadap validasi step 0. Validasi ini berbeda dari run R1 terdahulu; nilainya tidak bisa langsung dibandingkan dengan tabel R1 lama.

Sesudah training, bandingkan checkpoint baru dengan R1 pada Mix.wav dan pemeriksaan dengar Nicker/Mond/Sam. Ukur pula clean/noisy held-out conversations dan uji jalur streaming sebelum mengganti model web app. Evaluation pack lama belum tersedia di `local_work/eval_pack`; trainer akan mencatat bahwa pack tersebut dilewati.

Mix.wav dan rekaman enrollment pengguna **tidak dipakai sebagai label training**: tidak ada sumber target bersih yang selaras. Run ini memperbaiki ekstraktor WeSep; discovery SepFormer tidak ikut dilatih. Belum ada klaim bahwa hasil Nicker atau noise lingkungan sudah membaik.

## Hasil run utama

Run selesai 12 Oktober 2026 pukul 03.09 WIB, 5.000 langkah dan export berhasil. Model yang diekspor adalah **checkpoint langkah 500**, dipilih berdasarkan loss validasi terendah, bukan bobot langkah 5.000.

| Ukuran validasi | R1 sebelum training | Checkpoint export, langkah 500 | Langkah 5.000 |
| --- | ---: | ---: | ---: |
| Loss, lebih rendah lebih baik | 0,720 | -0,985 | -0,799 |
| SI-SDR improvement saat target aktif, dB | 3,796 | 4,992 | 5,185 |
| Peredaman saat target diam, dB | 7,016 | 4,422 | 4,227 |
| Kesalahan energi per frame, dB | 10,907 | 8,731 | 7,498 |

Validasi menunjukkan perbaikan pemisahan dan energi, disertai penurunan peredaman saat target diam. Belum ada kesimpulan kualitas pada rekaman pengguna. Causality check trainer menunjukkan perubahan prefix 0; speed check full-sequence 0,077 detik per 1 detik audio pada MPS. Pemeriksaan tersebut dilakukan pada model dalam memori setelah training; checkpoint export langkah 500 perlu diperiksa melalui jalur inference/streaming secara terpisah.

Perbandingan rekaman pengguna disiapkan melalui `evaluation/wesep_eval/compare_user_checkpoints.py`. Script membaca checkpoint export langsung tanpa mengganti model aktif web app, memakai saved discovery dan enrollment yang sama untuk R1/model baru, streaming CPU 64 ms, serta penguatan volume yang sama dalam setiap pasangan A/B. Versi penguatan individual juga disediakan untuk memeriksa suara pelan.
