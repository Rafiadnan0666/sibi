# SIBI Translator: SIBI ke Bahasa Indonesia

Peragakan isyarat alfabet SIBI di depan webcam. Aplikasi mengenali hurufnya,
merangkai jadi kata, memprediksi kalimat Indonesia, lalu bisa diucapkan (TTS)
dan disimpan. Isyarat SIBI adalah inputnya. Gerakan jari (point, pinch, fist)
hanya menekan tombol di layar, bukan pengganti bahasa isyarat.

```
              WEBCAM
                 |
                 v
        Hand tracking (MediaPipe)
                 |
                 v
          SIBI recognition (TF.js)
                 |
                 v
       SIGN jadi WORD atau PHRASE
                 |
                 v
       +---------+---------+
       v                   v
  Text prediction       Sentence
       |                   |
       +---------+---------+
                 |
                 v
           Indonesian text ke suara atau simpanan
```

## Dataset dan sumber data

Terlatih pada data SIBI terbuka (setara alfabet Kaggle
[`alvinbintang/sibi-dataset`](https://www.kaggle.com/datasets/alvinbintang/sibi-dataset),
yang butuh token login, lihat `static/models/README.md`):
272 sampel sendi + 480 foto, 24 huruf statis A-I dan K-Y, dari
[AJustiago/SIBI-Recognition](https://github.com/AJustiago/SIBI-Recognition) (MIT,
kamus SIBI resmi). Punya `kaggle.json`? Latih ulang persis di dataset Kaggle via
`python/train_sibi.py --data ./SIBI`.

## Stack

| Lapisan | Teknologi |
|---|---|
| Frontend | SvelteKit + TypeScript + Tailwind |
| Vision | MediaPipe `tasks-vision` HandLandmarker |
| SIBI ML | MLP sendi + Conv1D baseline + MobileNetV2, semua jadi TensorFlow.js lokal |
| Citra | ROI crop, resize 128, normalisasi, augmentasi, uji grayscale |
| Bahasa | n-gram dan frekuensi Indonesia (tanpa server) |
| Suara | Web Speech API `speechSynthesis` (`id-ID`) |
| Simpan | IndexedDB (riwayat, pengaturan) + fallback localStorage |
| Hosting | Rp0 di Cloudflare Pages (static) |

## Menjalankan

```sh
npm install
npm run dev      # dev server
npm run check    # type-check (harus 0 error)
npm run build    # output statis di build/
```

## Model: 3 yang nyata

| Model | Input | Val acc | Latih dengan |
|---|---|---|---|
| MLP sendi (primer) | 63 angka sendi ternormalisasi | **87,1%** | `python/retrain_joint.py` |
| Conv1D baseline (vote kedua) | sendi piksel mentah | 74,2% | AJustiago/SIBI-Recognition (MIT) |
| MobileNetV2 (cek visual) | ROI 128x128 | 73,8% | `python/train_image.py` |

Fusi di browser: keyakinan sendi naik 7% untuk tiap model lain yang setuju,
lalu temporal smoothing, lalu huruf. J dan Z dikecualikan (isyarat gerak).
Verifikasi numerik TF.js vs Keras: selisih maks 7e-7 (`scripts/verify-models.mjs`).
Detail: `static/models/README.md`.

## Deploy (Cloudflare Pages, Rp0)

1. Push ke GitHub. 2. Cloudflare Pages, Create project, pilih repo ini.
3. Build command `npm run build`, output `build`. 4. Selesai, dapat `*.pages.dev`.

## Kontrol kamera (pengatur tombol, BUKAN isyarat SIBI)

| Gerakan | Fungsi |
|---|---|
| Point | Gerakkan kursor udara |
| Pinch | Klik atau pilih prediksi |
| Fist | Hapus huruf atau kata terakhir |
| Peace | Spasi (kunci huruf jadi kata) |
| Thumbs up | Ucapkan kalimat |

## Struktur

```
src/lib/  sibiLabels, preprocessing, sibiClassifier (+smoothing),
          jointFeatures, handTracker, languageModel, storage, tts
src/routes/+page.svelte   UI utama (kamera, kalimat, prediksi, riwayat)
python/  retrain_joint.py, train_image.py, train_sibi.py, export + verifikasi
static/models/  3 model TF.js + labels.json
scripts/  verifikasi numerik model di Node
```

## Evaluasi (usulan laporan)

Akurasi, presisi, recall, F1 + confusion matrix. Ketahanan: latar, cahaya, jarak,
user, webcam. Real time: FPS, latensi, stabilitas prediksi. UI: akurasi gestur
dan angka salah klik.
