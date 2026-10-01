# 🤟 SIBI Translator — SIBI → Bahasa Indonesia

**Asisten komunikasi SIBI (MVP pengenalan alfabet)**: peragakan isyarat alfabet SIBI di
depan webcam → dikenali jadi huruf → dirangkai jadi kata → diprediksi jadi kalimat
Indonesia → bisa diucapkan (TTS) dan disimpan. Isyarat SIBI adalah *input*-nya;
gestur jari (point/pinch/fist/…) hanya *pengatur UI*, bukan pengganti bahasa isyarat.

> Klaim jujur untuk laporan: ini **sistem pengenalan alfabet SIBI + prediksi kalimat**,
> bukan "penerjemah SIBI penuh". Kata/kalimat SIBI dinamis butuh dataset + model kedua.

```
              WEBCAM
                 │
                 ▼
        Hand + body detection (MediaPipe)
                 │
                 ▼
          SIBI recognition (CNN, TF.js)
                 │
                 ▼
       SIGN → WORD / PHRASE
                 │
                 ▼
       ┌─────────┴─────────┐
       ▼                   ▼
  Text prediction       Sentence
       │                   │
       └─────────┬─────────┘
                 ▼
           Indonesian text → 🔊 Speak / 💾 Save
```

## Dataset & sumber data

Terlatih pada data SIBI terbuka (setara alfabet Kaggle
[`alvinbintang/sibi-dataset`](https://www.kaggle.com/datasets/alvinbintang/sibi-dataset),
yang butuh token login — lihat `static/models/README.md`):
272 sampel sendi + 480 foto, 24 huruf statis A–I/K–Y, dari
[AJustiago/SIBI-Recognition](https://github.com/AJustiago/SIBI-Recognition) (MIT,
kamus SIBI resmi). Punya `kaggle.json`? Latih ulang persis di dataset Kaggle via
`python/train_sibi.py --data ./SIBI`.

## Stack

| Lapisan | Teknologi |
|---|---|
| Frontend | SvelteKit + TypeScript + Tailwind (neobrutalism ☀️ kuning/putih) |
| Vision | MediaPipe `tasks-vision` HandLandmarker (landmark + gestur kontrol) |
| SIBI ML | CNN / MobileNetV2 transfer learning → TensorFlow.js (browser, lokal) |
| Citra | ROI crop → resize 128 → normalisasi → augmentasi → (uji grayscale) |
| Bahasa | n-gram / frekuensi Indonesia (tanpa server, tanpa LLM) |
| Suara | Web Speech API `speechSynthesis` (`id-ID`) |
| Simpan | IndexedDB (riwayat, pengaturan) + fallback localStorage |
| Hosting | Rp0 — Cloudflare Pages (static) |

## Menjalankan

```sh
npm install
npm run dev      # dev server
npm run check    # type-check (harus 0 error)
npm run build    # output statis di build/
```

## Model — 3 yang nyata, 0 yang demo

| Model | Input | Val acc | Latih dengan |
|---|---|---|---|
| 🧠 MLP sendi (primer) | 63 angka sendi ternormalisasi | **87,1%** | `python/retrain_joint.py` |
| 🧪 Conv1D baseline (vote-2) | sendi piksel mentah | 74,2% | AJustiago/SIBI-Recognition (MIT) |
| 📷 MobileNetV2 (cross-check) | ROI 128×128 | 73,8% | `python/train_image.py` |

Fusi di browser: keyakinan sendi +7% per model yang setuju → temporal smoothing →
huruf. J/Z dikecualikan (isyarat dinamis). Verifikasi numerik TF.js vs Keras:
selisih maks 7e-7 (`scripts/verify-models.mjs`). Detail: `static/models/README.md`.

## Deploy (Cloudflare Pages, Rp0)

1. Push ke GitHub. 2. Cloudflare Pages → *Create project* → repo ini.
3. Build command `npm run build`, output `build`. 4. Selesai — dapat `*.pages.dev`.

## Gestur kontrol (pengatur UI, BUKAN isyarat SIBI)

| Gestur | Fungsi |
|---|---|
| ☝ Point | Gerakkan kursor udara |
| 🤏 Pinch | Klik / pilih prediksi |
| ✊ Fist | Hapus huruf/kata terakhir |
| ✌️ Peace | Spasi (kunci huruf → kata) |
| 🖐️ Palm | Jeda / lanjut kamera |
| 👍 Thumbs up | Ucapkan kalimat |

## Struktur

```
src/lib/  sibiLabels · preprocessing · sibiClassifier (+smoothing) ·
          handTracker · languageModel · storage · tts
src/routes/+page.svelte   UI utama (kamera, kalimat, prediksi, riwayat)
python/train_sibi.py      training CNN + evaluasi + export TF.js
static/models/            slot model.json
```

## Evaluasi (usulan laporan)

Akurasi/presisi/recall/F1 + confusion matrix · robustness (latar, cahaya, jarak,
user, webcam) · FPS/latensi/stabilitas real-time · akurasi gestur UI &
accidental-click rate.
