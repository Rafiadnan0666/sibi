# SIBI Translator: SIBI ke Bahasa Indonesia

Peragakan isyarat alfabet SIBI di depan webcam. Aplikasi mengenali hurufnya,
merangkai jadi kata, memprediksi kalimat Indonesia, lalu bisa diucapkan (TTS)
dan disimpan. Kalau kamera tidak bisa dipakai, ketik manual tetap bisa.

```
              WEBCAM
                 |
                 v
        Hand tracking (MediaPipe)
                 |
                 v
     Fitur kanonis v2 (puter + cerminkan tangan ke pose baku)
                 |
                 v
          SIBI recognition (TF.js, model sendi v2)
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

## Model v2: kenapa sekarang lebih akurat

Keluhan lama: tangan kiri vs kanan beda hasil, kamera miring beda hasil,
orang lain beda hasil, huruf mirip (M/N, S/T, E/O) ketukar.

Akar masalahnya ada di `src/lib/jointFeatures.ts` + `python/retrain_joint.py`:
fitur v1 hanya `tangan relatif pergelangan + bagi ukuran tangan`. Itu tahan
geser dan tahan jarak, tapi TIDAK tahan putaran kamera dan TIDAK tahan tangan
kiri/kanan. Jadi huruf yang sama dari tangan/sudut/orang berbeda = vektor
berbeda = huruf ketebak beda.

Perbaikan v2 (di `python/retrain_joint.py` fungsi `normalize`, dicerminkan
persis di `src/lib/jointFeatures.ts` fungsi `landmarksToFeatures`):

| Dari (v1) | Ke (v2) | Cara kerja |
|---|---|---|
| `kurang pergelangan, bagi ukuran` | sama, lalu **puter** | `phi = atan2(mx, my)` dari sendi jari tengah, seluruh tangan diputar balik sebesar itu sehingga jari tengah selalu menunjuk +Y. Kamera miring ±30° = vektor identik |
| `tangan kiri/kanan = data beda` | **cerminkan** | kalau pangkal jempol (sendi 1) di -X, seluruh X dikali -1. Tangan kiri jadi tangan kanan otomatis |
| `augmentasi putar ±14°, skala 0,9–1,1` | **augmentasi orang + sudut** | putar ±45°, skala 0,8–1,25, shear, jitter, plus tiap jari dipanjang/pendekkan ±10% (simulasi orang lain) |
| `MLP 256-128, dropout besar` | `MLP 256-128-64 + label smoothing 0,05 + class weight` | seimbang untuk kelas kecil (P cuma 7 sampel) |

Angka jujur (validasi 124 sampel, 24 huruf A-I K-Y):

| Kondisi | v2 |
|---|---|
| Bersih | **90,3%** (v1: 87,1%) |
| Tangan lain (mirror) | 90,3% (fitur persis sama) |
| Kamera miring ±30° | 90,3% (fitur persis sama) |
| Jarak dekat/jauh 0,7x–1,4x | 90,3% (fitur persis sama) |
| Simulasi orang lain (jari ±8%) | 86–90% |
| TF.js vs Keras | selisih maks 2,1e-7 (`scripts/verify-models.mjs`) |
| TS vs Python | selisih 0,0 (`scripts/verify-norm.mjs`) |

Plus penjaga huruf mirip di `src/routes/+page.svelte` (`MARGIN_MIN = 0,08`):
top1 harus mengalahkan top2 minimal 0,08, kalau tidak bingkai itu dibuang
(tidak masuk smoother, tidak jadi huruf). Lalu temporal smoother tetap jalan:
7 dari 12 bingkai harus setuju + keyakinan cukup + jeda antar huruf.

Batas jujur yang masih ada: data latih cuma 272 sampel (±11 per huruf), jadi
latar/cahaya ekstrem dan tangan yang sangat berbeda masih bisa salah. J dan Z
dikecualikan di semua model karena isyarat gerak, bukan handshape statis.
Tidak ada model yang 100%: yang ada adalah salah makin jarang.

## Naikkan akurasi dengan tangan sendiri (satu-satunya jalan jujur)

Di panel `Debug sendi dan citra (opsional)` ada `Rekam tangan sendiri`:
pilih huruf, tahan isyarat, tekan Rekam (1 sampel per ±150 ms, maks 2000).
Kumpulkan tiap huruf dari tangan kiri/kanan, sudut, dan jarak berbeda, lalu
`Unduh CSV` dan gabung ke latih ulang:

```sh
python python/retrain_joint.py --extra ./sibi-extra.csv
```

Lalu export ulang ke `static/models/sibi-joint` (perintah di bawah) dan
`npm run build`. Setiap 20–30 sampel per huruf dari tanganmu biasanya
menggeser akurasi lebih dari semua trik di atas.

## Dataset dan sumber data

Terlatih pada data SIBI terbuka (setara alfabet Kaggle
[`alvinbintang/sibi-dataset`](https://www.kaggle.com/datasets/alvinbintang/sibi-dataset),
yang butuh token login, lihat `static/models/README.md`):
272 sampel sendi dari
[AJustiago/SIBI-Recognition](https://github.com/AJustiago/SIBI-Recognition) (MIT,
kamus SIBI resmi).

Latih ulang:

```sh
pip install tensorflow scikit-learn pandas numpy
python python/retrain_joint.py          # MLP sendi v2, ±3 mnt CPU
$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python'
tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model joint_sm static/models/sibi-joint
```

Punya `kaggle.json` dan mau data Kaggle persisnya? Pakai CNN citra:

```sh
python python/train_sibi.py --data ./SIBI --epochs 30 --img 128
```

## Cara ketik dan hapus (3 jalan)

1. **Kamera**: tahan isyarat sampai bilah stabil 100% = 1 huruf masuk.
2. **Kotak "Ketik manual"** di bagian Kalimat Indonesia: klik lalu ketik A–Z.
3. **Keyboard fisik** (di mana saja kecuali saat kursor di kotak/form):
   `A–Z` tambah huruf, `Backspace` hapus huruf/kata terakhir,
   `Spasi` kunci huruf jadi kata, `Enter` ucapkan kalimat.
4. **Hapus**: tombol `Hapus`, atau gestur `Fist` (✊) di depan kamera,
   atau klik kata untuk menghapus kata itu saja.

## Yang dibuang (bloat) dan yang dibuat cepat

Dari `src/routes/+page.svelte` (sebelumnya 1030 baris):

| Dibuang | Alasan |
|---|---|
| Kursor udara + klik pakai pinch (`point`/`pinch`) | sumber salah klik; tombol dan keyboard lebih pasti |
| Kartu status 5 jari + grafik vektor 63 angka di layar utama | pindah ke panel `Debug sendi dan citra (opsional)`, tertutup default |
| Kanvas peta sendi + preview citra tiap frame | hanya digambar kalau panel debug dibuka |
| Bagian model 3 kartu besar + bagian riset panjang | diringkas jadi badge + 1 bagian pendek |

| Dibuat cepat | Dari → Ke | Cara kerja |
|---|---|---|
| Model saat startup | 3 model ±7 MB → 1 model 236 KB | `SibiEngine.load()` cuma memuat sendi v2; `loadSlot()` memuat baseline/citra sekali saat checkbox dicentang (`src/lib/sibiClassifier.ts`, `onToggleVote` di page) |
| Vote baseline/citra | selalu jalan tiap 200–250 ms → mati default | cuma inferensi kalau dicentang; fusi +0,07 per vote yang setuju tetap sama |
| Inferensi sendi | tiap 90 ms → tiap 110 ms | cukup responsif, CPU/GPU lebih adem |

## Stack

| Lapisan | Teknologi |
|---|---|
| Frontend | SvelteKit + TypeScript + Tailwind |
| Vision | MediaPipe `tasks-vision` HandLandmarker |
| SIBI ML | MLP sendi v2 (utama) + Conv1D baseline + MobileNetV2 (2 vote opsional), TensorFlow.js lokal |
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

## Kontrol kamera (pengatur tombol, BUKAN isyarat SIBI)

| Gerakan | Fungsi |
|---|---|
| Fist ✊ | Hapus huruf atau kata terakhir |
| Peace ✌️ | Spasi (kunci huruf jadi kata) |
| Thumbs up 👍 | Ucapkan kalimat |

## Struktur

```
src/lib/  sibiLabels, preprocessing, sibiClassifier (+smoothing, loadSlot),
          jointFeatures (kanonis v2), handTracker, languageModel, storage, tts
src/routes/+page.svelte   UI utama (kamera, kalimat + ketik manual, riwayat)
python/  retrain_joint.py (v2 kanonis), train_image.py, train_sibi.py, export + verifikasi
static/models/  model sendi v2 TF.js + 2 vote opsional + labels.json
scripts/  verifikasi numerik model di Node + verifikasi normalisasi
```

## Evaluasi (usulan laporan)

Akurasi, presisi, recall, F1 + confusion matrix (`metrics.json` tiap latih).
Ketahanan: tangan (mirror), sudut (±30°), jarak, simulasi orang, webcam.
Real time: FPS, latensi (`inferMs` live di UI), stabilitas prediksi.
UI: uji ketik manual vs kamera untuk kalimat yang sama.
