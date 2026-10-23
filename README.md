# SIBI Translator: SIBI ke Bahasa Indonesia

Peragakan isyarat alfabet SIBI di depan webcam. Aplikasi mengenali hurufnya,
merangkai jadi kata, memprediksi kalimat Indonesia, lalu bisa diucapkan
(Text-to-Speech) dan disimpan. Kalau kamera tidak bisa dipakai, ketik manual
tetap bisa.

Penggunaan singkat: nyalakan kamera, tahan isyarat sampai bilah stabil penuh,
huruf masuk otomatis. Kumpulkan huruf jadi kata (Spasi), kumpulkan kata jadi
kalimat, tekan Ucapkan untuk suara.

---

## Daftar isi

1. Ringkasan alur data
2. Pelacakan tangan (MediaPipe)
3. Ekstraksi fitur kanonis v2
4. Klasifikasi dan fusi (3 model)
5. Penghalusan temporal (smoother)
6. Penyusunan teks (huruf, kata, kalimat)
7. Model bahasa (prediksi kata)
8. Kontrol gerakan tangan (gesture)
9. Keyboard dan input manual
10. Tata letak antarmuka
11. Pengaturan (settings)
12. Penyimpanan (storage)
13. Text-to-Speech
14. Perekam sampel tangan sendiri
15. Struktur proyek
16. Menjalankan dan build
17. Pelatihan ulang model
18. Evaluasi
19. Pemecahan masalah

---

## 1. Ringkasan alur data

```
WEBCAM
  |
  v
Hand tracking (MediaPipe HandLandmarker, 1 tangan)
  |
  v
Fitur kanonis v2 (wrist-relative, scale, rotasi, cerminkan tangan)
  |
  v
SIBI recognition (TF.js, model sendi v2 + 2 vote opsional)
  |
  v
SIGN menjadi HURUF (commit setelah stabil)
  |
  v
HURUF menjadi KATA (Spasi / peace gesture)
  |
  v
KATA menjadi KALIMAT (kumpulan kata)
  |
  v
Prediksi kata berikutnya + auto-koreksi
  |
  v
Teks Indonesia ke suara (TTS) atau simpan (IndexedDB)
```

Semua inferensi berjalan di browser. Tidak ada server, tidak ada data yang
dikirim keluar.

---

## 2. Pelacakan tangan (MediaPipe)

File: `src/lib/handTracker.ts`

- Model: `@mediapipe/tasks-vision` HandLandmarker, mode VIDEO, 1 tangan
  (numHands = 1). Deteksi satu tangan saja supaya tangan kedua tidak
  mencemari klasifikasi.
- Delegate: GPU dulu, fallback CPU kalau GPU gagal.
- Confidence: minHandDetection 0.5, minHandPresence 0.6, minTracking 0.6.
- Timestamp monotonic: rAF timestamp dijaga agar tidak menurun (syarat
  VIDEO mode).
- Hanya tangan dengan skor handedness >= 0.5 yang diterima.
- Pemilihan tangan: jika lebih dari satu terdeteksi, yang skor tertinggi
  yang dipakai.
- Smoothing temporal: landmark dihaluskan dengan alpha 0.45 (exponential
  moving average) untuk mengurangi jitter antar frame. Reset saat tangan
  hilang.
- ROI box: bounding box piksel dari landmark (min/max x,y).
- Overlay: digambar di canvas overlay (cermin X supaya cocok dengan video
  yang di-mirror via CSS).

Fungsi utama:

- `ensureHandLandmarker()` memuat model (lazy, CDN wasm).
- `detectHands(video, nowMs)` mengembalikan `{ hands, raw }`.
- `drawHandOverlay(ctx, hands, W, H)` menggambar skeleton + box.
- `decodeControlGesture(lm)` mengubah landmark jadi gestur kontrol.

---

## 3. Ekstraksi fitur kanonis v2

File: `src/lib/jointFeatures.ts`, fungsi `landmarksToFeatures`.

Kontrak fitur harus sama persis dengan `python/retrain_joint.py` fungsi
`normalize`. 21 sendi x 3 koordinat = 63 angka.

Langkah:

1. Pixel coords: `P[i] = (x_i * W, y_i * H, z_i)` (z tetap raw MediaPipe).
2. Wrist-relative: `d[i] = P[i] - P[0]`.
3. Scale: `size = ||P[9] - P[0]|| + 1e-6` (pergelangan ke middle MCP).
   `d[i] /= size`.
4. Rotasi XY: `phi = atan2(d[9].x, d[9].y)`, putar seluruh tangan supaya
   jari tengah menunjuk +Y. Kamera miring +/-30 derajat menghasilkan vektor
   identik.
5. Cerminkan X: kalau pangkal jempol (sendi 1) di -X, seluruh X dikali -1.
   Tangan kiri jadi tangan kanan otomatis.
6. Flatten ke 63 angka.

Fitur ini tahan geser, jarak, putaran kamera, dan tangan kiri/kanan. Yang
tersisa adalah perbedaan anatomi jari antar orang, disimulasikan saat latih
dengan variasi panjang jari +/-10%.

Verifikasi: `scripts/verify-norm.mjs` memastikan TS dan Python menghasilkan
angka identik (selisih 0.0).

---

## 4. Klasifikasi dan fusi (3 model)

File: `src/lib/sibiClassifier.ts`, kelas `SibiEngine`.

Tiga slot model:

| Slot | Key | Model | Val/Test | Status default |
|---|---|---|---|---|
| 0 (utama) | joint | MLP kanonis v2 (63 fitur) | val 90,3% | dimuat saat startup |
| 1 (vote) | baseline | Conv1D raw pixel joints (upstream) | val 74,2% | lazy (checkbox) |
| 2 (vote) | image | MobileNetV2 128px pada foto SIBI | test 84,8% | lazy (checkbox) |

- `load()` memuat model utama saja saat startup (cepat). Vote dimuat saat
  checkbox dicentang (`loadSlot`).
- `predictJoint(feats)` mengembalikan label, confidence, top3, latency.
- `predictBaseline(px)` Conv1D pada 63,1.
- `predictImage(tensor)` MobileNetV2 pada 128x128x3 [-1,1].

Fusi confidence di `+page.svelte` `classifyFrame`:

```
fused = joint.conf + (baseline agree ? 0.07 : 0) + (image agree ? 0.07 : 0)
fused = min(0.99, fused)
```

Vote image dijalankan pada ROI asli dan ROI yang dicerminkan horizontal,
lalu dipilih yang confidence-nya lebih tinggi (invarian tangan kiri/kanan).

Guard huruf mirip: `MARGIN_MIN = 0.2`. Top1 harus mengalahkan top2 minimal
0.2, kalau tidak bingkai dibuang (tidak masuk smoother).

---

## 5. Penghalusan temporal (smoother)

File: `src/lib/sibiClassifier.ts`, kelas `PredictionSmoother`.

- Window 15 frame, minFrames 9, minConfidence 0.6 (default), cooldown 1500 ms.
- Push label + confidence. Hitung mayoritas dalam window.
- Commit hanya jika: mayoritas >= 9 frame, rata-rata confidence >= minConf,
  5 frame terakhir semua label yang sama, dan cooldown terpenuhi.
- Setelah commit, window direset, progress 0.
- `progress` (0..1) dipakai untuk bilah stabil di UI.

Motion gate di `classifyFrame`: rata-rata perpindahan landmark antar frame
harus <= 0.02, kalau tidak frame dibuang (tangan sedang bergerak).

Area gate: box tangan harus 0.06-0.8 dari area frame. Lighting gate:
luminansi ROI 45-215.

---

## 6. Penyusunan teks

State di `+page.svelte`:

- `prefix`: buffer huruf yang sedang diketik (maks 24).
- `words`: array kata yang sudah dikunci.
- `recentChars`: 10 huruf terakhir (debug).
- `sentence`: derived `toSentence(words + prefix)`.

Operasi:

- `commitChar(c)`: tambah 1 huruf ke prefix.
- `deleteLast()`: hapus 1 huruf dari prefix; kalau prefix kosong, hapus kata
  terakhir.
- `commitSpace()`: kunci prefix jadi kata (dengan auto-koreksi opsional),
  prefix dikosongkan.
- `removeWord(i)`: hapus kata ke-i.
- `moveWord(i, dir)`: geser kata kiri/kanan.
- `applySuggestion(s)`: ambil saran jadi kata.
- `clearAll()`: kosongkan semua.
- `speak()`: TTS kalimat.
- `saveCurrent()`: simpan ke riwayat.

Auto-koreksi (`autoCorrectWord` di `languageModel.ts`): snap kata ke kata
kamus terdekat (Levenshtein, threshold <= 3 dan <= 1/4 panjang). Toggle di
pengaturan, default aktif.

---

## 7. Model bahasa

File: `src/lib/languageModel.ts`.

- VOCAB: ~60 kata Indonesia umum dengan frekuensi.
- BIGRAMS: kata lanjutan yang likely setelah kata tertentu.
- `completeWord(prefix)`: kata kamus yang diawali prefix.
- `predictNext(words)`: berdasarkan kata terakhir, bigram + fallback
  frekuensi.
- `suggest(words, prefix)`: kalau prefix non-kosong, mode "lengkapi kata";
  kalau tidak, mode "kata berikutnya".
- `toSentence(words)`: gabung dengan spasi, kapitalisasi awal.
- `autoCorrectWord(input)`: Levenshtein ke kata kamus terdekat.

---

## 8. Kontrol gerakan tangan (gesture)

File: `src/lib/handTracker.ts`, `decodeControlGesture`.

Deteksi jari extended via jarak tip ke wrist vs pip ke wrist. Gestur:

| Gestur | Kondisi | Fungsi |
|---|---|---|
| fist | semua jari tertutup | hapus huruf/kata terakhir |
| peace | telunjuk + tengah naik | spasi (kunci kata) |
| thumbsup | hanya jempol naik | ucapkan kalimat |
| palm | >= 4 jari naik | jeda (skip klasifikasi) |
| point | hanya telunjuk naik | (tidak dipakai untuk commit) |
| pinch | thumb + index dekat | (tidak dipakai) |

Guard: gestur harus tahan 8 frame berturut-turut sebelum aksi dijalankan,
plus cooldown (fist 1800 ms, peace 1800 ms, thumbsup 2500 ms). Ini mencegah
isyarat transisi memicu hapus/spasi/ucap.

---

## 9. Keyboard dan input manual

- Keyboard fisik (di mana saja kecuali kursor di input/textarea/select):
  A-Z tambah huruf, Backspace hapus, Spasi kunci kata, Enter ucapkan.
  Modifier (Ctrl/Meta/Alt) diabaikan.
- Kotak "Ketik manual": hanya A-Z yang lolos, maks 24 char, otomatis
  uppercase.
- Tombol di layar: Spasi, Hapus, Ucapkan, Simpan, Bersihkan.

---

## 10. Tata letak antarmuka

Layout flex (responsive):

```
+--------------------------------------------------------------+
| HEADER: judul + badge model + status tracking                 |
+--------------------------------------------------------------+
| +--------+  +-----------------------------+  +--------+       |
| | KIRI   |  | TENGAH (kamera besar)        |  | KANAN  |       |
| | hapus  |  | video + overlay + badge huruf|  | masuk  |       |
| | huruf  |  | kontrol kamera              |  | huruf  |       |
| | riwayat|  |                              |  | ketik  |       |
| | huruf  |  |                              |  | manual |       |
| +--------+  +-----------------------------+  +--------+       |
|                                                              |
| BAWAH: saran kata (3 kolom)                                  |
+--------------------------------------------------------------+
| KONTROL KAMERA: fist/peace/thumbsup/palm                      |
| RIWAYAT, MODEL STATUS, DATA/MODEL, DEBUG (opsional)            |
+--------------------------------------------------------------+
```

- Kamera: aspect-video, max-h 80vh, mirror via CSS.
- Badge huruf besar di pojok kanan atas kamera saat tangan terdeteksi.
- Hint framing (terlalu jauh/kecil, gelap/terang) di atas kamera.
- Status badge: FPS, latensi sendi, jumlah tangan, gestur aktif.

---

## 11. Pengaturan (settings)

Di panel "Pengaturan pengenalan":

- Keyakinan minimum (0.3-0.9, step 0.05). Default 0.8.
- Jeda antar huruf (600-2500 ms). Default 1500.
- Vote baseline (Conv1D) on/off. Default on.
- Cek silang CNN citra on/off. Default on.
- Uji grayscale on/off. Default off.
- Normalisasi citra: -1..1 (MobileNet) atau 0..1. Default -1..1.
- Koreksi otomatis kata on/off. Default on.
- Tombol "Pulihkan default": reset semua ke nilai awal.

Pengaturan disimpan di IndexedDB (fallback localStorage) via `src/lib/storage.ts`.

---

## 12. Penyimpanan (storage)

File: `src/lib/storage.ts`.

- `loadHistory` / `saveHistoryEntry` / `clearHistory` / `deleteHistoryEntry`:
  riwayat kalimat (teks + kata-kata + timestamp).
- `loadSetting(key, default)` / `saveSetting(key, value)`: pengaturan.
- Backend: IndexedDB, fallback localStorage.

---

## 13. Text-to-Speech

File: `src/lib/tts.ts`.

- Web Speech API `speechSynthesis`, lang `id-ID`.
- Pilih voice Indonesia jika tersedia, fallback voice default.
- `speakIndonesian(text, {rate, pitch})`, `stopSpeaking()`.
- Dipicu oleh tombol Ucapkan atau gestur thumbsup.

---

## 14. Perekam sampel tangan sendiri

Di panel "Debug sendi dan citra (opsional)":

1. Pilih huruf (A-Y, 24 huruf).
2. Tekan "Rekam": setiap ~150 ms 1 sampel (vektor 63 kanonis) disimpan.
   Maks 2000 sampel.
3. "Unduh CSV": file `sibi-extra.csv` (kolom class_type + f0..f62).
4. Latih ulang: `python python/retrain_joint.py --extra ./sibi-extra.csv`.
5. Export ulang ke `static/models/sibi-joint`, lalu `npm run build`.

---

## 15. Struktur proyek

```
src/lib/
  sibiLabels.ts        label 24 huruf, ukuran input 128, URL model
  preprocessing.ts     cropROI, preprocessToTensor, drawPreview
  sibiClassifier.ts    SibiEngine (3 slot), PredictionSmoother
  jointFeatures.ts     landmarksToFeatures (kanonis v2), rawPixelFeatures
  handTracker.ts       MediaPipe wrapper, decodeControlGesture, smoothing
  languageModel.ts     n-gram Indonesia, autoCorrectWord
  storage.ts           IndexedDB riwayat + settings
  tts.ts               Web Speech API
  types.ts             tipe TS
  index.ts             re-export
src/routes/
  +page.svelte         UI utama
  +layout.svelte       layout
python/
  retrain_joint.py     latih MLP sendi v2 (kanonis)
  train_sibi.py        latih CNN citra (Kaggle)
  train_sibi_advanced.py  MobileNetV2 80/20
  train_image.py       latih CNN citra sederhana
  export_savedmodel.py export SavedModel
  dump_verify.py       verifikasi model
scripts/
  verify-norm.mjs      verifikasi normalisasi TS vs Python
  verify-models.mjs    verifikasi TF.js vs Keras
static/models/
  sibi-joint/          model utama (236 KB)
  sibi-joint-baseline/ vote baseline (4 MB)
  sibi-advanced/       vote citra (12 MB)
  sibi-image/          mirror backward-compat
  README.md            info dataset
```

---

## 16. Menjalankan dan build

```sh
npm install
npm run dev      # dev server (http://localhost:5173)
npm run check    # type-check (harus 0 error)
npm run build    # output statis di build/
npm run preview  # preview build
```

Catatan: buka via http (dev/preview), bukan file://, supaya fetch model dan
kamera berfungsi.

---

## 17. Pelatihan ulang model

Dependensi Python:

```sh
pip install tensorflow scikit-learn pandas numpy
```

MLP sendi v2:

```sh
python python/retrain_joint.py
$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python'
tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model joint_sm static/models/sibi-joint
```

Dengan data sendiri:

```sh
python python/retrain_joint.py --extra ./sibi-extra.csv
```

CNN citra (Kaggle, butuh kaggle.json):

```sh
python python/train_sibi.py --data ./SIBI --epochs 30 --img 128
```

CNN citra advanced (80/20):

```sh
python python/train_sibi_advanced.py --data "C:\path\SIBI" --out ./training_output/sibi_advanced_80_20 --epochs 30 --img 128 --batch 32
```

---

## 18. Evaluasi

- Akurasi, presisi, recall, F1 + confusion matrix (`metrics.json` tiap latih).
- Ketahanan: tangan (mirror), sudut (+/-30 derajat), jarak, simulasi orang,
  webcam.
- Real time: FPS, latensi (inferMs live di UI), stabilitas prediksi.
- UI: uji ketik manual vs kamera untuk kalimat yang sama.

Hasil v2 (validasi 124 sampel, 24 huruf A-I K-Y):

| Kondisi | Akurasi |
|---|---|
| Bersih | 90,3% |
| Tangan lain (mirror) | 90,3% |
| Kamera miring +/-30 derajat | 90,3% |
| Jarak 0,7x-1,4x | 90,3% |
| Simulasi orang lain (jari +/-8%) | 86-90% |

TF.js vs Keras selisih maks 2,1e-7. TS vs Python selisih 0,0.

---

## 19. Pemecahan masalah

- Model gagal dimuat: cek koneksi internet (model diunduh dari CDN/storage).
  Tombol "Coba lagi" di bagian Model.
- Kamera kosong: tutup aplikasi lain yang memakai kamera, atau ganti
  perangkat di dropdown.
- Tidak ada tangan terdeteksi: cahaya cukup, latar polos, tangan di tengah.
- Huruf tidak masuk: tahan isyarat sampai bilah stabil 100%; hindari gerakan.
- Terlalu gelap/terang: tambah cahaya atau kurangi exposure.
- Akurasi rendah: rekam tangan sendiri (panel debug) dan latih ulang.
- Riwayat tidak tersimpan: periksa IndexedDB/localStorage tidak diblokir.