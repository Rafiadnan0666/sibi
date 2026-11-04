# SIBI Translator: SIBI ke Bahasa Indonesia

Peragakan isyarat alfabet SIBI di depan webcam. Aplikasi mengenali hurufnya,
merangkai huruf jadi kata, memprediksi kata berikutnya, lalu membacakan
kalimat dengan Text-to-Speech. Kalau kamera tidak bisa dipakai, input manual
tetap bisa.

Cara pakai singkat:

1. Nyalakan kamera.
2. Tahan satu isyarat sampai kartu huruf di pojok kanan menunjukkan
   **Terkunci** (biasanya 0,4-0,8 detik).
3. Ulangi untuk huruf berikutnya. Tekan **Spasi** (atau gestur peace) untuk
   mengunci kata.
4. Tekan **Ucapkan** (atau gestur thumbs up) untuk membacakan kalimat.

J dan Z adalah isyarat dinamis: gambar huruf J dengan gerakan kelingking,
atau Z dengan gerakan telunjuk. Kalau gerakan sulit terbaca, tombol
**J (gerakan)** dan **Z (gerakan)** di panel teks bisa dipakai.

---

## Daftar isi

1. Ringkasan alur data
2. Peta matematika: di mana rumusnya?
3. Lapisan 1: kamera, 21 sendi, dan kotak tangan
4. Lapisan 2: fitur kanonis 63 angka
5. Lapisan 3: model neural network dan fusi
6. Lapisan 4: smoother temporal (kunci huruf)
7. Lapisan 5: huruf dinamis J dan Z
8. Gestur kontrol (bukan isyarat SIBI)
9. Citra ROI dan normalisasi gambar
10. Bahasa: kata, kalimat, dan auto-koreksi
11. Waktu, FPS, dan anggaran frame
12. Penyusunan teks dan antarmuka
13. Pengaturan dan penyimpanan
14. Text-to-Speech
15. Perekam sampel tangan sendiri
16. Struktur proyek
17. Menjalankan dan build
18. Pelatihan ulang dan matematika pelatihan
19. Evaluasi dan batasan
20. Pemecahan masalah

---

## 1. Ringkasan alur data

```text
WEBCAM (640x480 atau lebih)
  |
  |  MediaPipe HandLandmarker, 1 tangan
  v
21 landmark (x, y, z) + skor handedness
  |
  |  EMA alpha = 0.6, pilih tangan skor tertinggi
  v
Landmark halus + bounding box piksel
  |
  |  Gerbang area 0.04..0.90 dan gerbang gerak 0.065
  v
Fitur kanonis 63 angka
  |
  |  MLP 63 -> 256 -> 128 -> 64 -> 24 softmax
  v
P(joint), P(baseline), P(image)  +  fusi keyakinan
  |
  |  Smoother: window 5, weighted vote, hold adaptif 350..805 ms
  v
SATU HURUF (commit)
  |
  |  prefix huruf -> kata (Spasi) -> kalimat
  v
Saran kata (n-gram + frekuensi) dan auto-koreksi (Levenshtein)
  |
  v
TTS Bahasa Indonesia / simpan ke IndexedDB
```

Semua inferensi berjalan di browser. Tidak ada server aplikasi dan tidak ada
data yang dikirim keluar. Model diunduh dari folder `static/models`; WASM
MediaPipe diunduh dari CDN.

---

## 2. Peta matematika: di mana rumusnya?

Tabel ini adalah indeks cepat. Bagian setelahnya menjelaskan setiap baris.

| Tahap | File | Fungsi | Input -> Output | Matematika inti |
|---|---|---|---|---|
| Pelacakan tangan | `src/lib/handTracker.ts` | `detectHands` | video -> 21 landmark | MediaPipe, pilih skor max, EMA |
| Kotak tangan | `src/lib/handTracker.ts` | `handBoundingBox` | landmark -> box piksel | min/max, lebar x tinggi |
| Gerbang area | `src/routes/+page.svelte` | `classifyFrame` | box -> lolos/tidak | `A = w*h/(W*H)`, `0.04 <= A <= 0.9` |
| Gerbang gerak | `src/routes/+page.svelte` | `classifyFrame` | landmark 2 frame | rata-rata jarak Euclidean |
| Fitur kanonis | `src/lib/jointFeatures.ts` | `landmarksToFeatures` | 21x3 -> 63 | translasi, skala, rotasi, mirror |
| Fitur baseline | `src/lib/jointFeatures.ts` | `rawPixelFeatures` | 21x3 -> 63 | koordinat piksel mentah |
| Sudut sendi | `src/lib/jointFeatures.ts` | `jointAngleDeg` | 3 titik -> derajat | `acos(dot/(|a||b|))` |
| Deteksi jari | `src/lib/handTracker.ts` | `fingerExtended`, `fingerPattern` | landmark -> boolean | rasio jarak ke pergelangan |
| Klasifikasi sendi | `src/lib/sibiClassifier.ts` | `predictJoint` | 63 -> 24 probabilitas | dense + ReLU + softmax |
| Klasifikasi baseline | `src/lib/sibiClassifier.ts` | `predictBaseline` | 63x1 -> 24 | Conv1D |
| Klasifikasi citra | `src/lib/sibiClassifier.ts` | `predictImage` | 128x128x3 -> 24 | MobileNetV2 + softmax |
| Fusi model | `src/routes/+page.svelte` | `classifyFrame` | 3 prediksi -> 1 skor | `p + 0.07*setuju + 0.07*setuju` |
| Smoother | `src/lib/predictionSmoother.ts` | `push` | frame -> huruf | weighted vote + hold adaptif |
| Huruf J/Z | `src/lib/motionLetters.ts` | `detectJ`, `detectZ` | jejak ujung jari | vektor, sudut 45 derajat |
| Gestur | `src/lib/handTracker.ts` | `decodeControlGesture` | landmark -> gestur | rasio jarak, aturan jari |
| ROI citra | `src/lib/preprocessing.ts` | `cropROI` | box -> canvas persegi | `side = 1.5*max(w,h)` |
| Tensor citra | `src/lib/preprocessing.ts` | `preprocessToTensor` | canvas -> Float32Array | resize 128, normalisasi |
| Saran kata | `src/lib/languageModel.ts` | `completeWord`, `predictNext` | prefix/kata -> skor | frekuensi + bigram |
| Auto-koreksi | `src/lib/languageModel.ts` | `autoCorrectWord` | kata -> kata | Levenshtein |
| Waktu/FPS | `src/routes/+page.svelte` | `loop`, `classifyFrame` | timestamp -> jadwal | periode, EMA |
| Pelatihan | `python/retrain_joint.py` | `normalize`, `augment` | CSV -> model | sama seperti runtime + augmentasi |

Notasi yang dipakai di dokumen ini:

```text
i, j          indeks sendi (0..20) atau indeks frame
t             waktu dalam milidetik
W, H          lebar dan tinggi frame video (piksel)
p_i           landmark ternormalisasi MediaPipe: (x_i, y_i)
P_i           landmark piksel: (x_i*W, y_i*H, z_i)
||v||         panjang Euclidean: sqrt(vx^2 + vy^2 + vz^2)
1[kondisi]    bernilai 1 kalau kondisi benar, 0 kalau salah
clamp(x,a,b)  max(a, min(b, x))
argmax        indeks nilai terbesar
```

---

## 3. Lapisan 1: kamera, 21 sendi, dan kotak tangan

File: `src/lib/handTracker.ts`

### 3.1 Koordinat MediaPipe

MediaPipe mengembalikan 21 sendi per tangan. Koordinat `x` dan `y` sudah
dinormalisasi ke `0..1` terhadap lebar dan tinggi gambar:

```text
x = 0 di kiri frame, x = 1 di kanan frame
y = 0 di atas frame,  y = 1 di bawah frame   <-- y bertambah ke bawah
z = kedalaman relatif MediaPipe (bukan piksel)
```

Indeks penting:

```text
0  pergelangan (wrist)
1  pangkal jempol (thumb_cmc)
4  ujung jempol
5  pangkal telunjuk (index_mcp)
8  ujung telunjuk
9  pangkal jari tengah (middle_mcp)
12 ujung jari tengah
16 ujung jari manis
20 ujung kelingking
```

Konversi ke piksel di `jointFeatures.ts`:

```text
P_i = (x_i * W, y_i * H, z_i)
```

Perhatikan: `x` dan `y` dikali ukuran frame, tetapi `z` tetap nilai mentah
MediaPipe. Ini disengaja agar sama dengan data latih Python.

### 3.2 Kotak tangan dan gerbang area

`handBoundingBox` mengambil nilai minimum dan maksimum dari 21 landmark:

```text
minX = min(x_i), maxX = max(x_i)
minY = min(y_i), maxY = max(y_i)

box.x = minX * W
box.y = minY * H
box.w = max(1, (maxX - minX) * W)
box.h = max(1, (maxY - minY) * H)
```

`max(1, ...)` mencegah kotak berukuran nol saat tangan sangat kecil.

Luas relatif terhadap frame:

```text
A = (box.w * box.h) / (W * H)
```

`classifyFrame` hanya memproses tangan jika:

```text
0.04 <= A <= 0.9
```

Artinya tangan harus menempati minimal 4% dan maksimal 90% area frame.

Contoh pada frame 640x480 (307.200 piksel):

```text
A = 0.04 -> 12.288 piksel^2 -> kotak persegi sekitar 111x111 piksel
A = 0.90 -> 276.480 piksel^2
```

Kalau `A < 0.04`, UI menampilkan "Terlalu jauh atau kecil. Dekatkan tangan
ke kamera." Gerbang ini mencegah tangan yang terlalu kecil menjadi huruf
sampah.

### 3.3 Smoothing EMA (exponential moving average)

Landmark mentah bergetar sedikit setiap frame. `detectHands` menjalankan
filter low-pass orde satu:

```text
s_i(t) = (1 - alpha) * s_i(t-1) + alpha * p_i(t)
alpha  = 0.6
```

Untuk `x`, `y`, dan `z` dengan rumus yang sama. Ini disebut EMA atau IIR
filter. Sifatnya:

- `alpha = 1` berarti tidak ada smoothing (langsung pakai data baru).
- `alpha = 0` berarti landmark tidak pernah berubah.
- `alpha = 0.6` berarti 60% nilai baru dan 40% nilai lama setiap frame.

Pada 30 fps, konstanta waktu filter sekitar:

```text
tau = -delta_t / ln(1 - alpha)
    = -33.3 ms / ln(0.4)
    = 36.4 ms
```

Frekuensi cutoff kira-kira 4,4 Hz. Jadi getaran cepat diredam, tetapi
gerakan tangan yang disengaja tetap lolos. Filter di-reset (`prevLm = null`)
saat tangan hilang, supaya tangan baru tidak "berbayang" dengan posisi lama.
Kotak tangan dihitung ulang dari landmark yang sudah dihaluskan.

### 3.4 Memilih satu tangan

Aplikasi hanya memakai satu tangan supaya tangan kedua tidak mengubah
klasifikasi. Dari semua deteksi:

```text
score_i = skor handedness tangan ke-i
pilih i = argmax(score_i)
terima jika score >= 0.5 (atau score = 0 jika metadata kosong)
```

Tangan dengan skor di bawah 0.5 dibuang.

### 3.5 Timestamp monoton

MediaPipe mode VIDEO menolak timestamp yang tidak naik. Karena itu:

```text
ts = max(now, lastTs + 1)
lastTs = ts
```

Ini mencegah error saat frame rAF terduplikasi atau saat tab kembali aktif.

### 3.6 Gerbang gerak rata-rata

Sebelum klasifikasi, `classifyFrame` membandingkan landmark frame ini dengan
frame sebelumnya:

```text
m = (1/21) * sum_i ||p_i(t) - p_i(t-1)||
```

Jika `m > 0.065`, frame dibuang (dianggap tangan sedang berpindah isyarat).
Nilai 0.065 adalah rata-rata perpindahan ternormalisasi per frame; pada
640x480 kira-kira 30 piksel per sendi. Ambang ini sengaja longgar supaya
gemetar kecil saat berbicara tidak menghentikan pengetikan.

---

## 4. Lapisan 2: fitur kanonis 63 angka

File: `src/lib/jointFeatures.ts`, fungsi `landmarksToFeatures`.
Kontrak ini harus sama persis dengan `normalize()` di
`python/retrain_joint.py`.

Tujuannya: huruf yang sama harus menghasilkan vektor yang sama walaupun:

- tangan digeser ke kiri/kanan/atas/bawah,
- tangan lebih dekat atau lebih jauh dari kamera,
- kamera dimiringkan,
- yang dipakai tangan kiri atau tangan kanan.

### 4.1 Langkah perhitungan

**Langkah 1 - piksel.**

```text
P_i = (x_i * W, y_i * H, z_i)
```

**Langkah 2 - relatif terhadap pergelangan.**

```text
d_i = P_i - P_0
```

Sekarang pergelangan menjadi titik (0,0,0). Translasi hilang.

**Langkah 3 - normalisasi ukuran.**

```text
size = ||P_9 - P_0|| + 1e-6
d_i  = d_i / size
```

`P_9` adalah pangkal jari tengah. Jarak pergelangan ke pangkal jari tengah
dipakai sebagai "satuan panjang" tangan. Tangan besar dan kecil menjadi
sebanding. `1e-6` mencegah pembagian nol.

Karena `P_9 - P_0` memakai tiga koordinat, `size` ikut memperhitungkan
kedalaman `z`. Ini sama dengan implementasi Python:

```python
size = np.linalg.norm(hands[:, 9:10, :] - wrist, axis=-1, keepdims=True) + 1e-6
```

**Langkah 4 - rotasi roll kamera.**

Ambil arah pangkal jari tengah setelah normalisasi:

```text
mx = d_9.x
my = d_9.y
phi = atan2(mx, my)
```

Lalu putar semua titik `(x, y)` dengan matriks rotasi:

```text
x' = x * cos(phi) - y * sin(phi)
y' = x * sin(phi) + y * cos(phi)
```

Mengapa `atan2(mx, my)` dan bukan `atan2(my, mx)`? Karena kita ingin
memutar vektor `(mx, my)` tepat ke sumbu +Y. Substitusi `cos(phi) = my/r`
dan `sin(phi) = mx/r` menghasilkan:

```text
x' = mx*my/r - my*mx/r = 0
y' = mx*mx/r + my*my/r = r
```

Jadi pangkal jari tengah selalu berada di `(0, r)`. Kamera yang miring
tidak lagi mengubah fitur. Sumbu `z` tidak diputar.

**Langkah 5 - mirror tangan kiri/kanan.**

Kalau koordinat `x` dari sendi 1 (pangkal jempol) negatif:

```text
jika out[1*3] < 0:
    untuk semua i: out[i*3] = -out[i*3]
```

Setelah mirror, pangkal jempol selalu di sisi +X. Tangan kiri dan tangan
kanan dengan bentuk isyarat yang sama menghasilkan vektor yang sama.

**Langkah 6 - ratakan.**

```text
out[3*i + 0] = x_i'
out[3*i + 1] = y_i'
out[3*i + 2] = z_i'
```

Total `21 * 3 = 63` angka (`Float32Array`).

### 4.2 Contoh angka

Misalkan frame 640x480, lalu:

```text
P_0 (pergelangan)  = (320, 240, 0)
P_9 (middle MCP)   = (368, 304, 0)
```

Maka:

```text
d_9   = (48, 64, 0)
size  = sqrt(48^2 + 64^2) = 80
d_9n  = (48/80, 64/80) = (0.6, 0.8)
phi   = atan2(0.6, 0.8) = 0.6435 rad = 36.87 derajat
```

Vektor `(0.6, 0.8)` diputar menjadi `(0, 1)`. Sekarang ambil ujung telunjuk:

```text
P_8 = (400, 280, 0)
d_8 = (80, 40, 0)
d_8n = (1.0, 0.5, 0)
x' = 1.0*cos(36.87) - 0.5*sin(36.87) = 0.8 - 0.3 = 0.5
y' = 1.0*sin(36.87) + 0.5*cos(36.87) = 0.6 + 0.4 = 1.0
```

Jadi ujung telunjuk relatif menjadi `(0.5, 1.0)`. Kalau pangkal jempol
berada di `x < 0`, semua `x` dibalik.

### 4.3 Apa yang invarian dan apa yang tidak

| Perubahan | Invarian? | Penyebab |
|---|---|---|
| Tangan digeser | Ya | dikurangi `P_0` |
| Jarak ke kamera | Ya | dibagi `size` |
| Kamera miring (roll) | Ya | rotasi `phi` |
| Tangan kiri vs kanan | Ya | mirror sumbu X |
| Sudut pandang 3D (yaw/pitch) | Tidak sepenuhnya | proyeksi perspektif mengubah bentuk |
| Proporsi jari tiap orang | Tidak sepenuhnya | dinamis, didekati augmentasi latih |
| Bentuk isyarat | Ya, itu yang diklasifikasi | - |

### 4.4 Fitur mentah untuk model baseline

`rawPixelFeatures` tidak melakukan normalisasi apa pun:

```text
out[3*i + 0] = P_i.x
out[3*i + 1] = P_i.y
out[3*i + 2] = P_i.z
```

Model baseline (Conv1D) melihat koordinat piksel mentah. Karena itu model
ini kurang tahan terhadap posisi, jarak, dan tangan kiri/kanan. Ia hanya
dipakai sebagai suara tambahan, bukan penentu utama.

### 4.5 Sudut sendi dan curl (untuk UI/gestur)

`jointAngleDeg(a, b, c)` menghitung sudut di titik `b`:

```text
v1 = a - b
v2 = c - b
cos(theta) = (v1 . v2) / (||v1|| * ||v2||)
theta = acos(clamp(cos(theta), -1, 1)) * 180 / pi
```

Contoh: kalau `v1` dan `v2` searah, `cos = 1`, `theta = 0` derajat. Kalau
tegak lurus, `theta = 90` derajat.

`fingerStates` memakai dua sudut (`mcp-pip-dip` dan `pip-dip-tip`) untuk
menghitung curl:

```text
curl = clamp(1 - (theta1 + theta2 - 60) / 220, 0, 1)
```

Jari dianggap extended jika:

```text
d(tip, wrist) > d(pip, wrist) * 1.06
dan curl < 0.55
```

Ini dipakai untuk visualisasi dan aturan gestur, bukan untuk label SIBI.

---

## 5. Lapisan 3: model neural network dan fusi

File: `src/lib/sibiClassifier.ts`

Tiga model dimuat saat startup:

| Slot | Key | Model | Input | Akurasi offline | Dipakai |
|---|---|---|---|---|---|
| 0 utama | `joint` | MLP kanonis v2 | 63 angka | val 90,3% | selalu |
| 1 vote | `baseline` | Conv1D sendi mentah | 63x1 | val 74,2% | default ON |
| 2 vote | `image` | MobileNetV2 | 128x128x3 | test 84,8% | default OFF |

Ketiganya dimuat ke memori saat startup supaya tidak ada jeda saat vote
diaktifkan. Yang menentukan huruf adalah model `joint`. Dua model lain hanya
menambah keyakinan jika setuju.

### 5.1 Arsitektur MLP sendi

```text
Input 63
  -> Dense 256, ReLU, Dropout 0.5
  -> Dense 128, ReLU, Dropout 0.4
  -> Dense 64,  ReLU, Dropout 0.25
  -> Dense 24,  Softmax
Output 24 probabilitas (A-I, K-Y)
```

Saat inference, Dropout dimatikan (identitas), jadi hanya dense + ReLU +
softmax yang dihitung.

Satu dense layer:

```text
z = W * x + b
```

`W` adalah matriks bobot, `b` bias. ReLU:

```text
ReLU(z) = max(0, z)
```

Softmax pada 24 logit:

```text
p_c = exp(z_c) / sum_j exp(z_j)
```

Sifat softmax:

```text
0 < p_c < 1
sum_c p_c = 1
```

Contoh sederhana dengan 3 logit `[2, 1, 0]`:

```text
exp(2) = 7.389
exp(1) = 2.718
exp(0) = 1.000
jumlah = 11.107
p = [0.665, 0.245, 0.090]
```

Keyakinan (confidence) yang ditampilkan adalah `max(p_c)`. Top-3 adalah
tiga indeks dengan `p_c` terbesar. Sorting dilakukan dengan membandingkan
nilai probabilitas, bukan jarak fitur.

### 5.2 Loss dan bobot kelas saat latihan

Model dilatih dengan categorical cross-entropy dan label smoothing 0,05:

```text
q_c = (1 - epsilon) * y_c + epsilon / C
L = -sum_c q_c * log(p_c)
```

`y_c = 1` untuk kelas benar dan 0 untuk kelas lain, `C = 24`,
`epsilon = 0.05`. Label smoothing mencegah model terlalu percaya diri pada
data latih.

Bobot kelas menyeimbangkan jumlah sampel:

```text
cw_i = N / (C * n_i)
```

`N` total sampel, `n_i` jumlah sampel kelas `i`. Kelas yang jarang mendapat
bobot lebih besar.

### 5.3 Baseline Conv1D

Input 63 koordinat piksel disusun sebagai `63 x 1`, lalu Conv1D mempelajari
pola lokal antar sendi. Outputnya 26 nilai; 24 nilai pertama dipakai sebagai
huruf, indeks 24 dan 25 diabaikan. Karena inputnya piksel mentah, model ini
mudah berubah kalau tangan digeser atau kamera miring.

### 5.4 MobileNetV2 untuk citra

Input `128 x 128 x 3` dengan nilai `[-1, 1]`. Backbone MobileNetV2
menggunakan depthwise separable convolution, diikuti global average pooling
dan head dense 512 -> 256 -> 24 softmax. Model ini melihat bentuk tangan
dari gambar, bukan dari koordinat sendi.

### 5.5 Fusi keyakinan

Di `classifyFrame`:

```text
baseAgree = (label_baseline == label_joint)
imgAgree  = (label_image    == label_joint)

fused = p_joint
      + 0.07 * (baseAgree ? 1 : 0)
      + 0.07 * (imgAgree  ? 1 : 0)

fused = min(0.99, fused)
```

Contoh:

```text
p_joint = 0.72, baseline setuju, image tidak setuju
fused = 0.72 + 0.07 = 0.79

p_joint = 0.72, keduanya setuju
fused = 0.72 + 0.07 + 0.07 = 0.86
```

`fused` bukan probabilitas murni; nilainya bisa lebih dari 1 sebelum
di-clamp. Nilai ini hanya dipakai sebagai input smoother. Angka yang
ditampilkan di kartu kamera adalah `p_joint` mentah.

Penting: sejak versi smoother baru, **tidak ada lagi gerbang margin**
`top1 - top2 >= 0.2`. Bingkai ragu ditolak oleh konsistensi waktu di
smoother, bukan oleh satu frame.

### 5.6 Vote citra dengan mirror

Untuk setiap ROI, model citra dijalankan dua kali:

```text
p_asli   = P(model | ROI)
p_mirror = P(model | mirror(ROI))
p_pilih  = max(p_asli, p_mirror)   (per confidence, bukan per kelas)
```

Yang dipilih adalah hasil dengan confidence lebih tinggi. Ini membuat model
citra lebih toleran terhadap tangan kiri/kanan.

### 5.7 Ambang keyakinan yang benar-benar dipakai

```text
classifyFrame: hanya kirim ke smoother jika p_joint >= 0.18
smoother:      abaikan frame jika conf < 0.16
smoother:      commit hanya jika rata-rata conf >= 0.24
```

Tiga angka ini berbeda dan punya peran masing-masing. `minConfidence`
default 0.24, bukan 0.8 seperti versi lama.

---

## 6. Lapisan 4: smoother temporal (kunci huruf)

File: `src/lib/predictionSmoother.ts`, kelas `PredictionSmoother`.

Ini bagian matematika yang paling menentukan pengalaman mengetik. Tujuan:
satu isyarat yang ditahan menghasilkan tepat satu huruf, secepat mungkin,
tanpa huruf sampah saat tangan berpindah.

### 6.1 Window dan bobot

Window hanya menyimpan 5 prediksi terakhir (`windowSize = 5`). Untuk frame
ke-`i` di dalam window (0 = paling lama, `n-1` = paling baru):

```text
w_i = 1 + 1.2 * i / (n - 1)
```

Untuk `n = 5`, bobotnya:

| i | Umur | Bobot |
|---|---|---|
| 0 | paling lama | 1.0 |
| 1 | - | 1.3 |
| 2 | - | 1.6 |
| 3 | - | 1.9 |
| 4 | paling baru | 2.2 |
| | **total** | **8.0** |

Frame terbaru mendapat bobot 2,2 kali frame terlama. Ini yang membuat
perpindahan isyarat terasa cepat: bukti baru langsung mendominasi.

### 6.2 Weighted vote

Untuk setiap label `L` yang muncul di window:

```text
W_L   = sum_{i: label_i = L} w_i
S_L   = sum_{i: label_i = L} conf_i
n_L   = jumlah frame berlabel L
Avg_L = S_L / n_L
```

Label teratas:

```text
topLabel = argmax_L W_L
topW     = W_topLabel
topAvg   = Avg_topLabel
totalW   = sum_i w_i
share    = topW / totalW
```

`share` adalah porsi bobot yang dimiliki label teratas. Nilainya 0..1.

Contoh dengan 5 frame:

```text
label:  A  A  B  B  A   (A = terbaru)
w:      1  1.3 1.6 1.9 2.2

W_A = 1 + 1.3 + 2.2 = 4.5
W_B = 1.6 + 1.9     = 3.5
totalW = 8.0
share_A = 4.5 / 8.0 = 0.5625
```

Karena `share_A < 0.58`, frame seperti ini tidak boleh commit. Dua label
masih terlalu berimbang.

### 6.3 Aturan pindah label

Label saat ini (`currentLabel`) hanya diganti jika salah satu benar:

```text
1) dua frame terbaru sama-sama topLabel (streak), ATAU
2) W_topLabel > 1.35 * W_currentLabel
```

Aturan pertama membuat reaksi cepat setelah dua prediksi baru yang konsisten.
Aturan kedua adalah hysteresis: label baru harus menang jelas (35% lebih
besar) sebelum mengambil alih. Satu frame noise tidak bisa memindahkan
label.

Saat label berganti:

```text
currentLabel = topLabel
labelStartAt = now
progress     = 0
```

### 6.4 Hold adaptif

Setelah label stabil, smoother menghitung berapa lama label itu sudah
ditahan:

```text
heldMs = now - labelStartAt
```

Hold yang dibutuhkan bergantung pada keyakinan rata-rata:

```text
confNorm = clamp((topAvg - 0.25) / 0.55, 0, 1)
holdMs   = round(700 * (1.15 - 0.65 * confNorm))
```

Range hasilnya:

```text
topAvg <= 0.25  -> confNorm = 0 -> holdMs = 805 ms
topAvg = 0.40   -> confNorm = 0.273 -> holdMs = 681 ms
topAvg = 0.60   -> confNorm = 0.636 -> holdMs = 515 ms
topAvg = 0.80   -> confNorm = 1.0 -> holdMs = 350 ms
topAvg >= 0.80  -> tetap 350 ms
```

Jadi isyarat yang sangat yakin terkunci sekitar 0,35 detik; isyarat yang
ragu diberi waktu sampai 0,8 detik. Tidak ada lagi syarat 2 detik.

Progress bar di UI:

```text
progress = min(1, heldMs / holdMs)
```

### 6.5 Syarat commit lengkap

Huruf baru dikembalikan jika **semua** benar:

```text
1) currentLabel tidak kosong
2) topAvg >= 0.24
3) heldMs >= holdMs
4) now - lastCommitAt >= 450 ms            (cooldown)
5) share >= 0.58                           (tidak ambigu)
6) jika huruf sama dengan commit terakhir:
      now - lastCommitAt >= 900 ms          (repeat guard)
```

Jika lolos, smoother:

```text
commitLabel = currentLabel
lastCommitAt = now
lastEmitted = {label: commitLabel, conf: topAvg}
window = []
progress = 0
currentLabel = ''
labelStartAt = 0
return commitLabel
```

Window dikosongkan setelah commit. Artinya huruf berikutnya harus dibangun
dari bukti baru, bukan sisa window lama.

### 6.6 Cooldown, repeat guard, dan jeda bicara

Tiga pengaman waktu:

**Cooldown 450 ms.** Setelah satu commit, commit berikutnya tidak boleh
lebih cepat dari 450 ms. Ini mencegah satu isyarat yang sama menghasilkan
dua huruf dalam satu gerakan kecil.

**Repeat guard 900 ms.** Kalau huruf yang akan di-commit sama dengan huruf
commit terakhir, jaraknya harus minimal 900 ms. Contoh: menahan A terus
menerus menghasilkan A pada ~0,5 detik, lalu A berikutnya pada ~1,4 detik.
Untuk mengetik "AA", pengguna memang perlu menahan lebih lama atau
mengubah isyarat lalu kembali.

**Pause 700 ms.** Jika jarak antar prediksi lebih dari 700 ms:

```text
currentLabel = ''
labelStartAt = 0
window = []
progress = 0
```

Jeda bicara yang pendek (misalnya 600 ms) **tidak** menghapus progress.
Ini penting: pengguna bisa berbicara sambil menahan isyarat tanpa kehilangan
hitung mundur. Frame dengan confidence di bawah 0,16 tidak masuk window,
tetapi tetap dihitung sebagai "masih ada prediksi" untuk aturan pause.

### 6.7 Contoh timeline

Asumsi prediksi datang setiap 100 ms (10 Hz), `A` dengan confidence 0,85:

```text
t=100  push A 0.85 -> current=A, t0=100, held=0
t=200  push A 0.85 -> held=100, hold=350, progress=0.29
t=300  push A 0.85 -> held=200, progress=0.57
t=400  push A 0.85 -> held=300, progress=0.86
t=500  push A 0.85 -> held=400 >= 350 -> COMMIT A
```

Jadi commit pertama sekitar 0,4-0,5 detik setelah isyarat mulai stabil.
Dengan deteksi 30 Hz dan inferensi 10-14 Hz, pengalaman nyata biasanya
0,45-0,6 detik.

Contoh pindah isyarat `A -> B` (dari test):

```text
t=100..1000  A 0.85   -> commit A di t=500
t=1100..2000  B 0.85  -> dua frame B terbaru (streak), label pindah di t=1200
                         held=350 -> commit B di t=1600
```

Perhatikan: B tidak commit tepat saat frame B pertama, karena smoother
menunggu dua frame B dan hold adaptif. Ini trade-off yang disengaja:
lebih lambat ~0,2 detik, tetapi tidak salah huruf saat transisi.

Contoh flicker `A/B` bergantian:

```text
A B A B A B ...
share label teratas sekitar 0.53 < 0.58
-> tidak pernah commit
```

Contoh jeda bicara:

```text
t=100,200,300  D 0.80
t=900          D 0.80   (gap 600 ms < 700 ms, progress dipertahankan)
t=1000..1300   D 0.80
-> commit D di t=900
```

Kalau gap > 700 ms, hitungan dimulai dari nol lagi.

### 6.8 Mengapa bukan majority 15 frame

Versi lama memakai window 15 prediksi dan menuntut 9 frame mayoritas. Karena
inferensi sendi saat itu berjalan sekitar 9-10 Hz, 15 prediksi berarti sekitar
1,5 detik data. Efek inersianya besar: setelah isyarat berganti, label lama
masih mayoritas selama hampir satu detik. Akibatnya pengguna menahan huruf
baru, tetapi yang terkunci masih huruf lama.

Versi sekarang memakai 5 frame berbobot, aturan pindah 2 frame, dan hold
adaptif. Total waktu commit jauh lebih pendek, dan huruf lama tidak
"menempel" di window.

### 6.9 Jika ingin mengubah perilaku

| Yang ingin diubah | Lokasi | Efek |
|---|---|---|
| Window lebih panjang | `windowSize` di `predictionSmoother.ts` | lebih halus, tetapi pindah huruf lebih lambat |
| Bobot frame terbaru | angka `1.2` pada `w_i` | lebih besar = lebih reaktif, lebih kecil = lebih stabil |
| Hold dasar | `holdMs: 700` di `+page.svelte` / `baseHold` | menaikkan semua hold 350-805 ms secara proporsional |
| Keyakinan minimum | `minConfidence` / `minConf` | lebih rendah = lebih mudah commit, lebih banyak salah |
| Jeda antar huruf | `cooldownMs` 450 | lebih tinggi = lebih aman, lebih lambat |
| Huruf sama berulang | angka `900` pada repeat guard | lebih tinggi = lebih sulit mengetik AA |
| Toleransi jeda bicara | angka `700` pada pause | lebih tinggi = progress lebih tahan jeda |
| Ambang ambigu | `topShare < 0.58` | lebih tinggi = lebih konservatif |
| Kecepatan pindah label | rasio `1.35` | lebih tinggi = lebih sulit pindah |
| Gerbang keyakinan frame | `out.confidence >= 0.18` | lebih rendah = lebih banyak frame masuk |
| Gerbang area tangan | `area < 0.04` dan `area > 0.9` | lebih ketat = tangan harus lebih jelas |
| Gerbang gerak | `lastMotion > 0.065` | lebih tinggi = lebih toleran gerakan kecil |
| Ambang J/Z | angka di `detectJ` / `detectZ` | lihat tabel parameter di bagian 7.6 |
| Blokir statis saat J/Z | `trailNet > 0.06` | lebih tinggi = D/I lebih mudah, J/Z lebih rawan salah |

Semua nilai ini adalah trade-off antara kecepatan dan keamanan. Turunkan
hold/cooldown untuk mengetik lebih cepat; naikkan jika huruf sampah sering
muncul.

---

## 7. Lapisan 5: huruf dinamis J dan Z

File: `src/lib/motionLetters.ts`

Model statis hanya punya 24 kelas: A-I dan K-Y. J dan Z adalah isyarat
dinamis, jadi tidak ada di model. Aplikasi mengenalinya dari jejak ujung
jari:

- **Z**: telunjuk lurus (bentuk D), ujung telunjuk menggambar garis
  horizontal, diagonal balik, lalu horizontal lagi.
- **J**: kelingking lurus (bentuk I), ujung kelingking turun panjang lalu
  mengait ke samping.

### 7.1 Menyimpan jejak

`pushTrail(trail, point, maxMs=1800, maxPoints=140)` menyimpan sampel
`(x, y, t)` dari ujung jari. Sampel lama dibuang jika:

```text
t_sekarang - t_pertama > 1800 ms
atau jumlah titik > 140
```

Dengan deteksi 30 Hz, 1800 ms berisi sekitar 54 titik. Batas 140 titik
mencegah memori tumbuh tanpa batas.

### 7.2 Memecah jejak menjadi segmen lurus

`trailSegments` mengubah jejak menjadi potongan garis lurus:

1. Untuk setiap pasangan titik, hitung:

```text
dx = x_i - x_{i-1}
dy = y_i - y_{i-1}
jarak = sqrt(dx^2 + dy^2)
```

2. Jika `jarak < 0.0025`, abaikan (getaran sensor). Pada 640 piksel,
   0.0025 kira-kira 1,6 piksel.

3. Hitung arah dalam 8 keranjang 45 derajat:

```text
bucket = round(atan2(dy, dx) / (pi/4)) mod 8
```

Contoh: `atan2 = 0` -> bucket 0 (kanan); `pi/2` -> bucket 2 (bawah);
`pi` -> bucket 4 (kiri); `-pi/2` -> bucket 6 (atas).

4. Selama arah masih di bucket yang sama atau bertetangga (`diff <= 1`),
   titik digabung ke segmen yang sedang berjalan.

5. Jika arah berubah lebih dari satu bucket (`diff > 1`, artinya beda
   minimal 90 derajat), segmen ditutup dan segmen baru dimulai.

6. Segmen yang panjangnya `< 0.02` dibuang. Pada 640 piksel, 0.02 kira-kira
   13 piksel. Ini membuang goyangan kecil, bukan stroke sungguhan.

Hasilnya adalah daftar vektor `(dx, dy)` yang mewakili stroke.

### 7.3 Deteksi Z

Syarat `detectZ`:

```text
1) minimal 8 titik
2) durasi jejak 280..1800 ms
3) ada tiga segmen berurutan a, b, c

a: horizontal pertama
   |a.dx| > 0.05
   |a.dx| > 1.2 * |a.dy|

b: diagonal ke bawah, arah x berlawanan dengan a
   b.dy > 0.035
   |b.dx| > 0.02
   tanda(b.dx) != tanda(a.dx)

c: horizontal terakhir, searah dengan a
   |c.dx| > 0.03
   tanda(c.dx) == tanda(a.dx)
   |c.dx| > 0.8 * |c.dy|
```

Contoh nyata dari test (koordinat ternormalisasi, y ke bawah):

```text
(0.30,0.30) -> (0.46,0.30)   dx=+0.16, dy=0.00   (horizontal kanan)
(0.46,0.30) -> (0.34,0.42)   dx=-0.12, dy=+0.12   (diagonal kiri-bawah)
(0.34,0.42) -> (0.50,0.44)   dx=+0.16, dy=+0.02   (horizontal kanan)
```

Semua syarat lolos -> Z.

### 7.4 Deteksi J

Syarat `detectJ`:

```text
1) minimal 8 titik
2) durasi 280..1800 ms
3) ada dua segmen berurutan: down, hook

down:
   down.dy > 0.09
   down.dy > 1.2 * |down.dx|

hook:
   |hook.dx| > 0.05
   |hook.dx| > 1.4 * |hook.dy|
```

Contoh dari test:

```text
(0.50,0.20) -> (0.50,0.40)   dx=0, dy=+0.20     (turun)
(0.50,0.40) -> (0.37,0.42)   dx=-0.13, dy=+0.02 (kait kiri)
```

Lolos -> J. Kait boleh ke kiri atau ke kanan; yang penting horizontal.

### 7.5 Bentuk tangan yang mengaktifkan

`fingerPattern` memastikan gerakan hanya dilacak saat bentuk dasarnya benar:

```text
Z -> indexOnly:  telunjuk lurus, tengah/manis/kelingking terlipat
J -> pinkyOnly:  kelingking lurus, telunjuk/tengah/manis terlipat
```

Jempol sengaja diabaikan karena bentuk I dan D sering berbeda antar penanda.

Aturan `fingerExtended`:

```text
d(tip, wrist) > d(pip, wrist) * 1.08
dan tip.y < pip.y + 0.06
```

`tip.y < pip.y + 0.06` berarti ujung jari harus sedikit lebih tinggi dari
sendi pip (ingat y bertambah ke bawah). Toleransi 0.06 memberi ruang untuk
jari yang agak mendatar.

### 7.6 Pengaman agar J/Z tidak salah Picu

```text
- hanya saat bentuk indexOnly/pinkyOnly
- trail di-reset saat bentuk berubah atau tangan hilang
- butuh minimal 8 titik dan stroke lengkap
- cooldown antar huruf gerakan: 1500 ms
- setelah commit J/Z: klasifikasi statis diblokir 1200 ms
- smoother statis di-reset setelah commit J/Z
```

Selama gerakan J/Z aktif, `classifyFrame` juga melewati frame jika
`trailNet > 0.06`, yaitu perpindahan bersih ujung jari dalam 350 ms terakhir
lebih dari 0.06 (sekitar 38 piksel pada lebar 640). Ini mencegah model
statis mengunci D atau I di tengah gerakan.

Parameter gerakan:

| Parameter | Nilai | Arti |
|---|---|---|
| `maxMs` | 1800 | umur maksimum jejak |
| `maxPoints` | 140 | jumlah titik maksimum |
| jitter | 0.0025 | jarak minimum antar sampel |
| segmen minimum | 0.02 | buang stroke terlalu pendek |
| durasi valid | 280-1800 ms | cepat/lambat masih diterima |
| Z horizontal 1 | 0.05 | minimal 5% lebar frame |
| Z diagonal | 0.035 | minimal 3,5% tinggi frame |
| Z horizontal 2 | 0.03 | minimal 3% lebar frame |
| J turun | 0.09 | minimal 9% tinggi frame |
| J kait | 0.05 | minimal 5% lebar frame |
| cooldown J/Z | 1500 ms | jeda antar huruf gerakan |
| blokir statis | 1200 ms | setelah J/Z commit |

---

## 8. Gestur kontrol (bukan isyarat SIBI)

File: `src/lib/handTracker.ts`, `decodeControlGesture`.

Gestur di bawah ini **hanya mengontrol aplikasi**, bukan huruf SIBI. Huruf
tetap datang dari model/smoother.

### 8.1 Matematika jari

Jarak 2D:

```text
d(a,b) = sqrt((a.x-b.x)^2 + (a.y-b.y)^2)
```

Jari dianggap extended:

```text
d(tip, wrist) > d(pip, wrist) * 1.08
dan tip.y < pip.y + 0.06
```

Jempol extended:

```text
d(thumb_tip, middle_mcp) > d(thumb_ip, middle_mcp) * 1.15
```

Pinch:

```text
handSize  = d(wrist, middle_mcp)
pinchRatio = d(thumb_tip, index_tip) / max(1e-6, handSize)
pinch = pinchRatio < 0.42
```

### 8.2 Aturan gestur

Urutan pengecekan (yang pertama menang):

| Prioritas | Gestur | Kondisi | Aksi |
|---|---|---|---|
| 1 | fist | semua jari + jempol tertutup | hapus huruf/kata |
| 2 | pinch | pinch dan `upCount <= 2` | tidak dipakai untuk commit |
| 3 | point | hanya telunjuk | tidak dipakai untuk commit |
| 4 | peace | telunjuk + tengah | spasi / kunci kata |
| 5 | palm | `upCount >= 4` | jeda klasifikasi |
| 6 | thumbsup | hanya jempol | ucapkan kalimat |

`upCount` adalah jumlah dari telunjuk, tengah, manis, kelingking yang
extended.

### 8.3 Hold dan cooldown gestur

- Gestur harus bertahan **900 ms** sebelum aksi dijalankan.
- Cooldown fist: 2200 ms.
- Cooldown peace: 2200 ms.
- Cooldown thumbsup: 2500 ms.

Rumus hold:

```text
held = now - gestureStartAt
progress = min(1, held / 900)
aksi jika held >= 900 dan cooldown terpenuhi
```

`gestureStartAt` di-reset ke 0 saat gestur berubah atau tangan hilang.
Ini mencegah bentuk transisi (misalnya tangan yang sedang membuka dari fist
ke palm) memicu hapus atau spasi.

---

## 9. Citra ROI dan normalisasi gambar

File: `src/lib/preprocessing.ts`

Model citra memakai crop persegi di sekitar tangan, bukan seluruh frame.

### 9.1 Crop persegi

```text
side = max(24, max(box.w, box.h) * 1.5)
cx = box.x + box.w / 2
cy = box.y + box.h / 2

sx = round(cx - side/2)
sy = round(cy - side/2)
s  = round(side)
```

Lalu clamping agar crop tidak keluar frame:

```text
sx = clamp(sx, 0, W-1)
sy = clamp(sy, 0, H-1)
s  = max(1, min(W - sx, H - sy, s))
```

Faktor 1,5 memberi padding 25% di setiap sisi. Contoh box 200x240 di
tengah frame 640x480:

```text
side = 1.5 * 240 = 360
cx=320, cy=240
sx=140, sy=60
s = min(500, 420, 360) = 360
crop = 360x360 piksel
```

Crop kemudian digambar ke canvas persegi dan di-resize ke 128x128.

### 9.2 Normalisasi piksel

Dua mode:

```text
neg-one-one: v = c / 127.5 - 1      c in 0..255 -> v in -1..1
zero-one:    v = c / 255            c in 0..255 -> v in 0..1
```

Default: `neg-one-one`, karena MobileNetV2 dilatih dengan `[-1, 1]`.
Contoh: `c=0 -> -1`, `c=127.5 -> 0`, `c=255 -> 1`.

### 9.3 Grayscale (opsional)

```text
Y = 0.299*R + 0.587*G + 0.114*B
R = G = B = Y
```

Ini adalah luma BT.601. Default mati karena model dilatih dengan RGB.

### 9.4 Mirror

```text
ctx.translate(128, 0)
ctx.scale(-1, 1)
ctx.drawImage(crop, 0, 0, 128, 128)
```

Hasilnya adalah flip horizontal. Model dijalankan pada citra asli dan
citra mirror, lalu confidence tertinggi dipakai.

### 9.5 Laju vote citra

Vote citra berjalan setiap 200 ms (5 Hz) dan hanya jika `useImage = true`.
Vote baseline berjalan setiap 250 ms (4 Hz) dan default aktif. Keduanya
jauh lebih lambat dari model sendi karena lebih berat.

---

## 10. Bahasa: kata, kalimat, dan auto-koreksi

File: `src/lib/languageModel.ts`

Model bahasa di sini sengaja kecil dan offline: daftar kata dengan
frekuensi + bigram, bukan LLM.

### 10.1 Skor frekuensi

Setiap kata punya frekuensi `f` (lihat `VOCAB`). Kata yang tidak ada di
daftar memakai default 20.

### 10.2 Melengkapi kata

Untuk prefix `p`:

```text
kandidat = kata yang diawali p dan bukan p itu sendiri
skor    = f + 4 * panjang(p)
urutkan skor menurun
ambil 3 teratas
```

Contoh prefix `"MA"`:

```text
MAU    92 + 4*2 = 100
MAKAN  88 + 4*2 = 96
MAAF   63 + 4*2 = 71
```

Tiga saran teratas: MAU, MAKAN, MAAF.

### 10.3 Prediksi kata berikutnya

Kalau kata terakhir ada di `BIGRAMS`:

```text
skor = 120 + f / 10
alasan = 'setelah "KATA"'
```

Kalau tidak ada, fallback ke kata paling sering:

```text
skor = f / 10
alasan = 'sering dipakai'
```

Contoh setelah `"SAYA"`:

```text
MAU    120 + 9.2 = 129.2
AKAN   120 + 7.8 = 127.8
SUKA   120 + 7.0 = 127.0
SEDANG 120 + 6.5 = 126.5
INGIN  120 + 6.7 = 126.7
```

Tiga teratas: MAU, AKAN, SUKA.

### 10.4 Levenshtein untuk auto-koreksi

Jarak edit `D[i][j]` antara dua kata `a` dan `b`:

```text
D[0][j] = j
D[i][0] = i
D[i][j] = min(
    D[i-1][j] + 1,       // hapus
    D[i][j-1] + 1,       // sisip
    D[i-1][j-1] + cost   // ganti
)
cost = 0 jika a[i-1] == b[j-1], selain itu 1
```

Implementasi memakai dua baris (`prev`, `curr`), jadi memori
`O(len(b))` dan waktu `O(len(a) * len(b))`.

Aturan koreksi:

```text
threshold = max(1, floor(len(input) / 4))
koreksi hanya jika bestDistance <= threshold dan bestDistance <= 3
```

Contoh:

```text
"MAKN"  -> "MAKAN"  jarak 1, threshold max(1,1)=1 -> dikoreksi
"MAKANN" -> "MAKAN" jarak 1, threshold max(1,1)=1 -> dikoreksi
"XYZ"   -> tidak ada kandidat dekat -> dibiarkan
```

Koreksi hanya dijalankan saat kata dikunci (Spasi), bukan setiap huruf.
Tujuannya agar ketikan yang belum selesai tidak diubah di tengah jalan.

### 10.5 Kalimat

```text
sentence = kata-kata digabung dengan spasi
           -> lowercase
           -> huruf pertama uppercase
```

`prefix` yang belum dikunci ikut ditampilkan di akhir kalimat sebagai
kata yang sedang diketik.

---

## 11. Waktu, FPS, dan anggaran frame

File: `src/routes/+page.svelte`

Aplikasi menjaga 60 fps dengan membagi pekerjaan ke beberapa laju:

| Pekerjaan | Periode | Laju kira-kira | Catatan |
|---|---|---|---|
| Repaint overlay (rAF) | 16,7 ms | 60 Hz | gambar skeleton tiap frame |
| Deteksi MediaPipe | 32 ms | 31,25 Hz | `lastDetectAt` |
| Inferensi sendi | 70 ms | 14,3 Hz | jika `inferMs <= 45` |
| Inferensi sendi (lambat) | 110 ms | 9,1 Hz | jika `inferMs > 45` |
| Vote baseline | 250 ms | 4 Hz | default ON |
| Vote citra | 200 ms | 5 Hz | default OFF |
| Update label UI | 120 ms | 8,3 Hz | `jointLabel`, top3 |
| Update state tangan | 100 ms | 10 Hz | `handsFound`, gestur |
| Hint framing | 500 ms | 2 Hz | teks bantuan |
| Progress hold UI | 80 ms | 12,5 Hz | `holdProgressUI` |

### 11.1 FPS EMA

```text
fpsBaru = round(0.9 * fpsLama + 0.1 * (1000 / dt))
```

`dt` adalah jarak antar frame rAF dalam ms. Ini filter low-pass dengan
alpha 0.1. Konstanta waktunya:

```text
tau = -200 ms / ln(0.9) = 1898 ms
```

Jadi angka FPS bergerak halus dan butuh sekitar 2 detik untuk mengikuti
perubahan besar.

### 11.2 Mengapa deteksi tidak setiap frame

Deteksi MediaPipe dan inferensi TensorFlow sama-sama memakai GPU. Kalau
keduanya dijalankan setiap 16 ms, antrean GPU menumpuk dan UI tersendat.
Strategi di sini:

- MediaPipe 30 Hz (cukup untuk gerakan tangan).
- Inferensi sendi 9-14 Hz (cukup untuk smoother).
- Overlay tetap 60 Hz supaya skeleton dan progress bar terlihat mulus.
- `predictBusy` menjamin hanya satu inferensi sendi berjalan pada satu waktu.
- Hasil deteksi di-cache (`cachedHands`) di antara jadwal deteksi.

### 11.3 Gerbang gerak dan J/Z

Gerbang gerak memakai rata-rata perpindahan semua landmark:

```text
m = (1/21) * sum_i ||p_i(t) - p_i(t-1)||
m > 0.065 -> frame dibuang
```

Untuk J/Z, gerbang tambahan memakai perpindahan bersih ujung jari:

```text
trailNet(350 ms) > 0.06 -> klasifikasi statis dilewati
```

Keduanya mencegah huruf statis terkunci di tengah gerakan.

---

## 12. Penyusunan teks dan antarmuka

State teks di `+page.svelte`:

- `prefix`: buffer huruf yang sedang diketik, maksimum 24 huruf.
- `words`: kata yang sudah dikunci.
- `recentChars`: 10 huruf terakhir untuk debug.
- `sentence`: hasil `toSentence(words + prefix)`.

Operasi:

- `commitChar(c)`: tambah satu huruf; kalau sudah 24, tampilkan pesan
  "Kata sudah 24 huruf. Tekan Spasi untuk mengunci kata."
- `pickLetter(c)`: masukkan huruf dari chip kandidat, reset smoother,
  tampilkan badge Terkunci.
- `deleteLast()`: hapus huruf terakhir; kalau prefix kosong, hapus kata
  terakhir.
- `commitSpace()`: kunci prefix menjadi kata (dengan auto-koreksi jika
  aktif), lalu kosongkan prefix.
- `removeWord(i)`: hapus kata ke-i.
- `moveWord(i, dir)`: geser kata ke kiri/kanan.
- `applySuggestion(s)`: masukkan saran sebagai kata.
- `speak()`: bacakan kalimat.
- `saveCurrent()`: simpan kalimat ke riwayat.

Keyboard:

| Tombol | Aksi |
|---|---|
| `A-Z` | tambah huruf |
| `Backspace` | hapus huruf/kata terakhir |
| `Spasi` | kunci kata |
| `Enter` | ucapkan kalimat |
| `1-3` | pilih saran kata |

Input manual:

```text
value.toUpperCase().replace(/[^A-Z]/g, '').slice(0, 24)
```

Hanya huruf A-Z, maksimal 24 karakter.

Tata letak sekarang satu kolom (max-width 6xl):

```text
HEADER
  judul + status model + status tracking
CAMERA
  video mirror + overlay skeleton
  kartu huruf + confidence + progress + 3 chip kandidat
  badge hijau "Terkunci"
  chip FPS / latensi / status tangan / gestur
OUTPUT
  kalimat, input manual, tombol J/Z, saran kata, Spasi/Hapus/Ucapkan/Simpan
KONTROL KAMERA
  daftar gestur (fist, peace, thumbsup, palm)
RIWAYAT
  kalimat tersimpan lokal
MODEL
  status 3 model + tombol nyalakan/coba lagi
DATA & EVALUASI
  ringkasan dataset, akurasi, cara reproduksi
FOOTER
```

Tidak ada lagi placeholder "tangan di sini", kotak tengah, atau zona
Kiri/Tengah/Kanan. Semua keputusan huruf berasal dari smoother.

---

## 13. Pengaturan dan penyimpanan

File: `src/lib/storage.ts`, IndexedDB dengan fallback localStorage.

Pengaturan yang dibaca saat startup:

| Key | Default | Arti |
|---|---|---|
| `minConf` | 0.24 | keyakinan rata-rata minimum untuk commit |
| `cooldownMs` | 450 | jeda minimum antar commit |
| `grayscale` | false | ubah ROI jadi grayscale untuk vote citra |
| `useBaseline` | true | aktifkan vote Conv1D |
| `useImage` | false | aktifkan vote MobileNetV2 |
| `autoCorrect` | true | koreksi kata saat Spasi |

Catatan: panel pengaturan visual sudah tidak ada di UI. Nilai-nilai ini
dibaca dari storage; kalau belum pernah disimpan, default di atas yang
dipakai. `normMode` selalu `neg-one-one`.

Data yang disimpan:

- Riwayat kalimat: teks, daftar kata, timestamp.
- Pengaturan di atas.

Tidak ada data yang dikirim ke server.

---

## 14. Text-to-Speech

File: `src/lib/tts.ts`

- Web Speech API `speechSynthesis`, bahasa `id-ID`.
- Voice Indonesia dipilih jika tersedia; kalau tidak, voice default.
- `speakIndonesian(text, {rate, pitch})` dengan default `rate = 0.95` dan
  `pitch = 1`.
- `stopSpeaking()` untuk menghentikan.
- Dipicu tombol **Ucapkan** atau gestur thumbs up.

---

## 15. Perekam sampel tangan sendiri

Panel debug menyediakan perekam sampel untuk memperbaiki huruf yang sering
salah pada tangan pengguna.

Alur:

1. Pilih huruf (24 huruf statis, A-I dan K-Y).
2. Tekan **Rekam**.
3. Setiap ~150 ms satu vektor 63 kanonis disimpan. Maksimum 2000 sampel.
4. Tekan **Unduh CSV**: file `sibi-extra.csv` dengan kolom
   `class_type, f0..f62`.
5. Latih ulang:

```sh
python python/retrain_joint.py --extra ./sibi-extra.csv
```

6. Export ke `static/models/sibi-joint`, lalu `npm run build`.

Sampel dari app sudah kanonis, jadi di Python langsung digabung setelah
`normalize()` data utama.

---

## 16. Struktur proyek

```text
src/lib/
  sibiLabels.ts          label 24 huruf, ukuran input, URL model
  preprocessing.ts       cropROI, preprocessToTensor, drawPreview
  sibiClassifier.ts      SibiEngine (3 slot), re-export smoother
  predictionSmoother.ts  weighted vote + hold adaptif + cooldown
  jointFeatures.ts       landmarksToFeatures (kanonis v2), rawPixelFeatures
  handTracker.ts         MediaPipe wrapper, gesture, EMA, fingerPattern
  motionLetters.ts       jejak ujung jari, detectJ, detectZ
  languageModel.ts       frekuensi, bigram, Levenshtein
  storage.ts             IndexedDB riwayat + settings
  tts.ts                 Web Speech API
  types.ts               tipe TypeScript
  index.ts               re-export publik
src/routes/
  +page.svelte           UI utama, loop kamera, fusi, teks
  +layout.svelte         layout
tests/
  smoother.test.mjs      test timing commit, switch, flicker, pause
  motion.test.mjs        test Z/J sintetis dan kasus negatif
python/
  retrain_joint.py       latih MLP sendi v2 (kanonis)
  train_sibi.py          latih CNN citra (Kaggle)
  train_sibi_advanced.py MobileNetV2 80/20
  train_image.py         latih CNN citra sederhana
  export_savedmodel.py   export SavedModel
  dump_verify.py         verifikasi model
scripts/
  verify-norm.mjs        verifikasi normalisasi TS vs Python
  verify-models.mjs      verifikasi TF.js vs Keras
static/models/
  sibi-joint/            model utama (236 KB)
  sibi-joint-baseline/   vote baseline (4 MB)
  sibi-advanced/         vote citra (12 MB)
  sibi-image/            mirror backward-compat
  README.md              info dataset
```

---

## 17. Menjalankan dan build

```sh
npm install
npm run dev      # dev server http://localhost:5173
npm run check    # type-check (harus 0 error)
npm run build    # output statis di build/
npm run preview  # preview build
```

Test matematika/smoother:

```sh
node tests/smoother.test.mjs
node tests/motion.test.mjs
```

Catatan: buka lewat http (dev/preview), bukan `file://`, supaya fetch model
dan izin kamera berfungsi.

---

## 18. Pelatihan ulang dan matematika pelatihan

Dependensi Python:

```sh
pip install tensorflow scikit-learn pandas numpy
```

### 18.1 Normalisasi data latih

`python/retrain_joint.py` memakai `normalize()` yang sama dengan
`landmarksToFeatures`:

```python
d = hands - wrist
size = norm(hands[:, 9] - wrist) + 1e-6
d = d / size
phi = atan2(d[:, 9, 0], d[:, 9, 1])
x' = x*cos(phi) - y*sin(phi)
y' = x*sin(phi) + y*cos(phi)
if thumb_cmc.x < 0: mirror semua x
flatten -> 63
```

### 18.2 Augmentasi

Setiap sampel dibuat 32 variasi (total 33x termasuk asli):

```text
1) mirror horizontal dengan probabilitas 0.5
2) skala panjang tiap jari: f ~ U(0.90, 1.10)
   G_j = G_mcp + (G_j - G_mcp) * f
3) skala global: s ~ U(0.8, 1.25), G = s * G
4) rotasi: sudut ~ U(-45, 45 derajat)
5) shear: sh ~ U(-0.10, 0.10), x = x + sh * y
6) jitter: G = G + N(0, 0.02^2)
```

Tujuannya mensimulasikan orang lain (panjang jari berbeda), kamera miring,
jarak berbeda, dan noise tracking.

### 18.3 Loss dan optimisasi

```text
C = 24 kelas
epsilon = 0.05
q_c = (1 - epsilon) * y_c + epsilon / C
L = -sum_i cw_{y_i} * sum_c q_{i,c} * log(p_{i,c})
```

- Optimizer Adam, learning rate 3e-4.
- Batch 64.
- Dropout 0.5, 0.4, 0.25.
- Early stopping: patience 50 pada `val_accuracy`.
- ReduceLROnPlateau: factor 0.5, patience 15 pada `val_loss`.
- Maksimum 500 epoch.

### 18.4 Augmentasi citra

Untuk model MobileNetV2:

```text
RandomFlip("horizontal")
RandomRotation(0.10)        # 0.10 * 360 = 36 derajat
RandomZoom(0.15)
RandomTranslation(0.10, 0.10)
RandomBrightness(0.20)
RandomContrast(0.15)
Rescaling(1/127.5, offset=-1)
```

### 18.5 Metrik evaluasi

```text
accuracy     = benar / total
precision_c  = TP_c / (TP_c + FP_c)
recall_c     = TP_c / (TP_c + FN_c)
F1_c         = 2 * precision_c * recall_c / (precision_c + recall_c)
macro-F1     = rata-rata F1 semua kelas
top-3 acc    = fraksi sampel yang label benarnya ada di 3 probabilitas teratas
```

### 18.6 Perintah latih ulang

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

CNN citra (Kaggle, butuh `kaggle.json`):

```sh
python python/train_sibi.py --data ./SIBI --epochs 30 --img 128
```

CNN citra advanced (80/20):

```sh
python python/train_sibi_advanced.py --data "C:\path\SIBI" --out ./training_output/sibi_advanced_80_20 --epochs 30 --img 128 --batch 32
```

---

## 19. Evaluasi dan batasan

### 19.1 Angka offline

MLP sendi v2 (validasi 124 sampel, 24 huruf A-I K-Y):

| Kondisi | Akurasi |
|---|---|
| Bersih | 90,3% |
| Tangan lain (mirror) | 90,3% |
| Kamera miring +/-30 derajat | 90,3% |
| Jarak 0,7x-1,4x | 90,3% |
| Simulasi orang lain (jari +/-8%) | 86-90% |

Model citra advanced: test 84,8%, top-3 95,6%, macro-F1 84,8% pada split
80/20 (4224/1056, seed 42).

Baseline Conv1D: val 74,2%.

Angka ini adalah akurasi offline pada dataset, bukan jaminan akurasi live.
Kualitas webcam, cahaya, latar, dan bentuk tangan pengguna sangat
mempengaruhi hasil.

### 19.2 Batasan yang jujur

- Model statis hanya 24 huruf. J dan Z ditangani terpisah lewat gerakan.
- Confidence softmax tidak dikalibrasi sempurna. Nilai 0,6 bukan berarti
  "60% benar" dalam arti statistik.
- Fusi `+0.07` adalah heuristik, bukan model probabilistik.
- Isyarat yang mirip (M/N, S/T, E/O) masih bisa tertukar; chip kandidat
  top-3 disediakan untuk koreksi cepat.
- SIBI adalah bahasa isyarat berbasis alfabet, bukan bahasa isyarat alami
  dengan tata bahasa sendiri. Aplikasi ini menerjemahkan huruf, bukan
  memahami BISINDO.
- Kamera hanya memakai satu tangan; dua tangan sekaligus tidak didukung.
- Gerakan J/Z peka terhadap kecepatan dan ukuran gerakan. Kalau gagal,
  gunakan tombol J/Z.

---

## 20. Pemecahan masalah

**Model gagal dimuat.**
Cek koneksi internet (WASM MediaPipe dari CDN, model dari `static/models`).
Tekan **Coba lagi** di kartu model. Kalau `Warmup gagal`, model tidak
dilaporkan siap secara palsu.

**Kamera kosong.**
Tutup aplikasi lain yang memakai kamera, izinkan browser mengakses kamera,
lalu pilih perangkat lain di dropdown.

**Tangan tidak terdeteksi.**
Cahaya cukup, latar polos, tangan di tengah frame. Kalau tangan terlalu
kecil, dekatkan ke kamera (area minimal 4% frame).

**Huruf tidak masuk.**
Tahan isyarat 0,4-0,8 detik tanpa gerakan besar. Lihat progress bar:
kalau penuh tetapi tidak commit, kemungkinan:
- dua label masih berimbang (`share < 0.58`),
- cooldown 450 ms belum lewat,
- huruf sama dengan commit terakhir dan repeat guard 900 ms belum lewat.

**Huruf yang sama tidak mau berulang.**
Itu disengaja. Untuk "AA", tahan lebih lama (>= 900 ms sejak commit
terakhir) atau ubah isyarat lalu kembali.

**Satu huruf tertentu selalu salah.**
Itu biasanya keterbatasan model pada bentuk tangan pengguna, bukan timing.
Gunakan chip kandidat top-3, atau rekam sampel huruf tersebut di panel debug
lalu latih ulang.

**J/Z tidak terbaca.**
Pastikan bentuk dasar benar (telunjuk untuk Z, kelingking untuk J), gerakan
cukup besar (lihat tabel parameter), dan tidak terlalu cepat/lambat
(280-1800 ms). Tombol J/Z selalu tersedia.

**Terlalu gelap/terang.**
Tambah cahaya atau kurangi exposure. Tidak ada gerbang luminance otomatis
di versi ini; kualitas gambar langsung mempengaruhi MediaPipe.

**Riwayat tidak tersimpan.**
Periksa IndexedDB/localStorage tidak diblokir mode incognito.

**UI tersendat.**
Lihat chip FPS dan `sendi ... ms`. Kalau inferensi > 45 ms, interval otomatis
turun ke 110 ms. Mematikan vote citra (default memang off) membantu mesin
lambat.
