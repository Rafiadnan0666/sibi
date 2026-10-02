# Model SIBI: 1 utama + 2 vote opsional (TF.js nyata)

| Folder | Model | Val acc | Ukuran | Sumber |
|---|---|---|---|---|
| `sibi-joint/` | MLP kanonis v2 63-fitur sendi (UTAMA, dimuat saat startup) | **90,3%** | ~236 KB | `python/retrain_joint.py` |
| `sibi-joint-baseline/` | Conv1D upstream (vote opsional, lazy) | 74,2% | ~4,2 MB | AJustiago/SIBI-Recognition (MIT) |
| `sibi-advanced/` | MobileNetV2 128px ROI (vote opsional, lazy) — TRAINED ON YOUR DATA | **84,8%** (top-3 95,6%) | ~12 MB | `python/train_sibi_advanced.py` |
| `sibi-image/` | mirror of `sibi-advanced/` (legacy URL compat) | **84,8%** | ~12 MB | same as above |

24 kelas: A-I, K-Y. J dan Z dikecualikan karena isyarat gerak, bukan handshape statis.

Fitur kanonis v2 (tahan tangan/sudut/jarak): pergelangan-relatif, normalisasi
ukuran tangan, putar jari tengah ke +Y (`atan2`), cerminkan pangkal jempol ke
+X. Mirror / ±30° / jarak 0,7–1,4x = vektor identik (90,3% tetap); simulasi
orang lain 86–90%. Wajib sama persis dengan `landmarksToFeatures` di
`src/lib/jointFeatures.ts`.

## Reproduksi penuh

```sh
pip install tensorflow scikit-learn pandas numpy
python python/retrain_joint.py      # MLP sendi v2, ±3 mnt CPU -> joint_model/
python python/train_image.py        # MobileNetV2, ±15 mnt CPU -> image_model/ + image_sm/
python python/export_savedmodel.py  # .h5 -> SavedModel (joint_sm/)

$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python'
tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model joint_sm static/models/sibi-joint
```

Salin `labels.json` hasil latih ke folder tujuannya, lalu `npm run build`.

## Verifikasi numerik (wajib lolos sebelum klaim "model nyata")

```sh
python python/dump_verify.py        # vektor uji + probabilitas referensi Keras
node scripts/verify-models.mjs ...   # TF.js vs Keras (aktual v2: 2,1e-7)
node scripts/verify-norm.mjs        # normalisasi TS vs Python (aktual v2: 0,0)
```

## Data

Sampel sendi (272 latih / 124 uji) dari
[AJustiago/SIBI-Recognition](https://github.com/AJustiago/SIBI-Recognition) (MIT),
kamus SIBI resmi, setara alfabet Kaggle
[`alvinbintang/sibi-dataset`](https://www.kaggle.com/datasets/alvinbintang/sibi-dataset)
(Kaggle butuh token login; untuk data Kaggle persisnya pakai
`python/train_sibi.py --data ./SIBI`).
