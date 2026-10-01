# Model SIBI: 3 model TF.js nyata

| Folder | Model | Val acc | Ukuran | Sumber |
|---|---|---|---|---|
| `sibi-joint/` | MLP 63-fitur sendi (primer) | **87,1%** | ~215 KB | `python/retrain_joint.py` |
| `sibi-joint-baseline/` | Conv1D upstream (vote-2) | 74,2% | ~4,2 MB | AJustiago/SIBI-Recognition (MIT) |
| `sibi-image/` | MobileNetV2 ROI (cross-check) | 73,8% | ~2,7 MB | `python/train_image.py` |

24 kelas: A-I, K-Y. J dan Z dikecualikan karena isyarat gerak, bukan handshape statis.

## Reproduksi penuh

```sh
pip install tensorflow tensorflowjs scikit-learn pandas numpy
python python/retrain_joint.py      # MLP sendi, ±1 mnt CPU -> joint_model/
python python/train_image.py        # MobileNetV2, ±15 mnt CPU -> image_model/ + image_sm/
python python/export_savedmodel.py  # .h5 -> SavedModel (joint_sm/, baseline_sm/)

$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python'
tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model joint_sm static/models/sibi-joint
tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model baseline_sm static/models/sibi-joint-baseline
tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --quantization_bytes=1 image_sm static/models/sibi-image
```

Salin `labels.json` masing-masing ke folder tujuannya, lalu `npm run build`.

## Verifikasi numerik (wajib lolos sebelum klaim "model nyata")

```sh
python python/dump_verify.py        # vektor uji + probabilitas referensi Keras
node scripts/verify-models.mjs ...   # TF.js vs Keras, toleransi 2e-2 (aktual: 7e-7)
node scripts/verify-norm.mjs        # normalisasi TS vs Python (aktual: 2.4e-7)
```

## Data

Sampel sendi (272 latih / 124 uji) + foto (480/240) dari
[AJustiago/SIBI-Recognition](https://github.com/AJustiago/SIBI-Recognition) (MIT),
kamus SIBI resmi, setara alfabet Kaggle
[`alvinbintang/sibi-dataset`](https://www.kaggle.com/datasets/alvinbintang/sibi-dataset)
(Kaggle butuh token login; untuk data Kaggle persisnya pakai
`python/train_sibi.py --data ./SIBI`).
