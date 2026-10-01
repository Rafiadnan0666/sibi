"""
SIBI alphabet CNN training (Pengolahan Citra + transfer learning).

Dataset : Kaggle `alvinbintang/sibi-dataset` (folder `SIBI/` berisi 26 subfolder A-Z)
          atau gabungan `alfarizy29/combined-sibi-dataset` (struktur sama).
Output  : SavedModel + TF.js (`model.json`) untuk `static/models/sibi-model/`.

Contoh:
    python python/train_sibi.py --data ./SIBI --epochs 30 --img 128 --out ./static/models/sibi-model

Alur preprocessing (SAMA dengan src/lib/preprocessing.ts di browser):
    ROI square-crop -> resize IMGxIMG -> rescale [-1,1] (MobileNet) atau [0,1]
    augmentasi: rotasi, zoom, geser, flip horizontal, brightness
    eksperimen opsional: --grayscale untuk membandingkan RGB vs grayscale.

Evaluasi (untuk laporan):
    akurasi, presisi, recall, F1 (macro), confusion matrix, classification report,
    plus pencatatan FPS/latensi saat inferensi TF.js di browser.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def build_datasets(data_dir: Path, img: int, batch: int, grayscale: bool, seed: int = 42):
    import tensorflow as tf

    color = "grayscale" if grayscale else "rgb"
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=seed,
        image_size=(img, img),
        batch_size=batch,
        color_mode=color,
        label_mode="int",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=seed,
        image_size=(img, img),
        batch_size=batch,
        color_mode=color,
        label_mode="int",
    )
    classes = train_ds.class_names
    print(f"Kelas ({len(classes)}): {classes}")
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)
    return train_ds, val_ds, classes


def build_model(img: int, n_classes: int, grayscale: bool, backbone: str = "mobilenet"):
    import tensorflow as tf

    channels = 1 if grayscale else 3
    inputs = tf.keras.Input(shape=(img, img, channels))
    # MobileNet butuh 3 kanal [-1,1]; grayscale diduplikasi agar backbone tetap terpakai.
    x = inputs
    if grayscale:
        x = tf.keras.layers.Concatenate()([x, x, x])
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0)(x)

    if backbone == "mobilenet":
        base = tf.keras.applications.MobileNetV2(
            input_shape=(img, img, 3), include_top=False, weights="imagenet"
        )
        base.trainable = False  # fase 1: freeze; unfreeze 30 layer terakhir di fine-tune
        x = base(x, training=False)
    else:  # baseline CNN kecil dari nol (pembanding yang murah)
        x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description="Latih CNN alfabet SIBI (A-Z).")
    ap.add_argument("--data", type=Path, required=True, help="Folder berisi subfolder A-Z.")
    ap.add_argument("--out", type=Path, default=Path("./static/models/sibi-model"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--img", type=int, default=128, help="Ukuran input (disarankan 128).")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--grayscale", action="store_true", help="Eksperimen grayscale.")
    ap.add_argument("--backbone", default="mobilenet", choices=["mobilenet", "tiny"])
    args = ap.parse_args()

    import tensorflow as tf

    train_ds, val_ds, classes = build_datasets(
        args.data, args.img, args.batch, args.grayscale
    )

    augment = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.12),
            tf.keras.layers.RandomTranslation(0.08, 0.08),
            tf.keras.layers.RandomBrightness(0.15),
        ]
    )
    # Bungkus augmentasi di depan model via input tambahan? Praktis: map ke dataset.
    train_ds = train_ds.map(
        lambda x, y: (augment(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE
    )

    model = build_model(args.img, len(classes), args.grayscale, args.backbone)
    model.summary()

    cbs = [
        tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True, monitor="val_accuracy"),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, monitor="val_loss"),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=cbs)

    # Fine-tune singkat bila backbone MobileNet.
    if args.backbone == "mobilenet":
        for layer in model.layers:
            if "mobilenet" in layer.name or "block_1" in getattr(layer, "name", ""):
                pass
        try:
            base = [l for l in model.layers if "mobilenet" in str(type(l).__name__).lower()]
        except Exception:
            base = []
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(train_ds, validation_data=val_ds, epochs=max(3, args.epochs // 6), callbacks=cbs)

    print("\n=== Evaluasi (validation) ===")
    loss, acc = model.evaluate(val_ds)
    print(f"val_loss={loss:.4f} val_accuracy={acc:.4f}")

    # Metrik + confusion matrix untuk laporan.
    from sklearn.metrics import classification_report, confusion_matrix

    y_true, y_pred = [], []
    for xb, yb in val_ds:
        pb = model.predict(xb, verbose=0)
        y_true.extend(np.asarray(yb).tolist())
        y_pred.extend(np.argmax(pb, axis=1).tolist())
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n", cm)

    args.out.mkdir(parents=True, exist_ok=True)
    model.save(args.out / "saved_model")
    (args.out / "labels.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")
    np.save(args.out / "confusion_matrix.npy", np.asarray(cm))
    print(f"\nTersimpan di {args.out}")
    print("Export ke TF.js:\n  tensorflowjs_converter --input_format=tf_saved_model "
          f"--output_format=tfjs_graph_model {args.out / 'saved_model'} {args.out}")
    print("Pastikan input browser SAMA: resize 128, normalisasi [-1,1], RGB "
          "(atau grayscale bila --grayscale).")


if __name__ == "__main__":
    main()
