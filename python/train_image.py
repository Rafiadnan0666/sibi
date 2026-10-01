"""Train SIBI alphabet image CNN (transfer learning) on open SIBI hand images.

Data: AJustiago/SIBI-Recognition Dataset_Training (480 img) / Dataset_Validation
(240 img), 24 classes A-I,K-Y (J/Z are dynamic SIBI). Images originate from the
official SIBI dictionary (pmpk.kemdikbud.go.id/sibi), same alphabet as the Kaggle
`alvinbintang/sibi-dataset` (Kaggle download needs a login token; to retrain on
the exact Kaggle folders use python/train_sibi.py --data ./SIBI).

Pipeline (mirrors src/lib/preprocessing.ts): resize IMG -> rescale [-1,1].
Saves: sibi_image_best.h5, SavedModel, labels.json, metrics.json.
TF.js conversion via CLI (see bottom).
"""

import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

REPO = r"C:\Users\BRAVO\AppData\Local\Temp\opencode\sibi-repo"
WORK = r"C:\Users\BRAVO\AppData\Local\Temp\opencode\sibi-train"
OUT = os.path.join(WORK, "image_model")
os.makedirs(OUT, exist_ok=True)

IMG = 128
BATCH = 32
SEED = 7


def make_ds(subdir, training):
    ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(REPO, subdir),
        image_size=(IMG, IMG),
        batch_size=BATCH,
        shuffle=training,
        seed=SEED,
    )
    return ds


def main():
    train_ds = make_ds("Dataset_Training", True)
    val_ds = make_ds("Dataset_Validation", False)
    classes = train_ds.class_names
    print("classes:", classes)
    assert classes == val_ds.class_names

    augment = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.12),
            tf.keras.layers.RandomTranslation(0.08, 0.08),
            tf.keras.layers.RandomBrightness(0.15),
            tf.keras.layers.RandomContrast(0.1),
        ]
    )
    norm = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0)
    AUT = tf.data.AUTOTUNE
    train_ds = train_ds.map(
        lambda x, y: (augment(x, training=True), y), num_parallel_calls=AUT
    ).map(lambda x, y: (norm(x), y), num_parallel_calls=AUT).prefetch(AUT)
    val_ds = val_ds.map(lambda x, y: (norm(x), y), num_parallel_calls=AUT).prefetch(AUT)

    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG, IMG, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False
    inputs = tf.keras.Input(shape=(IMG, IMG, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(len(classes), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    cbs = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=6, restore_best_weights=True, mode="max"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=25, callbacks=cbs, verbose=1)

    # short fine-tune of the top of the backbone
    base.trainable = True
    for layer in base.layers[:-25]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=6, callbacks=cbs, verbose=1)

    y_true, y_pred = [], []
    for xb, yb in val_ds:
        pb = model.predict(xb, verbose=0)
        y_true.extend(np.asarray(yb).tolist())
        y_pred.extend(np.argmax(pb, axis=1).tolist())
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))
    cm = confusion_matrix(y_true, y_pred)
    print("confusion matrix:\n", cm)

    model.save(os.path.join(OUT, "sibi_image_best.h5"))
    model.export(os.path.join(WORK, "image_sm"))
    with open(os.path.join(OUT, "labels.json"), "w") as f:
        json.dump(classes, f)
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(
            {
                "val_accuracy": float(np.mean(np.array(y_pred) == np.array(y_true))),
                "classes": classes,
                "input": f"{IMG}x{IMG} RGB [-1,1]",
            },
            f,
            indent=2,
        )
    print("saved to", OUT)


if __name__ == "__main__":
    main()
