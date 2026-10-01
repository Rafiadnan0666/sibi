"""Retrain SIBI joint classifier on upstream CSVs with PROPER normalization.

Upstream flaw: raw pixel coords (mixed image scales) -> overfit (98% train / 74% val).
Fix: wrist-relative + hand-size normalized features (translation & scale invariant),
mirror augmentation (left/right-hand robust), jitter, dropout, early stopping.

Input : 63 landmark values in CSV column order (= MediaPipe indices 0..20, xyz)
Output: 24 softmax classes (A-I, K-Y; J/Z are dynamic SIBI, excluded like upstream)
Saves : sibi_joint_best.h5, labels.json, metrics.json
"""

import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

WORK = r"C:\Users\BRAVO\AppData\Local\Temp\opencode\sibi-train"
OUT = os.path.join(WORK, "joint_model")
os.makedirs(OUT, exist_ok=True)

# 21 joints x xyz in CSV column order
JOINTS = (
    ["wrist"]
    + [f"thumb_{p}" for p in ("Cmc", "Mcp", "Ip", "Tip")]
    + [f"{f}_{p}" for f in ("index", "middle", "ring", "pinky") for p in ("Mcp", "Pip", "Dip", "Tip")]
)
COLS = [f"{j}{a}" for j in JOINTS for a in ("X", "Y", "Z")]
assert len(COLS) == 63


def load(csv_path):
    df = pd.read_csv(csv_path).sort_values("class_type").reset_index(drop=True)
    labels = sorted(df["class_type"].unique().tolist())
    code = {c: i for i, c in enumerate(labels)}
    y = df["class_type"].map(code).to_numpy()
    X = df[[c for c in COLS]].to_numpy(dtype=np.float64)
    return X.reshape(-1, 21, 3), y, labels


def normalize(hands):
    """hands: (N,21,3) pixel coords -> (N,63) invariant features."""
    wrist = hands[:, 0:1, :]
    d = hands - wrist
    size = np.linalg.norm(hands[:, 9:10, :] - wrist, axis=-1, keepdims=True) + 1e-6
    return (d / size).reshape(len(hands), -1).astype(np.float32)


def augment(feats, rng, copies=8):
    """feats: (N,63) normalized. Mirror + jitter + scale + planar rotation."""
    F = feats.reshape(-1, 21, 3)
    out = [F]
    for _ in range(copies):
        G = F.copy()
        if rng.random() < 0.5:  # left/right hand mirror
            G[:, :, 0] *= -1
        s = rng.uniform(0.9, 1.1, size=(len(G), 1, 1))
        G = G * s
        ang = rng.uniform(-0.25, 0.25, size=len(G))
        ca, sa = np.cos(ang), np.sin(ang)
        x = G[:, :, 0] * ca[:, None] - G[:, :, 1] * sa[:, None]
        y = G[:, :, 0] * sa[:, None] + G[:, :, 1] * ca[:, None]
        G[:, :, 0], G[:, :, 1] = x, y
        G += rng.normal(0, 0.012, size=G.shape)  # tracking jitter
        out.append(G)
    return np.concatenate(out, axis=0).reshape(-1, 63).astype(np.float32)


def main():
    rng = np.random.default_rng(7)
    Xtr_raw, ytr, labels = load(os.path.join(WORK, "train.csv"))
    Xva_raw, yva, labels_va = load(os.path.join(WORK, "val.csv"))
    assert labels == labels_va, (labels, labels_va)
    print("classes:", labels)

    Xtr = normalize(Xtr_raw)
    Xva = normalize(Xva_raw)
    ytr_aug = np.tile(ytr, 9)
    Xtr_aug = augment(Xtr, rng, copies=8)
    print("train:", Xtr_aug.shape, "val:", Xva.shape)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(63,)),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.45),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(len(labels), activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(3e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    cbs = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=40, restore_best_weights=True, mode="max"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", patience=12, factor=0.5, verbose=1
        ),
    ]
    model.fit(
        Xtr_aug, ytr_aug, epochs=400, batch_size=64,
        validation_data=(Xva, yva), callbacks=cbs, verbose=1,
    )

    for name, X, y in [("train_clean", Xtr, ytr), ("val", Xva, yva)]:
        p = model.predict(X, verbose=0).argmax(1)
        print(f"{name} acc: {(p == y).mean():.4f}")
    pv = model.predict(Xva, verbose=0).argmax(1)
    print(classification_report(yva, pv, target_names=labels, digits=4))
    print("confusion matrix:\n", confusion_matrix(yva, pv))

    model.save(os.path.join(OUT, "sibi_joint_best.h5"))
    with open(os.path.join(OUT, "labels.json"), "w") as f:
        json.dump(labels, f)
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(
            {
                "val_accuracy": float((pv == yva).mean()),
                "train_clean_accuracy": float(
                    (model.predict(Xtr, verbose=0).argmax(1) == ytr).mean()
                ),
                "features": "wrist-relative / hand-size normalized 63-vector",
                "classes": labels,
            },
            f,
            indent=2,
        )
    print("saved to", OUT)


if __name__ == "__main__":
    main()
