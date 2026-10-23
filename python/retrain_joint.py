"""Retrain SIBI joint classifier v2 — invariant to hand, angle, and person.

Problem v1 (val 87.1% on clean CSV, but brittle live):
  same letter + other hand (left/right) -> different vector
  same letter + tilted camera -> different vector
  same letter + other person (finger ratios) -> different vector
  -> confusing alphabets (M/N, S/T, E/O ...)

Fix v2 — bake the invariance into the FEATURES (not hoped from data):
  1. wrist-relative (translation invariant, same as v1)
  2. scale by wrist->middle_mcp (size/distance invariant, same as v1)
  3. NEW: rotate in XY so middle_mcp points +Y (camera-roll invariant)
  4. NEW: mirror X so thumb base is always +X (left/right-hand invariant)
  5. NEW: augment per-finger bone length +-10% (simulates other persons),
     rotation +-45 deg (tilted camera), scale 0.8-1.25, shear, jitter.

Input : 63 landmark values in CSV column order (= MediaPipe indices 0..20, xyz)
Output: 24 softmax classes (A-I, K-Y; J/Z are dynamic SIBI, excluded like upstream)
Saves : sibi_joint_best.h5, labels.json, metrics.json

Data: AJustiago/SIBI-Recognition hands_SIBI_*.csv (MIT, official SIBI dict,
same alphabet as Kaggle alvinbintang/sibi-dataset which needs a login token).
To retrain on the exact Kaggle image folders use python/train_sibi.py instead.

Must stay in sync with src/lib/jointFeatures.ts landmarksToFeatures().
"""

import argparse
import json
import os
import shutil

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

WORK = os.environ.get("SIBI_WORK", r"C:\Users\BRAVO\AppData\Local\Temp\opencode\sibi-train")
REPO = os.environ.get("SIBI_REPO", r"C:\Users\BRAVO\AppData\Local\Temp\opencode\sibi-repo")
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

# finger groups (joint indices) for per-person bone-length simulation
FINGERS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
MCP_OF = {1: 1, 2: 1, 3: 1, 4: 1, 5: 5, 6: 5, 7: 5, 8: 5, 9: 9, 10: 9,
          11: 9, 12: 9, 13: 13, 14: 13, 15: 13, 16: 13, 17: 17, 18: 17, 19: 17, 20: 17}


def ensure_csvs():
    for name in ("hands_SIBI_training.csv", "hands_SIBI_validation.csv"):
        dst = os.path.join(WORK, "train.csv" if "training" in name else "val.csv")
        if not os.path.exists(dst):
            src = os.path.join(REPO, name)
            if not os.path.exists(src):
                raise SystemExit(f"missing {src}: clone AJustiago/SIBI-Recognition to {REPO}")
            shutil.copy(src, dst)
            print(f"copied {src} -> {dst}")


def load(csv_path):
    df = pd.read_csv(csv_path).sort_values("class_type").reset_index(drop=True)
    labels = sorted(df["class_type"].unique().tolist())
    code = {c: i for i, c in enumerate(labels)}
    y = df["class_type"].map(code).to_numpy()
    X = df[[c for c in COLS]].to_numpy(dtype=np.float64)
    return X.reshape(-1, 21, 3), y, labels


def load_extra(paths, code):
    """Merge user-recorded canonical CSVs (class_type,f0..f62 from the app
    sample collector). Already canonical -> concatenated AFTER normalize()."""
    Xs, ys = [], []
    for p in paths:
        df = pd.read_csv(p)
        if "class_type" not in df.columns:
            raise SystemExit(f"{p}: missing class_type column")
        fcols = [c for c in df.columns if c != "class_type"]
        if len(fcols) != 63:
            raise SystemExit(f"{p}: expected 63 feature cols, got {len(fcols)}")
        bad = sorted(set(df["class_type"].unique().tolist()) - set(code))
        if bad:
            raise SystemExit(f"{p}: unknown labels {bad} (want subset of {sorted(code)})")
        Xs.append(df[fcols].to_numpy(dtype=np.float32))
        ys.append(df["class_type"].map(code).to_numpy())
        print(f"extra {p}: {len(df)} rows")
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)


def normalize(hands):
    """hands: (N,21,3) pixel coords -> (N,63) canonical invariant features.

    Steps (MUST match landmarksToFeatures in src/lib/jointFeatures.ts):
      d = joints - wrist; d /= ||middle_mcp - wrist||
      rotate XY by atan2(mx,my) so middle_mcp lands on +Y axis
      if thumb_cmc.x < 0: mirror X (left hand becomes right hand)
    """
    wrist = hands[:, 0:1, :]
    d = hands - wrist
    size = np.linalg.norm(hands[:, 9:10, :] - wrist, axis=-1, keepdims=True) + 1e-6
    d = d / size
    mx, my = d[:, 9, 0], d[:, 9, 1]
    phi = np.arctan2(mx, my)
    ca, sa = np.cos(phi), np.sin(phi)
    x = d[:, :, 0] * ca[:, None] - d[:, :, 1] * sa[:, None]
    y = d[:, :, 0] * sa[:, None] + d[:, :, 1] * ca[:, None]
    d[:, :, 0], d[:, :, 1] = x, y
    flip = d[:, 1, 0] < 0  # thumb base on -X -> mirror to +X
    d[flip, :, 0] *= -1
    return d.reshape(len(hands), -1).astype(np.float32)


def augment(feats, rng, copies=32):
    """feats: (N,63) canonical features. Simulate new angles/persons/noise."""
    F = feats.reshape(-1, 21, 3)
    out = [F]
    n = len(F)
    for _ in range(copies):
        G = F.copy()
        if rng.random() < 0.5:
            G[:, :, 0] *= -1  # other hand (redundant w/ canonical, keeps robustness)
        # per-person finger length ratios: scale each finger from its MCP
        for joints in FINGERS:
            f = rng.uniform(0.90, 1.10, size=(n, 1, 1))
            mcp = joints[0]
            G[:, joints, :] = G[:, mcp:mcp + 1, :] + (G[:, joints, :] - G[:, mcp:mcp + 1, :]) * f
        s = rng.uniform(0.8, 1.25, size=(n, 1, 1))  # distance / hand size
        G = G * s
        ang = rng.uniform(-0.785, 0.785, size=n)  # +-45 deg tilted camera
        ca, sa = np.cos(ang), np.sin(ang)
        x = G[:, :, 0] * ca[:, None] - G[:, :, 1] * sa[:, None]
        y = G[:, :, 0] * sa[:, None] + G[:, :, 1] * ca[:, None]
        G[:, :, 0], G[:, :, 1] = x, y
        sh = rng.uniform(-0.10, 0.10, size=(n, 1, 1))  # perspective shear
        G[:, :, 0] = G[:, :, 0] + sh[:, :, 0] * G[:, :, 1]
        G += rng.normal(0, 0.02, size=G.shape)  # tracking jitter
        out.append(G)
    return np.concatenate(out, axis=0).reshape(-1, 63).astype(np.float32)


def probe_acc(model, Xva_raw, yva, rng):
    """Robustness probes: accuracy under synthetic hand/angle/person shifts."""
    Xc = normalize(Xva_raw)
    base = (model.predict(Xc, verbose=0).argmax(1) == yva).mean()
    res = {"clean": float(base)}
    F = Xc.reshape(-1, 21, 3)
    # other hand
    G = F.copy()
    G[:, :, 0] *= -1
    res["mirror_hand"] = float((model.predict(G.reshape(-1, 63), verbose=0).argmax(1) == yva).mean())
    # tilted camera +-30 deg
    for deg in (30, -30):
        a = np.deg2rad(deg)
        ca, sa = np.cos(a), np.sin(a)
        G = F.copy()
        x = G[:, :, 0] * ca - G[:, :, 1] * sa
        y = G[:, :, 0] * sa + G[:, :, 1] * ca
        G[:, :, 0], G[:, :, 1] = x, y
        res[f"rot{deg}"] = float((model.predict(G.reshape(-1, 63), verbose=0).argmax(1) == yva).mean())
    # other person: finger ratios +-8%
    G = F.copy()
    for joints in FINGERS:
        mcp = joints[0]
        G[:, joints, :] = G[:, mcp:mcp + 1, :] + (G[:, joints, :] - G[:, mcp:mcp + 1, :]) * 1.08
    res["person_long"] = float((model.predict(G.reshape(-1, 63), verbose=0).argmax(1) == yva).mean())
    G = F.copy() + rng.normal(0, 0.02, size=F.shape)
    res["jitter"] = float((model.predict(G.reshape(-1, 63).astype(np.float32), verbose=0).argmax(1) == yva).mean())
    return res


def main():
    ap = argparse.ArgumentParser(description="Latih MLP sendi SIBI v2 (kanonis).")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="CSV kanonis dari kolektor sampel app (class_type,f0..f62).")
    args = ap.parse_args()

    ensure_csvs()
    rng = np.random.default_rng(7)
    Xtr_raw, ytr, labels = load(os.path.join(WORK, "train.csv"))
    Xva_raw, yva, labels_va = load(os.path.join(WORK, "val.csv"))
    assert labels == labels_va, (labels, labels_va)
    print("classes:", labels)

    Xtr = normalize(Xtr_raw)
    Xva = normalize(Xva_raw)
    n_extra = 0
    if args.extra:
        code = {c: i for i, c in enumerate(labels)}
        Xe, ye = load_extra(args.extra, code)
        Xtr = np.concatenate([Xtr, Xe], axis=0)
        ytr = np.concatenate([ytr, ye], axis=0)
        n_extra = len(ye)
    print(f"train clean: {len(ytr)} (+{n_extra} extra)")

    Xtr = normalize(Xtr_raw)
    Xva = normalize(Xva_raw)
    ytr_aug = np.tile(ytr, 33)
    Xtr_aug = augment(Xtr, rng, copies=32)
    print("train:", Xtr_aug.shape, "val:", Xva.shape)

    n_cls = len(labels)
    ytr_oh = tf.keras.utils.to_categorical(ytr_aug, n_cls)
    yva_oh = tf.keras.utils.to_categorical(yva, n_cls)
    counts = np.bincount(ytr, minlength=n_cls)
    cw = {i: float(len(ytr) / (n_cls * max(1, c))) for i, c in enumerate(counts)}

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(63,)),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(len(labels), activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(3e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )
    cbs = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=50, restore_best_weights=True, mode="max"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", patience=15, factor=0.5, verbose=1
        ),
    ]
    model.fit(
        Xtr_aug, ytr_oh, epochs=500, batch_size=64, class_weight=cw,
        validation_data=(Xva, yva_oh), callbacks=cbs, verbose=1,
    )

    for name, X, y in [("train_clean", Xtr, ytr), ("val", Xva, yva)]:
        p = model.predict(X, verbose=0).argmax(1)
        print(f"{name} acc: {(p == y).mean():.4f}")
    pv = model.predict(Xva, verbose=0).argmax(1)
    print(classification_report(yva, pv, target_names=labels, digits=4))
    print("confusion matrix:\n", confusion_matrix(yva, pv))
    probes = probe_acc(model, Xva_raw, yva, rng)
    print("robustness probes:", json.dumps(probes, indent=2))

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
                "robustness": probes,
                "n_extra_user_samples": n_extra,
                "features": "canonical v2: wrist-relative / hand-size norm + rotate middle_mcp to +Y + mirror thumb to +X",
                "classes": labels,
            },
            f,
            indent=2,
        )
    print("saved to", OUT)


if __name__ == "__main__":
    main()
