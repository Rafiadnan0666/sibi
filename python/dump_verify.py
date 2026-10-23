"""Dump verification vectors: features + Keras reference probabilities."""

import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import tensorflow as tf

WORK = r"C:\Users\BRAVO\AppData\Local\Temp\opencode\sibi-train"
COLS = (
    ["wrist"]
    + ["thumb_Cmc", "thumb_Mcp", "thumb_Ip", "thumb_Tip"]
    + [f"{f}_{p}" for f in ("index", "middle", "ring", "pinky") for p in ("Mcp", "Pip", "Dip", "Tip")]
)
COLS = [f"{j}{a}" for j in COLS for a in ("X", "Y", "Z")]


def normalize(hands):
    """Canonical v2 (MUST match retrain_joint.normalize + jointFeatures.ts)."""
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
    flip = d[:, 1, 0] < 0
    d[flip, :, 0] *= -1
    return d.reshape(len(hands), -1).astype(np.float32)


va = pd.read_csv(os.path.join(WORK, "val.csv")).sort_values("class_type").reset_index(drop=True)
labels = sorted(va["class_type"].unique().tolist())
Xraw = va[COLS].to_numpy(dtype=np.float64).reshape(-1, 21, 3)
Xnorm = normalize(Xraw)

joint = tf.keras.models.load_model(os.path.join(WORK, "joint_model", "sibi_joint_best.h5"))
try:
    base = tf.keras.models.load_model(os.path.join(WORK, "upstream_model.h5"), compile=False)
except Exception as e:
    print("baseline skipped (optional):", e)
    base = None

# 2 samples per class for a solid check (48 vectors)
idx = []
for c in labels:
    rows = va.index[va["class_type"] == c].tolist()[:2]
    idx.extend(rows)

out = {"labels": labels, "joint": [], "baseline": []}
pj = joint.predict(Xnorm[idx], verbose=0)
pb = (
    base.predict(Xraw[idx].astype(np.float32).reshape(len(idx), 63, 1), verbose=0)
    if base is not None
    else None
)
for n, i in enumerate(idx):
    out["joint"].append(
        {"feat": Xnorm[i].tolist(), "probs": pj[n].tolist(), "label": va.loc[i, "class_type"]}
    )
    if pb is not None:
        out["baseline"].append(
            {
                "feat": Xraw[i].astype(np.float32).reshape(-1).tolist(),
                "probs": pb[n].tolist(),
                "label": va.loc[i, "class_type"],
            }
        )

with open(os.path.join(WORK, "verify_vectors.json"), "w") as f:
    json.dump(out, f)
print("wrote", len(idx), "vectors x2 models")
pj_acc = np.mean(pj.argmax(1) == np.array([labels.index(va.loc[i, 'class_type']) for i in idx]))
print("joint subset acc:", pj_acc)
