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
    wrist = hands[:, 0:1, :]
    d = hands - wrist
    size = np.linalg.norm(hands[:, 9:10, :] - wrist, axis=-1, keepdims=True) + 1e-6
    return (d / size).reshape(len(hands), -1).astype(np.float32)


va = pd.read_csv(os.path.join(WORK, "val.csv")).sort_values("class_type").reset_index(drop=True)
labels = sorted(va["class_type"].unique().tolist())
Xraw = va[COLS].to_numpy(dtype=np.float64).reshape(-1, 21, 3)
Xnorm = normalize(Xraw)

joint = tf.keras.models.load_model(os.path.join(WORK, "joint_model", "sibi_joint_best.h5"))
base = tf.keras.models.load_model(os.path.join(WORK, "upstream_model.h5"), compile=False)

# 2 samples per class for a solid check (48 vectors)
idx = []
for c in labels:
    rows = va.index[va["class_type"] == c].tolist()[:2]
    idx.extend(rows)

out = {"labels": labels, "joint": [], "baseline": []}
pj = joint.predict(Xnorm[idx], verbose=0)
pb = base.predict(Xraw[idx].astype(np.float32).reshape(len(idx), 63, 1), verbose=0)
for n, i in enumerate(idx):
    out["joint"].append(
        {"feat": Xnorm[i].tolist(), "probs": pj[n].tolist(), "label": va.loc[i, "class_type"]}
    )
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
