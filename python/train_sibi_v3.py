"""SIBI v3: full A-Z (26 classes, incl. J+Z) MobileNet-224 — 80/20 stratified.

Adapts the 98% Kaggle notebook (Denty Nirwana Bintang, Apache 2.0):
  MobileNet(v1)-imagenet 224, last 80 layers unfrozen from epoch 1,
  head GlobalMaxPooling2D/BN/Dense1024/Dropout0.3/BN/Dense512/softmax,
  Adam 1e-4, batch 32, <=50 epochs, EarlyStopping val_loss(p5)+ReduceLROnPlateau
  factor 0.2(p3), strong augmentation (rot20/shift.15/zoom.2/bright/flip).
Tweaks for this project: tf.data pipeline (no ImageDataGenerator),
  preprocess_input at train time, balanced class weights (J=45,Z=49 vs ~270),
  same stratified 80/20 seed 42, ROC-AUC + per-letter reports for proof.

In : C:\\Users\\BRAVO\\Downlo\\archive\\SIBI_v3_224\\A-Z (offline hand-crop cache)
Out: training_output/sibi_v3_az_80_20/ (metrics, reports, plots, h5, SavedModel)
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np


def collect(data_dir: Path):
    letters = sorted([d.name for d in data_dir.iterdir() if d.is_dir()
                      and len(d.name) == 1])
    paths, labels = [], []
    code = {c: i for i, c in enumerate(letters)}
    for c in letters:
        for f in sorted((data_dir / c).iterdir()):
            if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                paths.append(str(f))
                labels.append(code[c])
    return np.array(paths), np.array(labels), letters


def datasets(tr_p, tr_y, te_p, te_y, batch, seed, n_cls):
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet import preprocess_input

    def decode(p, y):
        im = tf.image.decode_jpeg(tf.io.read_file(p), channels=3)
        im = tf.image.resize(im, (224, 224))
        return tf.cast(im, tf.float32), tf.one_hot(y, n_cls)

    aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(20.0 / 360.0),
        tf.keras.layers.RandomTranslation(0.15, 0.15),
        tf.keras.layers.RandomZoom(0.20),
        tf.keras.layers.RandomBrightness(0.30),
        tf.keras.layers.RandomContrast(0.20),
    ])
    AUT = tf.data.AUTOTUNE
    tr = tf.data.Dataset.from_tensor_slices((tr_p, tr_y))
    tr = tr.shuffle(len(tr_p), seed=seed, reshuffle_each_iteration=True)
    tr = tr.map(decode, num_parallel_calls=AUT)
    tr = tr.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUT)
    tr = tr.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUT)
    tr = tr.batch(batch).prefetch(AUT)
    te = tf.data.Dataset.from_tensor_slices((te_p, te_y))
    te = te.map(decode, num_parallel_calls=AUT)
    te = te.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUT)
    te = te.batch(batch).prefetch(AUT)
    return tr, te


def build(n_cls):
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNet

    base = MobileNet(include_top=False, weights="imagenet",
                     input_shape=(224, 224, 3))
    base.trainable = False  # phase 1: frozen; unfreeze last 40 for fine-tune
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalMaxPooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    outputs = tf.keras.layers.Dense(n_cls, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss="categorical_crossentropy",
                  metrics=["accuracy",
                           tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3")])
    return model, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path,
                    default=Path(r"C:\Users\BRAVO\Downlo\archive\SIBI_v3_224"))
    ap.add_argument("--out", type=Path,
                    default=Path("./training_output/sibi_v3_az_80_20"))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import tensorflow as tf
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 f1_score, precision_score, recall_score,
                                 roc_auc_score)
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    paths, labels, letters = collect(args.data)
    print(f"classes ({len(letters)}): {letters}", flush=True)
    print(f"total: {len(paths)}", flush=True)
    assert len(letters) == 26 and "J" in letters and "Z" in letters

    tr_p, te_p, tr_y, te_y = train_test_split(
        paths, labels, test_size=0.20, random_state=args.seed, stratify=labels)
    print(f"TRAIN {len(tr_p)} TEST {len(te_p)} (80/20 stratified seed={args.seed})",
          flush=True)
    with open(out / "split_counts.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "index", "total", "train", "test"])
        for i, c in enumerate(letters):
            w.writerow([c, i, int((labels == i).sum()),
                        int((tr_y == i).sum()), int((te_y == i).sum())])

    tr_ds, te_ds = datasets(tr_p, tr_y, te_p, te_y, args.batch, args.seed,
                            len(letters))
    cw = compute_class_weight("balanced", classes=np.arange(len(letters)), y=tr_y)
    cw = {i: float(v) for i, v in enumerate(cw)}
    print("class_weight J/Z:", cw[letters.index("J")], cw[letters.index("Z")],
          flush=True)

    model, base = build(len(letters))
    model.summary(print_fn=lambda s: print(s, flush=True))
    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                         restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2,
                                             patience=3, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(str(out / "best_v3_phase1.keras"),
                                           monitor="val_loss",
                                           save_best_only=True, verbose=1),
    ]
    h1 = model.fit(tr_ds, validation_data=te_ds, epochs=args.epochs,
                   class_weight=cw, callbacks=cbs, verbose=1)

    # ---- phase 2: fine-tune last 40 backbone layers at low LR (the frozen
    # head is trained by now, so unfreezing is safe) ----
    base.trainable = True
    for layer in base.layers[:-40]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss="categorical_crossentropy",
                  metrics=["accuracy",
                           tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3")])
    ft_epochs = max(4, args.epochs // 4)
    print(f"=== fine-tune last 40 backbone layers, up to {ft_epochs} epochs "
          f"(lr=1e-5) ===", flush=True)
    cbs2 = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                         restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2,
                                             patience=2, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(str(out / "best_v3.keras"),
                                           monitor="val_loss",
                                           save_best_only=True, verbose=1),
    ]
    h2 = model.fit(tr_ds, validation_data=te_ds, epochs=ft_epochs,
                   class_weight=cw, callbacks=cbs2, verbose=1)

    loss, acc, top3 = model.evaluate(te_ds, verbose=1)
    print(f"test_loss={loss:.4f} test_acc={acc:.4f} test_top3={top3:.4f}", flush=True)

    t0 = time.time()
    yt, yp, pr = [], [], []
    for xb, yb in te_ds:
        pb = model.predict(xb, verbose=0)
        yt.extend(np.argmax(yb.numpy(), axis=1).tolist())
        yp.extend(np.argmax(pb, axis=1).tolist())
        pr.extend(pb.tolist())
    infer_ms = (time.time() - t0) / max(1, len(yt)) * 1000.0
    yt, yp, pr = np.array(yt), np.array(yp), np.array(pr)

    rep = classification_report(yt, yp, target_names=letters, digits=4)
    print(rep, flush=True)
    (out / "classification_report.txt").write_text(
        f"SIBI v3 A-Z 80/20 held-out (n={len(yt)})\n"
        f"test_loss={loss:.4f} test_acc={acc:.4f} test_top3={top3:.4f} "
        f"infer_ms_CPU={infer_ms:.2f}\n\n" + rep, encoding="utf-8")
    cm = confusion_matrix(yt, yp)
    np.save(out / "confusion_matrix.npy", cm)
    try:
        auc = float(roc_auc_score(np.eye(len(letters))[yt], pr, average="macro",
                                  multi_class="ovr"))
    except Exception as e:
        auc = -1.0
        print("AUC failed:", e, flush=True)
    print(f"macro AUC={auc:.4f}", flush=True)

    from sklearn.metrics import precision_recall_fscore_support
    prc, rcl, f1, sup = precision_recall_fscore_support(yt, yp, zero_division=0)
    per = [{"class": c, "index": i, "precision": float(prc[i]),
            "recall": float(rcl[i]), "f1": float(f1[i]),
            "support": int(sup[i]), "correct": int(cm[i, i])}
           for i, c in enumerate(letters)]
    metrics = {"dataset": str(args.data), "split": "stratified 80/20",
               "seed": args.seed, "img": 224, "batch": args.batch,
               "n_train": int(len(tr_y)), "n_test": int(len(te_y)),
               "classes": letters, "test_loss": float(loss),
               "accuracy": float(acc), "top3": float(top3), "macro_auc": auc,
               "infer_ms_per_img_CPU": float(infer_ms),
               "macro_precision": float(precision_score(yt, yp, average="macro",
                                                        zero_division=0)),
               "macro_recall": float(recall_score(yt, yp, average="macro",
                                                  zero_division=0)),
               "macro_f1": float(f1_score(yt, yp, average="macro",
                                           zero_division=0)),
               "per_class": per,
               "input": "224x224 MobileNet preprocess_input [-1,1] (hand-crop cache)",
               "model": "MobileNet-imagenet 224 phase1-frozen + GMP/BN/1024/Drop.3/BN/512, ft-last40 lr1e-5, Adam 1e-4"}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "labels.json").write_text(json.dumps(letters, indent=2), encoding="utf-8")
    with open(out / "per_class.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["class", "index", "precision", "recall",
                                          "f1", "support", "correct"])
        w.writeheader()
        w.writerows(per)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hist_acc = (list(h1.history.get("accuracy", []))
                  + list(h2.history.get("accuracy", [])))
    hist_val = (list(h1.history.get("val_accuracy", []))
                + list(h2.history.get("val_accuracy", [])))
    hist_loss = (list(h1.history.get("loss", []))
                 + list(h2.history.get("loss", [])))
    hist_vloss = (list(h1.history.get("val_loss", []))
                  + list(h2.history.get("val_loss", [])))
    ep = range(1, len(hist_acc) + 1)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(ep, hist_acc, label="train")
    plt.plot(ep, hist_val, label="val/test")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.plot(ep, hist_loss, label="train")
    plt.plot(ep, hist_vloss, label="val/test")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "training_curves.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(13, 11))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(letters)))
    ax.set_yticks(range(len(letters)))
    ax.set_xticklabels(letters)
    ax.set_yticklabels(letters)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"SIBI v3 A-Z confusion — test 20% (n={len(yt)}, acc={acc:.3f})")
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=120)
    plt.close()

    rng = np.random.default_rng(args.seed)
    fig, axes = plt.subplots(4, 7, figsize=(14, 8))
    idx = []
    for i in range(len(letters)):
        ii = np.where(yt == i)[0]
        if len(ii):
            idx.append(rng.choice(ii, size=1, replace=False)[0])
    idx = idx[:28]
    from PIL import Image
    for ax, k in zip(axes.flat, idx):
        im = Image.open(te_p[k]).convert("RGB").resize((128, 128))
        ax.imshow(im)
        ok = yp[k] == yt[k]
        ax.set_title(f"t:{letters[yt[k]]} p:{letters[yp[k]]} {pr[k].max():.2f}",
                     fontsize=8, color="green" if ok else "red")
        ax.axis("off")
    for ax in axes.flat[len(idx):]:
        ax.axis("off")
    plt.suptitle(f"v3 sample test predictions — acc {acc:.3f}")
    plt.tight_layout()
    plt.savefig(out / "sample_predictions.png", dpi=120)
    plt.close()

    model.save(out / "sibi_v3_best.h5")
    model.export(str(out / "saved_model"))
    print(f"FINAL test_acc={acc:.4f} top3={top3:.4f} macroF1={metrics['macro_f1']:.4f} "
          f"AUC={auc:.4f}", flush=True)


if __name__ == "__main__":
    main()
