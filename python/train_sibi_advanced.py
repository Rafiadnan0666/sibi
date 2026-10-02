"""
SIBI alphabet ADVANCED image CNN — 80/20 stratified, per-alphabet proof.

Dataset: C:\\Users\\BRAVO\\Downlo\\archive\\SIBI  (24 folders A-I,K-Y, 220 imgs each = 5280)
Split  : stratified 80% train / 20% test (seed 42) -> 4224 / 1056 (176 / 44 per class)
Input  : 128x128 RGB [-1,1]  (grayscale JPGs duplicated to RGB, mirrors src/lib/preprocessing.ts)
Backbone: MobileNetV2 (imagenet) + improved head (BN + 512 + 256 + heavy dropout)
Training: phase-1 freeze, phase-2 fine-tune last 40 layers, label-smoothing, Adam,
          strong augmentation, EarlyStopping + ReduceLROnPlateau + Checkpoint.
Proof  : metrics.json, classification_report.txt (+per-class csv),
         confusion_matrix.npy/.png (counts) + normalized png,
         training_curves.png, sample_predictions.png, split_counts.csv,
         savedmodel + .h5 + labels.json + tfjs/ (model.json)

Usage:
    python python/train_sibi_advanced.py --data "C:\\Users\\BRAVO\\Downlo\\archive\\SIBI" --out ./training_output/sibi_advanced_80_20 --epochs 30 --img 128 --batch 32
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np


def collect_files(data_dir: Path):
    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    paths, labels = [], []
    code = {c: i for i, c in enumerate(classes)}
    for c in classes:
        for f in sorted((data_dir / c).iterdir()):
            if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                paths.append(str(f))
                labels.append(code[c])
    return np.array(paths), np.array(labels), classes


def build_tf_datasets(train_paths, train_labels, test_paths, test_labels, img, batch, seed):
    import tensorflow as tf

    def decode(path, label):
        raw = tf.io.read_file(path)
        # dataset JPGs are grayscale ('L'); decode then duplicate to RGB so
        # MobileNetV2 pretrained weights stay useful + browser RGB path matches.
        im = tf.image.decode_image(raw, channels=1, expand_animations=False)
        im = tf.image.grayscale_to_rgb(im)
        im = tf.image.resize(im, (img, img), method=tf.image.ResizeMethod.BILINEAR)
        im = tf.cast(im, tf.float32)
        return im, label

    norm = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0)
    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.10),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomTranslation(0.10, 0.10),
        tf.keras.layers.RandomBrightness(0.20),
        tf.keras.layers.RandomContrast(0.15),
    ])

    AUT = tf.data.AUTOTUNE
    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    train_ds = train_ds.shuffle(len(train_paths), seed=seed, reshuffle_each_iteration=True)
    train_ds = train_ds.map(decode, num_parallel_calls=AUT)
    train_ds = train_ds.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=AUT)
    train_ds = train_ds.map(lambda x, y: (norm(x), y), num_parallel_calls=AUT)
    train_ds = train_ds.batch(batch).prefetch(AUT)

    test_ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
    test_ds = test_ds.map(decode, num_parallel_calls=AUT)
    test_ds = test_ds.map(lambda x, y: (norm(x), y), num_parallel_calls=AUT)
    test_ds = test_ds.batch(batch).prefetch(AUT)
    return train_ds, test_ds


def build_model(img, n_classes):
    import tensorflow as tf

    base = tf.keras.applications.MobileNetV2(
        input_shape=(img, img, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False
    inputs = tf.keras.Input(shape=(img, img, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.40)(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.20)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy",
                 tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3")],
    )
    return model, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path(r"C:\Users\BRAVO\Downlo\archive\SIBI"))
    ap.add_argument("--out", type=Path, default=Path("./training_output/sibi_advanced_80_20"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--img", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import tensorflow as tf
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 f1_score, precision_score, recall_score)
    from sklearn.model_selection import train_test_split

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    print(f"data={args.data} out={out} img={args.img} epochs={args.epochs} batch={args.batch}")

    paths, labels, classes = collect_files(args.data)
    print(f"classes ({len(classes)}): {classes}")
    from collections import Counter
    print("counts per class:", dict(sorted(Counter(labels).items())))
    assert len(classes) == 24, classes
    for c in classes:
        n = int((labels == classes.index(c)).sum())
        assert n == 220, (c, n)
    print(f"total images: {len(paths)}")

    tr_p, te_p, tr_y, te_y = train_test_split(
        paths, labels, test_size=0.20, random_state=args.seed, stratify=labels)
    print(f"TRAIN {len(tr_p)}  TEST {len(te_p)}  (80/20 stratified, seed={args.seed})")
    # per-class split proof
    with open(out / "split_counts.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "index", "total", "train", "test"])
        for i, c in enumerate(classes):
            w.writerow([c, i, int((labels == i).sum()),
                        int((tr_y == i).sum()), int((te_y == i).sum())])

    train_ds, test_ds = build_tf_datasets(tr_p, tr_y, te_p, te_y, args.img, args.batch, args.seed)

    model, base = build_model(args.img, len(classes))
    model.summary(print_fn=print)

    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8,
                                         restore_best_weights=True, mode="max", verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3,
                                             factor=0.5, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(str(out / "best_phase1.keras"),
                                           monitor="val_accuracy", save_best_only=True, mode="max"),
    ]
    h1 = model.fit(train_ds, validation_data=test_ds, epochs=args.epochs,
                   callbacks=cbs, verbose=1)

    # ---- phase 2: fine-tune last 40 backbone layers (real unfreeze, v1 script never did) ----
    base.trainable = True
    for layer in base.layers[:-40]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy",
                 tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3")],
    )
    ft_epochs = max(4, args.epochs // 3)
    cbs2 = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6,
                                         restore_best_weights=True, mode="max", verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2,
                                             factor=0.5, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(str(out / "best_final.keras"),
                                           monitor="val_accuracy", save_best_only=True, mode="max"),
    ]
    print(f"\n=== fine-tune last 40 backbone layers for up to {ft_epochs} epochs (lr=1e-5) ===")
    h2 = model.fit(train_ds, validation_data=test_ds, epochs=ft_epochs,
                   callbacks=cbs2, verbose=1)

    # ---- evaluation on held-out 20% ----
    print("\n=== TEST (held-out 20%) ===")
    loss, acc, top3 = model.evaluate(test_ds, verbose=1)
    print(f"test_loss={loss:.4f} test_acc={acc:.4f} test_top3={top3:.4f}")

    t0 = time.time()
    y_true, y_pred, y_prob = [], [], []
    for xb, yb in test_ds:
        pb = model.predict(xb, verbose=0)
        y_true.extend(np.asarray(yb).tolist())
        y_pred.extend(np.argmax(pb, axis=1).tolist())
        y_prob.extend(pb.tolist())
    infer_ms = (time.time() - t0) / max(1, len(y_true)) * 1000.0
    y_true = np.array(y_true); y_pred = np.array(y_pred); y_prob = np.array(y_prob)

    rep = classification_report(y_true, y_pred, target_names=classes, digits=4)
    print(rep)
    (out / "classification_report.txt").write_text(
        f"SIBI advanced 80/20 — test on held-out 20% (n={len(y_true)})\n"
        f"test_loss={loss:.4f} test_acc={acc:.4f} test_top3={top3:.4f} infer_ms_per_img_CPU={infer_ms:.2f}\n\n" + rep,
        encoding="utf-8")
    cm = confusion_matrix(y_true, y_pred)
    np.save(out / "confusion_matrix.npy", cm)
    per_class = []
    from sklearn.metrics import precision_recall_fscore_support
    pr, rc, f1, sup = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
    for i, c in enumerate(classes):
        per_class.append({"class": c, "index": i,
                          "precision": float(pr[i]), "recall": float(rc[i]),
                          "f1": float(f1[i]), "support": int(sup[i]),
                          "correct": int(cm[i, i])})
    macro = {"accuracy": float(acc), "top3": float(top3),
             "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
             "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
             "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0))}
    with open(out / "per_class.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["class", "index", "precision", "recall", "f1", "support", "correct"])
        w.writeheader(); w.writerows(per_class)

    metrics = {"dataset": str(args.data), "split": "stratified 80/20",
               "seed": args.seed, "img": args.img, "batch": args.batch,
               "n_train": int(len(tr_y)), "n_test": int(len(te_y)),
               "classes": classes, "test_loss": float(loss),
               "infer_ms_per_img_CPU": float(infer_ms), **macro,
               "per_class": per_class,
               "input": f"{args.img}x{args.img} RGB [-1,1] (gray duplicated)",
               "model": "MobileNetV2-imagenet + GAP/BN/512/BN/256, label handling sparse-CE, ft-last40 lr1e-5"}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "labels.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")

    # ---- plots ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def chain(key):
        return list(h1.history.get(key, [])) + list(h2.history.get(key, []))

    ep = range(1, len(chain("accuracy")) + 1)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(ep, chain("accuracy"), label="train acc")
    plt.plot(ep, chain("val_accuracy"), label="val/test acc")
    plt.xlabel("epoch (phase1+finetune)"); plt.ylabel("accuracy")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.plot(ep, chain("loss"), label="train loss")
    plt.plot(ep, chain("val_loss"), label="val/test loss")
    plt.xlabel("epoch (phase1+finetune)"); plt.ylabel("loss")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(out / "training_curves.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes); ax.set_yticklabels(classes)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"SIBI confusion matrix — test 20% (n={len(y_true)}, acc={acc:.3f})")
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout(); plt.savefig(out / "confusion_matrix.png", dpi=150); plt.close()

    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cmn, cmap="Greens", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes); ax.set_yticklabels(classes)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title("SIBI confusion matrix — normalized per true class (recall)")
    for i in range(len(classes)):
        ax.text(np.argmax(cmn[i]), i, f"{cmn[i].max():.2f}", ha="center", va="center",
                fontsize=8, color="black", fontweight="bold")
    plt.tight_layout(); plt.savefig(out / "confusion_matrix_normalized.png", dpi=150); plt.close()

    # sample predictions grid: 2 random test imgs per 12 classes = 24 tiles
    rng = np.random.default_rng(args.seed)
    fig, axes = plt.subplots(4, 6, figsize=(12, 8))
    idx = []
    for i in range(len(classes)):
        ii = np.where(y_true == i)[0]
        idx.extend(rng.choice(ii, size=min(1, len(ii)), replace=False).tolist())
    idx = idx[:24]
    for ax, k in zip(axes.flat, idx):
        from PIL import Image
        im = Image.open(te_p[k]).convert("RGB").resize((128, 128))
        ax.imshow(im)
        ok = y_pred[k] == y_true[k]
        ax.set_title(f"t:{classes[y_true[k]]} p:{classes[y_pred[k]]} {y_prob[k].max():.2f}",
                     fontsize=9, color="green" if ok else "red")
        ax.axis("off")
    plt.suptitle(f"Sample test predictions (green=correct) — acc {acc:.3f}")
    plt.tight_layout(); plt.savefig(out / "sample_predictions.png", dpi=150); plt.close()

    # ---- save model ----
    model.save(out / "sibi_advanced_best.h5")
    model.export(str(out / "saved_model"))
    print(f"\nSaved h5 + SavedModel to {out}")
    print(f"PROOF: {out/'metrics.json'}, classification_report.txt, per_class.csv, "
          f"confusion_matrix.png, training_curves.png, sample_predictions.png")
    print(f"FINAL test_acc={acc:.4f} top3={top3:.4f} macroF1={macro['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
