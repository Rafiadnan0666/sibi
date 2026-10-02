"""v3 offline preprocessing: MediaPipe hand-crop cache at 224px.

Adapts the 98% Kaggle notebook recipe (Denty Nirwana Bintang, Apache 2.0):
  hand-crop (pad 40, center fallback) -> seeded random-background blend
  -> gray + GaussianBlur(5,5) -> RGB -> resize 224.
Tweaks vs notebook: static_image_mode=True (still photos, better recall),
  detect on downscaled<=800px copy (speed), blend seeded by file path
  (deterministic cache). preprocess_input ([-1,1]) is applied at TRAIN time,
  not baked in, so the cache stays uint8 JPGs.

In : C:\\Users\\BRAVO\\Downlo\\archive\\SIBI_tidy\\A-Z (6587 unique)
Out: C:\\Users\\BRAVO\\Downlo\\archive\\SIBI_v3_224\\A-Z + cache_report.json
"""
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

SRC = Path(r"C:\Users\BRAVO\Downlo\archive\SIBI_tidy")
DST = Path(r"C:\Users\BRAVO\Downlo\archive\SIBI_v3_224")
LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def skin_hand_crop(img: np.ndarray):
    """YCrCb skin mask -> largest contour bbox + pad. None if unusable
    (grayscale-tight crops from the top set skip detection upstream)."""
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, np.array([0, 133, 77], dtype=np.uint8),
                       np.array([255, 173, 127], dtype=np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    h, w, _ = img.shape
    if cv2.contourArea(c) < 0.02 * h * w:
        return None
    x, y, bw, bh = cv2.boundingRect(c)
    pad = 40
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
    return img[y0:y1, x0:x1] if x1 > x0 and y1 > y0 else None


def blend_background(crop: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w, _ = crop.shape
    background = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    alpha = float(rng.uniform(0.1, 0.3))
    return cv2.addWeighted(crop, 1 - alpha, background, alpha, 0)


def main() -> None:
    report: dict = {"per_letter": {}, "hand_ok": 0, "fallback": 0,
                    "note": ("MediaPipe solutions is absent from the installed "
                             "mediapipe 0.10.33 build, so hand-crop uses YCrCb "
                             "skin segmentation; tight grayscale crops use the "
                             "notebook's center fallback directly.")}
    for L in LETTERS:
        sdir = SRC / L
        files = sorted([f for f in sdir.iterdir()
                        if f.is_file() and f.suffix.lower() in EXTS])
        outdir = DST / L
        outdir.mkdir(parents=True, exist_ok=True)
        n_ok = n_fb = 0
        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                continue
            h, w, _ = img.shape
            crop = None
            # tight grayscale crops (top set): channels are identical, skin
            # segmentation is meaningless -> notebook center fallback.
            is_gray = (np.abs(img[:, :, 0].astype(int)
                              - img[:, :, 1].astype(int)).max() == 0
                       and np.abs(img[:, :, 1].astype(int)
                                  - img[:, :, 2].astype(int)).max() == 0)
            if not is_gray:
                crop = skin_hand_crop(img)
                if crop is not None:
                    n_ok += 1
            if crop is None or crop.size == 0:
                n_fb += 1
                x0, y0 = w // 4, h // 4
                crop = img[y0:3 * h // 4, x0:3 * w // 4]
            seed = int(hashlib.md5(str(f).encode()).hexdigest()[:8], 16)
            crop = blend_background(crop, seed)
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            rgb = cv2.cvtColor(blur, cv2.COLOR_GRAY2RGB)
            resized = cv2.resize(rgb, (224, 224))
            cv2.imwrite(str(outdir / (f.stem + ".jpg")), resized,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
        report["per_letter"][L] = {"cached": n_ok + n_fb, "hand_ok": n_ok,
                                   "fallback": n_fb}
        report["hand_ok"] += n_ok
        report["fallback"] += n_fb
        print(f"{L}: cached={n_ok + n_fb} hand_ok={n_ok} fallback={n_fb}", flush=True)
    (DST / "cache_report.json").write_text(json.dumps(report, indent=2))
    print(f"TOTAL hand_ok={report['hand_ok']} fallback={report['fallback']}")


if __name__ == "__main__":
    sys.exit(main())
