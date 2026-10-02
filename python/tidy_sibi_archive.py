"""Tidy the messy SIBI archive into neat per-letter folders (v3 dataset).

Reads (all under C:\\Users\\BRAVO\\Downlo\\archive\\SIBI):
  top-level A-Y (24x220 gray 250px), Combine_SIBI (A-Z),
  LEMLITBANG V02 training+validation (A-Z), RAW training+validation (A-Z).
  Combine_SIBI (3) is skipped: md5-proven 100% byte-duplicate subset.

Writes:
  C:\\Users\\BRAVO\\Downlo\\archive\\SIBI_tidy\\A-Z  (deduped by md5, PIL-verified)
  tidy_report.json + tidy_report.csv (per-letter per-source counts, dupes dropped)

Then deletes ONLY junk (no real photos harmed):
  *.ini, *_copy.* inside SIBI/, and the duplicate tree Combine_SIBI (3)/.
"""
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

SRC = Path(r"C:\Users\BRAVO\Downlo\archive\SIBI")
DST = Path(r"C:\Users\BRAVO\Downlo\archive\SIBI_tidy")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]

SOURCES: list[tuple[str, Path]] = [
    ("top", SRC),
    ("combine", SRC / "Combine_SIBI"),
    ("v02-train", SRC / "SIBI_datasets_LEMLITBANG_SIBI_R_90.10_V02"
     / "SIBI_datasets_LEMLITBANG_SIBI_R_90.10_V02" / "training"),
    ("v02-val", SRC / "SIBI_datasets_LEMLITBANG_SIBI_R_90.10_V02"
     / "SIBI_datasets_LEMLITBANG_SIBI_R_90.10_V02" / "validation"),
    ("raw-train", SRC / "SIBI_datasets_LEMLITBANG_SIBI_R_90.10_RAW"
     / "SIBI_datasets_LEMLITBANG_SIBI_R_90.10_RAW" / "training"),
    ("raw-val", SRC / "SIBI_datasets_LEMLITBANG_SIBI_R_90.10_RAW"
     / "SIBI_datasets_LEMLITBANG_SIBI_R_90.10_RAW" / "validation"),
]


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    from PIL import Image

    DST.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    per_letter_source: dict[str, dict[str, int]] = {L: {} for L in LETTERS}
    per_letter_final: dict[str, int] = {L: 0 for L in LETTERS}
    dupes = corrupt = 0

    for sname, sdir in SOURCES:
        if not sdir.exists():
            print(f"SKIP missing {sname}: {sdir}", flush=True)
            continue
        for L in LETTERS:
            ldir = sdir / L
            if not ldir.is_dir():
                continue
            files = sorted([f for f in ldir.iterdir()
                            if f.is_file() and f.suffix.lower() in IMG_EXTS])
            per_letter_source[L][sname] = len(files)
            outdir = DST / L
            outdir.mkdir(exist_ok=True)
            for f in files:
                try:
                    digest = md5(f)
                except OSError:
                    corrupt += 1
                    continue
                if digest in seen:
                    dupes += 1
                    continue
                try:
                    with Image.open(f) as im:
                        im.verify()
                except Exception:
                    corrupt += 1
                    continue
                seen.add(digest)
                per_letter_final[L] += 1
                shutil.copy2(f, outdir / f"{L}_{per_letter_final[L]:04d}{f.suffix.lower()}")

    report = {
        "sources": [s for s, _ in SOURCES],
        "per_letter_source": per_letter_source,
        "per_letter_final": per_letter_final,
        "total_final": sum(per_letter_final.values()),
        "dupes_dropped": dupes,
        "corrupt_dropped": corrupt,
    }
    (DST / "tidy_report.json").write_text(json.dumps(report, indent=2))
    with open(DST / "tidy_report.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["letter"] + [s for s, _ in SOURCES] + ["final"])
        for L in LETTERS:
            w.writerow([L] + [per_letter_source[L].get(s, 0) for s, _ in SOURCES]
                       + [per_letter_final[L]])
    print(json.dumps({k: v for k, v in report.items() if k != "per_letter_source"}, indent=2))
    print("per-letter final:", per_letter_final)

    # ---- junk removal (only provable junk) ----
    removed_ini = removed_copy = 0
    for f in SRC.rglob("*.ini"):
        if f.is_file():
            f.unlink()
            removed_ini += 1
    for f in SRC.rglob("*_copy.*"):
        if f.is_file() and f.suffix.lower() in IMG_EXTS:
            f.unlink()
            removed_copy += 1
    dupe_tree = SRC / "Combine_SIBI (3)"
    dupe_tree_gone = False
    if dupe_tree.is_dir():
        shutil.rmtree(dupe_tree)
        dupe_tree_gone = True
    print(f"junk removed: ini={removed_ini} copy={removed_copy} "
          f"dupe_tree={dupe_tree_gone}")


if __name__ == "__main__":
    sys.exit(main())
