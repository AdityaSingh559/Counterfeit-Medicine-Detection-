"""Clean data/raw: drop corrupt/tiny/duplicate/non-image files, standardise format, cap count.

Usage:
    python src/clean_dataset.py [--max-images 1000] [--min-size 64] [--quality 95]

All surviving images are re-saved as RGB JPEG at a single fixed quality
(one consistent pipeline, so file format is never a hint for the model).
Images are written to data/raw/ as img_00001.jpg, ...; originals are removed.
"""
import argparse
import hashlib
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SUMMARY_DIR = config.OUTPUTS_DIR / "metrics"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_valid(raw_dir: Path, min_size: int):
    """Return (valid_paths, removed_counts) after dropping bad files and exact duplicates."""
    removed = {"non_image": 0, "corrupt": 0, "too_small": 0, "duplicate": 0}
    seen = set()
    valid = []
    for p in sorted(raw_dir.rglob("*")):
        if not p.is_file() or p.name == ".gitkeep":
            continue
        if p.suffix.lower() not in IMAGE_EXTS:
            removed["non_image"] += 1
            continue
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                w, h = im.size
        except (UnidentifiedImageError, OSError, SyntaxError):
            removed["corrupt"] += 1
            continue
        if min(w, h) < min_size:
            removed["too_small"] += 1
            continue
        digest = sha256_file(p)
        if digest in seen:
            removed["duplicate"] += 1
            continue
        seen.add(digest)
        valid.append(p)
    return valid, removed


def make_montage(paths, out_path: Path, n: int = 16, tile: int = 128):
    paths = paths[:n]
    cols = 4
    rows = (len(paths) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * tile, rows * tile), "white")
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            im = im.convert("RGB").resize((tile, tile))
        canvas.paste(im, ((i % cols) * tile, (i // cols) * tile))
    canvas.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-images", type=int, default=1000)
    ap.add_argument("--min-size", type=int, default=64)
    ap.add_argument("--quality", type=int, default=95)
    args = ap.parse_args()

    config.set_seeds()
    raw = config.RAW_DIR
    if not raw.exists() or not any(raw.rglob("*")):
        sys.exit(f"{raw} is empty. Download a Kaggle pill/medicine image dataset and unzip it there first.")

    valid, removed = collect_valid(raw, args.min_size)
    found = len(valid)
    if found == 0:
        sys.exit("No valid images found.")

    rng = random.Random(config.SEED)
    rng.shuffle(valid)
    capped = valid[: args.max_images]

    # Re-save through one pipeline into a temp folder, then replace data/raw contents.
    tmp = raw.parent / "_raw_clean_tmp"
    tmp.mkdir(exist_ok=True)
    sizes = []
    for i, p in enumerate(capped, start=1):
        with Image.open(p) as im:
            im = im.convert("RGB")
            sizes.append(im.size)
            im.save(tmp / f"img_{i:05d}.jpg", format="JPEG", quality=args.quality)

    for p in sorted(raw.rglob("*"), reverse=True):
        if p.is_file() and p.name != ".gitkeep":
            p.unlink()
        elif p.is_dir():
            try:
                p.rmdir()
            except OSError:
                pass
    final = []
    for p in sorted(tmp.iterdir()):
        target = raw / p.name
        p.replace(target)
        final.append(target)
    tmp.rmdir()

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    make_montage(final, SUMMARY_DIR / "dataset_montage.png")

    ws = np.array([s[0] for s in sizes])
    hs = np.array([s[1] for s in sizes])
    lines = [
        f"Valid images found:   {found}",
        f"Kept (cap {args.max_images}):    {len(final)}",
        f"Removed: {removed}",
        f"Width  min/median/max: {ws.min()} / {int(np.median(ws))} / {ws.max()}",
        f"Height min/median/max: {hs.min()} / {int(np.median(hs))} / {hs.max()}",
        f"Format: RGB JPEG, quality {args.quality}",
    ]
    summary = "\n".join(lines)
    print(summary)
    (SUMMARY_DIR / "dataset_summary.txt").write_text(summary + "\n", encoding="utf-8")
    print(f"Montage saved to {SUMMARY_DIR / 'dataset_montage.png'}")
    if not 500 <= len(final) <= 1000:
        print("WARNING: plan expects 500-1000 images; consider a different/larger dataset.")


if __name__ == "__main__":
    main()
