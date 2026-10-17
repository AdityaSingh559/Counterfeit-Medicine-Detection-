"""Split data/raw 70/15/15, then simulate one counterfeit per genuine image.

Usage:
    python src/generate_fakes.py [--raw-dir data/raw] [--out-dir data] [--preview-dir outputs/fakes_preview]

Order matters: genuine originals are split FIRST, then each split's fakes are
derived only from that split's genuine images, so nothing leaks across splits.
Genuine and counterfeit images are both saved as RGB JPEG at the same quality,
so file format / compression pipeline is not a hint for the model.

Each fake gets 1-3 random alterations at one random strength (low/medium/high):
    color_shift        global   HSV hue / saturation / value shift
    blur_imprint       local    gaussian blur or inpaint over a box
    wrong_text         local    wrong / rotated characters drawn in a box
    shape_change       global   scale, aspect change, mild perspective warp
    noise_compression  global   gaussian noise then low-quality JPEG round trip
"""
import argparse
import csv
import json
import random
import string
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

JPEG_QUALITY = 95
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
ALTERATIONS = ["color_shift", "blur_imprint", "wrong_text", "shape_change", "noise_compression"]
LOCALIZED = {"blur_imprint", "wrong_text"}
STRENGTH_T = {"low": 0.25, "medium": 0.6, "high": 1.0}
SPLITS = ("train", "val", "test")
MANIFEST_COLUMNS = ["fake_filename", "source_filename", "split", "alterations", "strength", "localized", "bbox"]


# ---------------------------------------------------------------- alterations
def color_shift(img: np.ndarray, t: float, rng: random.Random) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    sign = lambda: rng.choice((-1, 1))  # noqa: E731
    hue = (5 + 15 * t) * rng.uniform(0.7, 1.0) * sign()           # degrees
    sat = 1 + (0.05 + 0.20 * t) * rng.uniform(0.7, 1.0) * sign()
    val = 1 + (0.05 + 0.20 * t) * rng.uniform(0.7, 1.0) * sign()
    hsv[..., 0] = (hsv[..., 0] + hue / 2) % 180                     # OpenCV hue is 0-179
    hsv[..., 1] = np.clip(hsv[..., 1] * sat, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * val, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def pick_box(h: int, w: int, t: float, rng: random.Random, pill_box=None):
    """Box for a localised alteration.

    With a known pill box: centred on the pill (small jitter), sized 0.6-1.1x the pill
    (at least 24 px) so the edit lands on the tablet face without wiping the whole pill. Without one: central (50%) or random
    box, 15-40% of the image side, scaled by strength.
    """
    if pill_box is not None:
        px1, py1, px2, py2 = pill_box
        size = max(24, int(max(px2 - px1, py2 - py1) * (0.6 + 0.5 * t)))
        size = min(size, w, h)
        cx = (px1 + px2) / 2 + rng.uniform(-0.1, 0.1) * size
        cy = (py1 + py2) / 2 + rng.uniform(-0.1, 0.1) * size
        x1 = int(max(0, min(cx - size / 2, w - size)))
        y1 = int(max(0, min(cy - size / 2, h - size)))
        return x1, y1, x1 + size, y1 + size
    frac = 0.15 + 0.25 * t
    bw, bh = max(8, int(w * frac)), max(8, int(h * frac))
    if rng.random() < 0.5:
        cx = int(w / 2 + rng.uniform(-0.1, 0.1) * w)
        cy = int(h / 2 + rng.uniform(-0.1, 0.1) * h)
        x1, y1 = cx - bw // 2, cy - bh // 2
    else:
        x1, y1 = rng.randint(0, max(0, w - bw)), rng.randint(0, max(0, h - bh))
    x1, y1 = max(0, min(x1, w - bw)), max(0, min(y1, h - bh))
    return x1, y1, x1 + bw, y1 + bh


def blur_imprint(img: np.ndarray, t: float, rng: random.Random, pill_box=None):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = pick_box(h, w, t, rng, pill_box)
    out = img.copy()
    if rng.random() < 0.5:
        k = int(5 + 20 * t) | 1                                       # odd kernel, strong enough to erase print
        out[y1:y2, x1:x2] = cv2.GaussianBlur(img[y1:y2, x1:x2], (k, k), 0)
    else:
        # Erase: fill with the median colour of the surrounding ring, feathered at the edge.
        pad = max(4, (x2 - x1) // 4)
        ox1, oy1, ox2, oy2 = max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)
        ring = img[oy1:oy2, ox1:ox2]
        outside = np.ones(ring.shape[:2], bool)
        outside[(y1 - oy1):(y2 - oy1), (x1 - ox1):(x2 - ox1)] = False
        valid = ring[outside]
        fill = np.median(valid, axis=0) if len(valid) else np.array([128, 128, 128])
        mask = np.zeros((h, w), np.float32)
        mask[y1:y2, x1:x2] = 1.0
        mask = cv2.GaussianBlur(mask, (0, 0), max(1.5, (x2 - x1) / 12))[..., None]
        out = (img * (1 - mask) + fill.reshape(1, 1, 3) * mask).astype(np.uint8)
    return out, (x1, y1, x2, y2)


def wrong_text(img: np.ndarray, t: float, rng: random.Random, pill_box=None):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = pick_box(h, w, t, rng, pill_box)
    bw, bh = x2 - x1, y2 - y1
    text = "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(rng.randint(2, 4)))
    try:
        font = ImageFont.load_default(size=max(8, int(bh * 0.5)))
    except TypeError:  # very old Pillow
        font = ImageFont.load_default()

    region = img[y1:y2, x1:x2]
    lum = float(region.mean())
    color = (30, 30, 30) if lum > 127 else (235, 235, 235)            # contrast with local brightness
    alpha = int(255 * (0.6 + 0.4 * t))

    layer = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.text((bw * 0.1, bh * 0.2), text, font=font, fill=color + (alpha,))
    angle = rng.uniform(-1, 1) * (5 + 15 * t)
    layer = layer.rotate(angle, resample=Image.BICUBIC, expand=False)

    base = Image.fromarray(img).convert("RGBA")
    base.alpha_composite(layer, dest=(x1, y1))
    return np.array(base.convert("RGB")), (x1, y1, x2, y2)


def shape_change(img: np.ndarray, t: float, rng: random.Random):
    """Return (warped image, 3x3 matrix) so boxes can be mapped through the same warp."""
    h, w = img.shape[:2]
    sx = 1 + rng.choice((-1, 1)) * (0.03 + 0.12 * t) * rng.uniform(0.6, 1.0)
    sy = 1 + rng.choice((-1, 1)) * (0.03 + 0.12 * t) * rng.uniform(0.6, 1.0)
    cx, cy = w / 2, h / 2
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[cx + (x - cx) * sx, cy + (y - cy) * sy] for x, y in src])
    jit = (0.02 + 0.08 * t) * np.array([w, h], np.float32)
    dst += np.float32([[rng.uniform(-1, 1) * jit[0], rng.uniform(-1, 1) * jit[1]] for _ in range(4)])
    m = cv2.getPerspectiveTransform(src, dst)
    # Black border matches the dataset's own letterbox padding (no replicate-streak artefacts).
    return cv2.warpPerspective(img, m, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)), m


def map_box(box, m, w, h):
    x1, y1, x2, y2 = box
    pts = np.float32([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]])
    out = cv2.perspectiveTransform(pts, m).reshape(-1, 2)
    nx1, ny1 = np.clip(out.min(axis=0), 0, [w - 1, h - 1])
    nx2, ny2 = np.clip(out.max(axis=0), 0, [w - 1, h - 1])
    return int(nx1), int(ny1), int(nx2), int(ny2)


def noise_compression(img: np.ndarray, t: float, rng: random.Random, np_rng: np.random.RandomState) -> np.ndarray:
    sigma = 2 + 10 * t
    noisy = np.clip(img.astype(np.float32) + np_rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)
    quality = int(70 - 45 * t)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(noisy, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def make_fake(img: np.ndarray, rng: random.Random, np_rng: np.random.RandomState, pill_box=None):
    """Return (fake_img, alteration_names, strength, bbox_or_None).

    pill_box (x1,y1,x2,y2) in the ORIGINAL image, if known, steers localised edits onto the pill.
    """
    k = rng.randint(1, 3)
    chosen = rng.sample(ALTERATIONS, k)
    strength = rng.choice(list(STRENGTH_T))
    t = STRENGTH_T[strength]

    boxes = []
    # Fixed order so bboxes stay valid: geometry first, local edits, then global colour/noise.
    if "shape_change" in chosen:
        img, m = shape_change(img, t, rng)
        if pill_box is not None:
            pill_box = map_box(pill_box, m, img.shape[1], img.shape[0])
    if "blur_imprint" in chosen:
        img, box = blur_imprint(img, t, rng, pill_box)
        boxes.append(box)
    if "wrong_text" in chosen:
        img, box = wrong_text(img, t, rng, pill_box)
        boxes.append(box)
    if "color_shift" in chosen:
        img = color_shift(img, t, rng)
    if "noise_compression" in chosen:
        img = noise_compression(img, t, rng, np_rng)

    bbox = None
    if boxes:
        bbox = (min(b[0] for b in boxes), min(b[1] for b in boxes),
                max(b[2] for b in boxes), max(b[3] for b in boxes))
    names = [a for a in ALTERATIONS if a in chosen]
    return img, names, strength, bbox


# ------------------------------------------------------------------- pipeline
def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))


def save_jpeg(arr: np.ndarray, path: Path) -> None:
    """The single save pipeline used for BOTH classes."""
    Image.fromarray(arr).save(path, format="JPEG", quality=JPEG_QUALITY)


def split_names(names, seed):
    names = sorted(names)
    random.Random(seed).shuffle(names)
    n = len(names)
    n_train = int(round(config.SPLIT[0] * n))
    n_val = int(round(config.SPLIT[1] * n))
    return {"train": names[:n_train], "val": names[n_train:n_train + n_val], "test": names[n_train + n_val:]}


def clear_outputs(out_dir: Path):
    for split in SPLITS:
        for cls in ("genuine", "counterfeit"):
            d = out_dir / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for f in d.iterdir():
                if f.is_file() and f.name != ".gitkeep":
                    f.unlink()


def make_preview(rows, out_path: Path, tile: int = 256, n_pairs: int = 8):
    """rows: list of (genuine_path, fake_path, bbox_or_None). Genuine | fake, two pairs per row."""
    rows = rows[:n_pairs]
    cols_pairs = 2
    n_rows = (len(rows) + cols_pairs - 1) // cols_pairs
    canvas = Image.new("RGB", (cols_pairs * 2 * tile, n_rows * tile), "white")
    for i, (g, f, bbox) in enumerate(rows):
        gi = Image.open(g).convert("RGB")
        fi = Image.open(f).convert("RGB")
        if bbox:
            ImageDraw.Draw(fi).rectangle(bbox, outline=(255, 0, 0), width=max(2, fi.width // 100))
        gi, fi = gi.resize((tile, tile)), fi.resize((tile, tile))
        r, c = divmod(i, cols_pairs)
        canvas.paste(gi, (c * 2 * tile, r * tile))
        canvas.paste(fi, (c * 2 * tile + tile, r * tile))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def run_assertions(out_dir: Path, split_map, manifest_rows):
    # 1. No source image appears in two splits.
    sets = {s: set(v) for s, v in split_map.items()}
    for a in SPLITS:
        for b in SPLITS:
            if a < b:
                assert not (sets[a] & sets[b]), f"leakage between {a} and {b}: {sets[a] & sets[b]}"
    # 2. Counts match and classes are balanced, files exist.
    for s in SPLITS:
        g = [p for p in (out_dir / s / "genuine").iterdir() if p.is_file() and p.name != ".gitkeep"]
        c = [p for p in (out_dir / s / "counterfeit").iterdir() if p.is_file() and p.name != ".gitkeep"]
        assert len(g) == len(c) == len(split_map[s]), f"{s}: genuine={len(g)} counterfeit={len(c)} expected={len(split_map[s])}"
    # 3. Manifest completeness.
    assert len(manifest_rows) == sum(len(v) for v in split_map.values()), "manifest row count mismatch"
    for r in manifest_rows:
        assert (out_dir / r["fake_filename"]).exists(), f"missing {r['fake_filename']}"
        assert r["source_filename"] in sets[r["split"]], f"{r['source_filename']} not in split {r['split']}"
    print("Assertions passed: no split leakage, classes balanced, manifest complete.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=config.RAW_DIR)
    ap.add_argument("--out-dir", type=Path, default=config.DATA_DIR)
    ap.add_argument("--preview-dir", type=Path, default=config.OUTPUTS_DIR / "fakes_preview")
    args = ap.parse_args()

    config.set_seeds()
    rng = random.Random(config.SEED)
    np_rng = np.random.RandomState(config.SEED)

    raw_files = {p.name: p for p in args.raw_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS}
    if len(raw_files) < 10:
        sys.exit(f"Only {len(raw_files)} images in {args.raw_dir}. Add the Kaggle dataset and run clean_dataset.py first.")

    boxes_file = args.raw_dir.parent / "raw_boxes.json"
    pill_boxes = json.loads(boxes_file.read_text(encoding="utf-8")) if boxes_file.exists() else {}
    print(f"Pill boxes available for {sum(n in pill_boxes for n in raw_files)}/{len(raw_files)} images"
          if pill_boxes else "No raw_boxes.json: localised edits use random/central boxes")

    split_map = split_names(raw_files.keys(), config.SEED)
    print({s: len(v) for s, v in split_map.items()}, "genuine images per split")
    clear_outputs(args.out_dir)

    manifest_rows, preview_rows = [], []
    for split in SPLITS:
        for name in split_map[split]:
            arr = load_rgb(raw_files[name])
            stem = Path(name).stem
            g_path = args.out_dir / split / "genuine" / f"{stem}.jpg"
            save_jpeg(arr, g_path)

            fake, alts, strength, bbox = make_fake(arr, rng, np_rng, pill_boxes.get(name))
            f_rel = f"{split}/counterfeit/{stem}_fake.jpg"
            f_path = args.out_dir / f_rel
            save_jpeg(fake, f_path)

            manifest_rows.append({
                "fake_filename": f_rel,
                "source_filename": name,
                "split": split,
                "alterations": ";".join(alts),
                "strength": strength,
                "localized": str(bbox is not None).lower(),
                "bbox": "" if bbox is None else ",".join(map(str, bbox)),
            })
            if len(preview_rows) < 8 and split == "train":
                preview_rows.append((g_path, f_path, bbox))

    manifest_path = args.out_dir / "fakes_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        w.writerows(manifest_rows)

    make_preview(preview_rows, args.preview_dir / "preview_grid.png")
    run_assertions(args.out_dir, split_map, manifest_rows)

    n_local = sum(r["localized"] == "true" for r in manifest_rows)
    print(f"Manifest: {manifest_path} ({len(manifest_rows)} fakes, {n_local} localised)")
    print(f"Preview:  {args.preview_dir / 'preview_grid.png'}")


if __name__ == "__main__":
    main()
