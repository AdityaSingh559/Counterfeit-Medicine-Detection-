"""Import a sample of MEDISEG (32-pills) into data/raw plus a pill-box sidecar file.

MEDISEG: City, University of London, CC BY 4.0, https://doi.org/10.25383/city.28574786.v2
Download MEDISEG.tar.gz from figshare and extract it first, then:

    python src/import_mediseg.py --src <extracted>/MEDISEG/32pills [--n 1100] [--min-pill 40]

Why a sidecar: counterfeit simulation should alter the PILL, not random background.
Writes data/raw_boxes.json  {filename: [x1, y1, x2, y2]} (the largest annotated pill).
Also de-duplicates Roboflow augmentations (same source photo, different '.rf.<hash>').
"""
import argparse
import json
import random
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="folder containing images/ and annotations.json")
    ap.add_argument("--n", type=int, default=1100)
    ap.add_argument("--min-pill", type=int, default=40, help="min shorter side (px) of the largest pill box")
    args = ap.parse_args()

    coco = json.loads((args.src / "annotations.json").read_text(encoding="utf-8"))
    by_image = {}
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        by_image.setdefault(a["image_id"], []).append((x, y, x + w, y + h))

    seen_sources, candidates = set(), []
    for im in sorted(coco["images"], key=lambda i: i["file_name"]):
        boxes = by_image.get(im["id"])
        if not boxes:
            continue
        source = re.split(r"\.rf\.", im["file_name"])[0]
        if source in seen_sources:
            continue
        x1, y1, x2, y2 = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        if min(x2 - x1, y2 - y1) < args.min_pill:
            continue
        seen_sources.add(source)
        candidates.append((im["file_name"], (round(x1), round(y1), round(x2), round(y2))))

    print(f"{len(candidates)} distinct source photos with a pill >= {args.min_pill}px")
    random.Random(config.SEED).shuffle(candidates)
    chosen = candidates[: args.n]

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    boxes_out = {}
    for i, (fname, box) in enumerate(chosen):
        new = f"medi_{i:05d}.jpg"
        shutil.copy(args.src / "images" / fname, config.RAW_DIR / new)
        boxes_out[new] = box
    (config.DATA_DIR / "raw_boxes.json").write_text(json.dumps(boxes_out), encoding="utf-8")
    print(f"Copied {len(chosen)} images to {config.RAW_DIR} and wrote data/raw_boxes.json")


if __name__ == "__main__":
    main()
