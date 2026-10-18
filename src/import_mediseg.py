"""Import a sample of MEDISEG (32-pills) into data/raw plus a pill-box sidecar file.

MEDISEG: City, University of London, CC BY 4.0, https://doi.org/10.25383/city.28574786.v2
Download MEDISEG.tar.gz from figshare and extract it first, then:

    python src/import_mediseg.py --src <extracted>/MEDISEG/32pills [--n 600] [--min-pill 40]

Why a sidecar: counterfeit simulation should alter the PILL, not random background.
Writes data/raw_boxes.json
    {filename: {"box": [x1,y1,x2,y2], "tight": true, "drug_id": "HK-05565", "drug_name": "LASIX TAB 40MG",
                "src": "<original MEDISEG file name>"}}
for the largest annotated pill. "src" lets prepare_drugid.py exclude these photos from the drug-ID
training set (so drug matching is never evaluated on images it trained on).
Also de-duplicates Roboflow augmentations (same source photo, different '.rf.<hash>').
"""
import argparse
import csv
import json
import random
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


def load_candidates(src: Path, min_pill: int):
    """Distinct source photos whose largest pill has shorter side >= min_pill.

    Returns a list of dicts (sorted by file name): file_name, box, drug_id, drug_name.
    """
    coco = json.loads((src / "annotations.json").read_text(encoding="utf-8"))
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    meta = {}
    meta_path = src.parent / "metadata.csv"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8-sig") as fh:
            meta = {r["id"]: r["name"] for r in csv.DictReader(fh)}

    by_image = {}
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        by_image.setdefault(a["image_id"], []).append((x, y, x + w, y + h, a["category_id"]))

    seen, out = set(), []
    for im in sorted(coco["images"], key=lambda i: i["file_name"]):
        boxes = by_image.get(im["id"])
        if not boxes:
            continue
        source = re.split(r"\.rf\.", im["file_name"])[0]
        if source in seen:
            continue
        x1, y1, x2, y2, cat = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        if min(x2 - x1, y2 - y1) < min_pill:
            continue
        seen.add(source)
        drug_id = cat_name.get(cat, str(cat))
        out.append({"file_name": im["file_name"], "box": (round(x1), round(y1), round(x2), round(y2)),
                    "drug_id": drug_id, "drug_name": meta.get(drug_id, drug_id)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="folder containing images/ and annotations.json")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--min-pill", type=int, default=40, help="min shorter side (px) of the largest pill box")
    args = ap.parse_args()

    candidates = load_candidates(args.src, args.min_pill)
    print(f"{len(candidates)} distinct source photos with a pill >= {args.min_pill}px")
    random.Random(config.SEED).shuffle(candidates)
    chosen = candidates[: args.n]

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    for old in config.RAW_DIR.glob("medi_*.jpg"):
        old.unlink()
    boxes_out = {}
    for i, c in enumerate(chosen):
        new = f"medi_{i:05d}.jpg"
        shutil.copy(args.src / "images" / c["file_name"], config.RAW_DIR / new)
        boxes_out[new] = {"box": list(c["box"]), "tight": True, "drug_id": c["drug_id"],
                          "drug_name": c["drug_name"], "src": c["file_name"]}

    # Merge into the shared sidecar (import_drugpacks.py owns the box_* keys).
    path = config.DATA_DIR / "raw_boxes.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data = {k: v for k, v in data.items() if not k.startswith("medi_")}
    data.update(boxes_out)
    path.write_text(json.dumps(data), encoding="utf-8")
    print(f"Copied {len(chosen)} images to {config.RAW_DIR} and updated data/raw_boxes.json")


if __name__ == "__main__":
    main()
