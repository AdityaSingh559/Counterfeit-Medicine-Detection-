"""Build the drug-identification dataset (data/drugid/{train,val,test}/<drug_id>/*.jpg).

    python src/prepare_drugid.py --src <extracted>/MEDISEG/32pills [--min-per-class 30]

Uses MEDISEG pill photos that are NOT in the counterfeit-detection set (data/raw_boxes.json "src"),
so the drug matcher is never evaluated on photos it trained on. One photo per source (no augmentation
duplicates). Label = drug of the largest pill in the photo. Drugs with fewer than --min-per-class
photos are dropped (they are reported as "unknown" at inference time).
Split 80/10/10 by source photo, seeded. Also writes data/drugid/drugs.json (id -> name).
"""
import argparse
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from import_mediseg import load_candidates  # noqa: E402

OUT = config.DATA_DIR / "drugid"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--min-pill", type=int, default=40)
    ap.add_argument("--min-per-class", type=int, default=30)
    args = ap.parse_args()

    boxes_file = config.DATA_DIR / "raw_boxes.json"
    if not boxes_file.exists():
        sys.exit("Run import_mediseg.py first (need data/raw_boxes.json to know which photos to exclude).")
    used = {v["src"] for v in json.loads(boxes_file.read_text(encoding="utf-8")).values() if "src" in v}

    cands = [c for c in load_candidates(args.src, args.min_pill) if c["file_name"] not in used]
    counts = Counter(c["drug_id"] for c in cands)
    keep = {d for d, n in counts.items() if n >= args.min_per_class}
    cands = [c for c in cands if c["drug_id"] in keep]
    print(f"{len(cands)} photos, {len(keep)} drugs kept "
          f"(dropped {sorted(set(counts) - keep)} with < {args.min_per_class} photos)")

    if OUT.exists():
        shutil.rmtree(OUT)
    rng = random.Random(config.SEED)
    rng.shuffle(cands)
    names = {}
    for i, c in enumerate(cands):
        r = i / len(cands)
        split = "train" if r < 0.8 else "val" if r < 0.9 else "test"
        d = OUT / split / c["drug_id"]
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(args.src / "images" / c["file_name"], d / f"{i:05d}.jpg")
        names[c["drug_id"]] = c["drug_name"]
    (OUT / "drugs.json").write_text(json.dumps(names, indent=1), encoding="utf-8")
    for s in ("train", "val", "test"):
        print(s, sum(1 for _ in (OUT / s).rglob("*.jpg")))


if __name__ == "__main__":
    main()
