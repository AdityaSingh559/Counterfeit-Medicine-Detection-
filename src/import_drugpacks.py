"""Import medicine BOX photos (Mobile-Captured Drug Packs) into data/raw.

Source: Mendeley Data "Mobile-Captured Drug Packs", CC BY 4.0
        https://data.mendeley.com/datasets/bsmy5jjysy  (doi 10.17632/bsmy5jjysy.3)

    python src/import_drugpacks.py [--n 400] [--per-group 4]

* Downloads a seeded sample (each file's SHA-256 is checked against the dataset listing).
* Letterboxes each photo to 640x640 on black (same look as the MEDISEG pill images).
* Writes box_NNNNN.jpg to data/raw and merges two sidecars:
    data/raw_boxes.json   {name: {"box": [x1,y1,x2,y2], "tight": false}}  -> region of the pack photo
    data/raw_groups.json  {name: group_id}   -> photos of (roughly) the same pack share a group,
                                                 so generate_fakes.py never splits a pack across train/test.
Groups are a heuristic: files named YYYYMMDD_HHMMSS are grouped by capture gaps > 10 s,
IMG_NNNN files in consecutive blocks of 12 (the dataset has ~12 shots per pack).
"""
import argparse
import datetime
import hashlib
import json
import random
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

API = "https://data.mendeley.com/public-api/datasets/bsmy5jjysy/files?folder_id=root&version=3"
UA = {"User-Agent": "Mozilla/5.0 (research dataset import; counterfeit-medicine-detection)"}  # default python UA gets 403
HEADERS = {"Accept": "application/vnd.mendeley-public-dataset.1+json", **UA}
GAP_SECONDS = 10
IMG_BLOCK = 12
SIDE = 640
PREFIX = "box_"


def http_get(url: str, headers: dict) -> bytes:
    """GET bytes. Mendeley sits behind Cloudflare, which can 403 Python's TLS fingerprint
    ("Just a moment..."); the system curl (native TLS stack) gets through, so fall back to it."""
    r = requests.get(url, headers=headers, timeout=120)
    if r.ok:
        return r.content
    cmd = ["curl", "-sSL", "--fail", "--retry", "2", url]
    for k, v in headers.items():
        cmd += ["-H", f"{k}: {v}"]
    return subprocess.run(cmd, check=True, capture_output=True).stdout


def list_files():
    data = json.loads(http_get(API, HEADERS))
    return [{"name": f["filename"], "url": f["content_details"]["download_url"],
             "sha256": f["content_details"]["sha256_hash"]} for f in data]


def make_groups(files):
    """Return {filename: group_id}."""
    groups = {}
    stamped = sorted((f["name"] for f in files if re.match(r"^\d{8}_\d{6}", f["name"])))
    prev, gid = None, 0
    for n in stamped:
        t = datetime.datetime.strptime(n[:15], "%Y%m%d_%H%M%S")
        if prev is not None and (t - prev).total_seconds() > GAP_SECONDS:
            gid += 1
        groups[n] = f"ts{gid:03d}"
        prev = t
    seq = sorted((f["name"] for f in files if re.match(r"^IMG_\d+", f["name"])),
                 key=lambda n: int(re.findall(r"\d+", n)[0]))
    for i, n in enumerate(seq):
        groups[n] = f"img{i // IMG_BLOCK:03d}"
    return groups


def fetch(item, dest: Path):
    out = dest / item["name"]
    if out.exists() and hashlib.sha256(out.read_bytes()).hexdigest() == item["sha256"]:
        return out
    for _ in range(3):
        try:
            content = http_get(item["url"], UA)
        except (requests.RequestException, subprocess.CalledProcessError):
            continue
        if hashlib.sha256(content).hexdigest() == item["sha256"]:
            out.write_bytes(content)
            return out
    raise RuntimeError(f"download/checksum failed for {item['name']}")


def letterbox(path: Path):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        scale = SIDE / max(im.size)
        w, h = max(1, round(im.width * scale)), max(1, round(im.height * scale))
        im = im.resize((w, h), Image.LANCZOS)
    canvas = Image.new("RGB", (SIDE, SIDE), (0, 0, 0))
    x, y = (SIDE - w) // 2, (SIDE - h) // 2
    canvas.paste(im, (x, y))
    return canvas, (x, y, x + w, y + h)


def merge_json(path: Path, updates: dict, drop_prefix: str):
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data = {k: v for k, v in data.items() if not k.startswith(drop_prefix)}
    data.update(updates)
    path.write_text(json.dumps(data), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--per-group", type=int, default=4, help="max photos taken from one pack group")
    ap.add_argument("--cache", type=Path, default=config.DATA_DIR / "_cache" / "drugpacks")
    args = ap.parse_args()

    files = list_files()
    groups = make_groups(files)
    by_group = {}
    for f in files:
        if f["name"] in groups:
            by_group.setdefault(groups[f["name"]], []).append(f)
    print(f"{len(files)} files listed in {len(by_group)} pack groups")

    rng = random.Random(config.SEED)
    chosen = []
    for gid in sorted(by_group):
        items = sorted(by_group[gid], key=lambda f: f["name"])
        rng.shuffle(items)
        chosen += [(gid, f) for f in items[: args.per_group]]
    rng.shuffle(chosen)
    chosen = chosen[: args.n]
    print(f"Downloading {len(chosen)} photos from {len({g for g, _ in chosen})} groups ...")

    args.cache.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=8) as ex:
        paths = list(ex.map(lambda gf: fetch(gf[1], args.cache), chosen))

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    for old in config.RAW_DIR.glob(f"{PREFIX}*.jpg"):
        old.unlink()
    boxes, grp = {}, {}
    for i, ((gid, _), p) in enumerate(zip(chosen, paths)):
        canvas, region = letterbox(p)
        name = f"{PREFIX}{i:05d}.jpg"
        canvas.save(config.RAW_DIR / name, format="JPEG", quality=95)
        boxes[name] = {"box": list(region), "tight": False}
        grp[name] = gid
    merge_json(config.DATA_DIR / "raw_boxes.json", boxes, PREFIX)
    merge_json(config.DATA_DIR / "raw_groups.json", grp, PREFIX)
    print(f"Wrote {len(boxes)} images to {config.RAW_DIR} and updated raw_boxes.json / raw_groups.json")


if __name__ == "__main__":
    main()
