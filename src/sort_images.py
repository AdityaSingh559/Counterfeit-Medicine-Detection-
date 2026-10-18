"""Segregate a mixed pool of unlabeled images using the trained model.

    python src/sort_images.py [--input data/test] [--out outputs/sorted] [--t-low 0.15] [--t-high 0.85]

Input is ONE mixed folder with no labels. Each image gets P(counterfeit); then
    p >= t-high  -> counterfeit/   (FLAG)
    p <= t-low   -> genuine/       (APPROVE)
    otherwise    -> escalate/      (needs a human / lab)
Images are copied (never moved) so the input pool stays intact. If the files are listed in
data/labels.csv, a scorecard compares the sorting with the ground truth (used for testing only).
"""
import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from dataset import eval_transform  # noqa: E402
from train import build_model  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def load_model(device):
    ck = torch.load(config.MODEL_PATH, map_location="cpu")
    model = build_model(pretrained=False)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck


@torch.no_grad()
def score_images(paths, model, ck, device, batch_size=32):
    tf = eval_transform(ck["img_size"])
    cf = ck["class_to_idx"]["counterfeit"]
    probs = []
    for i in range(0, len(paths), batch_size):
        x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + batch_size]]).to(device)
        probs += torch.softmax(model(x), dim=1)[:, cf].cpu().tolist()
    return probs


def action_for(p, t_low, t_high):
    return "counterfeit" if p >= t_high else "genuine" if p <= t_low else "escalate"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=config.DATA_DIR / "test")
    ap.add_argument("--out", type=Path, default=config.OUTPUTS_DIR / "sorted")
    ap.add_argument("--t-low", type=float, default=config.T_LOW)
    ap.add_argument("--t-high", type=float, default=config.T_HIGH)
    args = ap.parse_args()

    if not config.MODEL_PATH.exists():
        sys.exit(f"{config.MODEL_PATH} not found. Train first: python src/train.py")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = load_model(device)

    paths = sorted(p for p in args.input.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    print(f"{len(paths)} images in {args.input}")
    probs = score_images(paths, model, ck, device)

    if args.out.exists():
        shutil.rmtree(args.out)
    for cls in ("genuine", "counterfeit", "escalate"):
        (args.out / cls).mkdir(parents=True)
    rows = []
    for p, pr in zip(paths, probs):
        act = action_for(pr, args.t_low, args.t_high)
        shutil.copy(p, args.out / act / p.name)
        rows.append({"filename": p.name, "p_counterfeit": round(pr, 4), "sorted_as": act})
    rep = pd.DataFrame(rows)
    rep.to_csv(args.out / "sorted_report.csv", index=False)
    print("sorted into:", rep.sorted_as.value_counts().to_dict(), "->", args.out)

    # Scorecard against ground truth (only if these files are in labels.csv).
    lab = pd.read_csv(config.DATA_DIR / "labels.csv", keep_default_na=False)
    rel = f"{args.input.name}/" + rep.filename
    lab = lab.set_index("filename").reindex(rel)
    if lab.label.notna().all() and (lab.label != "").all():
        rep["truth"] = lab.label.to_numpy()
        auto = rep[rep.sorted_as != "escalate"]
        wrong = auto[auto.sorted_as != auto.truth]
        print(f"auto-decided {len(auto)}/{len(rep)} ({len(auto) / len(rep):.0%}), "
              f"correct {len(auto) - len(wrong)} ({1 - len(wrong) / max(1, len(auto)):.1%})")
        print(f"  fakes wrongly auto-APPROVED (missed): {int(((auto.sorted_as == 'genuine') & (auto.truth == 'counterfeit')).sum())}")
        print(f"  genuine wrongly auto-FLAGGED (false alarm): {int(((auto.sorted_as == 'counterfeit') & (auto.truth == 'genuine')).sum())}")
        print("  escalated by truth:", rep[rep.sorted_as == "escalate"].truth.value_counts().to_dict())
        rep.to_csv(args.out / "sorted_report.csv", index=False)


if __name__ == "__main__":
    main()
