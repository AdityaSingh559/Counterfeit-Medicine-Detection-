"""Phase 6: do the heatmaps point at what was actually edited?

    python src/validate_explanations.py --select                   # pick method/layer on VALIDATION data
    python src/validate_explanations.py --evaluate --split test    # report with the chosen explainer
    python src/validate_explanations.py --figures                  # curated TP/TN/FP/FN figures

For every LOCALISED fake (the manifest has a bbox of the edit) we measure, on the heatmap:
  mass     share of heatmap energy inside the edited box     (chance level = box area share)
  lift     mass / chance                                      (>1: attention concentrated on the edit)
  iou20    IoU between the top-20% heatmap pixels and the box
  hit      does the single hottest pixel fall inside the box ("pointing game")
Controls: the same box on the genuine TWIN of that photo (should be ~chance), and
  black_lift = share of heatmap energy on black pixels / share of black pixels  (>1: attention on black
  borders/letterbox, a possible shortcut), plus the same for the outer 8% frame band.
Global edits (colour shift, noise, warp) have no meaningful box, so they are excluded from overlap scores.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from dataset import eval_transform, load_labels  # noqa: E402
from explain import EXPLAINER_FILE, build_engine, cam_batch, load_classifier, load_explainer_config  # noqa: E402
from explain import METHODS  # noqa: E402

METRICS = config.OUTPUTS_DIR / "metrics"
SRC_SIZE = 640                      # all dataset images are 640x640
CANDIDATE_LAYERS = [18, 17, 13, 10]  # MobileNetV2 features[...]: stride 32 (x3) ... stride 16


def load_pairs(split: str):
    """DataFrame of localised fakes with bbox and the filename of their genuine twin."""
    lab = load_labels()
    lab = lab[lab.split == split]
    man = pd.read_csv(config.MANIFEST_PATH, keep_default_na=False).set_index("fake_filename")
    fakes = lab[lab.label == "counterfeit"].join(man[["alterations", "strength", "localized", "bbox"]], on="filename")
    fakes = fakes[fakes.bbox != ""].copy()
    genuine = lab[lab.label == "genuine"].set_index("source_filename").filename
    fakes["twin"] = fakes.source_filename.map(genuine)
    fakes["box"] = fakes.bbox.map(lambda s: [int(v) for v in s.split(",")])
    return fakes.reset_index(drop=True)


def box_mask(box, size):
    s = size / SRC_SIZE
    x1, y1, x2, y2 = [int(round(v * s)) for v in box]
    m = np.zeros((size, size), bool)
    m[max(0, y1):min(size, y2), max(0, x1):min(size, x2)] = True
    return m


def load_batch(files, size, device):
    tf = eval_transform(size)
    x = torch.stack([tf(Image.open(config.DATA_DIR / f).convert("RGB")) for f in files]).to(device)
    rgb = np.stack([np.asarray(Image.open(config.DATA_DIR / f).convert("RGB").resize((size, size))) for f in files])
    return x, rgb


def cams_for(files, engine, cf, size, device, bs=24):
    cams, rgbs = [], []
    for i in range(0, len(files), bs):
        x, rgb = load_batch(files[i:i + bs], size, device)
        cams.append(cam_batch(engine, x, cf))
        rgbs.append(rgb)
    return np.concatenate(cams), np.concatenate(rgbs)


def region_stats(cam, mask):
    total = cam.sum() + 1e-9
    mass = float(cam[mask].sum() / total)
    chance = float(mask.mean())
    top = cam >= np.quantile(cam, 0.8)
    return {"mass": mass, "chance": chance, "lift": mass / max(chance, 1e-9),
            "iou20": float((top & mask).sum() / max(1, (top | mask).sum())),
            "hit": bool(mask[np.unravel_index(cam.argmax(), cam.shape)])}


def border_stats(cam, rgb):
    """Attention on black pixels / outer frame, relative to how much of the image they cover."""
    total = cam.sum() + 1e-9
    black = rgb.max(axis=2) < 12
    size = cam.shape[0]
    band = np.zeros_like(black)
    k = int(0.08 * size)
    band[:k], band[-k:], band[:, :k], band[:, -k:] = True, True, True, True
    out = {"black_frac": float(black.mean()), "edge_frac": float(band.mean()),
           "edge_lift": float(cam[band].sum() / total / band.mean())}
    out["black_lift"] = float(cam[black].sum() / total / black.mean()) if black.mean() > 0.01 else np.nan
    return out


def evaluate(split, method, layer, model, ck, device, extra_all=True):
    cf = ck["class_to_idx"]["counterfeit"]
    size = ck["img_size"]
    engine = build_engine(model, method, layer)
    pairs = load_pairs(split)
    fcams, frgb = cams_for(list(pairs.filename), engine, cf, size, device)
    gcams, grgb = cams_for(list(pairs.twin), engine, cf, size, device)
    rows = []
    for i, r in pairs.iterrows():
        m = box_mask(r.box, size)
        f, g = region_stats(fcams[i], m), region_stats(gcams[i], m)
        rows.append({"filename": r.filename, "kind": r.kind, "strength": r.strength, "alterations": r.alterations,
                     **{f"fake_{k}": v for k, v in f.items()},
                     **{f"twin_{k}": v for k, v in g.items()},
                     **{f"fake_{k}": v for k, v in border_stats(fcams[i], frgb[i]).items()},
                     **{f"twin_{k}": v for k, v in border_stats(gcams[i], grgb[i]).items()}})
    df = pd.DataFrame(rows)
    summ = {"n_localised_fakes": int(len(df)),
            "fake": {k: float(df[f"fake_{k}"].mean()) for k in ("mass", "chance", "lift", "iou20", "hit")},
            "genuine_twin_control": {k: float(df[f"twin_{k}"].mean()) for k in ("mass", "lift", "iou20", "hit")},
            "border": {"fake_black_lift": float(df.fake_black_lift.mean()), "twin_black_lift": float(df.twin_black_lift.mean()),
                       "fake_edge_lift": float(df.fake_edge_lift.mean()), "twin_edge_lift": float(df.twin_edge_lift.mean())}}
    return summ, df


def select(model, ck, device):
    rows = []
    for method, layer in itertools.product(METHODS, CANDIDATE_LAYERS):
        s, _ = evaluate("val", method, layer, model, ck, device)
        rows.append({"method": method, "layer": layer, "iou20": s["fake"]["iou20"], "lift": s["fake"]["lift"],
                     "hit": s["fake"]["hit"], "mass": s["fake"]["mass"], "twin_lift": s["genuine_twin_control"]["lift"],
                     "fake_black_lift": s["border"]["fake_black_lift"], "twin_black_lift": s["border"]["twin_black_lift"]})
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    df = pd.DataFrame(rows)
    # Selection criterion: how much MORE the heatmap concentrates on the edited box for the fake than for its
    # genuine twin (a heatmap that lights up the same box on genuine photos is not showing the edit).
    df["lift_gap"] = df["lift"] - df["twin_lift"]
    df = df.sort_values("lift_gap", ascending=False)
    df.to_csv(METRICS / "explain_selection_val.csv", index=False)
    best = df.iloc[0]
    EXPLAINER_FILE.write_text(json.dumps({"method": best.method, "layer": int(best.layer), "selected_on": "validation lift_gap (fake lift - genuine-twin lift)"}),
                              encoding="utf-8")
    print(f"selected: {best.method} @ features[{int(best.layer)}]")
    return best


def figures(model, ck, device, method, layer, n_each=2):
    """Curated TP / TN / FP / FN triptychs from the TEST pool using test predictions."""
    from explain import show_cam_on_image  # noqa: E402
    cf = ck["class_to_idx"]["counterfeit"]
    size = ck["img_size"]
    pred_file = METRICS / "test_final_predictions.csv"
    pr = pd.read_csv(pred_file)
    pr["fake"] = pr.label == "counterfeit"
    pr["pred_fake"] = pr.p_cf >= 0.5
    man = pd.read_csv(config.MANIFEST_PATH, keep_default_na=False).set_index("fake_filename")
    pr = pr.join(man[["alterations", "bbox"]], on="filename")
    groups = {"TP (fake caught)": pr[pr.fake & pr.pred_fake].sort_values("p_cf", ascending=False),
              "TN (genuine passed)": pr[~pr.fake & ~pr.pred_fake].sort_values("p_cf"),
              "FP (genuine flagged)": pr[~pr.fake & pr.pred_fake].sort_values("p_cf", ascending=False),
              "FN (fake missed)": pr[pr.fake & ~pr.pred_fake].sort_values("p_cf")}
    engine = build_engine(model, method, layer)
    out_dir = config.OUTPUTS_DIR / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):
        old.unlink()
    cases = []
    for name, df in groups.items():
        for _, r in df.head(n_each).iterrows():
            cases.append((name, r))
    fig, axes = plt.subplots(len(cases), 2, figsize=(7, 3.4 * len(cases)))
    for ax_row, (name, r) in zip(axes, cases):
        x, rgb = load_batch([r.filename], size, device)
        cam = cam_batch(engine, x, cf)[0]
        overlay = show_cam_on_image(rgb[0].astype(np.float32) / 255.0, cam, use_rgb=True)
        ax_row[0].imshow(rgb[0])
        if isinstance(r.bbox, str) and r.bbox:
            x1, y1, x2, y2 = [v * size / SRC_SIZE for v in map(int, r.bbox.split(","))]
            ax_row[0].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="lime", lw=2))
        ax_row[0].set_title(f"{name}: {r.kind}, true={r.label}", fontsize=9)
        ax_row[1].imshow(overlay)
        ax_row[1].set_title(f"heatmap ({method}@{layer}), P(fake)={r.p_cf:.2f}", fontsize=9)
        for a in ax_row:
            a.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "gradcam_cases.png", dpi=110)
    plt.close()
    print(f"saved {out_dir / 'gradcam_cases.png'} ({len(cases)} cases; green box = where the edit was made)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--split", default="test")
    ap.add_argument("--method")
    ap.add_argument("--layer", type=int)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    model, ck, device = load_classifier()
    if args.select:
        select(model, ck, device)
    cfg = load_explainer_config()
    method, layer = args.method or cfg["method"], cfg["layer"] if args.layer is None else args.layer
    if args.evaluate:
        summ, df = evaluate(args.split, method, layer, model, ck, device)
        summ["explainer"] = {"method": method, "layer": layer}
        by_strength = df.groupby("strength")[["fake_iou20", "fake_lift", "fake_hit"]].mean().round(3).to_dict()
        by_kind = df.groupby("kind")[["fake_iou20", "fake_lift", "fake_hit"]].mean().round(3).to_dict()
        summ["by_strength"], summ["by_kind"] = by_strength, by_kind
        out = METRICS / f"explain_validation_{args.split}{args.tag}.json"
        out.write_text(json.dumps(summ, indent=2), encoding="utf-8")
        df.to_csv(METRICS / f"explain_validation_{args.split}{args.tag}_rows.csv", index=False)
        print(json.dumps(summ, indent=1))
    if args.figures:
        figures(model, ck, device, method, layer)


if __name__ == "__main__":
    main()
