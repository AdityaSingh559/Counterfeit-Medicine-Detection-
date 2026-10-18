"""Phase 5 sanity checks: trivial baseline, shortcut tests, overfit-16-images.

    python src/sanity_checks.py

1. Baseline: colour histogram + logistic regression (no CNN). The CNN must beat this to mean anything.
2. Shortcuts: can file size / image dimensions / JPEG quantisation alone predict the class? (They should not.)
3. Overfit: the full model must reach ~100% on 16 training images, proving the training loop works.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from dataset import CLASS_TO_IDX, MedicineDataset, eval_transform, load_labels  # noqa: E402
from train import build_model, set_mode, set_trainable  # noqa: E402

OUT = config.OUTPUTS_DIR / "metrics" / "sanity_checks.json"


def hist_features(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        a = np.asarray(im.convert("RGB").resize((64, 64)), dtype=np.float32)
    h, _ = np.histogramdd(a.reshape(-1, 3), bins=(6, 6, 6), range=((0, 256),) * 3)
    return (h / h.sum()).ravel()


def file_features(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        w, h = im.size
        q = im.quantization if hasattr(im, "quantization") else {}
        qmean = float(np.mean([np.mean(t) for t in q.values()])) if q else 0.0
    size = path.stat().st_size
    return np.array([size, size / (w * h), w, h, qmean], dtype=np.float64)


def fit_eval(df, featfn, name, res):
    X, y = {}, {}
    for split in ("train", "test"):
        d = df[df.split == split]
        X[split] = np.stack([featfn(config.DATA_DIR / f) for f in d.filename])
        y[split] = (d.label == "counterfeit").astype(int).to_numpy()
    sc = StandardScaler().fit(X["train"])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(X["train"]), y["train"])
    p = clf.predict_proba(sc.transform(X["test"]))[:, 1]
    pred = (p >= 0.5).astype(int)
    d = df[df.split == "test"]
    out = {"auc": float(roc_auc_score(y["test"], p)), "accuracy": float((pred == y["test"]).mean()),
           "cf_recall": float(pred[y["test"] == 1].mean())}
    for k in sorted(d.kind.unique()):
        m = (d.kind == k).to_numpy()
        out[f"auc_{k}"] = float(roc_auc_score(y["test"][m], p[m]))
    res[name] = out
    print(name, {k: round(v, 3) for k, v in out.items()})
    return X, y


def single_feature_auc(X, y, names, res):
    aucs = {}
    for i, n in enumerate(names):
        a = roc_auc_score(y["test"], X["test"][:, i]) if np.ptp(X["test"][:, i]) > 0 else 0.5
        aucs[n] = float(max(a, 1 - a))                     # direction-free
    res["single_file_feature_auc"] = aucs
    print("single-feature AUC (0.5 = no signal):", {k: round(v, 3) for k, v in aucs.items()})


def overfit_test(res, device):
    config.set_seeds()
    df = load_labels()
    ds = MedicineDataset("train", eval_transform(config.IMG_SIZE), df)
    idx = list(range(0, len(ds), max(1, len(ds) // 16)))[:16]
    xs = torch.stack([ds[i][0] for i in idx]).to(device)
    ys = torch.tensor([ds[i][1] for i in idx]).to(device)
    model = build_model().to(device)
    set_trainable(model, 0)                                   # unfreeze everything
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for step in range(1, 41):
        set_mode(model, 0)
        opt.zero_grad()
        loss = loss_fn(model(xs), ys)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(xs).argmax(1) == ys).float().mean().item()
    res["overfit_16"] = {"final_loss": float(loss.item()), "train_acc_on_16": acc}
    print(f"overfit 16 images: loss {loss.item():.4f}, acc {acc:.2f}")


def main():
    df = load_labels()
    res = {}
    print("-- baseline: colour histogram + logistic regression")
    fit_eval(df, hist_features, "baseline_colour_histogram", res)
    print("-- shortcut check: file size / dimensions / JPEG quantisation only")
    X, y = fit_eval(df, file_features, "shortcut_file_features", res)
    single_feature_auc(X, y, ["file_size", "bytes_per_pixel", "width", "height", "jpeg_quant_mean"], res)
    print("-- overfit 16 images")
    overfit_test(res, "cuda" if torch.cuda.is_available() else "cpu")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
