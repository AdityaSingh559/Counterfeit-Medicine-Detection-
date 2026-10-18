"""Train MobileNetV2 (transfer learning) to detect counterfeit medicine images.

    python src/train.py [--epochs 8] [--ft-epochs 4] [--unfreeze-from 14] [--img-size 224]

Stage 1: backbone frozen, head only (lr 1e-3).   Stage 2: unfreeze features[unfreeze-from:] (lr 1e-4).
Checkpoint rule (plan: best validation counterfeit recall, F1 tie-break), with one guard: epochs whose
counterfeit precision is below 0.80 rank below all others, so "flag everything" can never win.
Model selection uses VALIDATION only; the test set is scored once at the end.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,  # noqa: E402
                             log_loss, precision_recall_fscore_support, roc_auc_score)
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from dataset import CLASS_TO_IDX, make_loaders  # noqa: E402

METRICS_DIR = config.OUTPUTS_DIR / "metrics"
CF = CLASS_TO_IDX["counterfeit"]


def build_model(pretrained: bool = True) -> nn.Module:
    m = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
    m.classifier[1] = nn.Linear(m.last_channel, 2)          # 1280 -> 2
    return m


def set_trainable(model: nn.Module, unfreeze_from: int | None):
    """unfreeze_from=None: head only. Otherwise features[unfreeze_from:] + head train."""
    for p in model.features.parameters():
        p.requires_grad = False
    if unfreeze_from is not None:
        for blk in model.features[unfreeze_from:]:
            for p in blk.parameters():
                p.requires_grad = True


def set_mode(model: nn.Module, unfreeze_from: int | None):
    """Frozen blocks stay in eval mode so their BatchNorm statistics are not disturbed."""
    model.train()
    cut = len(model.features) if unfreeze_from is None else unfreeze_from
    for blk in model.features[:cut]:
        blk.eval()


@torch.no_grad()
def predict(model, loader, device):
    """Return (P(counterfeit) array, true class-index array) in dataset order."""
    model.eval()
    probs, ys = [], []
    for x, y, _ in loader:
        p = torch.softmax(model(x.to(device)), dim=1)[:, CF]
        probs.append(p.cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(probs), np.concatenate(ys)


def quick_metrics(p_cf, y, thr=0.5):
    pred = np.where(p_cf >= thr, CF, 1 - CF)
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, labels=[CF], zero_division=0)
    return {"acc": float(accuracy_score(y, pred)), "cf_precision": float(pr[0]),
            "cf_recall": float(rc[0]), "cf_f1": float(f1[0])}


def epoch_score(m):
    return (m["cf_precision"] >= 0.80, m["cf_recall"], m["cf_f1"])


def run_epochs(model, loaders, device, n_epochs, lr, unfreeze_from, tag, history, best):
    set_trainable(model, unfreeze_from)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    _, train_loader = loaders["train"]
    _, val_loader = loaders["val"]
    for ep in range(1, n_epochs + 1):
        t0 = time.time()
        set_mode(model, unfreeze_from)
        run_loss, correct, n = 0.0, 0, 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
        p_val, y_val = predict(model, val_loader, device)
        vm = quick_metrics(p_val, y_val)
        vloss = float(log_loss((y_val == CF).astype(int), p_val, labels=[0, 1]))
        rec = {"stage": tag, "epoch": len(history) + 1, "train_loss": run_loss / n, "train_acc": correct / n,
               "val_loss": vloss, **{f"val_{k}": v for k, v in vm.items()}}
        history.append(rec)
        print(f"[{tag}] ep {rec['epoch']:2d}  loss {rec['train_loss']:.3f} acc {rec['train_acc']:.3f} | "
              f"val acc {vm['acc']:.3f} cf-recall {vm['cf_recall']:.3f} cf-prec {vm['cf_precision']:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if best["score"] is None or epoch_score(vm) > best["score"]:
            best["score"], best["epoch"], best["metrics"] = epoch_score(vm), rec["epoch"], vm
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def full_report(p_cf, y, df, out_dir: Path, name: str):
    """Metrics on a split + breakdowns (kind, strength, localised, alteration)."""
    pred = np.where(p_cf >= 0.5, CF, 1 - CF)
    prec, rec, f1, sup = precision_recall_fscore_support(y, pred, labels=[CF, 1 - CF], zero_division=0)
    res = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score((y == CF).astype(int), p_cf)),
        "counterfeit": {"precision": float(prec[0]), "recall": float(rec[0]), "f1": float(f1[0]), "support": int(sup[0])},
        "genuine": {"precision": float(prec[1]), "recall": float(rec[1]), "f1": float(f1[1]), "support": int(sup[1])},
        "confusion_matrix_[cf,gen]x[cf,gen]": confusion_matrix(y, pred, labels=[CF, 1 - CF]).tolist(),
    }
    d = df.copy()
    d["y"], d["pred"], d["p_cf"] = y, pred, p_cf
    man = pd.read_csv(config.MANIFEST_PATH, keep_default_na=False).set_index("fake_filename")
    d = d.join(man[["alterations", "strength", "localized"]], on="filename")
    d["localized"] = d["localized"].astype(str).str.lower()          # pandas may parse true/false as bool

    def cf_recall(sub):
        return None if len(sub) == 0 else {"n": int(len(sub)), "recall": float((sub.pred == CF).mean())}

    fakes = d[d.y == CF]
    res["by_kind"] = {k: {"cf_recall": cf_recall(fakes[fakes.kind == k]),
                          "genuine_recall": float((d[(d.kind == k) & (d.y != CF)].pred != CF).mean())}
                      for k in sorted(d.kind.unique())}
    res["cf_recall_by_strength"] = {s: cf_recall(fakes[fakes.strength == s]) for s in ("low", "medium", "high")}
    res["cf_recall_by_localized"] = {s: cf_recall(fakes[fakes.localized == s]) for s in ("true", "false")}
    res["cf_recall_by_alteration"] = {a: cf_recall(fakes[fakes.alterations.str.contains(a)])
                                      for a in ("color_shift", "blur_imprint", "wrong_text", "shape_change", "noise_compression")}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}_metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    d[["filename", "label", "kind", "y", "p_cf"]].to_csv(out_dir / f"{name}_predictions.csv", index=False)
    ConfusionMatrixDisplay(confusion_matrix(y, pred, labels=[CF, 1 - CF]),
                           display_labels=["counterfeit", "genuine"]).plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion matrix ({name})")
    plt.savefig(out_dir / f"confusion_matrix_{name}.png", dpi=130, bbox_inches="tight")
    plt.close()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--ft-epochs", type=int, default=4)
    ap.add_argument("--unfreeze-from", type=int, default=14)
    ap.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    ap.add_argument("--head-lr", type=float, default=config.LR)
    ap.add_argument("--ft-lr", type=float, default=1e-4)
    ap.add_argument("--skip-test", action="store_true", help="experiments: score val only, never touch test")
    ap.add_argument("--tag", default="", help="suffix for output files (experiments)")
    ap.add_argument("--no-save", action="store_true", help="do not overwrite models/best_model.pt")
    args = ap.parse_args()

    config.set_seeds()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  img_size: {args.img_size}")
    loaders = make_loaders(args.img_size)
    for s, (ds, _) in loaders.items():
        print(s, len(ds), ds.df.label.value_counts().to_dict())

    model = build_model().to(device)
    history, best = [], {"score": None}
    run_epochs(model, loaders, device, args.epochs, args.head_lr, None, "head", history, best)
    if args.ft_epochs:
        run_epochs(model, loaders, device, args.ft_epochs, args.ft_lr, args.unfreeze_from, "finetune", history, best)

    print(f"best epoch {best['epoch']}  val {best['metrics']}")
    model.load_state_dict(best["state"])
    suffix = f"_{args.tag}" if args.tag else ""
    if not args.no_save:
        config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best["state"], "class_to_idx": CLASS_TO_IDX, "img_size": args.img_size,
                    "mean": config.IMAGENET_MEAN, "std": config.IMAGENET_STD, "epoch": best["epoch"],
                    "val_metrics": best["metrics"]}, config.MODEL_PATH)
        print(f"saved {config.MODEL_PATH}")

    # Final scoring: val (for threshold tuning later) and test (once).
    for split in (("val",) if args.skip_test else ("val", "test")):
        ds, loader = loaders[split]
        p, y = predict(model, loader, device)
        res = full_report(p, y, ds.df, METRICS_DIR, f"{split}{suffix}")
        if split == "test":
            print(json.dumps({k: res[k] for k in ("accuracy", "roc_auc", "counterfeit", "genuine", "by_kind",
                                                  "cf_recall_by_strength", "cf_recall_by_localized",
                                                  "cf_recall_by_alteration")}, indent=1))

    h = pd.DataFrame(history)
    h.to_csv(METRICS_DIR / f"training_history{suffix}.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(h.epoch, h.train_loss, label="train"); ax[0].plot(h.epoch, h.val_loss, label="val")
    ax[0].set_title("loss"); ax[0].legend()
    ax[1].plot(h.epoch, h.train_acc, label="train"); ax[1].plot(h.epoch, h.val_acc, label="val")
    ax[1].plot(h.epoch, h.val_cf_recall, label="val cf recall", ls="--"); ax[1].set_title("accuracy / recall"); ax[1].legend()
    for a in ax:
        a.set_xlabel("epoch")
    plt.tight_layout(); plt.savefig(METRICS_DIR / f"training_curves{suffix}.png", dpi=130); plt.close()


if __name__ == "__main__":
    main()
