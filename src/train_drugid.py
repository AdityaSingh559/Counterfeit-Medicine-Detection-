"""Train the drug-identification model and build the genuine reference gallery.

    python src/prepare_drugid.py --src <MEDISEG>/32pills      # once
    python src/train_drugid.py [--epochs 6] [--ft-epochs 8]

* MobileNetV2 fine-tuned to name the drug (MEDISEG registration code) from a pill photo.
  No colour jitter: colour is part of a drug's identity.
* Saves models/drug_model.pt (weights + class list) and models/drug_gallery.pt
  (L2-normalised 1280-d embeddings of the genuine TRAIN photos of every drug) for reference matching.
* Trained only on genuine photos that are not in the counterfeit-detection dataset.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from dataset import eval_transform  # noqa: E402
from train import build_model, set_mode, set_trainable  # noqa: E402

DRUG_DIR = config.DATA_DIR / "drugid"
DRUG_MODEL = config.ROOT / "models" / "drug_model.pt"
DRUG_GALLERY = config.ROOT / "models" / "drug_gallery.pt"


def drug_train_tf(size):
    return T.Compose([T.Resize((size, size)), T.RandomHorizontalFlip(), T.RandomRotation(15), T.ToTensor(),
                      T.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD)])


def embed(model, x):
    """Pooled 1280-d feature vector (the layer before the classifier)."""
    f = model.features(x)
    return torch.flatten(nn.functional.adaptive_avg_pool2d(f, 1), 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, n = 0, 0
    for x, y in loader:
        correct += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
        n += len(y)
    return correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--ft-epochs", type=int, default=8)
    ap.add_argument("--unfreeze-from", type=int, default=10)
    ap.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    args = ap.parse_args()

    config.set_seeds()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = datasets.ImageFolder(DRUG_DIR / "train", drug_train_tf(args.img_size))
    val_ds = datasets.ImageFolder(DRUG_DIR / "val", eval_transform(args.img_size))
    test_ds = datasets.ImageFolder(DRUG_DIR / "test", eval_transform(args.img_size))
    classes = train_ds.classes
    print(f"{len(classes)} drugs; train {len(train_ds)} val {len(val_ds)} test {len(test_ds)}")
    tl = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    vl = DataLoader(val_ds, batch_size=64, num_workers=0)
    sl = DataLoader(test_ds, batch_size=64, num_workers=0)

    counts = np.bincount(train_ds.targets, minlength=len(classes))
    w = torch.tensor((counts.max() / counts) ** 0.5, dtype=torch.float32, device=device)   # soften imbalance
    loss_fn = nn.CrossEntropyLoss(weight=w)

    model = build_model().to(device)
    model.classifier[1] = nn.Linear(model.last_channel, len(classes)).to(device)
    best_acc, best_state = -1.0, None
    for stage, n_ep, lr, unfreeze in (("head", args.epochs, 1e-3, None),
                                      ("finetune", args.ft_epochs, 1e-4, args.unfreeze_from)):
        set_trainable(model, unfreeze)
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
        for ep in range(1, n_ep + 1):
            t0 = time.time()
            set_mode(model, unfreeze)
            tot, n = 0.0, 0
            for x, y in tl:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss = loss_fn(model(x), y)
                loss.backward()
                opt.step()
                tot += loss.item() * len(y)
                n += len(y)
            acc = evaluate(model, vl, device)
            print(f"[{stage}] ep {ep:2d} loss {tot / n:.3f} val acc {acc:.3f} ({time.time() - t0:.0f}s)", flush=True)
            if acc > best_acc:
                best_acc, best_state = acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_acc = evaluate(model, sl, device)
    print(f"best val acc {best_acc:.3f}  TEST top-1 acc {test_acc:.3f}")

    # Reference gallery: embeddings of genuine TRAIN photos (no augmentation), per drug.
    gal_ds = datasets.ImageFolder(DRUG_DIR / "train", eval_transform(args.img_size))
    gl = DataLoader(gal_ds, batch_size=64, num_workers=0)
    model.eval()
    embs, labs = [], []
    with torch.no_grad():
        for x, y in gl:
            e = nn.functional.normalize(embed(model, x.to(device)), dim=1)
            embs.append(e.cpu()); labs.append(y)
    embs, labs = torch.cat(embs), torch.cat(labs)
    gallery = {classes[i]: embs[labs == i] for i in range(len(classes))}

    names = json.loads((DRUG_DIR / "drugs.json").read_text(encoding="utf-8"))
    DRUG_MODEL.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "classes": classes, "names": names, "img_size": args.img_size,
                "val_acc": best_acc, "test_acc": test_acc}, DRUG_MODEL)
    torch.save(gallery, DRUG_GALLERY)
    print(f"saved {DRUG_MODEL} and {DRUG_GALLERY}")


if __name__ == "__main__":
    main()
