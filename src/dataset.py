"""Datasets, transforms and loaders for the genuine-vs-counterfeit classifier.

data/{train,val,test}/ are mixed pools of anonymous images; data/labels.csv holds the ground truth.
Class indices follow ImageFolder's alphabetical convention: counterfeit = 0, genuine = 1, so
P(counterfeit) = softmax(logits)[:, 0]. The checkpoint stores class_to_idx and everything uses it.
"""
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

CLASS_TO_IDX = {"counterfeit": 0, "genuine": 1}
LABELS_CSV = config.DATA_DIR / "labels.csv"


def train_transform(img_size: int):
    # Light augmentation only. No colour jitter: colour shift is one of the counterfeit cues.
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ToTensor(),
        T.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])


def eval_transform(img_size: int):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])


def load_labels() -> pd.DataFrame:
    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"{LABELS_CSV} not found. Run src/generate_fakes.py first.")
    return pd.read_csv(LABELS_CSV, keep_default_na=False)


class MedicineDataset(Dataset):
    """One split of data/labels.csv. Returns (image_tensor, class_index, row_index)."""

    def __init__(self, split: str, transform, df: pd.DataFrame | None = None, kinds=None):
        df = load_labels() if df is None else df
        df = df[df["split"] == split]
        if kinds is not None:
            df = df[df["kind"].isin(kinds)]
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        with Image.open(config.DATA_DIR / row["filename"]) as im:
            img = self.transform(im.convert("RGB"))
        return img, CLASS_TO_IDX[row["label"]], i


def make_loaders(img_size: int = config.IMG_SIZE, batch_size: int = config.BATCH_SIZE, kinds=None):
    """Return {split: (dataset, loader)}. num_workers=0 keeps Windows happy."""
    df = load_labels()
    out = {}
    for split in ("train", "val", "test"):
        tf = train_transform(img_size) if split == "train" else eval_transform(img_size)
        ds = MedicineDataset(split, tf, df, kinds)
        out[split] = (ds, DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"), num_workers=0))
    return out
