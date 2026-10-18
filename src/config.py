"""Central configuration for the counterfeit medicine detection project."""
import random
from pathlib import Path

import numpy as np

# --- Reproducibility ---
SEED = 42

# --- Model / training ---
IMG_SIZE = 224          # plan: 160 on CPU, 224 on GPU (RTX 4060 available; small edits need the resolution)
BATCH_SIZE = 32
EPOCHS = 8
LR = 1e-3
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --- Decision thresholds ---
T_HIGH = 0.85           # p >= T_HIGH  -> FLAG     (verdict 1)
T_LOW = 0.15            # p <= T_LOW   -> APPROVE  (verdict 0); otherwise ESCALATE (verdict 2)

# --- Data ---
SPLIT = (0.70, 0.15, 0.15)

# --- Audit backend ---
AUDIT_BACKEND = "chain"  # or "hashchain"
RPC_URL = "http://127.0.0.1:8545"
CHAIN_ID = 31337
SOLC_VERSION = "0.8.20"

# --- Paths (pathlib keeps Windows happy) ---
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MANIFEST_PATH = DATA_DIR / "fakes_manifest.csv"
OUTPUTS_DIR = ROOT / "outputs"
HEATMAP_DIR = OUTPUTS_DIR / "heatmaps"
CONTRACT_SOURCE = ROOT / "contracts" / "MedicineAudit.sol"
CONTRACT_FILE = ROOT / "contract.json"
MODEL_PATH = ROOT / "models" / "best_model.pt"
HASHCHAIN_LOG = OUTPUTS_DIR / "hashchain_log.csv"


def set_seeds(seed: int = SEED) -> None:
    """Fix random seeds for random, numpy and torch (if installed)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
