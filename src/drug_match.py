"""Drug identification + reference matching for pill photos.

    from drug_match import DrugMatcher
    m = DrugMatcher()
    m.analyse(pil_images) -> [{"drug_id", "drug_name", "drug_conf", "similarity", "top3"}, ...]

1. A fine-tuned MobileNetV2 names the drug (MEDISEG registration code; name/ingredient from metadata).
2. The photo's embedding is compared (cosine, mean of the top-k) with embeddings of KNOWN GENUINE photos
   of that drug (models/drug_gallery.pt). Low similarity = the pill does not look like genuine examples
   of the drug it appears to be: extra evidence of a counterfeit, independent of the CNN.
Only meaningful for loose pills; boxes are not covered (no drug labels for boxes).
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from dataset import eval_transform  # noqa: E402
from train import build_model  # noqa: E402
from train_drugid import DRUG_GALLERY, DRUG_MODEL, embed  # noqa: E402

TOP_K = 5


class DrugMatcher:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(DRUG_MODEL, map_location="cpu")
        self.classes, self.names = ck["classes"], ck["names"]
        self.model = build_model(pretrained=False)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, len(self.classes))
        self.model.load_state_dict(ck["state_dict"])
        self.model.to(self.device).eval()
        self.tf = eval_transform(ck["img_size"])
        gal = torch.load(DRUG_GALLERY, map_location="cpu")
        self.gallery = {d: g.to(self.device) for d, g in gal.items()}

    @torch.no_grad()
    def analyse_tensors(self, x: torch.Tensor):
        x = x.to(self.device)
        emb = nn.functional.normalize(embed(self.model, x), dim=1)
        probs = torch.softmax(self.model(x), dim=1)
        out = []
        for i in range(len(x)):
            conf, idx = probs[i].max(0)
            drug = self.classes[int(idx)]
            sims = emb[i] @ self.gallery[drug].T
            sim = sims.topk(min(TOP_K, len(sims))).values.mean().item()
            top3 = [(self.classes[int(j)], float(probs[i, j])) for j in probs[i].topk(3).indices]
            out.append({"drug_id": drug, "drug_name": self.names.get(drug, drug), "drug_conf": float(conf),
                        "similarity": sim, "top3": top3})
        return out

    def analyse(self, images, batch_size: int = 32):
        """images: list of PIL images (RGB)."""
        res = []
        for i in range(0, len(images), batch_size):
            x = torch.stack([self.tf(im.convert("RGB")) for im in images[i:i + batch_size]])
            res += self.analyse_tensors(x)
        return res
