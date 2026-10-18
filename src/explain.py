"""Grad-CAM heatmaps for the counterfeit classifier.

STATUS: basic heatmap only (enough for the demo UI). Still to do in Phase 6: overlap validation against
the manifest bboxes (CAM mass / top-20% IoU), a SHAP/LIME comparison, and curated TP/TN/FP/FN figures.
Grad-CAM shows which image regions pushed the model toward "counterfeit"; it is a hint, not proof.
"""
import io
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from dataset import eval_transform  # noqa: E402
from train import build_model  # noqa: E402


def load_classifier(device=None):
    """Return (model, checkpoint, device) for models/best_model.pt."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(config.MODEL_PATH, map_location="cpu")
    model = build_model(pretrained=False)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck, device


def predict_with_heatmap(pil_image: Image.Image, model, ck, device):
    """Return (p_counterfeit, overlay_rgb uint8, cam float[0,1] HxW) for one image."""
    size = ck["img_size"]
    cf = ck["class_to_idx"]["counterfeit"]
    x = eval_transform(size)(pil_image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        p_cf = torch.softmax(model(x), dim=1)[0, cf].item()
    # Heatmap for the COUNTERFEIT class (where the model "looks" for evidence of a fake).
    cam_engine = GradCAM(model=model, target_layers=[model.features[-1]])
    cam = cam_engine(input_tensor=x, targets=[ClassifierOutputTarget(cf)])[0]
    rgb = np.asarray(pil_image.convert("RGB").resize((size, size)), dtype=np.float32) / 255.0
    overlay = show_cam_on_image(rgb, cam, use_rgb=True)
    return p_cf, overlay, cam


def png_bytes(arr: np.ndarray) -> bytes:
    """Encode an RGB uint8 array as PNG bytes (this is what gets hashed and logged on-chain later)."""
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()
