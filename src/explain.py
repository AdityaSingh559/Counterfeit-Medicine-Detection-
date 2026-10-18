"""Heatmaps (Grad-CAM family) for the counterfeit classifier.

A heatmap shows which image regions pushed the model toward "counterfeit". It is a hint about the
model's attention, NOT proof; src/validate_explanations.py measures how well it matches the real edits.

    p, overlay, cam = predict_with_heatmap(pil_image, model, ck, device)   # uses models/explainer.json
    cams = cam_batch(engine, x_batch, class_index)                          # (B, H, W) in [0, 1]
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from dataset import eval_transform  # noqa: E402
from train import build_model  # noqa: E402

METHODS = {"gradcam": GradCAM, "gradcam++": GradCAMPlusPlus, "layercam": LayerCAM}
EXPLAINER_FILE = config.ROOT / "models" / "explainer.json"
DEFAULT_EXPLAINER = {"method": "gradcam", "layer": -1}


def load_classifier(device=None):
    """Return (model, checkpoint, device) for models/best_model.pt."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(config.MODEL_PATH, map_location="cpu")
    model = build_model(pretrained=False)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck, device


def load_explainer_config():
    """The (method, layer) chosen on validation data by validate_explanations.py --select."""
    if EXPLAINER_FILE.exists():
        return json.loads(EXPLAINER_FILE.read_text(encoding="utf-8"))
    return dict(DEFAULT_EXPLAINER)


def build_engine(model, method="gradcam", layer=-1):
    return METHODS[method](model=model, target_layers=[model.features[layer]])


def cam_batch(engine, x: torch.Tensor, class_index: int) -> np.ndarray:
    """CAM maps in [0, 1], shape (B, H, W), for the given class (no torch.no_grad: needs gradients)."""
    return engine(input_tensor=x, targets=[ClassifierOutputTarget(class_index)] * len(x))


def predict_with_heatmap(pil_image: Image.Image, model, ck, device, method=None, layer=None):
    """Return (p_counterfeit, overlay_rgb uint8, cam float[0,1] HxW) for one image."""
    cfg = load_explainer_config()
    method, layer = method or cfg["method"], cfg["layer"] if layer is None else layer
    size = ck["img_size"]
    cf = ck["class_to_idx"]["counterfeit"]
    x = eval_transform(size)(pil_image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        p_cf = torch.softmax(model(x), dim=1)[0, cf].item()
    cam = cam_batch(build_engine(model, method, layer), x, cf)[0]
    rgb = np.asarray(pil_image.convert("RGB").resize((size, size)), dtype=np.float32) / 255.0
    return p_cf, show_cam_on_image(rgb, cam, use_rgb=True), cam


def png_bytes(arr: np.ndarray) -> bytes:
    """Encode an RGB uint8 array as PNG bytes (this is what gets hashed and logged on-chain later)."""
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()
