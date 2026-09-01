"""
gradcam.py — Grad-CAM mbi CNN-në (EfficientNet-B0, dl_layer) — tregon
CILAT zona të imazhit ndikuan më shumë në vendimin e CNN-së. Shpjegim,
jo klasifikues i katërt.
"""

import sys
from pathlib import Path
from typing import cast, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.dl_layer.model import build_model

MODEL_PATH = Path("models/dl_layer_best.pt")
DEVICE = torch.device("cpu")

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
])

_model: Optional[torch.nn.Module] = None
_cam: Optional[GradCAM] = None


def _load_model_and_cam():
    global _model, _cam
    if _model is None:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        _model = build_model(num_unfrozen_blocks=checkpoint["num_unfrozen_blocks"])
        _model.load_state_dict(checkpoint["model_state_dict"])
        _model.to(DEVICE)
        _model.eval()

        # Shtresa e fundit konvolucionale e EfficientNet-B0 — pikërisht
        # ajo çka Grad-CAM përdor për të llogaritur gradientët spaciale
        target_layer = cast(nn.Sequential, _model.features)[-1]
        _cam = GradCAM(model=_model, target_layers=[target_layer])

    return _model, _cam


def compute_gradcam_map(image_path: str, target_class: int = 1) -> np.ndarray:
    """
    Kthen hartën Grad-CAM (H, W), normalizuar [0,1] — target_class=1
    ('fake') si default, sepse na intereson pse CNN mendon se një
    imazh është i manipuluar.
    """
    model, cam = _load_model_and_cam()

    image = Image.open(image_path).convert("RGB").resize((224, 224))
    input_tensor = cast(torch.Tensor, _transform(image)).unsqueeze(0).to(DEVICE)

    targets = [ClassifierOutputTarget(target_class)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]  # type: ignore

    return grayscale_cam  # tashmë [0,1] nga vetë libraria


def save_gradcam_overlay(image_path: str, output_path: str, target_class: int = 1):
    """Ruaj overlay Grad-CAM mbi imazhin origjinal si file."""
    image = Image.open(image_path).convert("RGB").resize((224, 224))
    image_np = np.array(image).astype(np.float32) / 255.0

    grayscale_cam = compute_gradcam_map(image_path, target_class)
    overlay = show_cam_on_image(image_np, grayscale_cam, use_rgb=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(output_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Përdorim: python gradcam.py path\\te\\imazhi.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    out_path = Path("outputs/gradcam") / (Path(image_path).stem + "_gradcam.png")

    save_gradcam_overlay(image_path, str(out_path))
    print(f"Grad-CAM overlay u ruajt te: {out_path}")