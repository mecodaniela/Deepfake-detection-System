"""
predict.py — Merr një imazh të vetëm, kthen CNN score (prob. "fake", 0-1).
Ky është funksioni që do të thirret nga src/fusion/ më vonë.

Ekzekutim direkt (test): python src\\dl_layer\\predict.py path\\te\\imazhi.jpg
"""

import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from typing import cast

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

_model = None  # ngarkohet një herë, ripërdoret (lazy loading)


def _load_model():
    global _model
    if _model is None:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        _model = build_model(num_unfrozen_blocks=checkpoint["num_unfrozen_blocks"])
        _model.load_state_dict(checkpoint["model_state_dict"])
        _model.to(DEVICE)
        _model.eval()
    return _model


@torch.no_grad()
def predict_cnn_score(image_path: str) -> float:
    """
    Merr path te një imazh (fytyrë e cropped, çfarëdo madhësie), kthen
    probabilitetin që imazhi të jetë 'fake' (0.0 = real, 1.0 = fake).
    """
    model = _load_model()

    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    tensor = cast(torch.Tensor, _transform(image)).unsqueeze(0).to(DEVICE)

    output = model(tensor)
    prob_fake = torch.softmax(output, dim=1)[0, 1].item()

    return prob_fake


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Përdorim: python predict.py path\\te\\imazhi.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    score = predict_cnn_score(image_path)
    print(f"CNN score (probabiliteti fake): {score:.4f}")
    print(f"Verdikt: {'FAKE' if score > 0.5 else 'REAL'}")