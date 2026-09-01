"""
social_media_simulation.py — Kombinon resize + JPEG compression për
simulim kushtesh tipike degradimi online. S'pretendon riprodhim të
saktë të ndonjë platforme specifike (WhatsApp/Instagram/Facebook).
"""
import sys
from pathlib import Path
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.degradation.resize import apply_resize_degradation
from src.degradation.jpeg_compression import apply_jpeg_compression

PRESETS = {
    "light": {"scale_percent": 90, "jpeg_quality": 85},
    "medium": {"scale_percent": 75, "jpeg_quality": 65},
    "heavy": {"scale_percent": 50, "jpeg_quality": 40},
}


def apply_social_media_degradation(image: Image.Image, preset: str = "medium") -> Image.Image:
    if preset not in PRESETS:
        raise ValueError(f"Preset i panjohur: {preset}. Zgjidh nga {list(PRESETS.keys())}")
    params = PRESETS[preset]
    degraded = apply_resize_degradation(image, params["scale_percent"])
    return apply_jpeg_compression(degraded, params["jpeg_quality"])