"""
ela.py — Error Level Analysis (ELA) për deepfake/manipulation detection.

Output: ela_score (0-1, probabilitet manipulimi) + ela_map (për vizualizim/
overlay me Grad-CAM më vonë, siç e kishim planifikuar te Kreu V).

KUJDES: normalizimi (SCORE_NORMALIZATION) është HEURISTIK/EMPIRIK — duhet
kalibruar duke e testuar mbi disa imazhe real/fake të njohura para se ta
konsiderosh score-in "final". Kjo është e ndryshme nga CNN, ku threshold-i
u nxor nga trainimi vetë.
"""

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops

JPEG_QUALITY = 90       # cilësia e rikompresimit
SCALE_FACTOR = 15       # amplifikim vetëm për vizualizim, jo për score
SCORE_NORMALIZATION = 40.0  # kalibrim fillestar — RREGULLO pas testeve reale


def compute_ela_map(image_path: str, quality: int = JPEG_QUALITY) -> np.ndarray:
    """Kthen ELA map (H, W) — diferenca absolute mes origjinalit dhe
    versionit të rikompresuar JPEG, mesatarizuar mbi kanalet RGB."""
    original = Image.open(image_path).convert("RGB")

    buffer = io.BytesIO()
    original.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer)

    diff = ImageChops.difference(original, recompressed)
    diff_array = np.array(diff).astype(np.float32)

    return diff_array.mean(axis=2)  # grayscale error map


def compute_ela_score(image_path: str, quality: int = JPEG_QUALITY) -> float:
    """
    Kthen score 0-1 (probabilitet manipulimi). Kombinon mesataren globale
    të gabimit me std/max, sepse manipulimi priret të krijojë zona LOKALE
    me error të lartë, jo uniform mbi gjithë imazhin.
    """
    ela_map = compute_ela_map(image_path, quality)

    mean_error = ela_map.mean()
    std_error = ela_map.std()
    max_error = ela_map.max()

    raw_score = (0.5 * mean_error) + (0.3 * std_error) + (0.2 * (max_error / 10))
    score = min(raw_score / SCORE_NORMALIZATION, 1.0)

    return float(score)


def save_ela_visualization(image_path: str, output_path: str, quality: int = JPEG_QUALITY):
    """Ruaj version të amplifikuar të ELA map — për figurën 4-panel
    (original, ELA heatmap, Grad-CAM, overlay) te Kreu V."""
    ela_map = compute_ela_map(image_path, quality)
    amplified = np.clip(ela_map * SCALE_FACTOR, 0, 255).astype(np.uint8)
    Image.fromarray(amplified).save(output_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Përdorim: python ela.py path\\te\\imazhi.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    score = compute_ela_score(image_path)
    print(f"ELA score (probabiliteti manipulim): {score:.4f}")
    print(f"Verdikt: {'MANIPULUAR' if score > 0.5 else 'AUTENTIK'}")

    out_path = Path(image_path).with_stem(Path(image_path).stem + "_ela_map")
    save_ela_visualization(image_path, str(out_path))
    print(f"ELA map u ruajt te: {out_path}")