"""
calibrate_forensic_score.py — Llogarit mean/std të CFA dhe DCT mbi
CALIBRATION split (jo test!), për normalizim korrekt te forensic_score.py.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.forensic_layer.cfa import compute_cfa_score
from src.forensic_layer.dct import compute_dct_score

CALIB_DIR = Path("data/frames/val")
SAMPLE_SIZE = 100  # real+fake bashkë, mostër më e madhe për kalibrim të qëndrueshëm


def collect_scores(compute_fn, n=SAMPLE_SIZE):
    scores = []
    for label in ["real", "fake"]:
        folder = CALIB_DIR / label
        files = sorted(folder.glob("*.jpg"))[:n // 2]
        for f in files:
            scores.append(compute_fn(str(f)))
    return np.array(scores)


def main():
    print("Duke llogaritur CFA mbi calibration split...")
    cfa_scores = collect_scores(compute_cfa_score)
    print(f"CFA_MEAN, CFA_STD = {cfa_scores.mean():.4f}, {cfa_scores.std():.4f}")

    print("Duke llogaritur DCT mbi calibration split...")
    dct_scores = collect_scores(compute_dct_score)
    print(f"DCT_MEAN, DCT_STD = {dct_scores.mean():.4f}, {dct_scores.std():.4f}")

    print("\nKopjo këto vlera te forensic_score.py (CFA_MEAN/STD, DCT_MEAN/STD).")


if __name__ == "__main__":
    main()