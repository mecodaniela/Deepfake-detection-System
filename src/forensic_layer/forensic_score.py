"""
forensic_score.py — Kombinon ELA, CFA, DCT në forensic_score, me
normalizim z-score (jo mesatare e papërpunuar) për të shmangur
dominimin nga score-t me shkallë të ndryshme (p.sh. DCT pranë 1.0).

Konstantet CFA_MEAN/STD, DCT_MEAN/STD llogariten mbi CALIBRATION split
(jo test!) — shih calibrate_forensic_score.py.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.forensic_layer.ela import compute_ela_score
from src.forensic_layer.cfa import compute_cfa_score
from src.forensic_layer.dct import compute_dct_score

WEIGHTS = {"ela": 0.0, "cfa": 0.5, "dct": 0.5}

# --- Konstante normalizimi (VENDOSËSE PËRKOHËSISHT — rifresko me
# calibrate_forensic_score.py mbi calibration split) ---
CFA_MEAN, CFA_STD = 0.3073, 0.0364
DCT_MEAN, DCT_STD = 0.9396, 0.0968


def _zscore_sigmoid(x: float, mean: float, std: float) -> float:
    """Standardizon x, pastaj e shtrydh në 0-1 me sigmoid — bën score
    të ndryshme të krahasueshme përpara mesatarizimit."""
    import math
    z = (x - mean) / (std + 1e-8)
    return 1.0 / (1.0 + math.exp(-z))


def compute_forensic_score(image_path: str) -> dict:
    raw = {
        "ela": compute_ela_score(image_path),
        "cfa": compute_cfa_score(image_path),
        "dct": compute_dct_score(image_path),
    }

    normalized = {
        "ela": raw["ela"],  # peshë 0, s'ka nevojë normalizimi
        "cfa": _zscore_sigmoid(raw["cfa"], CFA_MEAN, CFA_STD),
        "dct": _zscore_sigmoid(raw["dct"], DCT_MEAN, DCT_STD),
    }

    active_weights = {k: WEIGHTS[k] for k in normalized if WEIGHTS[k] > 0}
    weight_sum = sum(active_weights.values())
    forensic_score = sum(
        normalized[k] * active_weights[k] for k in active_weights
    ) / weight_sum

    result = dict(raw)
    result["forensic_score"] = forensic_score
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Përdorim: python forensic_score.py <path_to_image>")
        sys.exit(1)
    result = compute_forensic_score(sys.argv[1])
    print("\nScore individuale (raw):")
    for key, val in result.items():
        if key != "forensic_score":
            print(f"  {key.upper()}: {val:.4f}")
    print(f"\nForensic Score (i kombinuar, normalizuar): {result['forensic_score']:.4f}")