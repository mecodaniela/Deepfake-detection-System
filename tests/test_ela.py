"""
test_ela.py — Unit tests për src/forensic_layer/ela.py.
Kontrollon shape, range, dhe konsistencë të output-it — jo saktësinë
statistikore (ajo verifikohet te 02_forensic_analysis.ipynb / batch tests).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.forensic_layer.ela import compute_ela_map, compute_ela_score

SAMPLE_DIR = PROJECT_ROOT / "data" / "frames" / "test" / "real"
SAMPLE_IMAGE = next(iter(SAMPLE_DIR.glob("*.jpg")), None)

pytestmark = pytest.mark.skipif(
    SAMPLE_IMAGE is None, reason="Asnjë imazh mostër nuk u gjet te data/frames/test/real"
)

def test_compute_ela_map_returns_ndarray():
    ela_map = compute_ela_map(str(SAMPLE_IMAGE))
    assert isinstance(ela_map, np.ndarray)

def test_compute_ela_map_shape_matches_grayscale_2d():
    ela_map = compute_ela_map(str(SAMPLE_IMAGE))
    assert ela_map.ndim == 2
    assert ela_map.shape[0] > 0 and ela_map.shape[1] > 0

def test_compute_ela_map_non_negative():
    ela_map = compute_ela_map(str(SAMPLE_IMAGE))
    assert (ela_map >= 0).all()

def test_compute_ela_score_returns_float():
    score = compute_ela_score(str(SAMPLE_IMAGE))
    assert isinstance(score, float)

def test_compute_ela_score_is_finite():
    score = compute_ela_score(str(SAMPLE_IMAGE))
    assert np.isfinite(score)

def test_compute_ela_score_deterministic():
    """Njëjti imazh duhet të japë njëjtin rezultat në thirrje të njëpasnjëshme."""
    score1 = compute_ela_score(str(SAMPLE_IMAGE))
    score2 = compute_ela_score(str(SAMPLE_IMAGE))
    assert score1 == score2

def test_compute_ela_score_quality_parameter_changes_result():
    """Cilësi të ndryshme JPEG duhet të japin rezultate të ndryshme (jo identike)."""
    score_default = compute_ela_score(str(SAMPLE_IMAGE))
    score_low_quality = compute_ela_score(str(SAMPLE_IMAGE), quality=50)
    assert score_default != score_low_quality