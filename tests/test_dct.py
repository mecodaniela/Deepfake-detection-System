"""
test_dct.py — Unit tests për src/forensic_layer/dct.py.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.forensic_layer.dct import compute_dct_score

SAMPLE_DIR = PROJECT_ROOT / "data" / "frames" / "test" / "real"
SAMPLE_IMAGE = next(iter(SAMPLE_DIR.glob("*.jpg")), None)

pytestmark = pytest.mark.skipif(
    SAMPLE_IMAGE is None, reason="Asnjë imazh mostër nuk u gjet te data/frames/test/real"
)

def test_compute_dct_score_returns_float():
    score = compute_dct_score(str(SAMPLE_IMAGE))
    assert isinstance(score, float)

def test_compute_dct_score_is_finite():
    score = compute_dct_score(str(SAMPLE_IMAGE))
    assert np.isfinite(score)

def test_compute_dct_score_deterministic():
    score1 = compute_dct_score(str(SAMPLE_IMAGE))
    score2 = compute_dct_score(str(SAMPLE_IMAGE))
    assert score1 == score2

def test_compute_dct_score_within_documented_range():
    score = compute_dct_score(str(SAMPLE_IMAGE))
    assert 0.0 <= score <= 1.0