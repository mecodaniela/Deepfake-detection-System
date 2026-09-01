"""
test_cfa.py — Unit tests për src/forensic_layer/cfa.py.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.forensic_layer.cfa import compute_cfa_score

SAMPLE_DIR = PROJECT_ROOT / "data" / "frames" / "test" / "real"
SAMPLE_IMAGE = next(iter(SAMPLE_DIR.glob("*.jpg")), None)

pytestmark = pytest.mark.skipif(
    SAMPLE_IMAGE is None, reason="Asnjë imazh mostër nuk u gjet te data/frames/test/real"
)

def test_compute_cfa_score_returns_float():
    score = compute_cfa_score(str(SAMPLE_IMAGE))
    assert isinstance(score, float)

def test_compute_cfa_score_is_finite():
    score = compute_cfa_score(str(SAMPLE_IMAGE))
    assert np.isfinite(score)

def test_compute_cfa_score_deterministic():
    score1 = compute_cfa_score(str(SAMPLE_IMAGE))
    score2 = compute_cfa_score(str(SAMPLE_IMAGE))
    assert score1 == score2

def test_compute_cfa_score_within_documented_range():
    """
    Bazuar te testi batch i dokumentuar: real mean 0.2901 (range 0.2433-0.6177),
    fake mean 0.3326 (range 0.2666-0.5119). Lejojmë një marzhe të gjerë sepse kjo është vetëm një mostër e vetme, jo statistikë batch.
    """
    score = compute_cfa_score(str(SAMPLE_IMAGE))
    assert 0.0 <= score <= 1.0