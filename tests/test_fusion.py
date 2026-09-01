"""
test_fusion.py — Unit tests për src/fusion/logistic_fusion.py.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.fusion.logistic_fusion import (
    compute_feature_vector, load_fusion_model, predict_fusion_probability
)

SAMPLE_DIR = PROJECT_ROOT / "data" / "frames" / "test" / "real"
SAMPLE_IMAGE = next(iter(SAMPLE_DIR.glob("*.jpg")), None)

pytestmark = pytest.mark.skipif(
    SAMPLE_IMAGE is None, reason="Asnjë imazh mostër nuk u gjet te data/frames/test/real"
)

def test_compute_feature_vector_returns_list():
    features = compute_feature_vector(str(SAMPLE_IMAGE))
    assert isinstance(features, list)

def test_compute_feature_vector_length_matches_feature_names():
    """
    Sipas models/fusion_thresholds.json, feature_names = [cnn, cfa, dct, fft]
    (ELA u hoq nga fusion) — pra 4 elementë.
    """
    features = compute_feature_vector(str(SAMPLE_IMAGE))
    assert len(features) == 4

def test_compute_feature_vector_all_finite():
    features = compute_feature_vector(str(SAMPLE_IMAGE))
    assert all(np.isfinite(f) for f in features)

def test_load_fusion_model_does_not_raise():
    model, feature_names = load_fusion_model()
    assert model is not None
    assert isinstance(feature_names, list)

def test_predict_fusion_probability_returns_float_in_range():
    prob = predict_fusion_probability(str(SAMPLE_IMAGE))
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0

def test_predict_fusion_probability_deterministic():
    prob1 = predict_fusion_probability(str(SAMPLE_IMAGE))
    prob2 = predict_fusion_probability(str(SAMPLE_IMAGE))
    assert prob1 == prob2

def test_predict_fusion_probability_with_preloaded_model_matches_default():
    """Kalimi eksplicit i modelit duhet të japë të njëjtin rezultat si ngarkimi automatik.

    load_fusion_model() kthen (model, feature_names) — predict_fusion_probability
    pret vetë modelin, jo tuple-in e plotë.
    """
    model, _ = load_fusion_model()
    prob_explicit = predict_fusion_probability(str(SAMPLE_IMAGE), model=model)
    prob_default = predict_fusion_probability(str(SAMPLE_IMAGE))
    assert prob_explicit == prob_default