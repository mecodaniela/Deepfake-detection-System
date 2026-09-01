"""
test_thresholds.py — Unit tests për logjikën e pragjeve (T1/T2).
VËREJTJE: find_threshold() ndodhet te src/fusion/thresholds.py
(përdoret vetëm gjatë kalibrimit, jo në kohë inference), ndërsa
load_thresholds() dhe classify_with_thresholds() ndodhen te src/inference_pipeline.py (përdoren në kohë inference/klasifikimi).
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.fusion.thresholds import find_threshold
from src.inference_pipeline import load_thresholds, classify_with_thresholds

def test_load_thresholds_returns_dict():
    thresholds = load_thresholds()
    assert isinstance(thresholds, dict)

def test_load_thresholds_has_expected_keys():
    """Bazuar te models/fusion_thresholds.json i konfirmuar."""
    thresholds = load_thresholds()
    for key in ["t1_real", "t2_fake", "target_precision", "feature_names"]:
        assert key in thresholds

def test_load_thresholds_t1_less_than_t2():
    thresholds = load_thresholds()
    assert thresholds["t1_real"] < thresholds["t2_fake"]

def test_load_thresholds_matches_documented_values():
    """T1=0.6015, T2=0.8170 siç u dokumentua në sesionin e kalibrimit."""
    thresholds = load_thresholds()
    assert thresholds["t1_real"] == pytest.approx(0.6015, abs=1e-3)
    assert thresholds["t2_fake"] == pytest.approx(0.8170, abs=1e-3)

def test_classify_with_thresholds_returns_dict():
    thresholds = load_thresholds()
    result = classify_with_thresholds(0.5, thresholds["t1_real"], thresholds["t2_fake"])
    assert isinstance(result, dict)

def test_classify_with_thresholds_low_vs_high_prob_differ():
    """
    Një probabilitet shumë i ulët (afër 0.0, real me besim të lartë) dhe
    një shumë i lartë (afër 1.0, fake me besim të lartë) duhet të japin
    rezultate të ndryshme nga classify_with_thresholds.
    """
    thresholds = load_thresholds()
    t1, t2 = thresholds["t1_real"], thresholds["t2_fake"]
    result_low = classify_with_thresholds(0.01, t1, t2)
    result_high = classify_with_thresholds(0.99, t1, t2)
    assert result_low != result_high

def test_classify_with_thresholds_uncertain_zone_between_t1_t2():
    """Një probabilitet ndërmjet T1 dhe T2 duhet të japë rezultat të ndryshëm
    nga rastet e qarta (real/fake me besim të lartë)."""
    thresholds = load_thresholds()
    t1, t2 = thresholds["t1_real"], thresholds["t2_fake"]
    midpoint = (t1 + t2) / 2
    result_uncertain = classify_with_thresholds(midpoint, t1, t2)
    result_certain_real = classify_with_thresholds(0.01, t1, t2)
    assert result_uncertain != result_certain_real

def test_find_threshold_runs_without_error():
    """Test i thjeshtë sanity mbi të dhëna sintetike të vogla."""
    y_true = [0, 0, 0, 1, 1, 1, 1, 1]
    scores = [0.05, 0.1, 0.2, 0.6, 0.7, 0.8, 0.9, 0.95]
    threshold = find_threshold(y_true, scores, target_precision=0.8)
    assert isinstance(threshold, (float, int))
    assert 0.0 <= threshold <= 1.0