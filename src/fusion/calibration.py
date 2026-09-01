# calibration.py
"""
calibration.py — Vlerësuam Platt scaling dhe isotonic regression si hap
para thresholds.py, dhe vendosëm t'i lëmë mënjanë nga pipeline-i default:
  1) LogisticRegression optimizon direkt log-loss, kështu që output-i i
     saj ËSHTË probabilitet by construction.
  2) Isotonic regression kërkon zakonisht >1000 mostra për të shmangur
     overfitting mbi kurbën e kalibrimit, holdout-i ynë (~160-200 mostra)
     është nën këtë prag.
  3) T1/T2 (thresholds.py) gjenden direkt nga precision_recall_curve mbi
     probabilitetin e papërpunuar, çka jep garanci empirike precision-i
     pavarësisht sa "teorikisht i kalibruar" është vetë probabiliteti.
Funksioni më poshtë mbahet vetëm si mjet krahasimi eksperimental
Ekzekuti,i: python src\\fusion\\calibration.py
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.fusion.train_fusion import build_dataset
from src.fusion.logistic_fusion import load_fusion_model

RANDOM_STATE = 42
TEST_SIZE = 0.2

def fit_isotonic_calibrator(base_model, X_val, y_val):
    """Kthen një version isotonic-kalibruar të modelit, vetëm për krahasim."""
    calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv="prefit")
    calibrated.fit(X_val, y_val)
    return calibrated

def main():
    print("Duke krahasuar probabilitetin e papërpunuar me atë isotonic-kalibruar...")
    X, y = build_dataset()
    _, X_val, _, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model, _ = load_fusion_model()
    raw_probs = model.predict_proba(X_val)[:, 1]

    calibrated_model = fit_isotonic_calibrator(model, X_val, y_val)
    calibrated_probs = calibrated_model.predict_proba(X_val)[:, 1]

    brier_raw = brier_score_loss(y_val, raw_probs)
    brier_calibrated = brier_score_loss(y_val, calibrated_probs)

    print("\n" + "=" * 50)
    print("KRAHASIMI: probabilitet i papërpunuar vs isotonic-kalibruar")
    print("=" * 50)
    print(f"Brier score (papërpunuar):       {brier_raw:.4f}")
    print(f"Brier score (isotonic-kalibruar): {brier_calibrated:.4f}")
    print("\n(Brier score më i ulët = probabilitete më të kalibruara. "
          "Nëse dallimi është minimal, kjo mbështet vendimin për të mos "
          "shtuar isotonic si hap default.)")

if __name__ == "__main__":
    main()