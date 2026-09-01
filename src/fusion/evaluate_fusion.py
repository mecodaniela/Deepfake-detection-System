# evaluate_fusion.py
"""
evaluate_fusion.py — Vlerësimi final i sistemit të fusion-it (CNN + CFA +
DCT + FFT + pragjet T1/T2) mbi test split-in, i cili mbetet plotësisht i
paprekur deri në këtë pikë (as CNN, as fusion, as thresholds nuk e kanë parë).

Jep dy grupe rezultatesh:
  1) Metrika standarde binare (accuracy/precision/recall/F1/ROC-AUC, prag
     0.5) — për krahasueshmëri direkte me dl_layer standalone.
  2) Shpërndarja sipas sistemit dy-prag T1/T2 (real i sigurt / fake i
     sigurt / i paqartë) + accuracy vetëm mbi rastet "e sigurta"
     (coverage-based) — numri që ka kuptim real për kontekst gjyqësor.

Ekzekutim: python src\\fusion\\evaluate_fusion.py
"""

import sys
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.fusion.train_fusion import build_dataset
from src.fusion.logistic_fusion import load_fusion_model, FEATURE_NAMES

THRESHOLDS_PATH = Path("models/fusion_thresholds.json")
TEST_DIR = Path("data/frames/test")
RESULTS_PATH = Path("evaluation/test_results_fusion.json")


def main():
    print(f"Duke ndërtuar feature-t mbi test split-in ({TEST_DIR})...")
    X_test, y_test = build_dataset(dir_path=TEST_DIR)
    print(f"Test set: {len(y_test)} mostra ({int(np.sum(y_test == 0))} real, "
          f"{int(np.sum(y_test == 1))} fake)")

    print("Duke ngarkuar modelin e fusion-it...")
    model, _ = load_fusion_model()

    print(f"Duke ngarkuar pragjet nga {THRESHOLDS_PATH}...")
    with open(THRESHOLDS_PATH, "r", encoding="utf-8") as f:
        thresholds = json.load(f)
    t1, t2 = thresholds["t1_real"], thresholds["t2_fake"]

    probs = model.predict_proba(X_test)[:, 1]

    # --- 1) Metrika standarde (prag 0.5) ---
    preds_05 = (probs >= 0.5).astype(int)

    metrics_05 = {
        "accuracy": accuracy_score(y_test, preds_05),
        "precision": precision_score(y_test, preds_05),
        "recall": recall_score(y_test, preds_05),
        "f1": f1_score(y_test, preds_05),
        "roc_auc": roc_auc_score(y_test, probs),
        "confusion_matrix": confusion_matrix(y_test, preds_05).tolist(),
    }

    print("\n" + "=" * 50)
    print("METRIKA STANDARDE (prag = 0.5)")
    print("=" * 50)
    for k, v in metrics_05.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v:.4f}")
    print(f"  confusion_matrix (rows=actual, cols=pred, 0=real/1=fake): "
          f"{metrics_05['confusion_matrix']}")
    print("\n" + str(classification_report(y_test, preds_05, target_names=["real", "fake"]))
    )
    # --- 2) Sistemi dy-prag T1/T2 ---
    is_real_certain = probs < t1
    is_fake_certain = probs >= t2
    is_uncertain = ~is_real_certain & ~is_fake_certain

    n_total = len(probs)
    n_real_c = int(np.sum(is_real_certain))
    n_fake_c = int(np.sum(is_fake_certain))
    n_unc = int(np.sum(is_uncertain))

    certain_mask = is_real_certain | is_fake_certain
    certain_preds = np.where(is_real_certain[certain_mask], 0, 1)
    certain_true = y_test[certain_mask]
    acc_certain = accuracy_score(certain_true, certain_preds) if certain_mask.sum() > 0 else None

    print("\n" + "=" * 50)
    print(f"SISTEMI DY-PRAG (T1={t1:.4f}, T2={t2:.4f}) mbi TEST ({n_total} mostra)")
    print("=" * 50)
    print(f"  REAL (besim i lartë):  {n_real_c} ({100*n_real_c/n_total:.1f}%)")
    print(f"  FAKE (besim i lartë):  {n_fake_c} ({100*n_fake_c/n_total:.1f}%)")
    print(f"  E PAQARTË:             {n_unc} ({100*n_unc/n_total:.1f}%)")
    if acc_certain is not None:
        print(f"  Accuracy mbi rastet e sigurta (coverage = {100*certain_mask.sum()/n_total:.1f}%): "
              f"{acc_certain:.4f}")

    # --- Ruajtja e rezultateve ---
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "metrics_threshold_05": metrics_05,
            "two_threshold_system": {
                "t1_real": t1,
                "t2_fake": t2,
                "n_total": n_total,
                "n_real_certain": n_real_c,
                "n_fake_certain": n_fake_c,
                "n_uncertain": n_unc,
                "coverage": certain_mask.sum() / n_total,
                "accuracy_on_certain": acc_certain,
            },
            "feature_names": FEATURE_NAMES,
        }, f, indent=2)
    print(f"\nRezultatet u ruajtën te: {RESULTS_PATH.resolve()}")


if __name__ == "__main__":
    main()