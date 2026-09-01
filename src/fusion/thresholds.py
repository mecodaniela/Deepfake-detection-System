# thresholds.py
"""
thresholds.py — Gjen pragjet T1 (real, besim i lartë) dhe T2 (fake, besim
i lartë) mbi holdout-in e brendshëm 20% (i njëjti split si train_fusion.py,
random_state=42), duke kërkuar precision >= TARGET_PRECISION për secilën
klasë. Mes T1 dhe T2: zonë "e paqartë" — kërkon shqyrtim njerëzor.

(Ish calibration.py — riemërtuar sepse kjo është pikërisht puna që
specifikimi i arkitekturës ia cakton thresholds.py. Shih calibration.py
për arsyen pse Platt/isotonic scaling u lanë mënjanë si hap i veçantë.)

Ekzekutim: python src\\fusion\\thresholds.py
"""

import sys
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.fusion.train_fusion import build_dataset
from src.fusion.logistic_fusion import load_fusion_model, FEATURE_NAMES

THRESHOLDS_PATH = Path("models/fusion_thresholds.json")

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_PRECISION = 0.95  # sa "të sigurt" duam të jemi për T1/T2


def find_threshold(y_true_binary, scores, target_precision):
    """Kthen pragun minimal ku precision >= target_precision, ose None."""
    precision, recall, thresholds = precision_recall_curve(y_true_binary, scores)
    for p, t in zip(precision[:-1], thresholds):
        if p >= target_precision:
            return float(t)
    return None


def main():
    print("Duke rindërtuar dataset-in dhe split-in e brendshëm (i njëjti si train_fusion.py)...")
    X, y = build_dataset()
    _, X_val, _, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print("Duke ngarkuar modelin e fusion-it...")
    model, _ = load_fusion_model()

    probs = model.predict_proba(X_val)[:, 1]  # probabiliteti i "fake"

    # --- T2: prag mbi të cilin klasifikojmë "fake" me precision >= target ---
    t2 = find_threshold(y_val, probs, TARGET_PRECISION)

    # --- T1: prag nën të cilin klasifikojmë "real" me precision >= target ---
    y_real = 1 - y_val
    scores_real = 1 - probs
    t_real = find_threshold(y_real, scores_real, TARGET_PRECISION)
    t1 = (1.0 - t_real) if t_real is not None else None

    print("\n" + "=" * 50)
    print(f"PRAGJET (target precision = {TARGET_PRECISION:.0%})")
    print("=" * 50)
    print(f"T1 (nën këtë = 'REAL' me besim {TARGET_PRECISION:.0%}): "
          f"{t1:.4f}" if t1 is not None else "T1: S'U ARRIT")
    print(f"T2 (mbi këtë  = 'FAKE' me besim {TARGET_PRECISION:.0%}): "
          f"{t2:.4f}" if t2 is not None else "T2: S'U ARRIT")

    if t1 is not None and t2 is not None:
        if t1 >= t2:
            print("\n⚠ T1 >= T2 — zona 'e paqartë' është bosh ose negative. "
                  "Ul TARGET_PRECISION dhe rifute.")
        else:
            certain_real = np.sum(probs < t1)
            certain_fake = np.sum(probs >= t2)
            uncertain = np.sum((probs >= t1) & (probs < t2))
            total = len(probs)

            print(f"\nShpërndarja mbi {total} mostra holdout:")
            print(f"  REAL (besim i lartë):  {certain_real} ({100*certain_real/total:.1f}%)")
            print(f"  FAKE (besim i lartë):  {certain_fake} ({100*certain_fake/total:.1f}%)")
            print(f"  E PAQARTË (mes T1-T2): {uncertain} ({100*uncertain/total:.1f}%)")

            THRESHOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(THRESHOLDS_PATH, "w", encoding="utf-8") as f:
                json.dump({
                    "t1_real": t1,
                    "t2_fake": t2,
                    "target_precision": TARGET_PRECISION,
                    "feature_names": FEATURE_NAMES,
                }, f, indent=2)
            print(f"\nPragjet u ruajtën te: {THRESHOLDS_PATH.resolve()}")


if __name__ == "__main__":
    main()