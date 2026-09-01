"""
train_fusion.py — Trajnon regresion logjistik mbi features (CNN, CFA, DCT,
FFT) → probabilitet fusion final. Trajnohet mbi VAL split (jo train, jo test).
Feature-t dhe modeli jetojnë tani te logistic_fusion.py — ky file merret
vetëm me ndërtimin e dataset-it dhe trajnimin.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.fusion.logistic_fusion import compute_feature_vector, FEATURE_NAMES, MODEL_PATH

TRAIN_DIR = Path("data/frames/val")
SAMPLE_PER_CLASS = 500  # rregullo nëse do trajnim më i qëndrueshëm (më shumë kohë)

def build_dataset(dir_path: Path = TRAIN_DIR, sample_per_class: int = SAMPLE_PER_CLASS):
    X, y = [], []

    for label_name, label_val in [("real", 0), ("fake", 1)]:
        folder = dir_path / label_name
        files = sorted(folder.glob("*.jpg"))[:sample_per_class]

        print(f"Duke përpunuar {len(files)} imazhe nga '{label_name}' ({dir_path})...")
        for i, f in enumerate(files):
            features = compute_feature_vector(str(f))
            X.append(features)
            y.append(label_val)

            if (i + 1) % 50 == 0:
                print(f"  {label_name}: {i + 1}/{len(files)}...")

    return np.array(X), np.array(y)

def main():
    print("Duke ndërtuar dataset-in e features...")
    X, y = build_dataset()

    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    print("\nMatrica e korrelacionit mes features:")
    print(df.corr().round(3))

    print("\nKorrelacioni i çdo feature me label-in (y):")
    for i, name in enumerate(FEATURE_NAMES):
        corr = np.corrcoef(X[:, i], y)[0, 1]
        print(f"  {name}: {corr:+.4f}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrajnim mbi {len(X_train)} mostra, validim i brendshëm mbi {len(X_val)}...")

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)

    print(f"\nAccuracy (validim i brendshëm 80/20 mbi mostrën val): {acc:.4f}")
    print("\nRaport i detajuar:")
    print(classification_report(y_val, y_pred, target_names=["real", "fake"]))

    print("\nKoeficientët e mësuar (peshat e secilit feature):")
    for name, coef in zip(FEATURE_NAMES, model.coef_[0]):
        print(f"  {name}: {coef:+.4f}")
    print(f"  intercept: {model.intercept_[0]:+.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_names": FEATURE_NAMES}, MODEL_PATH)
    print(f"\nModeli u ruajt te: {MODEL_PATH.resolve()}")


if __name__ == "__main__":
    main()