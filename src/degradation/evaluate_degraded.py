"""
evaluate_degraded.py — Vlerëson CNN-në vetëm DHE Fusion-in e plotë mbi
çdo kusht të degraduar (nga create_degraded_test_set.py), prodhon tabelë krahasuese Accuracy/F1/ROC-AUC.
"""
import sys
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.dl_layer.predict import predict_cnn_score
from src.forensic_layer.cfa import compute_cfa_score
from src.forensic_layer.dct import compute_dct_score
from src.frequency_layer.fft_transform import compute_fft_score

DEGRADED_ROOT = Path("data/degraded")
FUSION_MODEL_PATH = Path("models/fusion_logistic.pkl")
OUTPUT_PATH = Path("evaluation/degradation_comparison.json")

CONDITIONS = ["original", "jpeg_q90", "jpeg_q70", "jpeg_q50", "resize_75", "resize_25", "social_media"]

def compute_fusion_probs(model, feature_names, image_path: str) -> float:
    feature_map = {
        "cnn": predict_cnn_score(image_path),
        "cfa": compute_cfa_score(image_path),
        "dct": compute_dct_score(image_path),
        "fft": compute_fft_score(image_path),
    }
    features = np.array([[feature_map[name] for name in feature_names]])
    return float(model.predict_proba(features)[0, 1])

def evaluate_condition(condition_name: str, fusion_model, feature_names) -> dict:
    y_true, cnn_probs, fusion_probs = [], [], []

    for label_name, label_val in [("real", 0), ("fake", 1)]:
        folder = DEGRADED_ROOT / condition_name / label_name
        files = sorted(folder.glob("*.jpg"))

        for f in files:
            cnn_score = predict_cnn_score(str(f))
            fusion_score = compute_fusion_probs(fusion_model, feature_names, str(f))

            y_true.append(label_val)
            cnn_probs.append(cnn_score)
            fusion_probs.append(fusion_score)

    y_true = np.array(y_true)
    cnn_probs = np.array(cnn_probs)
    fusion_probs = np.array(fusion_probs)

    cnn_preds = (cnn_probs >= 0.5).astype(int)
    fusion_preds = (fusion_probs >= 0.5).astype(int)

    return {
        "n": len(y_true),
        "cnn": {
            "accuracy": accuracy_score(y_true, cnn_preds),
            "f1": f1_score(y_true, cnn_preds),
            "roc_auc": roc_auc_score(y_true, cnn_probs),
        },
        "fusion": {
            "accuracy": accuracy_score(y_true, fusion_preds),
            "f1": f1_score(y_true, fusion_preds),
            "roc_auc": roc_auc_score(y_true, fusion_probs),
        },
    }

def main():
    print("Duke ngarkuar modelin e fusion-it...")
    saved = joblib.load(FUSION_MODEL_PATH)
    fusion_model = saved["model"]
    feature_names = saved["feature_names"]

    results = {}
    for condition in CONDITIONS:
        print(f"\nDuke vlerësuar kushtin: {condition}...")
        results[condition] = evaluate_condition(condition, fusion_model, feature_names)
        c = results[condition]
        print(f"  n={c['n']}")
        print(f"  CNN    — acc: {c['cnn']['accuracy']:.4f}, f1: {c['cnn']['f1']:.4f}, auc: {c['cnn']['roc_auc']:.4f}")
        print(f"  Fusion — acc: {c['fusion']['accuracy']:.4f}, f1: {c['fusion']['f1']:.4f}, auc: {c['fusion']['roc_auc']:.4f}")

    print("\n" + "=" * 90)
    print("TABELA PËRMBLEDHËSE — CNN vs FUSION NËN DEGRADIM")
    print("=" * 90)
    header = f"{'Test':<15} {'CNN Acc':>9} {'CNN F1':>9} {'CNN AUC':>9} | {'Fus Acc':>9} {'Fus F1':>9} {'Fus AUC':>9}"
    print(header)
    print("-" * len(header))
    for condition in CONDITIONS:
        c = results[condition]
        print(f"{condition:<15} "
              f"{c['cnn']['accuracy']:>9.4f} {c['cnn']['f1']:>9.4f} {c['cnn']['roc_auc']:>9.4f} | "
              f"{c['fusion']['accuracy']:>9.4f} {c['fusion']['f1']:>9.4f} {c['fusion']['roc_auc']:>9.4f}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nRezultatet u ruajtën te: {OUTPUT_PATH.resolve()}")

if __name__ == "__main__":
    main()