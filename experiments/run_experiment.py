"""
run_experiment.py — Trajnon fusion (logistic regression) me nënbashkësi
të ndryshme features (sipas config YAML), duke rifolur SCORE-t ekzistuese
(cnn/ela/cfa/dct/fft) — jo ritrajnim CNN. Vlerëson mbi test (njëherë për config, jo iterativisht).
Ekzekutim: python src\\experiments\\run_experiment.py
"""
import sys
import json
from pathlib import Path

import yaml
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.utils.app_logging import get_logger
from src.utils.seed import set_seed
from src.dl_layer.predict import predict_cnn_score
from src.forensic_layer.ela import compute_ela_score
from src.forensic_layer.cfa import compute_cfa_score
from src.forensic_layer.dct import compute_dct_score
from src.frequency_layer.fft_transform import compute_fft_score

log = get_logger("run_experiment")

CONFIGS_DIR = Path("experiments/configs")
RESULTS_DIR = Path("experiments/results")

TRAIN_SAMPLE_DIR = Path("data/frames/val")   # mbahet konsistent me train_fusion.py
TEST_DIR = Path("data/frames/test")
SAMPLE_PER_CLASS_TRAIN = 1000
SAMPLE_PER_CLASS_TEST = 500

ALL_FEATURE_FNS = {
    "cnn": predict_cnn_score,
    "ela": compute_ela_score,
    "cfa": compute_cfa_score,
    "dct": compute_dct_score,
    "fft": compute_fft_score,
}

def compute_all_scores(image_path: str) -> dict:
    return {name: fn(image_path) for name, fn in ALL_FEATURE_FNS.items()}

def build_full_dataset(base_dir: Path, n_per_class: int) -> tuple[list[dict], np.ndarray]:
    """Llogarit TË GJITHA score-t (jo vetëm ato të config-ut aktual) — ripërdoren për çdo config."""
    all_scores, labels = [], []
    for label_name, label_val in [("real", 0), ("fake", 1)]:
        files = sorted((base_dir / label_name).glob("*.jpg"))[:n_per_class]
        for f in files:
            all_scores.append(compute_all_scores(str(f)))
            labels.append(label_val)
    return all_scores, np.array(labels)

def run_config(config_path: Path, train_scores, y_train, test_scores, y_test) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    name = config["name"]
    features = config["features"]
    log.info(f"Eksperimenti: {name} — features: {features}")

    X_train = np.array([[s[feat] for feat in features] for s in train_scores])
    X_test = np.array([[s[feat] for feat in features] for s in test_scores])

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    out_dir = RESULTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"config": config, "metrics": metrics}, f, indent=2)

    joblib.dump({"model": model, "feature_names": features}, out_dir / "model.pkl")

    log.info(f"  {name}: accuracy={metrics['accuracy']:.4f}, roc_auc={metrics['roc_auc']:.4f}")
    return {"name": name, **metrics}

def main():
    set_seed(42)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Duke llogaritur TË GJITHA score-t (train/val + test) — një herë, ripërdoren për çdo config...")
    train_scores, y_train = build_full_dataset(TRAIN_SAMPLE_DIR, SAMPLE_PER_CLASS_TRAIN)
    test_scores, y_test = build_full_dataset(TEST_DIR, SAMPLE_PER_CLASS_TEST)

    configs = sorted(CONFIGS_DIR.glob("*.yaml"))
    log.info(f"U gjetën {len(configs)} config: {[c.stem for c in configs]}")

    all_results = []
    for config_path in configs:
        result = run_config(config_path, train_scores, y_train, test_scores, y_test)
        all_results.append(result)

    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    log.info("\n" + "=" * 60)
    log.info("PËRMBLEDHJE E EKSPERIMENTEVE")
    log.info("=" * 60)
    for r in all_results:
        log.info(f"{r['name']:20s} accuracy={r['accuracy']:.4f}  roc_auc={r['roc_auc']:.4f}  f1={r['f1']:.4f}")

    log.info(f"\nPërmbledhja u ruajt te: {summary_path.resolve()}")

if __name__ == "__main__":
    main()