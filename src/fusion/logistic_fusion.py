# logistic_fusion.py
"""
logistic_fusion.py — Moduli qendror i fusion-it: llogarit vektorin e
features për një imazh dhe ngarkon/aplikon modelin e trajnuar për të
kthyer P(fake). Përdoret nga train_fusion.py, thresholds.py, dhe
evaluate_fusion.py — një burim i vetëm i së vërtetës për feature-t.
"""
import sys
from pathlib import Path

import joblib

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.dl_layer.predict import predict_cnn_score
from src.forensic_layer.cfa import compute_cfa_score
from src.forensic_layer.dct import compute_dct_score
from src.frequency_layer.fft_transform import compute_fft_score

FEATURE_NAMES = ["cnn", "cfa", "dct", "fft"]
MODEL_PATH = Path("models/fusion_logistic.pkl")


def compute_feature_vector(image_path: str) -> list[float]:
    return [
        predict_cnn_score(image_path),
        compute_cfa_score(image_path),
        compute_dct_score(image_path),
        compute_fft_score(image_path),
    ]


def load_fusion_model(model_path: Path = MODEL_PATH):
    saved = joblib.load(model_path)
    return saved["model"], saved["feature_names"]


def predict_fusion_probability(image_path: str, model=None) -> float:
    """P(fake) për një imazh të vetëm — pika e hyrjes për predict.py më vonë."""
    if model is None:
        model, _ = load_fusion_model()
    features = [compute_feature_vector(image_path)]
    return float(model.predict_proba(features)[0, 1])