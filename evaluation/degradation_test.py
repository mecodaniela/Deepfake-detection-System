"""
degradation_test.py:Teston sa mirë performon sistemi mbi imazhe të
degraduar (JPEG cilësi e ulët, resize, kombinim), krahasuar me
imazhet origjinale. Degradimi bëhet on-the-fly, jo kërkon dataset
paraprakisht të degraduar.
"""
import io
import sys
from pathlib import Path

import numpy as np
import joblib
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))
from evaluation.metrics import compute_metrics, print_metrics
from src.dl_layer.predict import predict_cnn_score, _load_model as _load_cnn_model
from src.forensic_layer.ela import compute_ela_score
from src.forensic_layer.cfa import compute_cfa_score
from src.forensic_layer.dct import compute_dct_score
from src.frequency_layer.fft_transform import compute_fft_score

TEST_DIR = Path("data/frames/test")
SAMPLE_PER_CLASS = 150
FUSION_MODEL_PATH = Path("models/fusion_logistic.pkl")
TEMP_DIR = Path("outputs/_degradation_temp")

DEGRADATION_LEVELS = {
    "original": None,
    "jpeg_low": {"jpeg_quality": 30},
    "resize_50pct": {"resize_factor": 0.5},
    "jpeg_low+resize": {"jpeg_quality": 30, "resize_factor": 0.5},
}

FEATURE_FNS = {
    "cnn": predict_cnn_score, "ela": compute_ela_score,
    "cfa": compute_cfa_score, "dct": compute_dct_score, "fft": compute_fft_score,
}
FEATURE_NAMES = list(FEATURE_FNS.keys())

def apply_degradation(image_path: str, config: dict | None, out_path: Path) -> str:
    """Aplikon degradim (resize + rikompresim JPEG) dhe ruan si skedar
    të ri dhe kthen path-in e imazhit të degraduar."""
    if config is None:
        return image_path

    image = Image.open(image_path).convert("RGB")

    if "resize_factor" in config:
        factor = config["resize_factor"]
        w, h = image.size
        small = image.resize((max(1, int(w * factor)), max(1, int(h * factor))))
        image = small.resize((w, h))  # rikthim te madhësia origjinale (simulon rezolucion të humbur)

    quality = config.get("jpeg_quality", 95)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, "JPEG", quality=quality)

    return str(out_path)

def compute_features(image_path: str) -> list[float]:
    return [FEATURE_FNS[name](image_path) for name in FEATURE_NAMES]

def main():
    fusion_data = joblib.load(FUSION_MODEL_PATH)
    fusion_model = fusion_data["model"]
    fusion_feature_names = fusion_data["feature_names"]  # p.sh. ['cnn', 'cfa', 'dct', 'fft'] — jo ELA

    files = []
    for label_name, label_val in [("real", 0), ("fake", 1)]:
        folder = TEST_DIR / label_name
        for f in sorted(folder.glob("*.jpg"))[:SAMPLE_PER_CLASS]:
            files.append((str(f), label_val))

    print(f"Duke testuar mbi {len(files)} imazhe, {len(DEGRADATION_LEVELS)} nivele degradimi...")
    results = {}
    for level_name, config in DEGRADATION_LEVELS.items():
        print(f"\nNiveli: {level_name}")

        cnn_preds, cnn_probs = [], []
        fusion_preds, fusion_probs = [], []
        y_true = []

        for i, (img_path, label) in enumerate(files):
            degraded_path = apply_degradation(
                img_path, config, TEMP_DIR / level_name / Path(img_path).name
            )

            all_scores = dict(zip(FEATURE_NAMES, compute_features(degraded_path)))
            cnn_score = all_scores["cnn"]

            cnn_probs.append(cnn_score)
            cnn_preds.append(1 if cnn_score > 0.5 else 0)

            fusion_input = [all_scores[name] for name in fusion_feature_names]  # vetëm features që pret modeli
            fusion_prob = fusion_model.predict_proba([fusion_input])[0, 1]
            fusion_probs.append(fusion_prob)
            fusion_preds.append(1 if fusion_prob > 0.5 else 0)

            y_true.append(label)

            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(files)}...")

        cnn_metrics = compute_metrics(y_true, cnn_preds, cnn_probs)
        fusion_metrics = compute_metrics(y_true, fusion_preds, fusion_probs)

        results[level_name] = {"cnn": cnn_metrics, "fusion": fusion_metrics}
        print_metrics(f"{level_name} — CNN only", cnn_metrics)
        print_metrics(f"{level_name} — Fusion", fusion_metrics)

    print("\n" + "=" * 70)
    print("PËRMBLEDHJE DEGRADATION TEST (Accuracy: CNN vs Fusion)")
    print("=" * 70)
    print(f"{'Niveli':<20}{'CNN Accuracy':<16}{'Fusion Accuracy':<16}")
    for level_name, r in results.items():
        print(f"{level_name:<20}{r['cnn']['accuracy']:<16.4f}{r['fusion']['accuracy']:<16.4f}")

if __name__ == "__main__":
    main()