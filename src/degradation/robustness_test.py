"""
robustness_test.py — Krahason performancën e CNN-së vetëm vs Fusion-it
të plotë nën kushte të ndryshme degradimi (JPEG, resize, social-media-style).

Ekzekutim: python src\n degradation/robustness_test.py
"""
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.dl_layer.predict import predict_cnn_score
from src.forensic_layer.cfa import compute_cfa_score
from src.forensic_layer.dct import compute_dct_score
from src.frequency_layer.fft_transform import compute_fft_score
from src.degradation.jpeg_compression import apply_jpeg_compression
from src.degradation.resize import apply_resize_degradation
from src.degradation.social_media_simulation import apply_social_media_degradation

TEST_DIR = Path("data/frames/test")
SAMPLE_PER_CLASS = 50  # 50+50=100 total; rrit nëse ke kohë, por 10 kushte x 100 = 1000 vlerësime
FUSION_MODEL_PATH = Path("models/fusion_logistic.pkl")

DEGRADATION_CONDITIONS = {
    "baseline": None,
    "jpeg_90": ("jpeg", 90),
    "jpeg_70": ("jpeg", 70),
    "jpeg_50": ("jpeg", 50),
    "resize_75": ("resize", 75),
    "resize_50": ("resize", 50),
    "resize_25": ("resize", 25),
    "social_light": ("social", "light"),
    "social_medium": ("social", "medium"),
    "social_heavy": ("social", "heavy"),
}

def apply_degradation(image: Image.Image, condition):
    if condition is None:
        return image
    kind, param = condition
    if kind == "jpeg":
        return apply_jpeg_compression(image, param)
    if kind == "resize":
        return apply_resize_degradation(image, param)
    if kind == "social":
        return apply_social_media_degradation(image, param)
    raise ValueError(f"Kushtim i panjohur: {condition}")

def load_sample_paths():
    samples = []
    for label_name, label_val in [("real", 0), ("fake", 1)]:
        folder = TEST_DIR / label_name
        files = sorted(folder.glob("*.jpg"))[:SAMPLE_PER_CLASS]
        for f in files:
            samples.append((str(f), label_val))
    return samples

def compute_fusion_prediction(model, feature_names, image_path: str) -> int:
    feature_map = {
        "cnn": predict_cnn_score(image_path),
        "cfa": compute_cfa_score(image_path),
        "dct": compute_dct_score(image_path),
        "fft": compute_fft_score(image_path),
    }
    features = np.array([[feature_map[name] for name in feature_names]])
    return int(model.predict(features)[0])

def evaluate_condition(condition, samples, fusion_model, feature_names, tmp_dir: Path):
    cnn_correct = 0
    fusion_correct = 0
    total = 0

    for image_path, true_label in samples:
        image = Image.open(image_path).convert("RGB")
        degraded = apply_degradation(image, condition)

        tmp_path = tmp_dir / "degraded_tmp.jpg"
        degraded.save(tmp_path, format="JPEG", quality=95)

        cnn_score = predict_cnn_score(str(tmp_path))
        cnn_correct += int((1 if cnn_score >= 0.5 else 0) == true_label)

        fusion_pred = compute_fusion_prediction(fusion_model, feature_names, str(tmp_path))
        fusion_correct += int(fusion_pred == true_label)

        total += 1

    return cnn_correct / total, fusion_correct / total

def main():
    samples = load_sample_paths()
    print(f"Mostër: {len(samples)} imazhe ({SAMPLE_PER_CLASS} real + {SAMPLE_PER_CLASS} fake)")

    saved = joblib.load(FUSION_MODEL_PATH)
    fusion_model = saved["model"]
    feature_names = saved["feature_names"]

    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for name, condition in DEGRADATION_CONDITIONS.items():
            print(f"\nDuke testuar: {name}...")
            cnn_acc, fusion_acc = evaluate_condition(condition, samples, fusion_model, feature_names, tmp_dir)
            results[name] = (cnn_acc, fusion_acc)
            print(f"  CNN accuracy:    {cnn_acc:.4f}")
            print(f"  Fusion accuracy: {fusion_acc:.4f}")

    print("\n" + "=" * 60)
    print("PËRMBLEDHJE — ROBUSTNESS TEST")
    print("=" * 60)
    print(f"{'Kushti':<15} {'CNN acc':>10} {'Fusion acc':>12} {'Fusion-CNN':>12}")
    for name, (cnn_acc, fusion_acc) in results.items():
        print(f"{name:<15} {cnn_acc:>10.4f} {fusion_acc:>12.4f} {fusion_acc - cnn_acc:>+12.4f}")

if __name__ == "__main__":
    main()