"""
cross_dataset.py — Mat generalizimin: trajno fusion mbi FaceForensics++
+ DFDC, testo mbi Celeb-DF (dataset plotësisht i patrajnuar mbi të).
Kërkon data/manifest.csv (nga dataset_scanner.py) për të filtruar sipas
dataset-it origjinal.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.append(str(Path(__file__).resolve().parents[1]))
from evaluation.metrics import compute_metrics, print_metrics
from src.dl_layer.predict import predict_cnn_score
from src.forensic_layer.ela import compute_ela_score
from src.forensic_layer.cfa import compute_cfa_score
from src.forensic_layer.dct import compute_dct_score
from src.frequency_layer.fft_transform import compute_fft_score

MANIFEST_PATH = Path("data/manifest.csv")
FRAMES_DIR = Path("data/frames")  # frames janë tashmë të ekstraktuara/split
SAMPLE_PER_CLASS = 300

FEATURE_FNS = {
    "cnn": predict_cnn_score, "ela": compute_ela_score,
    "cfa": compute_cfa_score, "dct": compute_dct_score, "fft": compute_fft_score,
}
FEATURE_NAMES = list(FEATURE_FNS.keys())

def compute_features(image_path: str) -> list[float]:
    return [FEATURE_FNS[name](image_path) for name in FEATURE_NAMES]

def collect_samples_by_dataset(split: str, dataset_filter: list[str], n_per_class: int):
    """
    Mbledh frame-t që i përkasin datasetit(ve) të specifikuar, duke
    përdorur emrin e skedarit (fillon me '<dataset>_...', siç e
    konfirmuam te struktura data/frames/).
    """
    X, y = [], []
    for label_name, label_val in [("real", 0), ("fake", 1)]:
        folder = FRAMES_DIR / split / label_name
        matching = [
            f for f in sorted(folder.glob("*.jpg"))
            if any(f.name.startswith(ds) for ds in dataset_filter)
        ][:n_per_class]

        for f in matching:
            X.append(compute_features(str(f)))
            y.append(label_val)

    return np.array(X), np.array(y)

def main():
    print("Duke mbledhur mostër TRAJNIMI nga FaceForensics++ + DFDC (val split)...")
    X_train, y_train = collect_samples_by_dataset(
        "val", ["faceforensics", "dfdc"], SAMPLE_PER_CLASS
    )
    print(f"  {len(X_train)} mostra gjithsej.")

    print("Duke mbledhur mostër TEST nga Celeb-DF (test split, i patrajnuar)...")
    X_test, y_test = collect_samples_by_dataset(
        "test", ["celebdf"], SAMPLE_PER_CLASS
    )
    print(f"  {len(X_test)} mostra gjithsej.")

    if len(X_train) == 0 or len(X_test) == 0:
        print("[GABIM] S'u gjetën mostra të mjaftueshme ...kontrollo emrat e "
              "dataset-eve (faceforensics/dfdc/celebdf) te data/frames/.")
        return

    print("\nDuke trajnuar fusion mbi FaceForensics+++DFDC...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_prob)
    print_metrics("Cross-Dataset: Train(FF+++DFDC) → Test(Celeb-DF)", metrics)

if __name__ == "__main__":
    main()