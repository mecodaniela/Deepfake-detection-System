"""
evaluate_video_level.py — Agregon parashikimet frame-level (nga CNN, dl_layer)
në një vendim të vetëm PËR VIDEO, mbi test split.

Arsyeja: një sistem forenzik/gjyqësor vendos për video, jo për frame të
veçantë — kjo është metrika "reale" për t'u raportuar te Kreu V.

Ekzekutim: python src\\dl_layer\\evaluate_video_level.py
"""

import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import cast

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.dl_layer.model import build_model
from src.data_pipeline.dataset_loader import DeepfakeFrameDataset, get_transforms

MODEL_PATH = Path("models/dl_layer_best.pt")
DEVICE = torch.device("cpu")

# Heq "_frameXXXX" (dhe çdo gjë pas tij, si .jpg) nga emri i skedarit,
# duke lënë vetëm ID-në e videos origjinale
FRAME_SUFFIX_PATTERN = re.compile(r"_frame\d+$")


def extract_video_id(img_path: Path) -> str:
    stem = img_path.stem  # emri pa .jpg
    video_id = FRAME_SUFFIX_PATTERN.sub("", stem)
    return video_id


@torch.no_grad()
def evaluate_video_level():
    print("Duke ngarkuar modelin...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model = build_model(num_unfrozen_blocks=checkpoint["num_unfrozen_blocks"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    print("Duke ngarkuar test dataset...")
    dataset = DeepfakeFrameDataset("test", transform=get_transforms("test"))

    # Grupon: video_id -> listë probabilitetesh "fake" nga frame-t e saj
    video_probs = defaultdict(list)
    video_true_label = {}

    print(f"Duke përpunuar {len(dataset)} frames...")
    for i in range(len(dataset)):
        img_path, label_idx = dataset.samples[i]
        video_id = extract_video_id(img_path)

        image, _ = dataset[i]  # tensor tashmë i transformuar
        tensor = cast(torch.Tensor, image).unsqueeze(0).to(DEVICE)
        output = model(tensor)
        prob_fake = torch.softmax(output, dim=1)[0, 1].item()

        video_probs[video_id].append(prob_fake)
        video_true_label[video_id] = label_idx  # e njëjta për çdo frame të videos

        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(dataset)} frames përpunuar...")

    print(f"\nTotal video unike: {len(video_probs)}")

    # --- Agregim: mesatare e probabiliteteve, dhe votim shumicë ---
    true_labels = []
    pred_mean = []
    pred_majority = []
    mean_probs_for_auc = []

    for video_id, probs in video_probs.items():
        true_labels.append(video_true_label[video_id])

        avg_prob = float(np.mean(probs))
        mean_probs_for_auc.append(avg_prob)
        pred_mean.append(1 if avg_prob > 0.5 else 0)

        frame_preds = [1 if p > 0.5 else 0 for p in probs]
        majority = 1 if sum(frame_preds) > len(frame_preds) / 2 else 0
        pred_majority.append(majority)

    def print_metrics(name, y_true, y_pred, y_probs=None):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)

        print(f"\n--- {name} ---")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-score:  {f1:.4f}")
        if y_probs is not None:
            auc = roc_auc_score(y_true, y_probs)
            print(f"ROC-AUC:   {auc:.4f}")
        print(f"Confusion Matrix:\n{cm}")

    print("\n" + "=" * 50)
    print("REZULTATE VIDEO-LEVEL (agreguar nga frame-level)")
    print("=" * 50)
    print_metrics("Agregim: Mesatare probabilitetesh", true_labels, pred_mean, mean_probs_for_auc)
    print_metrics("Agregim: Votim shumicë", true_labels, pred_majority)


if __name__ == "__main__":
    evaluate_video_level()