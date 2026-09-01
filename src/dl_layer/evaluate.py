"""
evaluate.py — Vlerëson modelin e ruajtur (dl_layer_best.pt) mbi test split.

KUJDES METODOLOGJIK: ekzekuto këtë vetëm NJËHERË, mbi modelin
PËRFUNDIMTAR të zgjedhur (pas krahasimit mes konfigurimeve të
ndryshme mbi val split). Test split duhet të mbetet "i paprekur"
deri në vendimin final — mos e rifut këtë skript për të krahasuar
variante të ndryshme modeli.

"""

import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.dl_layer.model import build_model
from src.data_pipeline.dataset_loader import get_dataloader

MODEL_PATH = Path("models/dl_layer_best.pt")
DEVICE = torch.device("cpu")


@torch.no_grad()
def evaluate_on_test():
    print("Duke ngarkuar modelin...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    model = build_model(num_unfrozen_blocks=checkpoint["num_unfrozen_blocks"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    print(f"Modeli u ngarkua (epoka {checkpoint['epoch']}, "
          f"val_accuracy gjatë trainimit: {checkpoint['val_accuracy']:.4f})")

    test_loader = get_dataloader("test", shuffle=False)

    all_labels = []
    all_preds = []
    all_probs = []  # probabiliteti i klasës "fake" (index 1)

    print("Duke vlerësuar mbi test split...")
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)

        preds = outputs.argmax(dim=1)

        all_labels.extend(labels.numpy())
        all_preds.extend(preds.numpy())
        all_probs.extend(probs[:, 1].numpy())

    # --- Metrikat ---
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)

    print("\n" + "=" * 50)
    print("REZULTATET FINALE — CNN (dl_layer) mbi TEST split")
    print("=" * 50)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print("\nConfusion Matrix:")
    print(f"                Predicted Real   Predicted Fake")
    print(f"Actual Real     {cm[0][0]:<16} {cm[0][1]}")
    print(f"Actual Fake     {cm[1][0]:<16} {cm[1][1]}")
    print("\nRaport i detajuar:")
    print(classification_report(all_labels, all_preds, target_names=["real", "fake"]))

    # --- Ruajtja e rezultateve për riprodueshmëri (Kreu V i tezës) ---
    results = {
        "accuracy": acc, "precision": precision, "recall": recall,
        "f1": f1, "roc_auc": auc, "confusion_matrix": cm.tolist(),
        "model_epoch": checkpoint["epoch"],
        "num_unfrozen_blocks": checkpoint["num_unfrozen_blocks"],
        "evaluated_at": datetime.now().isoformat(),
    }

    output_path = Path("evaluation") / "test_results_dl_layer.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nRezultatet u ruajtën te: {output_path.resolve()}")

    return results


if __name__ == "__main__":
    evaluate_on_test()