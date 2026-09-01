"""
create_notebook_04.py — Gjeneron 04_model_analysis.ipynb automatikisht
nga përmbajtja e përcaktuar këtu, pa nevojë krijimi manual qelize-për-qelizë.
Ekzekutim: python create_notebook_04.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = [
    ("markdown", """# Kapitulli V  Analiza e Modelit (CNN)

Ky notebook lexon rezultate TASHMË TË LLOGARITURA nga sesionet e
trajnimit/vlerësimit (`evaluation/`, `models/`) dhe nuk rillogarit asgjë
nga e para."""),

    ("code", """import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()

plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11"""),

    ("markdown", "## 1. Training Curves (CNN — dl_layer)\n\nTë dhëna nga logu i sesionit të trajnimit origjinal (epoch-level summary, baseline NUM_UNFROZEN_BLOCKS=3/LR=1e-4), jo rillogaritur."),

    ("code", """training_history = pd.DataFrame({
    "epoch": [1, 2, 3],
    "train_loss": [0.3032, 0.1712, 0.1239],
    "train_acc": [0.8638, 0.9287, 0.9488],
    "val_loss": [0.3525, 0.4063, 0.4633],
    "val_acc": [0.8690, 0.8561, 0.8370],
})

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(training_history["epoch"], training_history["train_loss"], "o-", label="Train Loss")
axes[0].plot(training_history["epoch"], training_history["val_loss"], "o-", label="Val Loss")
axes[0].axvline(x=1, color="green", linestyle="--", alpha=0.5, label="Best model (epoch 1)")
axes[0].set_xlabel("Epokë")
axes[0].set_ylabel("Loss")
axes[0].set_title("Loss Curves — CNN (EfficientNet-B0)")
axes[0].set_xticks(training_history["epoch"])
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(training_history["epoch"], training_history["train_acc"], "o-", label="Train Accuracy")
axes[1].plot(training_history["epoch"], training_history["val_acc"], "o-", label="Val Accuracy")
axes[1].axvline(x=1, color="green", linestyle="--", alpha=0.5, label="Best model (epoch 1)")
axes[1].set_xlabel("Epokë")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Accuracy Curves — CNN (EfficientNet-B0)")
axes[1].set_xticks(training_history["epoch"])
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PROJECT_ROOT / "outputs" / "training_curves.png", dpi=150)
plt.show()"""),

    ("markdown", "## 2. Frame-level vs Video-level Aggregation"),

    ("code", """comparison_data = pd.DataFrame({
    "accuracy": [0.8519, 0.8900, 0.9000],
    "f1": [0.8591, 0.8946, 0.9038],
    "roc_auc": [0.9235, 0.9423, None],
}, index=["Frame-level", "Video-level (mean-prob)", "Video-level (majority-vote)"])

comparison_data"""),

    ("code", """fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(comparison_data))
width = 0.35

ax.bar(x - width / 2, comparison_data["accuracy"], width, label="Accuracy")
ax.bar(x + width / 2, comparison_data["f1"], width, label="F1-score")

ax.set_xticks(x)
ax.set_xticklabels(comparison_data.index, rotation=10)
ax.set_ylabel("Vlera")
ax.set_title("Frame-level vs Video-level Aggregation")
ax.set_ylim(0.7, 1.0)
ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(PROJECT_ROOT / "outputs" / "frame_vs_video_level.png", dpi=150)
plt.show()"""),

    ("markdown", "## 3. Mean-Probability vs Majority-Vote (Video-level)"),

    ("code", """agg_comparison = pd.DataFrame({
    "accuracy": [0.8900, 0.9000],
    "precision": [0.8642, 0.8758],
    "recall": [0.9272, 0.9338],
    "f1": [0.8946, 0.9038],
}, index=["Mean-probability", "Majority-vote"])

fig, ax = plt.subplots(figsize=(9, 6))
agg_comparison.plot(kind="bar", ax=ax)
ax.set_title("Krahasimi i Metodave të Agregimit — Video-level")
ax.set_ylabel("Vlera")
ax.set_ylim(0.8, 1.0)
ax.tick_params(axis="x", rotation=0)
ax.legend(loc="lower right")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(PROJECT_ROOT / "outputs" / "aggregation_methods_comparison.png", dpi=150)
plt.show()"""),

    ("markdown", "## 4. ROC-AUC Summary (CNN Frame-level)\n\nShënim: kurba e plotë ROC kërkon `y_prob`/`y_true` të ruajtura nga `evaluate.py`. Nëse s'i kemi, mjaftohemi me vlerën përfundimtare të ROC-AUC."),

    ("code", """results_path = PROJECT_ROOT / "evaluation" / "test_results_dl_layer.json"

with open(results_path, "r", encoding="utf-8") as f:
    dl_results = json.load(f)

print(f"ROC-AUC (frame-level, test set): {dl_results['roc_auc']:.4f}")
print(f"Accuracy: {dl_results['accuracy']:.4f}")
print(f"F1-score: {dl_results['f1']:.4f}")"""),
("markdown", "## 5. Confusion Matrix — CNN Frame-level (Test Set)"),

("code", """cm_frame = np.array(dl_results["confusion_matrix"])

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm_frame, cmap="Blues")

labels = ["Real", "Fake"]
ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
ax.set_xlabel("Parashikuar"); ax.set_ylabel("Real (etiketa e vertetë)")
ax.set_title("Confusion Matrix — CNN Frame-level (test set)")

for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm_frame[i, j]), ha="center", va="center",
                color="white" if cm_frame[i, j] > cm_frame.max()/2 else "black", fontsize=14)

plt.colorbar(im, ax=ax, label="Numri i mostrave")
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "outputs" / "cnn_frame_confusion_matrix.png", dpi=150)
plt.show()"""),
]

nb["cells"] = [
    nbf.v4.new_markdown_cell(content) if kind == "markdown" else nbf.v4.new_code_cell(content)
    for kind, content in cells
]

output_path = "notebooks/04_model_analysis.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook u krijua te: {output_path}")