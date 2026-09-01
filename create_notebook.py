"""
create_notebook.py — Gjeneron 05_results_visualization.ipynb automatikisht
nga përmbajtja e përcaktuar këtu, pa nevojë krijimi manual qelize-për-qelizë.

Ekzekutim: python create_notebook.py
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = [
    ("markdown", """# Kapitulli V — Vizualizimi i Rezultateve

Ky notebook lexon rezultate TASHMË TË LLOGARITURA nga `experiments/`, 
`evaluation/`, dhe `models/` — nuk rillogarit asgjë nga e para."""),

    ("code", """import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()

plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11"""),

    ("markdown", "## 1. Ablation Study — CNN-only vs CNN+FFT vs CNN+Forensic vs Hybrid Full"),

    ("code", """summary_path = PROJECT_ROOT / "experiments" / "results" / "summary.json"

with open(summary_path, "r", encoding="utf-8") as f:
    experiment_results = json.load(f)

df_experiments = pd.DataFrame(experiment_results)
df_experiments = df_experiments.set_index("name")
df_experiments"""),

    ("code", """fig, ax = plt.subplots(figsize=(11, 6))

metrics_to_plot = ["accuracy", "f1", "roc_auc"]
x = np.arange(len(df_experiments))
width = 0.25

for i, metric in enumerate(metrics_to_plot):
    ax.bar(x + i * width, df_experiments[metric], width, label=metric.upper())

ax.set_xticks(x + width)
ax.set_xticklabels(df_experiments.index, rotation=15)
ax.set_ylabel("Vlera")
ax.set_title("Krahasimi i konfigurimeve — Ablation Study")
ax.legend()
ax.set_ylim(0, 1.0)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(PROJECT_ROOT / "outputs" / "ablation_comparison.png", dpi=150)
plt.show()"""),

    ("markdown", "## 2. Robustness ndaj Degradimit — CNN vs Fusion"),

    ("code", """degradation_path = PROJECT_ROOT / "evaluation" / "degradation_comparison.json"

with open(degradation_path, "r", encoding="utf-8") as f:
    degradation_results = json.load(f)

rows = []
for condition, values in degradation_results.items():
    rows.append({
        "condition": condition,
        "cnn_accuracy": values["cnn"]["accuracy"],
        "fusion_accuracy": values["fusion"]["accuracy"],
        "cnn_auc": values["cnn"]["roc_auc"],
        "fusion_auc": values["fusion"]["roc_auc"],
    })

df_degradation = pd.DataFrame(rows).set_index("condition")
df_degradation"""),

    ("code", """fig, axes = plt.subplots(1, 2, figsize=(15, 6))

df_degradation[["cnn_accuracy", "fusion_accuracy"]].plot(
    kind="bar", ax=axes[0], color=["#4C72B0", "#DD8452"]
)
axes[0].set_title("Accuracy nën Degradim")
axes[0].set_ylabel("Accuracy")
axes[0].legend(["CNN", "Fusion"])
axes[0].tick_params(axis="x", rotation=30)
axes[0].grid(axis="y", alpha=0.3)

df_degradation[["cnn_auc", "fusion_auc"]].plot(
    kind="bar", ax=axes[1], color=["#4C72B0", "#DD8452"]
)
axes[1].set_title("ROC-AUC nën Degradim")
axes[1].set_ylabel("ROC-AUC")
axes[1].legend(["CNN", "Fusion"])
axes[1].tick_params(axis="x", rotation=30)
axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(PROJECT_ROOT / "outputs" / "degradation_comparison.png", dpi=150)
plt.show()"""),

    ("markdown", "## 3. Fusion — Rezultatet Finale mbi Test Set (T1/T2 Dual-Threshold)"),

    ("code", """thresholds_path = PROJECT_ROOT / "models" / "fusion_thresholds.json"

with open(thresholds_path, "r", encoding="utf-8") as f:
    thresholds = json.load(f)

print(f"T1 (real, besim i lartë): {thresholds['t1_real']:.4f}")
print(f"T2 (fake, besim i lartë): {thresholds['t2_fake']:.4f}")
print(f"Target precision: {thresholds['target_precision']:.0%}")"""),
]

nb["cells"] = [
    nbf.v4.new_markdown_cell(content) if kind == "markdown" else nbf.v4.new_code_cell(content)
    for kind, content in cells
]

output_path = "notebooks/05_results_visualization.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook u krijua te: {output_path}")