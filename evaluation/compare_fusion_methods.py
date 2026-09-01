"""
compare_fusion_methods.py — Formaton rezultatet e experiments/run_experiment.py
(tashmë të llogaritura) në tabelë të lexueshme + grafik krahasues.
"""
import sys
import json
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))

SUMMARY_PATH = Path("experiments/results/summary.json")
OUTPUT_DIR = Path("outputs/evaluation")

def load_summary() -> list[dict]:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"S'u gjet {SUMMARY_PATH} ... ekzekuto experiments/run_experiment.py fillimisht."
        )
    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def print_comparison_table(results: list[dict]) -> None:
    print(f"\n{'Metoda':<18}{'Accuracy':<12}{'Precision':<12}{'Recall':<12}{'F1':<12}{'ROC-AUC':<10}")
    print("-" * 76)
    for r in results:
        print(f"{r['name']:<18}{r['accuracy']:<12.4f}{r['precision']:<12.4f}"
              f"{r['recall']:<12.4f}{r['f1']:<12.4f}{r['roc_auc']:<10.4f}")

def save_comparison_chart(results: list[dict], output_path: str) -> None:
    names = [r["name"] for r in results]
    accuracy = [r["accuracy"] for r in results]
    roc_auc = [r["roc_auc"] for r in results]

    x = range(len(names))
    fig, ax = plt.subplots(figsize=(9, 5))

    width = 0.35
    ax.bar([i - width / 2 for i in x], accuracy, width, label="Accuracy")
    ax.bar([i + width / 2 for i in x], roc_auc, width, label="ROC-AUC")

    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Vlera")
    ax.set_title("Krahasimi i Metodave: Accuracy vs ROC-AUC")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

def main():
    results = load_summary()
    print_comparison_table(results)

    chart_path = OUTPUT_DIR / "fusion_methods_comparison.png"
    save_comparison_chart(results, str(chart_path))
    print(f"\nGrafiku u ruajt te: {chart_path.resolve()}")

if __name__ == "__main__":
    main()