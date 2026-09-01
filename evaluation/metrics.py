"""
metrics.py — Funksione të ripërdorshme për llogaritjen e metrikave
standarde (accuracy, precision, recall, F1, ROC-AUC, confusion matrix).
Përdoret nga compare_fusion_methods.py, cross_dataset.py, degradation_test.py.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    """
    Kthen dict me metrika standarde. y_prob (probabilitete, jo vendime
    binare) është opsionale — nëse mungon, ROC-AUC nuk llogaritet.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics["roc_auc"] = None  # p.sh. nëse y_true ka vetëm 1 klasë

    return metrics

def print_metrics(name: str, metrics: dict) -> None:
    print(f"\n--- {name} ---")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1']:.4f}")
    if metrics.get("roc_auc") is not None:
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"Confusion Matrix: {metrics['confusion_matrix']}")