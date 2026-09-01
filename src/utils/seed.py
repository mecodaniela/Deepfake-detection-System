"""
seed.py — Vendos random seed në të gjitha librarite relevante
(Python, NumPy, PyTorch) për riprodueshmëri të eksperimenteve.
"""
import os
import random

import numpy as np
import torch

DEFAULT_SEED = 42

def set_seed(seed: int = DEFAULT_SEED) -> None:
    """
    Thirre në krye të çdo skripti trajnimi/eksperimenti
    (train.py, train_fusion.py, etj.) PARA çdo operacioni rastësor.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # s'dëmton edhe pa GPU, thjesht no-op

    os.environ["PYTHONHASHSEED"] = str(seed)

    # Sigurohet determinizëm më i fortë në operacione CUDA (nëse do
    # përdoret ndonjëherë GPU në të ardhmen) — pa efekt mbi CPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    set_seed(42)
    print(f"Seed u vendos në 42. Test: {random.random():.6f}, {np.random.rand():.6f}")