"""
dct.py — Discrete Cosine Transform (DCT) coefficient statistics analysis.

Parimi: JPEG punon në blloqe 8x8 me DCT; imazhe natyrale kanë shpërndarje
energjie karakteristike (e përqendruar te frekuencat e ulëta, rënie e
qëndrueshme). Manipulim/gjenerim artificial priret të prodhojë energji
të parregullt/të tepërt te frekuencat e larta (tekstura "shumë të pastra"
ose artefakte bllok-bazuara).

Output: dct_score (0-1, probabilitet manipulim) — heuristik, kërkon
kalibrim empirik si ELA/CFA.
"""

import numpy as np
from PIL import Image
from scipy.fftpack import dct

BLOCK_SIZE = 8


def _block_dct2(block: np.ndarray) -> np.ndarray:
    """DCT 2D mbi një bllok 8x8."""
    return dct(dct(block.T, norm="ortho").T, norm="ortho")


def _high_frequency_energy_ratio(gray: np.ndarray) -> float:
    """
    Ndan imazhin në blloqe 8x8, llogarit DCT për secilin, dhe kthen
    raportin mesatar të energjisë te frekuencat e LARTA (këndi
    poshtë-djathtas i bllokut DCT) kundrejt energjisë totale.

    Imazhe natyrale: energji e ulët në frekuenca të larta (pak detaje
    "të mprehta" në nivel bllok).
    Manipulim/GAN: mund të ketë energji anomalisht të lartë (tekstura
    artificiale, buzë të forta të gjeneruara).
    """
    h, w = gray.shape
    h_blocks = h // BLOCK_SIZE
    w_blocks = w // BLOCK_SIZE

    if h_blocks == 0 or w_blocks == 0:
        return 0.0

    ratios = []

    # Maskë për frekuenca "të larta": treshja poshtë-djathtas e bllokut 8x8
    high_freq_mask = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=bool)
    for i in range(BLOCK_SIZE):
        for j in range(BLOCK_SIZE):
            if i + j >= 8:  # diagonale që ndan frekuenca ulëta nga të larta
                high_freq_mask[i, j] = True

    for by in range(h_blocks):
        for bx in range(w_blocks):
            block = gray[
                by * BLOCK_SIZE:(by + 1) * BLOCK_SIZE,
                bx * BLOCK_SIZE:(bx + 1) * BLOCK_SIZE,
            ].astype(np.float32)

            coeffs = _block_dct2(block)
            energy = coeffs ** 2

            high_energy = energy[high_freq_mask].sum()
            total_energy = energy.sum() + 1e-8

            ratios.append(high_energy / total_energy)

    return float(np.mean(ratios))


def compute_dct_score(image_path: str) -> float:
    """
    Kthen score 0-1 (probabilitet manipulim). Score i LARTË = energji
    anomalisht e lartë te frekuencat e larta = gjasa më të larta për
    manipulim/gjenerim artificial.
    """
    image = Image.open(image_path).convert("L")
    gray = np.array(image)

    ratio = _high_frequency_energy_ratio(gray)

    # Log-transform: të dhënat shtrihen mbi disa rende madhësie
    # (nga 0.0001 deri 0.4+), log e stabilizon shkallën
    log_ratio = np.log1p(ratio * 1000)
    normalized = min(log_ratio / 5.0, 1.0)
    score = 1.0 - normalized  # kthim polariteti: energji e ULËT = score i LARTË (fake)

    return float(max(0.0, min(score, 1.0)))


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Përdorim: python dct.py path\\te\\imazhi.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    score = compute_dct_score(image_path)
    print(f"DCT score (probabiliteti manipulim): {score:.4f}")
    print(f"Verdikt: {'MANIPULUAR' if score > 0.5 else 'AUTENTIK'}")