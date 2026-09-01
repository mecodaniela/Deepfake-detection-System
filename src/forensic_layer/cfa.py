"""
cfa.py — Color Filter Array (CFA) / demosaicing artifact analysis.

Parimi: kamerat kapin 1 kanal/piksel (Bayer pattern), pjesa tjetër
interpolohet — kjo lë periodicitet statistikor 2x2 të matshëm në FFT
të mbetjes (residual) së parashikimit linear. Imazhe GAN-gjeneruara
priren të mos e kenë këtë "nënshkrim" fizik të kapjes reale.

Output: cfa_score (0-1, probabilitet manipulim/gjenerim artificial) —
KUJDES: si ELA, kjo formulë është heuristike, kërkon kalibrim empirik.
"""

import numpy as np
from PIL import Image


def _bilinear_residual(gray: np.ndarray) -> np.ndarray:
    """
    Model i thjeshtë parashikimi linear (mesatare e 4 fqinjëve),
    të ngjashëm me hapin e demosaicing. Mbetja (residual) e këtij
    parashikimi mban periodicitetin CFA nëse imazhi është kapur nga
    kamerë reale (jo pastërtisht sintetik).
    """
    predicted = np.zeros_like(gray)
    predicted[1:-1, 1:-1] = (
        gray[:-2, 1:-1].astype(np.float32) +   # sipër
        gray[2:, 1:-1].astype(np.float32) +    # poshtë
        gray[1:-1, :-2].astype(np.float32) +   # majtas
        gray[1:-1, 2:].astype(np.float32)      # djathtas
    ) / 4.0

    residual = gray.astype(np.float32) - predicted
    return residual[1:-1, 1:-1]  # heq buzët pa vlerë të vlefshme


def _periodicity_strength(residual: np.ndarray) -> float:
    """
    Llogarit fuqinë e periodicitetit 2x2 në spektrin e frekuencës
    (FFT) të residualit — CFA lë energji të përqendruar te frekuencat
    Nyquist (buzë e spektrit), jo e shpërndarë uniformisht.
    """
    if residual.size == 0 or residual.std() < 1e-6:
        return 0.0

    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(residual)))
    h, w = spectrum.shape
    cy, cx = h // 2, w // 2

    # Energjia te buzët e spektrit (frekuenca të larta, ku CFA
    # periodicity 2x2 shfaqet) kundrejt energjisë totale
    edge_band = 4  # gjerësia e brezit pranë buzëve Nyquist
    edge_energy = (
        spectrum[:edge_band, :].sum() + spectrum[-edge_band:, :].sum() +
        spectrum[:, :edge_band].sum() + spectrum[:, -edge_band:].sum()
    )
    total_energy = spectrum.sum() + 1e-8

    return float(edge_energy / total_energy)


def compute_cfa_score(image_path: str) -> float:
    """
    Kthen score 0-1 (probabilitet manipulim/gjenerim artificial).
    Score i LARTË = periodicitet CFA i DOBËT/mungon = gjasa më të
    larta për origjinë sintetike (GAN) ose manipulim të fortë lokal.
    """
    image = Image.open(image_path).convert("L")  # grayscale
    gray = np.array(image)

    residual = _bilinear_residual(gray)
    periodicity = _periodicity_strength(residual)

    # Periodicitet i FORTË (kamerë reale) -> score i ULËT (autentik)
    # Periodicitet i DOBËT (sintetik/manipuluar) -> score i LARTË
    # Normalizim: periodicity zakonisht bie 0.05-0.30 mbi imazhe reale
    score = min(periodicity / 0.15, 1.0)

    return float(max(0.0, min(score, 1.0)))


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Përdorim: python cfa.py path\\te\\imazhi.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    score = compute_cfa_score(image_path)
    print(f"CFA score (probabiliteti manipulim/sintetik): {score:.4f}")
    print(f"Verdikt: {'MANIPULUAR/SINTETIK' if score > 0.5 else 'AUTENTIK'}")