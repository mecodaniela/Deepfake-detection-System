"""
fft_transform.py — Global frequency-domain analysis via radial power spectrum.

Parimi: imazhe GAN-gjeneruara shpesh lënë "nënshkrim" periodik në
spektrin global të magnitudës (nga operacione upsampling/transposed
convolution brenda gjeneratorit) të dallueshëm si maja karakteristike
kur spektri mesatarizohet radialisht (azimuthal average), teknikë e
njohur në literaturën GAN-detection.
Ndryshe nga DCT (lokal, blloqe 8x8), FFT këtu operon mbi TË GJITHË
imazhin njëherësh dhe kap struktura periodike që s'shfaqen brenda një
blloku të vetëm 8x8.
"""
import numpy as np
from PIL import Image

def compute_radial_spectrum(image_path: str, n_bins: int = 50) -> np.ndarray:
    """
    Kthen spektrin e fuqisë mesatarizuar radialisht (1D, n_bins pika)
    energjia mesatare në çdo distancë nga qendra e spektrit FFT.
    """
    image = Image.open(image_path).convert("L")
    gray = np.array(image).astype(np.float32)

    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(gray))) ** 2
    h, w = spectrum.shape
    cy, cx = h // 2, w // 2

    y, x = np.indices((h, w))
    radius = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_radius = min(cy, cx)

    bin_edges = np.linspace(0, max_radius, n_bins + 1)
    radial_profile = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (radius >= bin_edges[i]) & (radius < bin_edges[i + 1])
        if mask.sum() > 0:
            radial_profile[i] = spectrum[mask].mean()

    return radial_profile

def compute_fft_raw(image_path: str) -> float:
    """Kthen peak_deviation të PAPËRPUNUAR (pa normalizim/klip) —
    përdoret për kalibrim mbi val split."""
    profile = compute_radial_spectrum(image_path)
    log_profile = np.log1p(profile)

    window = 5
    kernel = np.ones(window) / window
    smoothed = np.convolve(log_profile, kernel, mode="same")
    residual = log_profile - smoothed

    start = len(residual) // 2
    end = len(residual) - window // 2
    high_freq_residual = residual[start:end]

    if len(high_freq_residual) == 0:
        return 0.0

    return float(np.abs(high_freq_residual).max())

def compute_fft_score(image_path: str, mean: float = 0.2492, std: float = 0.0521) -> float:
    """Score 0-1 via z-score sigmoid — kalibruar mbi val split."""
    import math
    raw = compute_fft_raw(image_path)
    z = (raw - mean) / (std + 1e-8)
    return 1.0 / (1.0 + math.exp(-z))

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Përdorim: python fft.py path\\te\\imazhi.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    score = compute_fft_score(image_path)
    print(f"FFT score (probabiliteti manipulim/sintetik): {score:.4f}")
    print(f"Verdikt: {'MANIPULUAR/SINTETIK' if score > 0.5 else 'AUTENTIK'}")