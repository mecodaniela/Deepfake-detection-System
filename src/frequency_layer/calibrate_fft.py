"""calibrate_fft.py — Llogarit mean/std të peak_deviation mbi val split."""

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.frequency_layer.fft_transform import compute_fft_raw

CALIB_DIR = Path("data/frames/val")
SAMPLE_SIZE = 100


def main():
    scores = []
    for label in ["real", "fake"]:
        folder = CALIB_DIR / label
        files = sorted(folder.glob("*.jpg"))[:SAMPLE_SIZE // 2]
        for f in files:
            scores.append(compute_fft_raw(str(f)))

    arr = np.array(scores)
    print(f"FFT_MEAN, FFT_STD = {arr.mean():.4f}, {arr.std():.4f}")


if __name__ == "__main__":
    main()