"""
cfa_batch_test.py — Test statistikor: CFA score mbi mostër real/fake.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.forensic_layer.cfa import compute_cfa_score

TEST_DIR = Path("data/frames/test")
SAMPLE_SIZE = 50


def sample_scores(label: str, n: int = SAMPLE_SIZE) -> list[float]:
    folder = TEST_DIR / label
    files = sorted(folder.glob("*.jpg"))[:n]

    scores = []
    for i, f in enumerate(files):
        score = compute_cfa_score(str(f))
        scores.append(score)
        if (i + 1) % 10 == 0:
            print(f"  {label}: {i + 1}/{len(files)} përpunuar...")

    return scores


def main():
    print("Duke llogaritur CFA scores mbi mostër real...")
    real_scores = sample_scores("real")

    print("Duke llogaritur CFA scores mbi mostër fake...")
    fake_scores = sample_scores("fake")

    real_arr = np.array(real_scores)
    fake_arr = np.array(fake_scores)

    print("\n" + "=" * 50)
    print("SHPËRNDARJA E CFA SCORE — REAL vs FAKE")
    print("=" * 50)
    print(f"\nREAL (n={len(real_arr)}):")
    print(f"  Mesatare: {real_arr.mean():.4f}  Std: {real_arr.std():.4f}")
    print(f"  Min/Max:  {real_arr.min():.4f} / {real_arr.max():.4f}")

    print(f"\nFAKE (n={len(fake_arr)}):")
    print(f"  Mesatare: {fake_arr.mean():.4f}  Std: {fake_arr.std():.4f}")
    print(f"  Min/Max:  {fake_arr.min():.4f} / {fake_arr.max():.4f}")

    diff = fake_arr.mean() - real_arr.mean()
    print(f"\nDiferenca e mesatareve (fake - real): {diff:+.4f}")

    if abs(diff) < 0.02:
        print("\n⚠ Diferencë shumë e vogël — CFA s'duket të ketë sinjal dallues.")
    elif diff > 0:
        print("\n✓ Fake priret të ketë score më të lartë (drejtimi i pritur).")
    else:
        print("\n⚠ Fake priret të ketë score MË TË ULËT — kundërintuitive.")


if __name__ == "__main__":
    main()