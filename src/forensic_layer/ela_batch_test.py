"""
ela_batch_test.py — Test i shpejtë statistikor: llogarit ELA score mbi
një mostër real + fake nga test split, printon shpërndarjen krahasuese.

Ekzekutim: python src\\forensic_layer\\ela_batch_test.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.forensic_layer.ela import compute_ela_score

TEST_DIR = Path("data/frames/test")
SAMPLE_SIZE = 50  # sa imazhe nga secila klasë


def sample_scores(label: str, n: int = SAMPLE_SIZE) -> list[float]:
    folder = TEST_DIR / label
    files = sorted(folder.glob("*.jpg"))[:n]  # n e para (deterministe, jo random)

    scores = []
    for i, f in enumerate(files):
        score = compute_ela_score(str(f))
        scores.append(score)
        if (i + 1) % 10 == 0:
            print(f"  {label}: {i + 1}/{len(files)} përpunuar...")

    return scores


def main():
    print("Duke llogaritur ELA scores mbi mostër real...")
    real_scores = sample_scores("real")

    print("Duke llogaritur ELA scores mbi mostër fake...")
    fake_scores = sample_scores("fake")

    real_arr = np.array(real_scores)
    fake_arr = np.array(fake_scores)

    print("\n" + "=" * 50)
    print("SHPËRNDARJA E ELA SCORE — REAL vs FAKE")
    print("=" * 50)
    print(f"\nREAL (n={len(real_arr)}):")
    print(f"  Mesatare: {real_arr.mean():.4f}")
    print(f"  Std:      {real_arr.std():.4f}")
    print(f"  Min/Max:  {real_arr.min():.4f} / {real_arr.max():.4f}")

    print(f"\nFAKE (n={len(fake_arr)}):")
    print(f"  Mesatare: {fake_arr.mean():.4f}")
    print(f"  Std:      {fake_arr.std():.4f}")
    print(f"  Min/Max:  {fake_arr.min():.4f} / {fake_arr.max():.4f}")

    diff = fake_arr.mean() - real_arr.mean()
    print(f"\nDiferenca e mesatareve (fake - real): {diff:+.4f}")

    if abs(diff) < 0.02:
        print("\n⚠ Diferencë shumë e vogël — ELA duket se s'ka sinjal të "
              "dobishëm dallues mbi këtë dataset me formulën aktuale.")
    elif diff > 0:
        print("\n✓ Fake priret të ketë score më të lartë (drejtimi i pritur).")
    else:
        print("\n⚠ Fake priret të ketë score MË TË ULËT se real — "
              "kjo është kundërintuitive, kërkon rishikim.")


if __name__ == "__main__":
    main()