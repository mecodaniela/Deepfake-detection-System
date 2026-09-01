"""
video_level_split.py

Krijon splits (train/val/test) në nivel video, të balancuara 50/50
real/fake, duke ruajtur diversitetin e metodave manipuluese.
"""

import csv
import random
from pathlib import Path
from collections import Counter, defaultdict

# ============================================================
# KONFIGURIME
# ============================================================

MANIFEST_PATH = Path("data/manifest.csv")
SPLIT_DIR = Path("data/splits")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

# Metodat FF++ me pak video — i ruajmë TË GJITHA, pa i prekur me sampling
PRIORITY_METHODS = {"Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"}


# ============================================================
# LEXIMI I MANIFESTIT
# ============================================================

def load_manifest(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


# ============================================================
# BALANCIMI I KLASAVE
# ============================================================

def balance_classes(rows: list[dict], rng: random.Random) -> list[dict]:
    real_rows = [r for r in rows if r["label"] == "real"]
    fake_rows = [r for r in rows if r["label"] == "fake"]

    n_real = len(real_rows)
    fake_budget = n_real  # synim 50/50

    print(f"\nBuxheti total 'fake': {fake_budget} (= numri i 'real')")

    # 1. Ruaj TË GJITHA videot e metodave prioritare (FF++ të vogla)
    priority_fake = [r for r in fake_rows if r["method"] in PRIORITY_METHODS]
    remaining_pool = [r for r in fake_rows if r["method"] not in PRIORITY_METHODS]

    print(f"Metoda prioritare (ruhen të gjitha): {len(priority_fake)} video")
    for method in sorted(PRIORITY_METHODS):
        count = sum(1 for r in priority_fake if r["method"] == method)
        print(f"  {method}: {count}")

    remaining_budget = fake_budget - len(priority_fake)
    if remaining_budget < 0:
        # rast i pamundur me numrat aktualë, por mbrojtje sigurie
        print("[WARN] Metodat prioritare vetë kalojnë buxhetin; kufizohet.")
        priority_fake = rng.sample(priority_fake, fake_budget)
        remaining_budget = 0

    # 2. Shpërndaje pjesën e mbetur PROPORCIONALISHT mes metodave të tjera
    #    (dfdc, celeb-synthesis, etj.)
    method_counts = Counter(r["method"] for r in remaining_pool)
    total_pool = sum(method_counts.values())

    selected_remaining = []
    running_total = 0
    methods = sorted(method_counts.keys())

    for i, method in enumerate(methods):
        pool_for_method = [r for r in remaining_pool if r["method"] == method]

        if i == len(methods) - 1:
            # metoda e fundit merr çfarë ka mbetur, të mos humbasim video nga rrumbullakimi
            alloc = remaining_budget - running_total
        else:
            alloc = round(remaining_budget * method_counts[method] / total_pool)

        alloc = min(alloc, len(pool_for_method))
        alloc = max(alloc, 0)

        selected_remaining.extend(rng.sample(pool_for_method, alloc))
        running_total += alloc

    print(f"\nPlotësimi proporcional ({remaining_budget} video):")
    fill_counts = Counter(r["method"] for r in selected_remaining)
    for method, count in sorted(fill_counts.items()):
        print(f"  {method}: {count} (nga {method_counts[method]} në dispozicion)")

    selected_fake = priority_fake + selected_remaining
    selected_real = list(real_rows)  # përdoren të gjitha

    print(f"\nGjithsej fake i zgjedhur: {len(selected_fake)}")
    print(f"Gjithsej real i zgjedhur: {len(selected_real)}")

    return selected_real + selected_fake


# ============================================================
# SPLIT NË NIVEL VIDEO (stratifikuar sipas label + method)
# ============================================================

def split_by_class(rows: list[dict], rng: random.Random):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["label"], r["method"])].append(r)

    train, val, test = [], [], []

    for (label, method), group_rows in groups.items():
        rng.shuffle(group_rows)
        n = len(group_rows)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)

        train.extend(group_rows[:n_train])
        val.extend(group_rows[n_train:n_train + n_val])
        test.extend(group_rows[n_train + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


# ============================================================
# RUAJTJA E CSV-VE
# ============================================================

def save_csv(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["path", "dataset", "method", "label"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# RAPORT
# ============================================================

def print_split_report(name: str, rows: list[dict]):
    labels = Counter(r["label"] for r in rows)
    methods = Counter(r["method"] for r in rows)

    print(f"\n{name}")
    print("-" * 40)
    print(f"Total: {len(rows)}")
    print(f"Real:  {labels.get('real', 0)}")
    print(f"Fake:  {labels.get('fake', 0)}")

    print("\nSipas metodës:")
    for method, count in sorted(methods.items()):
        print(f"  {method}: {count}")


# ============================================================
# MAIN
# ============================================================

def main():
    rng = random.Random(RANDOM_SEED)

    print("Lexohet manifesti...")
    rows = load_manifest(MANIFEST_PATH)
    print(f"U gjetën gjithsej {len(rows)} video.")

    # 1. Balance real/fake (video-level, me prioritet për metodat e vogla)
    balanced_rows = balance_classes(rows, rng)

    # 2. Split stratifikuar sipas (label, method)
    train, val, test = split_by_class(balanced_rows, rng)

    # 3. Ruaj CSV-të
    train_path = SPLIT_DIR / "train.csv"
    val_path = SPLIT_DIR / "val.csv"
    test_path = SPLIT_DIR / "test.csv"

    save_csv(train, train_path)
    save_csv(val, val_path)
    save_csv(test, test_path)

    # 4. Raport
    print_split_report("TRAIN", train)
    print_split_report("VALIDATION", val)
    print_split_report("TEST", test)

    print("\n========================================")
    print("Split-et u krijuan me sukses.")
    print("========================================")

    print(f"\nTrain: {train_path.resolve()}")
    print(f"Val:   {val_path.resolve()}")
    print(f"Test:  {test_path.resolve()}")

    print(f"\nRandom seed: {RANDOM_SEED}")
    print(
        f"Raporti: "
        f"{TRAIN_RATIO:.0%} train / "
        f"{VAL_RATIO:.0%} val / "
        f"{TEST_RATIO:.0%} test"
    )


if __name__ == "__main__":
    main()