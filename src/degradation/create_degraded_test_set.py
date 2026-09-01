"""
create_degraded_test_set.py — Krijon kopje FIZIKE të përhershme të një
mostre të rastësishme nga test set, nën kushte të ndryshme degradimi.
Dataset-i origjinal (data/raw, data/frames) mbetet plotësisht i paprekur.

data/frames/test/{real,fake} (mostër e rastësishme, seed=42)
    │
    ├──→ data/degraded/original/{real,fake}/       (baseline, KOPJE EKZAKTE)
    ├──→ data/degraded/jpeg_q90/{real,fake}/
    ├──→ data/degraded/jpeg_q70/{real,fake}/
    ├──→ data/degraded/jpeg_q50/{real,fake}/
    ├──→ data/degraded/resize_75/{real,fake}/
    ├──→ data/degraded/resize_25/{real,fake}/
    └──→ data/degraded/social_media/{real,fake}/
"""
import sys
import random
import shutil
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.degradation.jpeg_compression import apply_jpeg_compression
from src.degradation.resize import apply_resize_degradation
from src.degradation.social_media_simulation import apply_social_media_degradation

from PIL import Image

TEST_DIR = Path("data/frames/test")
OUTPUT_ROOT = Path("data/degraded")
SAMPLE_PER_CLASS = 400
RANDOM_SEED = 42

CONDITIONS = {
    "original": None,
    "jpeg_q90": ("jpeg", 90),
    "jpeg_q70": ("jpeg", 70),
    "jpeg_q50": ("jpeg", 50),
    "resize_75": ("resize", 75),
    "resize_25": ("resize", 25),
    "social_media": ("social", "medium"),
}

def apply_condition(image: Image.Image, condition):
    if condition is None:
        return image
    kind, param = condition
    if kind == "jpeg":
        return apply_jpeg_compression(image, param)
    if kind == "resize":
        return apply_resize_degradation(image, param)
    if kind == "social":
        return apply_social_media_degradation(image, param)
    raise ValueError(f"Kushtim i panjohur: {condition}")

def sample_random_files(label: str, n: int, rng: random.Random) -> list[Path]:
    folder = TEST_DIR / label
    all_files = sorted(folder.glob("*.jpg"))  # sort fillimisht për determinizëm të mostrimit
    if len(all_files) < n:
        print(f"[WARN] Vetëm {len(all_files)} skedarë në {folder}, më pak se {n} të kërkuar.")
        n = len(all_files)
    return rng.sample(all_files, n)

def main():
    rng = random.Random(RANDOM_SEED)

    print(f"Duke zgjedhur mostër rastësore ({SAMPLE_PER_CLASS} real + {SAMPLE_PER_CLASS} fake, seed={RANDOM_SEED})...")
    samples = {
        "real": sample_random_files("real", SAMPLE_PER_CLASS, rng),
        "fake": sample_random_files("fake", SAMPLE_PER_CLASS, rng),
    }

    for condition_name, condition in CONDITIONS.items():
        print(f"\nDuke krijuar kushtin: {condition_name}...")
        for label, files in samples.items():
            out_dir = OUTPUT_ROOT / condition_name / label
            out_dir.mkdir(parents=True, exist_ok=True)

            for i, src_path in enumerate(files):
                out_path = out_dir / src_path.name

                if condition is None:
                    # "original" = KOPJE EKZAKTE, pa asnjë rikodim/rikompresim
                    shutil.copy2(src_path, out_path)
                else:
                    image = Image.open(src_path).convert("RGB")
                    degraded = apply_condition(image, condition)
                    degraded.save(out_path, format="JPEG", quality=95)

                if (i + 1) % 100 == 0:
                    print(f"  {label}: {i + 1}/{len(files)}...")

    print(f"\nGati. Struktura u krijua te: {OUTPUT_ROOT.resolve()}")

if __name__ == "__main__":
    main()