"""
dataset_loader.py — Skanon data/raw/ dhe krijon një manifest CSV me
path, dataset, method, label per çdo video te gjetur, si bazë per
video_level_split.py dhe image_extraction.py.

Ekzekutim: python src\\data_pipeline\\dataset_loader.py
"""

import csv
import json
from pathlib import Path
from collections import Counter

# Path-et relative — skripti pritet të ekzekutohet nga rrënja e projektit
# (deepfake-detection-system/), jo nga brenda src/data_pipeline/
RAW_DIR = Path("data/raw")
OUTPUT_CSV = Path("data/manifest.csv")

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}


def scan_faceforensics(base_path: Path) -> list[dict]:
    entries = []
    ff_root = base_path / "faceforensics"
    if not ff_root.exists():
        print(f"[INFO] S'u gjet {ff_root}, kapërcehet FaceForensics++.")
        return entries

    # Original (real)
    orig_dir = ff_root / "original_sequences" / "youtube" / "c23" / "videos"
    if orig_dir.exists():
        for f in sorted(orig_dir.iterdir()):
            if f.suffix.lower() in VIDEO_EXTENSIONS:
                entries.append({
                    "path": str(f.resolve()),
                    "dataset": "faceforensics",
                    "method": "original",
                    "label": "real",
                })
    else:
        print(f"[WARN] S'u gjet {orig_dir} — original videos mungojnë.")

    # Manipulated (fake) — çdo metodë e shkarkuar deri tani
    manip_root = ff_root / "manipulated_sequences"
    if manip_root.exists():
        for method_dir in sorted(manip_root.iterdir()):
            if not method_dir.is_dir():
                continue
            video_dir = method_dir / "c23" / "videos"
            if not video_dir.exists():
                continue
            for f in sorted(video_dir.iterdir()):
                if f.suffix.lower() in VIDEO_EXTENSIONS:
                    entries.append({
                        "path": str(f.resolve()),
                        "dataset": "faceforensics",
                        "method": method_dir.name,
                        "label": "fake",
                    })
    else:
        print(f"[INFO] {manip_root} mungon ende — vetëm 'original' u skanua.")

    return entries


def scan_dfdc(base_path: Path) -> list[dict]:
    entries = []
    dfdc_root = base_path / "dfdc"
    if not dfdc_root.exists():
        print(f"[INFO] S'u gjet {dfdc_root}, kapërcehet DFDC.")
        return entries

    # Kërko rekursivisht të gjithë .mp4, pavarësisht strukturës së nënfolderave
    # (dfdctrain/, dfdc_train_part_XX/, ose direkt te dfdc/)
    video_files = list(dfdc_root.rglob("*.mp4"))

    if not video_files:
        print(f"[WARN] S'u gjet asnjë .mp4 nën {dfdc_root}.")
        return entries

    skipped = 0
    for f in video_files:
        stem = f.stem.lower()
        if stem.startswith("fake-") or stem.startswith("fake_"):
            label = "fake"
        elif stem.startswith("real-") or stem.startswith("real_"):
            label = "real"
        else:
            skipped += 1
            continue

        entries.append({
            "path": str(f.resolve()),
            "dataset": "dfdc",
            "method": "dfdc",  # DFDC s'ndan sipas metode specifike si FaceForensics
            "label": label,
        })

    if skipped:
        print(f"[WARN] {skipped} skedarë .mp4 te DFDC s'u njohën nga emri "
              f"(as 'fake-' as 'real-' prefiks), u kapërcyen.")

    return entries


def scan_celebdf(base_path: Path) -> list[dict]:
    entries = []
    celeb_root = base_path / "celebdf"
    if not celeb_root.exists():
        print(f"[INFO] S'u gjet {celeb_root}, kapërcehet Celeb-DF.")
        return entries

    subfolder_labels = {
        "Celeb-real": "real",
        "YouTube-real": "real",
        "Celeb-synthesis": "fake",
    }

    for subfolder, label in subfolder_labels.items():
        folder = celeb_root / subfolder
        if not folder.exists():
            print(f"[WARN] S'u gjet {folder}, kapërcehet.")
            continue
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() in VIDEO_EXTENSIONS:
                entries.append({
                    "path": str(f.resolve()),
                    "dataset": "celebdf",
                    "method": subfolder,
                    "label": label,
                })

    return entries


def main():
    all_entries: list[dict] = []
    all_entries.extend(scan_faceforensics(RAW_DIR))
    all_entries.extend(scan_dfdc(RAW_DIR))
    all_entries.extend(scan_celebdf(RAW_DIR))

    if not all_entries:
        print("\n[GABIM] S'u gjet asnjë video gjithsej. Kontrollo që "
              "skripti ekzekutohet nga rrënja e projektit (deepfake-detection-system/) "
              "dhe që data/raw/ ka strukturën e pritur.")
        return

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "dataset", "method", "label"])
        writer.writeheader()
        writer.writerows(all_entries)

    real_count = sum(1 for e in all_entries if e["label"] == "real")
    fake_count = sum(1 for e in all_entries if e["label"] == "fake")

    print(f"\nManifest u krijua: {OUTPUT_CSV.resolve()}")
    print(f"Gjithsej: {len(all_entries)} video  ({real_count} real, {fake_count} fake)\n")

    by_dataset = Counter(e["dataset"] for e in all_entries)
    print("Sipas dataset-it:")
    for ds, count in sorted(by_dataset.items()):
        print(f"  {ds}: {count}")

    by_method = Counter(e["method"] for e in all_entries)
    print("\nSipas metodës:")
    for m, count in sorted(by_method.items()):
        print(f"  {m}: {count}")


if __name__ == "__main__":
    main()