"""
image_extraction.py

Nxjerr frame nga videot e train/val/test, zbulon fytyrën me MediaPipe,
e pret (crop), e ridimensionon, dhe e ruan si .jpg.

Mbështet RESUME: nëse procesi ndërpritet, rinisja e skriptit kapërcen
automatikisht videot tashmë të përpunuara plotësisht.
"""

import csv
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from pathlib import Path
from tqdm import tqdm

# ============================================================
# KONFIGURIME
# ============================================================

SPLIT_DIR = Path("data/splits")
OUTPUT_DIR = Path("data/frames")

FACE_MODEL_PATH = "models/blaze_face_short_range.tflite"
FRAMES_PER_VIDEO = 30
OUTPUT_SIZE = (224, 224)
FACE_PADDING = 0.25  # % shtesë përreth bounding box-it të fytyrës

MIN_DETECTION_CONFIDENCE = 0.5
MODEL_SELECTION = 1  # 0 = short-range (~2m), 1 = full-range (~5m, më i përshtatshëm për video të përziera)

SPLITS = ["train", "val", "test"]


# ============================================================
# NDIHMËSE — LEXIMI I CSV-VE
# ============================================================

def load_split(split_name: str) -> list[dict]:
    path = SPLIT_DIR / f"{split_name}.csv"
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ============================================================
# NDIHMËSE — PROGRESS TRACKING (PËR RESUME)
# ============================================================

def load_progress(split_name: str) -> set:
    progress_path = OUTPUT_DIR / split_name / "_progress.txt"
    if not progress_path.exists():
        return set()
    with open(progress_path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def mark_done(split_name: str, base_name: str):
    progress_path = OUTPUT_DIR / split_name / "_progress.txt"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_path, "a", encoding="utf-8") as f:
        f.write(base_name + "\n")
        f.flush()


# ============================================================
# NDIHMËSE — ZGJEDHJA E FRAME-VE UNIFORME
# ============================================================

def get_frame_indices(total_frames: int, n_frames: int) -> list[int]:
    if total_frames <= 0:
        return []
    n = min(n_frames, total_frames)
    return sorted(set(np.linspace(0, total_frames - 1, n, dtype=int)))


# ============================================================
# NDIHMËSE — ZBULIMI I FYTYRËS ME MEDIAPIPE
# ============================================================

def detect_face(detector, frame) -> tuple | None:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = detector.detect(mp_image)

    if not result.detections:
        return None

    best = max(result.detections, key=lambda d: d.categories[0].score)
    bbox = best.bounding_box

    h, w = frame.shape[:2]
    x1 = max(0, bbox.origin_x)
    y1 = max(0, bbox.origin_y)
    x2 = min(w, bbox.origin_x + bbox.width)
    y2 = min(h, bbox.origin_y + bbox.height)

    if x2 <= x1 or y2 <= y1:
        return None

    return (x1, y1, x2, y2)


# ============================================================
# NDIHMËSE — CROP FYTYRE ME PADDING
# ============================================================

def crop_face(frame, facial_area, padding_ratio: float):
    x1, y1, x2, y2 = facial_area
    w, h = x2 - x1, y2 - y1

    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)

    H, W = frame.shape[:2]
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(W, x2 + pad_x)
    y2 = min(H, y2 + pad_y)

    return frame[y1:y2, x1:x2]


# ============================================================
# PËRPUNIMI I NJË VIDEOJE
# ============================================================

def process_video(detector, video_path: str, out_dir: Path, base_name: str, stats: dict):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        stats["videos_failed"] += 1
        print(f"[WARN] S'u hap dot video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = get_frame_indices(total_frames, FRAMES_PER_VIDEO)

    frame_idx_set = set(indices)
    current_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if current_idx in frame_idx_set:
            facial_area = detect_face(detector, frame)

            if facial_area is None:
                stats["frames_no_face"] += 1
                current_idx += 1
                continue

            face_crop = crop_face(frame, facial_area, FACE_PADDING)
            if face_crop.size == 0:
                stats["frames_no_face"] += 1
                current_idx += 1
                continue

            face_crop = cv2.resize(face_crop, OUTPUT_SIZE)

            out_path = out_dir / f"{base_name}_frame{current_idx:04d}.jpg"
            cv2.imwrite(str(out_path), face_crop)

            saved_count += 1
            stats["frames_saved"] += 1

        current_idx += 1

    cap.release()

    if saved_count == 0:
        stats["videos_zero_frames"] += 1

    stats["videos_processed"] += 1


# ============================================================
# PËRPUNIMI I NJË SPLIT-I TË PLOTË (me resume)
# ============================================================

def process_split(detector, split_name: str):
    rows = load_split(split_name)
    completed = load_progress(split_name)

    stats = {
        "videos_processed": 0,
        "videos_skipped": 0,
        "videos_failed": 0,
        "videos_zero_frames": 0,
        "frames_saved": 0,
        "frames_no_face": 0,
    }

    print(f"\n{'=' * 50}")
    print(f"SPLIT: {split_name.upper()}  ({len(rows)} video)")
    if completed:
        print(f"Rifillim: {len(completed)} video tashmë të përpunuara, kapërcehen.")
    print(f"{'=' * 50}")

    for row in tqdm(rows, desc=split_name):
        video_path = row["path"]
        label = row["label"]
        dataset = row["dataset"]
        method = row["method"]

        video_stem = Path(video_path).stem
        base_name = f"{dataset}_{method}_{video_stem}"

        if base_name in completed:
            stats["videos_skipped"] += 1
            continue

        out_dir = OUTPUT_DIR / split_name / label
        out_dir.mkdir(parents=True, exist_ok=True)

        tqdm.write(f"Duke përpunuar: {base_name}")

        process_video(detector, video_path, out_dir, base_name, stats)
        mark_done(split_name, base_name)

    print(f"\nRaport — {split_name}:")
    print(f"  Video të kapërcyera (tashmë të bëra): {stats['videos_skipped']}")
    print(f"  Video të përpunuara (këtë herë): {stats['videos_processed']}")
    print(f"  Video të dështuara (s'u hapën): {stats['videos_failed']}")
    print(f"  Video pa asnjë frame të ruajtur: {stats['videos_zero_frames']}")
    print(f"  Frame të ruajtura (këtë herë): {stats['frames_saved']}")
    print(f"  Frame të kapërcyera (s'u gjet fytyrë): {stats['frames_no_face']}")

    return stats


# ============================================================
# MAIN
# ============================================================

def main():
    base_options = mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
    options = mp_vision.FaceDetectorOptions(
        base_options=base_options,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    )

    with mp_vision.FaceDetector.create_from_options(options) as detector:
        all_stats = {}
        for split_name in SPLITS:
            all_stats[split_name] = process_split(detector, split_name)

    print(f"\n{'=' * 50}")
    print("PËRMBLEDHJE E PLOTË")
    print(f"{'=' * 50}")

    total_frames = sum(s["frames_saved"] for s in all_stats.values())
    total_videos = sum(s["videos_processed"] for s in all_stats.values())

    for split_name, s in all_stats.items():
        print(f"{split_name}: {s['frames_saved']} frame nga {s['videos_processed']} video (këtë herë)")

    print(f"\nTotal frame të ruajtura (këtë herë): {total_frames}")
    print(f"Total video të përpunuara (këtë herë): {total_videos}")
    print(f"\nOutput: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()