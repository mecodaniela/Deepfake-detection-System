"""
evidence_overlay_batch_test.py — Test statistikor: llogarit IoU
agreement (ELA vs Grad-CAM) mbi një mostër real+fake, krahasuar me
bazën e pritur nga rastësia (chance baseline) për secilin imazh.
Ekzekutim: python src/explainability/evidence_overlay_batch_test.py
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.forensic_layer.ela import compute_ela_map
from src.forensic_layer.visualization import normalize_heatmap
from src.explainability.gradcam import compute_gradcam_map

TEST_DIR = Path("data/frames/test")
SAMPLE_SIZE = 50
TOP_PERCENT = 0.15

def top_k_binary(arr: np.ndarray, pct: float) -> np.ndarray:
    flat = arr.flatten()
    k = max(1, int(len(flat) * pct))
    threshold_val = np.partition(flat, -k)[-k]
    return arr >= threshold_val

def compute_iou_and_baseline(image_path: str) -> tuple[float, float]:
    """Kthen (iou_real, iou_chance_baseline) për një imazh."""
    ela_map = compute_ela_map(image_path, quality=90)
    ela_normalized = normalize_heatmap(ela_map)

    gradcam_map = compute_gradcam_map(image_path, target_class=1)

    if ela_normalized.shape != gradcam_map.shape:
        ela_img = Image.fromarray((ela_normalized * 255).astype(np.uint8))
        ela_img = ela_img.resize((gradcam_map.shape[1], gradcam_map.shape[0]))
        ela_normalized = np.array(ela_img).astype(np.float32) / 255.0

    ela_binary = top_k_binary(ela_normalized, TOP_PERCENT)
    gradcam_binary = top_k_binary(gradcam_map, TOP_PERCENT)

    p1 = ela_binary.mean()
    p2 = gradcam_binary.mean()

    intersection = np.logical_and(ela_binary, gradcam_binary).sum()
    union = np.logical_or(ela_binary, gradcam_binary).sum()
    iou_real = float(intersection / union) if union > 0 else 0.0

    # Baza e pritur nga rastësia (nëse ELA/Grad-CAM të pavarur)
    inter_chance = p1 * p2
    union_chance = p1 + p2 - inter_chance
    iou_chance = float(inter_chance / union_chance) if union_chance > 0 else 0.0

    return iou_real, iou_chance

def sample_scores(label: str, n: int = SAMPLE_SIZE) -> tuple[list[float], list[float]]:
    folder = TEST_DIR / label
    files = sorted(folder.glob("*.jpg"))[:n]

    ious, baselines = [], []
    for i, f in enumerate(files):
        try:
            iou, baseline = compute_iou_and_baseline(str(f))
            ious.append(iou)
            baselines.append(baseline)
        except Exception as e:
            print(f"  [WARN] Dështoi për {f.name}: {e}")
            continue

        if (i + 1) % 10 == 0:
            print(f"  {label}: {i + 1}/{len(files)} përpunuar...")

    return ious, baselines

def main():
    print("Duke llogaritur IoU (ELA vs Grad-CAM) mbi mostër real...")
    real_iou, real_baseline = sample_scores("real")

    print("Duke llogaritur IoU (ELA vs Grad-CAM) mbi mostër fake...")
    fake_iou, fake_baseline = sample_scores("fake")

    all_iou = np.array(real_iou + fake_iou)
    all_baseline = np.array(real_baseline + fake_baseline)

    print("\n" + "=" * 50)
    print("EVIDENCE OVERLAY — IoU vs BAZA E RASTËSISË")
    print("=" * 50)

    print(f"\nREAL (n={len(real_iou)}):")
    print(f"  IoU real:     mesatare={np.mean(real_iou):.4f}, std={np.std(real_iou):.4f}")
    print(f"  IoU rastësie: mesatare={np.mean(real_baseline):.4f}")

    print(f"\nFAKE (n={len(fake_iou)}):")
    print(f"  IoU real:     mesatare={np.mean(fake_iou):.4f}, std={np.std(fake_iou):.4f}")
    print(f"  IoU rastësie: mesatare={np.mean(fake_baseline):.4f}")

    print(f"\nGjithsej (n={len(all_iou)}):")
    print(f"  IoU real:     mesatare={all_iou.mean():.4f}, std={all_iou.std():.4f}")
    print(f"  IoU rastësie: mesatare={all_baseline.mean():.4f}")

    diff = all_iou.mean() - all_baseline.mean()
    print(f"\nDiferenca (IoU real - baza rastësie): {diff:+.4f}")

    if diff > 0.02:
        print("\n✓ ELA dhe Grad-CAM tregojnë marrëveshje hapësinore mbi rastësinë "
              "(sinjal i lehtë real, jo i fortë).")
    elif diff > -0.02:
        print("\n⚠ Diferencë e vogël — ELA/Grad-CAM praktikisht të pavarur "
              "hapësinisht (agreement ≈ rastësi).")
    else:
        print("\n⚠ IoU real ËSHTË NËN bazën e rastësisë — kundërintuitive, "
              "kërkon rishikim.")

if __name__ == "__main__":
    main()