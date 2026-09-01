"""
evidence_overlay.py — Kombinon ELA map + Grad-CAM map mbi të njëjtin
imazh, dhe llogarit një masë "agreement" (IoU-style) mes dy metodave.
Kjo është EVIDENCE CONSISTENCY, jo komponent i katërt i fusion-it.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.forensic_layer.ela import compute_ela_map
from src.forensic_layer.visualization import normalize_heatmap
from src.explainability.gradcam import compute_gradcam_map
import matplotlib.pyplot as plt

def save_individual_panels(image_path: str, output_dir: str) -> dict:
    """
    Ruan katër imazhe të veçanta (original, ELA, Grad-CAM, overlay) si
    PNG, plus agreement score — për t'i vendosur raporti në grid 2x2.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem

    original_img = Image.open(image_path).convert("RGB").resize((224, 224))
    ela_map = compute_ela_map(image_path, quality=90)
    ela_normalized = normalize_heatmap(ela_map)
    gradcam_map = compute_gradcam_map(image_path, target_class=1)

    if ela_normalized.shape != gradcam_map.shape:
        resized = Image.fromarray((ela_normalized * 255).astype(np.uint8)).resize(
            (gradcam_map.shape[1], gradcam_map.shape[0]))
        ela_normalized = np.array(resized).astype(np.float32) / 255.0

    agreement = compute_agreement_score(ela_map, gradcam_map)

    paths = {}

    original_path = out_dir / f"{stem}_original.png"
    original_img.save(original_path)
    paths["original"] = str(original_path)

    ela_path = out_dir / f"{stem}_ela.png"
    plt.imsave(ela_path, ela_normalized, cmap="jet")
    paths["ela"] = str(ela_path)

    gradcam_path = out_dir / f"{stem}_gradcam.png"
    plt.imsave(gradcam_path, gradcam_map, cmap="jet")
    paths["gradcam"] = str(gradcam_path)

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(original_img)
    ax.imshow(gradcam_map, cmap="jet", alpha=0.4)
    if ela_normalized.shape == gradcam_map.shape:
        ax.contour(ela_normalized, levels=[0.5], colors="white", linewidths=1.2)
    ax.axis("off")
    overlay_path = out_dir / f"{stem}_overlay.png"
    plt.savefig(overlay_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    paths["overlay"] = str(overlay_path)

    return {"paths": paths, "agreement_iou": agreement}
def compute_agreement_score(ela_map: np.ndarray, gradcam_map: np.ndarray, top_percent: float = 0.15) -> float:
    """
    Binarizon të dyja hartat duke marrë top_percent% e pikselave më aktivë
    të secilës (jo threshold fiks) — kështu të dyja hartat kanë madhësi
    krahasimi të barabartë, dhe IoU mat vetëm mbivendosjen HAPËSINORE, jo diferencën e densitetit natyral mes ELA dhe Grad-CAM.
    """
    if ela_map.shape != gradcam_map.shape:
        ela_img = Image.fromarray((normalize_heatmap(ela_map) * 255).astype(np.uint8))
        ela_img = ela_img.resize((gradcam_map.shape[1], gradcam_map.shape[0]))
        ela_normalized = np.array(ela_img).astype(np.float32) / 255.0
    else:
        ela_normalized = normalize_heatmap(ela_map)

    def top_k_binary(arr, pct):
        flat = arr.flatten()
        k = max(1, int(len(flat) * pct))
        threshold_val = np.partition(flat, -k)[-k]
        return arr >= threshold_val

    ela_binary = top_k_binary(ela_normalized, top_percent)
    gradcam_binary = top_k_binary(gradcam_map, top_percent)

    print(f"    [debug] ELA hot pixels: {ela_binary.sum()} ({100*ela_binary.sum()/ela_binary.size:.1f}%)")
    print(f"    [debug] Grad-CAM hot pixels: {gradcam_binary.sum()} ({100*gradcam_binary.sum()/gradcam_binary.size:.1f}%)")

    intersection = np.logical_and(ela_binary, gradcam_binary).sum()
    union = np.logical_or(ela_binary, gradcam_binary).sum()

    return float(intersection / union) if union > 0 else 0.0

def create_evidence_overlay_figure(image_path: str, output_path: str, quality: int = 90):
    """
    Krijon figurën 4-panele: original | ELA heatmap | Grad-CAM | overlay
    i kombinuar (ELA konturet mbi Grad-CAM), me agreement score si titull.
    """
    original_img = Image.open(image_path).convert("RGB").resize((224, 224))

    ela_map = compute_ela_map(image_path, quality)
    ela_normalized = normalize_heatmap(ela_map)

    gradcam_map = compute_gradcam_map(image_path, target_class=1)

    agreement = compute_agreement_score(ela_map, gradcam_map)

    # Overlay i kombinuar: Grad-CAM si sfond (jet), ELA si kontur (të bardhë)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(original_img)
    axes[0].set_title("Origjinal")
    axes[0].axis("off")

    axes[1].imshow(ela_normalized, cmap="jet")
    axes[1].set_title("ELA Heatmap")
    axes[1].axis("off")

    axes[2].imshow(gradcam_map, cmap="jet")
    axes[2].set_title("Grad-CAM (CNN)")
    axes[2].axis("off")

    axes[3].imshow(original_img)
    axes[3].imshow(gradcam_map, cmap="jet", alpha=0.4)
    if ela_normalized.shape == gradcam_map.shape:
        axes[3].contour(ela_normalized, levels=[0.5], colors="white", linewidths=1.5)
    axes[3].set_title(f"Overlay (Agreement IoU: {agreement:.3f})")
    axes[3].axis("off")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return agreement

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Përdorim: python evidence_overlay.py path\\te\\imazhi.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    out_path = Path("outputs/evidence_overlay") / (Path(image_path).stem + "_evidence.png")

    agreement = create_evidence_overlay_figure(image_path, str(out_path))
    print(f"Figura u ruajt te: {out_path}")
    print(f"Agreement Score (IoU): {agreement:.4f}")