"""
visualization.py — Krijon vizualizime të ELA-s: heatmap i normalizuar,
overlay mbi imazhin origjinal, dhe figurë krahasuese e ruajtur si file.

Themeli për figurën 4-panele (original, ELA heatmap, Grad-CAM, overlay)
të planifikuar për Kreun V — Grad-CAM shtohet më vonë nga explainability/.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.forensic_layer.cfa import _bilinear_residual, _periodicity_strength
from src.forensic_layer.dct import _block_dct2, BLOCK_SIZE
from src.forensic_layer.ela import compute_ela_map


def normalize_heatmap(ela_map: np.ndarray) -> np.ndarray:
    """
    Normalizon ELA map në [0, 1] duke përdorur min-max mbi VETË imazhin
    (jo shkallë fikse globale) — kështu çdo imazh e shfaq strukturën e
    vet relative, edhe nëse niveli absolut i gabimit ndryshon.
    """
    min_val = ela_map.min()
    max_val = ela_map.max()

    if max_val - min_val < 1e-6:
        return np.zeros_like(ela_map)

    return (ela_map - min_val) / (max_val - min_val)


def apply_colormap(normalized_map: np.ndarray, colormap: str = "jet") -> np.ndarray:
    """
    Kthen hartën e normalizuar [0,1] në RGB duke përdorur një colormap
    (jet: blu=gabim i ulët, kuq=gabim i lartë — konvencion i njohur
    forenzik/thermal imaging, i lexueshëm lehtë nga jo-teknikë).
    """
    cmap = plt.get_cmap(colormap)
    colored = cmap(normalized_map)  # kthen RGBA në [0,1]
    return (colored[:, :, :3] * 255).astype(np.uint8)  # heq alpha, kthen 0-255


def create_overlay(original_img: Image.Image, heatmap_rgb: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """
    Përzien heatmap-in RGB mbi imazhin origjinal (alpha blending).
    alpha=0.5 -> peshë e barabartë; rrit alpha për heatmap më dominant.
    """
    original_arr = np.array(original_img.convert("RGB")).astype(np.float32)
    heatmap_arr = heatmap_rgb.astype(np.float32)

    if original_arr.shape[:2] != heatmap_arr.shape[:2]:
        heatmap_img = Image.fromarray(heatmap_rgb).resize(
            (original_arr.shape[1], original_arr.shape[0])
        )
        heatmap_arr = np.array(heatmap_img).astype(np.float32)

    blended = (1 - alpha) * original_arr + alpha * heatmap_arr
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

def compute_cfa_visualization(image_path: str) -> np.ndarray:
    """
    Kthen spektrin e magnitudës (log-shkallë) të residualit CFA —
    periodiciteti demosaicing/checkerboard shfaqet si maja te buzët.
    """
    image = Image.open(image_path).convert("L")
    gray = np.array(image)

    residual = _bilinear_residual(gray)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(residual)))

    log_spectrum = np.log1p(spectrum)  # log për dukshmëri (energjia varion shumë)
    return normalize_heatmap(log_spectrum)

def compute_dct_block_visualization(image_path: str) -> np.ndarray:
    """
    Kthen një hartë (h_blocks x w_blocks) me raportin e energjisë së
    frekuencave të larta për SECILIN bllok 8x8 veç e veç — tregon
    vizualisht CILAT zona të imazhit kanë anomali frekuence, jo
    vetëm një mesatare të vetme si dct.py.
    """
    image = Image.open(image_path).convert("L")
    gray = np.array(image)

    h, w = gray.shape
    h_blocks = h // BLOCK_SIZE
    w_blocks = w // BLOCK_SIZE

    high_freq_mask = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=bool)
    for i in range(BLOCK_SIZE):
        for j in range(BLOCK_SIZE):
            if i + j >= 8:
                high_freq_mask[i, j] = True

    block_map = np.zeros((h_blocks, w_blocks))

    for by in range(h_blocks):
        for bx in range(w_blocks):
            block = gray[
                by * BLOCK_SIZE:(by + 1) * BLOCK_SIZE,
                bx * BLOCK_SIZE:(bx + 1) * BLOCK_SIZE,
            ].astype(np.float32)

            coeffs = _block_dct2(block)
            energy = coeffs ** 2
            high_energy = energy[high_freq_mask].sum()
            total_energy = energy.sum() + 1e-8

            block_map[by, bx] = high_energy / total_energy

    # Ridimenson te madhësia origjinale e imazhit (secili bllok bëhet
    # një "patch" i njëtrajtshëm 8x8 pixel në output vizual)
    block_map_upscaled = np.kron(block_map, np.ones((BLOCK_SIZE, BLOCK_SIZE)))
    return normalize_heatmap(block_map_upscaled)

def save_full_forensic_figure(image_path: str, output_path: str, quality: int = 90):
    """
    Figurë krahasuese 5-panele: original | ELA | CFA spectrum |
    DCT block-map (kontrolli 8x8) | overlay ELA.
    """
    original_img = Image.open(image_path).convert("RGB")

    ela_map = compute_ela_map(image_path, quality)
    ela_normalized = normalize_heatmap(ela_map)
    ela_rgb = apply_colormap(ela_normalized, colormap="jet")
    overlay_img = create_overlay(original_img, ela_rgb, alpha=0.5)

    cfa_vis = compute_cfa_visualization(image_path)
    dct_vis = compute_dct_block_visualization(image_path)

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))

    axes[0].imshow(original_img)
    axes[0].set_title("Origjinal")
    axes[0].axis("off")

    axes[1].imshow(ela_rgb)
    axes[1].set_title("ELA Heatmap")
    axes[1].axis("off")

    axes[2].imshow(cfa_vis, cmap="viridis")
    axes[2].set_title("CFA — Spektri i Periodicitetit")
    axes[2].axis("off")

    axes[3].imshow(dct_vis, cmap="inferno")
    axes[3].set_title("DCT — Energji per Bllok 8×8")
    axes[3].axis("off")

    axes[4].imshow(overlay_img)
    axes[4].set_title("Overlay (Origjinal + ELA)")
    axes[4].axis("off")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Përdorim: python visualization.py path\\te\\imazhi.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    OUTPUT_DIR = Path("outputs/forensic_visualizations")
    out_path = OUTPUT_DIR / (Path(image_path).stem + "_forensic_figure.png")

    save_full_forensic_figure(image_path, str(out_path))
    print(f"Figura u ruajt te: {out_path}")