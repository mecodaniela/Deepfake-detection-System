"""
resize.py — Ul rezolucionin e imazhit në % të madhësisë origjinale,
pastaj e rikthen në madhësinë origjinale — simulon humbje reale detajesh
(jo thjesht ndryshim madhësie hyrëse pa humbje).
"""
from PIL import Image

def apply_resize_degradation(image: Image.Image, scale_percent: int) -> Image.Image:
    original_size = image.size
    new_size = (
        max(1, int(original_size[0] * scale_percent / 100)),
        max(1, int(original_size[1] * scale_percent / 100)),
    )
    downscaled = image.resize(new_size, Image.Resampling.BILINEAR)
    return downscaled.resize(original_size, Image.Resampling.BILINEAR)