"""
jpeg_compression.py — Aplikon kompresim JPEG me kualitet të ndryshëm
mbi një imazh në memorie (pa ruajtje të domosdoshme në disk).
"""
import io
from PIL import Image

def apply_jpeg_compression(image: Image.Image, quality: int) -> Image.Image:
    """
    Ricikon imazhin nëpër kompresim JPEG me kualitetin e dhënë (1-100),
    kthen imazhin e dekompresuar (PIL Image).
    """
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")