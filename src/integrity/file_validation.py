"""
file_validation.py — Kontrollon konsistencën mes extension, MIME type,
dhe magic bytes të skedarit. Mospërputhje = flag integriteti (jo konkluzion automatik manipulimi).
"""
from pathlib import Path

# Magic bytes (nënshkrimi i parë i bajteve) për formatet kryesore
MAGIC_BYTES = {
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"GIF87a": "image/gif",
    b"GIF89a": "image/gif",
    b"RIFF": "video/avi",  # kërkon kontroll shtesë për WEBP vs AVI
    
}

EXTENSION_TO_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".mp4": "video/mp4",
    ".avi": "video/avi",
}


def detect_magic_type(file_path: str) -> str | None:
    """Lexon bajtet e para dhe kthen MIME type sipas magic bytes."""
    with open(file_path, "rb") as f:
        header = f.read(32)

    # MP4/MOV: box "ftyp" në offset 4-8, pavarësisht madhësisë së box-it
    if header[4:8] == b"ftyp":
        return "video/mp4"

    for signature, mime in MAGIC_BYTES.items():
        if header.startswith(signature):
            return mime
    return None


def validate_file(file_path: str) -> dict:
    """
    Kthen dict me rezultatin e validimit: extension i deklaruar,
    MIME i pritur nga extension, MIME real (nga magic bytes), dhe
    nëse janë konsistente.
    """
    path = Path(file_path)
    extension = path.suffix.lower()

    expected_mime = EXTENSION_TO_MIME.get(extension)
    actual_mime = detect_magic_type(file_path)

    is_consistent = (
        expected_mime is not None
        and actual_mime is not None
        and expected_mime == actual_mime
    )

    return {
        "file_path": str(path),
        "extension": extension,
        "expected_mime": expected_mime,
        "actual_mime_from_magic_bytes": actual_mime,
        "is_consistent": is_consistent,
        "flag": None if is_consistent else "INTEGRITY_MISMATCH",
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Përdorim: python file_validation.py <path_to_file>")
        sys.exit(1)

    result = validate_file(sys.argv[1])
    print(f"Skedari: {result['file_path']}")
    print(f"Extension: {result['extension']}")
    print(f"MIME i pritur (nga extension): {result['expected_mime']}")
    print(f"MIME real (nga magic bytes): {result['actual_mime_from_magic_bytes']}")
    print(f"Konsistent: {result['is_consistent']}")
    if result["flag"]:
        print(f"⚠ FLAG: {result['flag']} — mospërputhje e zbuluar (jo domosdoshmërisht manipulim)")