"""
hashing.py — Llogarit hash kriptografik (SHA-256) të skedarit bruto,
për identifikim të qëndrueshëm dhe verifikim integriteti të provës.
"""

import hashlib
from pathlib import Path


def compute_sha256(file_path: str) -> str:
    sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
    except (OSError, PermissionError) as e:
        raise RuntimeError(f"S'u lexua dot skedari për hashing: {file_path} ({e})") from e
    return sha256.hexdigest()


def verify_hash(file_path: str, expected_hash: str) -> bool:
    """
    Verifikon nëse hash-i aktual i skedarit përputhet me një hash të
    pritur (p.sh. atë të ruajtur më parë në provenance record) —
    zbulon çdo ndryshim, edhe të vetëm 1 bit.
    """
    actual_hash = compute_sha256(file_path)
    return actual_hash.lower() == expected_hash.lower()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Përdorim: python hashing.py <path_to_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    if not Path(file_path).exists():
        print(f"[GABIM] Skedari s'ekziston: {file_path}")
        sys.exit(1)

    hash_value = compute_sha256(file_path)
    print(f"Skedari: {file_path}")
    print(f"SHA-256: {hash_value}")