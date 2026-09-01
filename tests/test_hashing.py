"""
test_hashing.py — Unit tests për src/integrity/hashing.py.
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.integrity.hashing import compute_sha256, verify_hash

SAMPLE_DIR = PROJECT_ROOT / "data" / "frames" / "test" / "real"
SAMPLE_IMAGES = list(SAMPLE_DIR.glob("*.jpg"))[:2]

pytestmark = pytest.mark.skipif(
    len(SAMPLE_IMAGES) < 2, reason="Duhen të paktën 2 imazhe mostër te data/frames/test/real"
)

def test_compute_sha256_returns_string():
    file_hash = compute_sha256(str(SAMPLE_IMAGES[0]))
    assert isinstance(file_hash, str)

def test_compute_sha256_correct_length():
    """SHA-256 në hex ka gjithmonë 64 karaktere."""
    file_hash = compute_sha256(str(SAMPLE_IMAGES[0]))
    assert len(file_hash) == 64

def test_same_file_same_hash():
    """Njëjti skedar duhet të japë gjithmonë njëjtin SHA-256."""
    hash1 = compute_sha256(str(SAMPLE_IMAGES[0]))
    hash2 = compute_sha256(str(SAMPLE_IMAGES[0]))
    assert hash1 == hash2

def test_different_files_different_hash():
    """Dy skedarë të ndryshëm duhet të japin hash të ndryshëm (praktikisht gjithmonë)."""
    hash1 = compute_sha256(str(SAMPLE_IMAGES[0]))
    hash2 = compute_sha256(str(SAMPLE_IMAGES[1]))
    assert hash1 != hash2

def test_verify_hash_true_for_correct_hash():
    correct_hash = compute_sha256(str(SAMPLE_IMAGES[0]))
    assert verify_hash(str(SAMPLE_IMAGES[0]), correct_hash) is True

def test_verify_hash_false_for_wrong_hash():
    wrong_hash = "0" * 64
    assert verify_hash(str(SAMPLE_IMAGES[0]), wrong_hash) is False

def test_verify_hash_false_when_file_altered_hash_from_other_file():
    """Hash i një skedari tjetër s'duhet të validohet për këtë skedar."""
    other_hash = compute_sha256(str(SAMPLE_IMAGES[1]))
    assert verify_hash(str(SAMPLE_IMAGES[0]), other_hash) is False