"""
test_pipeline.py — Integration tests për src/inference_pipeline.py.
VËREJTJE: struktura e saktë e dict-it të kthyer nga run_inference_pipeline()
s'është konfirmuar plotësisht (emrat e çelësave si "verdict", "scores", etj.).
Testi kryesor këtu verifikon që pipeline-i ekzekutohet pa gabime dhe kthen
një dict — jo çdo fushë specifike. Shto asserte më të hollësishme pas
konfirmimit të skemës së saktë.
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.inference_pipeline import generate_evidence_id, run_inference_pipeline
from src.integrity.hashing import compute_sha256

SAMPLE_DIR = PROJECT_ROOT / "data" / "frames" / "test" / "real"
SAMPLE_IMAGE = next(iter(SAMPLE_DIR.glob("*.jpg")), None)

pytestmark = pytest.mark.skipif(
    SAMPLE_IMAGE is None, reason="Asnjë imazh mostër nuk u gjet te data/frames/test/real"
)

def test_generate_evidence_id_returns_string():
    sha256_hash = compute_sha256(str(SAMPLE_IMAGE))
    evidence_id = generate_evidence_id(sha256_hash)
    assert isinstance(evidence_id, str)
    assert len(evidence_id) > 0

def test_generate_evidence_id_deterministic():
    """Njëjti hash duhet të japë gjithmonë njëjtin evidence_id."""
    sha256_hash = compute_sha256(str(SAMPLE_IMAGE))
    id1 = generate_evidence_id(sha256_hash)
    id2 = generate_evidence_id(sha256_hash)
    assert id1 == id2

def test_generate_evidence_id_different_for_different_hashes():
    hash_a = "a" * 64
    hash_b = "b" * 64
    assert generate_evidence_id(hash_a) != generate_evidence_id(hash_b)

@pytest.mark.slow
def test_run_inference_pipeline_completes_without_error(tmp_path):
    """
    Test integrimi i plotë: kalon një imazh të vetëm nëpër gjithë pipeline-in
    (integrity → scores → fusion → explainability → JSON). Mund të marrë
    disa sekonda (CNN inference, CPU-only).
    """
    output_dir = str(tmp_path / "inference_output")
    result = run_inference_pipeline(str(SAMPLE_IMAGE), output_dir=output_dir)
    assert isinstance(result, dict)
    assert len(result) > 0

@pytest.mark.slow
def test_run_inference_pipeline_creates_output_files(tmp_path):
    output_dir = tmp_path / "inference_output"
    run_inference_pipeline(str(SAMPLE_IMAGE), output_dir=str(output_dir))
    produced_files = list(output_dir.rglob("*"))
    assert len(produced_files) > 0