"""
inference_pipeline.py — Orkestron gjithë procesin e analizës forenzike
mbi një imazh të vetëm: integritet, score individuale, fusion, klasifikim final i strukturuar (T1/T2), dhe explainability.
Ekzekutim: python src/inference_pipeline.py <path_to_image>
"""
import sys
import json
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.integrity.hashing import compute_sha256
from src.integrity.file_validation import validate_file
from src.integrity.provenance import create_provenance_record
from src.dl_layer.predict import predict_cnn_score
from src.forensic_layer.ela import compute_ela_score
from src.forensic_layer.cfa import compute_cfa_score
from src.forensic_layer.dct import compute_dct_score
from src.frequency_layer.fft_transform import compute_fft_score
from src.explainability.evidence_overlay import save_individual_panels
from src.reporting.chain_of_custody import initialize_custody_log, record_event, load_custody_log

FUSION_MODEL_PATH = Path("models/fusion_logistic.pkl")
THRESHOLDS_PATH = Path("models/fusion_thresholds.json")

MODEL_METADATA = {
    "cnn": {
        "architecture": "EfficientNet-B0",
        "version": "efficientnet_b0_v1",
        "framework": "PyTorch",
        "input_resolution": "224x224",
        "training_datasets": ["FaceForensics++", "DFDC", "Celeb-DF"],
        "checkpoint": "models/dl_layer_best.pt",
    },
    "fusion": {
        "algorithm": "Logistic Regression",
        "version": "fusion_logistic_v3_no_ela",
        "features": ["cnn", "cfa", "dct", "fft"],
    },
}

SCOPE_TEXT = (
    "Scope of examination: This analysis evaluates the supplied digital image for "
    "indicators consistent with manipulation or synthetic generation using the "
    "implemented forensic and machine-learning methods."
)
DISCLAIMER_TEXT = (
    "Important: The system output represents a probabilistic technical assessment "
    "and does not, by itself, establish the legal authenticity or authorship of the "
    "evidence. Admissibility depends on jurisdiction, procedural standards, expert "
    "testimony, and evidence handling/documentation practices."
)

def load_fusion_model():
    saved = joblib.load(FUSION_MODEL_PATH)
    return saved["model"], saved["feature_names"]

def load_thresholds() -> dict:
    with open(THRESHOLDS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def classify_with_thresholds(prob_fake: float, t1: float, t2: float) -> dict:
    """
    Kthen klasifikim të strukturuar: kategoria, besimi i vendimit, dhe
    rregulla e saktë e përdorur, jo vetëm etiketë e thjeshtë.
    """
    if prob_fake < t1:
        return {
            "classification": "AUTHENTIC",
            "confidence": "HIGH",
            "decision_rule": f"P(fake) < T1 ({t1:.4f})",
        }
    elif prob_fake >= t2:
        return {
            "classification": "MANIPULATED",
            "confidence": "HIGH",
            "decision_rule": f"P(fake) >= T2 ({t2:.4f})",
        }
    return {
        "classification": "SUSPICIOUS",
        "confidence": "MEDIUM",
        "decision_rule": f"T1 <= P(fake) < T2 ({t1:.4f} - {t2:.4f})",
    }

def generate_evidence_id(sha256_hash: str) -> str:
    year = datetime.now().year
    return f"EVD-{year}-{sha256_hash[:8].upper()}"

def run_inference_pipeline(image_path: str, output_dir: str = "outputs/inference") -> dict:
    image_path = str(image_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Integritet ---
    sha256_hash = compute_sha256(image_path)  # hash i skedarit ORIGJINAL, para çdo përpunimi
    evidence_id = generate_evidence_id(sha256_hash)
    provenance = create_provenance_record(image_path)
    processing_id = provenance["processing_id"]

    initialize_custody_log(processing_id, sha256_hash, image_path)

    validation = validate_file(image_path)
    record_event(processing_id, "integrity_validation", sha256_hash,
                 output_summary={"is_consistent": validation["is_consistent"]})

    # --- 2. Score individuale ---
    cnn_score = predict_cnn_score(image_path)
    record_event(processing_id, "cnn_analysis", sha256_hash,
                 output_summary={"cnn_score": cnn_score}, model_key="cnn")

    ela_score = compute_ela_score(image_path)
    cfa_score = compute_cfa_score(image_path)
    dct_score = compute_dct_score(image_path)
    fft_score = compute_fft_score(image_path)
    record_event(processing_id, "forensic_analysis", sha256_hash,
                 output_summary={"ela": ela_score, "cfa": cfa_score, "dct": dct_score, "fft": fft_score})

    # --- 3. Fusion ---
    model, feature_names = load_fusion_model()
    feature_map = {"cnn": cnn_score, "ela": ela_score, "cfa": cfa_score, "dct": dct_score, "fft": fft_score}
    features = np.array([[feature_map[name] for name in feature_names]])
    prob_fake = float(model.predict_proba(features)[0, 1])

    thresholds = load_thresholds()
    t1, t2 = thresholds["t1_real"], thresholds["t2_fake"]
    decision = classify_with_thresholds(prob_fake, t1, t2)
    record_event(processing_id, "fusion_classification", sha256_hash,
                 output_summary={"probability_fake": prob_fake, "decision": decision},
                 model_key="fusion")

    # --- 4. Explainability ---
    panels = save_individual_panels(image_path, str(out_dir))
    record_event(processing_id, "explainability_generated", sha256_hash,
                 output_summary={"agreement_iou": panels["agreement_iou"]})

    # --- 5. Rezultati i plotë ---
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evidence_id": evidence_id,
        "original_filename": Path(image_path).name,
        "internal_storage_reference": f"{evidence_id}/original.jpg",
        "file_size_bytes": provenance["file_size_bytes"],
        "scope_text": SCOPE_TEXT,
        "disclaimer_text": DISCLAIMER_TEXT,
        "integrity": {
            "sha256": sha256_hash,
            "file_validation": validation,
        },
        "scores": {
            "cnn": cnn_score,
            "ela": ela_score,
            "cfa": cfa_score,
            "dct": dct_score,
            "fft": fft_score,
        },
        "model_metadata": MODEL_METADATA,
        "fusion": {
            "probability_fake": prob_fake,
            "feature_names": feature_names,
            "t1_real_threshold": t1,
            "t2_fake_threshold": t2,
            "decision": decision,
        },
        "explainability": {
            "panel_paths": panels["paths"],
            "ela_gradcam_agreement_iou": panels["agreement_iou"],
        },
        "limitations": [
            "ELA u testua statistikisht (n=100-500 në disa faza, tre nivele analize) dhe u gjet pa sinjal dallues të pavarur, përfshihet vetëm për transparencë metodologjike.",
            "CFA/DCT/FFT japin sinjal të dobët individualisht (korrelacion ~0 me label-in). Kontributi kryesor i klasifikimit vjen nga CNN.",
            "Sistemi u trajnua/testua mbi FaceForensics++/DFDC/Celeb-DF, performanca mbi metoda manipulimi krejt të reja/të panjohura s'është e garantuar.",
           "Nën degradim (kompresim JPEG i ulët dhe/ose humbje rezolucioni), saktësia e përgjithshme bie ndjeshëm (nga ~93-95% drejt ~78-84%); sistemi hibrid s'ka treguar avantazh konsistent mbi CNN-në e vetme në asnjë nivel të testuar të degradimit — CNN mbetet e barabartë ose më e mirë në shumicën e kushteve.",
        ],
    }

    json_out_path = out_dir / f"{evidence_id}_inference.json"
    result["chain_of_custody"] = load_custody_log(processing_id)  # noqa — importuar më poshtë

    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    record_event(processing_id, "report_data_finalized", sha256_hash,
                 output_summary={"json_path": str(json_out_path)})

    result["_json_path"] = str(json_out_path.resolve())
    return result

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Përdorim: python inference_pipeline.py <path_to_image>")
        sys.exit(1)

    result = run_inference_pipeline(sys.argv[1])
    print(f"\nEvidence ID: {result['evidence_id']}")
    print(f"Rezultati u ruajt te: {result['_json_path']}")
    print(f"Klasifikimi: {result['fusion']['decision']['classification']} "
          f"(besim: {result['fusion']['decision']['confidence']})")
    print(f"P(fake): {result['fusion']['probability_fake']:.4f}")