"""
chain_of_custody.py — Regjistron kronologjikisht çdo hap përpunimi mbi
një provë specifike: timestamp, hash hyrës, hapi, versioni i software/
modelit, dhe përmbledhje e output-it. Krijon "audit trail" të plotë.

Një skedar JSON i vetëm për processing_id, me listë ngjarjesh.
"""

import json
from pathlib import Path
from datetime import datetime, timezone

CUSTODY_LOG_DIR = Path("outputs/chain_of_custody")

SOFTWARE_VERSION = "1.0.0"
MODEL_VERSIONS = {
    "cnn": "efficientnet_b0_epoch1_valacc0.869",
    "fusion": "logistic_regression_v3_no_ela",
}


def _log_path(processing_id: str) -> Path:
    CUSTODY_LOG_DIR.mkdir(parents=True, exist_ok=True)
    return CUSTODY_LOG_DIR / f"{processing_id}.json"


def initialize_custody_log(processing_id: str, input_hash: str, input_path: str) -> None:
    """Krijon skedarin e log-ut me ngjarjen e parë: marrja e provës."""
    log_path = _log_path(processing_id)
    entries = [{
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "step": "evidence_received",
        "input_hash": input_hash,
        "input_path": input_path,
        "software_version": SOFTWARE_VERSION,
        "model_version": None,
        "output_summary": None,
    }]
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


def record_event(processing_id: str, step: str, input_hash: str,
                  output_summary: dict | str | None = None,
                  model_key: str | None = None) -> None:
    """
    Shton një ngjarje të re në log-un ekzistues të processing_id-së.
    model_key: p.sh. "cnn" ose "fusion", për të regjistruar versionin
    përkatës nga MODEL_VERSIONS automatikisht.
    """
    log_path = _log_path(processing_id)

    if log_path.exists():
        with open(log_path, "r", encoding="utf-8") as f:
            entries = json.load(f)
    else:
        entries = []

    entries.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "step": step,
        "input_hash": input_hash,
        "software_version": SOFTWARE_VERSION,
        "model_version": MODEL_VERSIONS.get(model_key) if model_key else None,
        "output_summary": output_summary,
    })

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


def load_custody_log(processing_id: str) -> list[dict]:
    log_path = _log_path(processing_id)
    if not log_path.exists():
        return []
    with open(log_path, "r", encoding="utf-8") as f:
        return json.load(f)