"""
provenance.py:Krijon një record të strukturuar për çdo provë të analizuar: filename, madhësi, timestamp, metadata, hash, processing ID.
"""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.integrity.hashing import compute_sha256
from src.integrity.file_validation import validate_file

def create_provenance_record(file_path: str) -> dict:
    """
    Krijon record të plotë provenance për një skedar provë:
    identifikim unik, hash, validim integriteti, dhe metadata bazë.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Skedari s'ekziston: {file_path}")

    stat = path.stat()

    record = {
        "processing_id": str(uuid.uuid4()),
        "filename": path.name,
        "file_path": str(path.resolve()),
        "file_size_bytes": stat.st_size,
        "created_timestamp": datetime.now(timezone.utc).isoformat(),
        "file_modified_timestamp": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(),
        "sha256_hash": compute_sha256(file_path),
        "integrity_validation": validate_file(file_path),
    }

    return record

def save_provenance_record(file_path: str, output_dir: str = "outputs/provenance") -> str:
    """
    Krijon record-in dhe e ruan si JSON, emërtuar sipas processing_id
    (unik për çdo analizë), gati për referencë në raportin final.
    """
    record = create_provenance_record(file_path)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{record['processing_id']}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    return str(out_path)

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Përdorim: python provenance.py <path_to_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    output_path = save_provenance_record(file_path)

    print(f"Provenance record u krijua dhe u ruajt te: {output_path}")

    with open(output_path, "r", encoding="utf-8") as f:
        record = json.load(f)
    print("\nPërmbajtja:")
    print(json.dumps(record, indent=2, ensure_ascii=False))