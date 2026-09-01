"""
training_pipeline.py — Orchestrator i trajnimit të plotë. Thërret hapat ekzistues (jo rishkruan logjikën) sipas radhës:
Dataset -> Preprocessing -> CNN training -> Forensic/Frequency
extraction -> Fusion training -> Calibration -> Threshold selection.
Ekzekutim: python src\\training_pipeline.py
"""
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils.app_logging import get_logger
from src.utils.seed import set_seed
from src.utils.device import get_device, device_info

log = get_logger("training_pipeline")

STEPS = [
    ("Dataset scanning", ["python", "src/data_pipeline/dataset_scanner.py"]),
    ("Video-level split", ["python", "src/data_pipeline/video_level_split.py"]),
    ("Frame extraction (image_extraction)", ["python", "src/data_pipeline/image_extraction.py"]),
    ("CNN training (dl_layer)", ["python", "src/dl_layer/train.py"]),
    ("Forensic calibration (CFA/DCT z-score)", ["python", "src/forensic_layer/calibrate_forensic_score.py"]),
    ("Frequency calibration (FFT z-score)", ["python", "src/frequency_layer/calibrate_fft.py"]),
    ("Fusion training", ["python", "src/fusion/train_fusion.py"]),
    ("Threshold selection", ["python", "src/fusion/thresholds.py"]),
]

def run_step(name: str, command: list[str]) -> bool:
    log.info(f"Duke filluar: {name}")
    result = subprocess.run(command, capture_output=False)

    if result.returncode != 0:
        log.error(f"Hapi dështoi: {name} (kod dalje: {result.returncode})")
        return False

    log.info(f"Përfundoi me sukses: {name}")
    return True

def main():
    set_seed(42)
    log.info(f"Device: {device_info()}")
    log.info("Duke filluar training_pipeline i plotë...")

    for name, command in STEPS:
        success = run_step(name, command)
        if not success:
            log.error(f"Pipeline u ndal te hapi: {name}")
            sys.exit(1)

    log.info("Training pipeline përfundoi me sukses — të gjithë hapat u kryen.")

if __name__ == "__main__":
    main()