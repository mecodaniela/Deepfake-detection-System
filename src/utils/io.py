"""
io.py — Funksione ndihmëse të përgjithshme: krijim direktorish, save/load JSON, load config, trajtim i sigurt skedarësh.
"""
import json
from pathlib import Path
from typing import Any

def ensure_dir(path: str | Path) -> Path:
    """Krijon direktorinë (dhe prindërit e saj) nëse s'ekziston, kthen Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_json(data: Any, path: str | Path, indent: int = 2) -> Path:
    """Ruaj çdo strukturë JSON-serializueshme te path, krijon direktorinë nëse duhet."""
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    return p

def load_json(path: str | Path) -> Any:
    """Lexon dhe kthen përmbajtjen JSON të një skedari."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Skedari s'ekziston: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def load_config(path: str | Path = "configs/config.json") -> dict:
    """
    Ngarkon skedarin e konfigurimit qendror të projektit (nëse ekziston).
    Kthen dict bosh nëse s'ekziston ende — s'shkakton gabim.
    """
    p = Path(path)
    if not p.exists():
        return {}
    return load_json(p)

def safe_read_text(path: str | Path, default: str = "") -> str:
    """Lexon tekst nga një skedar, kthen default nëse skedari s'ekziston."""
    p = Path(path)
    if not p.exists():
        return default
    return p.read_text(encoding="utf-8")

if __name__ == "__main__":
    test_dir = ensure_dir("outputs/_io_test")
    test_path = save_json({"test": True, "value": 42}, test_dir / "test.json")
    loaded = load_json(test_path)
    print(f"Test i shpejtë: {loaded}")