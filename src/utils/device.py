"""
device.py — Zgjedh device-in (CUDA GPU nëse ka, përndryshe CPU) — një pikë e vetme e vërtetë për këtë vendim, në vend që ta përsërisim
`torch.device("cpu")` manualisht në çdo skedar (siç e kemi bërë deri tani te dl_layer, explainability, etj.).
"""
import torch

def get_device(force_cpu: bool = False) -> torch.device:
    """
    Kthen torch.device — CUDA nëse është e disponueshme (dhe force_cpu
    është False), përndryshe CPU. Projekti aktual funksionon vetëm
    mbi CPU, por kjo e bën kodin gati për migrim të lehtë nëse
    ndonjëherë përdoret GPU.
    """
    if not force_cpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def device_info() -> str:
    """Kthen një string përshkrues të device-it aktual, për logging."""
    device = get_device()
    if device.type == "cuda":
        return f"CUDA GPU: {torch.cuda.get_device_name(0)}"
    return "CPU"

if __name__ == "__main__":
    print(f"Device i zgjedhur: {get_device()}")
    print(f"Detaje: {device_info()}")