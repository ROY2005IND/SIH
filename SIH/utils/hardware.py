"""Hardware detection and memory management for MoonFlower AI."""
import os
import platform
import psutil
from typing import TypedDict

class HardwareTelemetry(TypedDict):
    device: str
    cuda_available: bool
    gpu_name: str
    cpu_cores: int
    cpu_threads: int
    ram_total_gb: float
    ram_available_gb: float
    platform_name: str

def get_hardware_info() -> HardwareTelemetry:
    """Safely detect system compute and memory specifications without crashing."""
    cuda_available = False
    gpu_name = "N/A (CPU Mode Active)"

    # Check for PyTorch CUDA if torch is present
    try:
        import torch
        if torch.cuda.is_available():
            cuda_available = True
            gpu_name = torch.cuda.get_device_name(0)
    except (ImportError, Exception):
        pass

    # Read CPU and RAM metrics via psutil
    try:
        cpu_cores = psutil.cpu_count(logical=False) or 1
        cpu_threads = psutil.cpu_count(logical=True) or 1
        mem = psutil.virtual_memory()
        ram_total = round(mem.total / (1024 ** 3), 2)
        ram_avail = round(mem.available / (1024 ** 3), 2)
    except Exception:
        cpu_cores = 1
        cpu_threads = 1
        ram_total = 0.0
        ram_avail = 0.0

    target_device = "cuda" if cuda_available else "cpu"

    return {
        "device": target_device,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "cpu_cores": cpu_cores,
        "cpu_threads": cpu_threads,
        "ram_total_gb": ram_total,
        "ram_available_gb": ram_avail,
        "platform_name": f"{platform.system()} {platform.release()}",
    }
