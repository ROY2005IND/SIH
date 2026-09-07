"""Filesystem path management for MoonFlower AI."""
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data subdirectories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "samples"
OUTPUTS_DIR = DATA_DIR / "outputs"
WEIGHTS_DIR = DATA_DIR / "weights"

# Configuration directory
CONFIG_DIR = PROJECT_ROOT / "config"

# Experiments directory
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
EXPERIMENT_RUNS_DIR = EXPERIMENTS_DIR / "runs"
EXPERIMENT_CONFIGS_DIR = EXPERIMENTS_DIR / "configs"
EXPERIMENT_RESULTS_DIR = EXPERIMENTS_DIR / "results"

def ensure_directories():
    """Ensure all expected local directories exist safely."""
    dirs = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        SAMPLES_DIR,
        OUTPUTS_DIR,
        WEIGHTS_DIR,
        EXPERIMENT_RUNS_DIR,
        EXPERIMENT_CONFIGS_DIR,
        EXPERIMENT_RESULTS_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

ensure_directories()
