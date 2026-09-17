# Configuration package
from pathlib import Path
import yaml

CONFIG_DIR = Path(__file__).resolve().parent
SETTINGS_FILE = CONFIG_DIR / "settings.yaml"

def load_settings() -> dict:
    """Load settings from config/settings.yaml with safe fallback defaults."""
    if not SETTINGS_FILE.exists():
        return {}
    with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
