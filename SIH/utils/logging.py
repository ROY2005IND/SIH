"""Structured scientific logging for MoonFlower AI."""
import logging
import sys
from pathlib import Path
from .paths import PROJECT_ROOT

LOGS_DIR = PROJECT_ROOT / "data" / "outputs" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_LOG_FILE = LOGS_DIR / "moonflower.log"

class AerospaceFormatter(logging.Formatter):
    """Clean aerospace telemetry logging format: [LEVEL] Component - Message"""
    def format(self, record):
        levelname = record.levelname
        component = getattr(record, "component", record.name)
        message = record.getMessage()
        return f"[{levelname:<5}] {component} - {message}"

def get_logger(component_name: str = "Core", log_file: Path | None = None) -> logging.Logger:
    """Get or configure a structured scientific logger."""
    logger = logging.getLogger(f"MoonFlower.{component_name}")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers
    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(AerospaceFormatter())
        logger.addHandler(ch)

        # File handler
        target_file = log_file or DEFAULT_LOG_FILE
        try:
            fh = logging.FileHandler(target_file, encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(AerospaceFormatter())
            logger.addHandler(fh)
        except Exception:
            pass  # Fall back to console only if file handler fails

    return logger
