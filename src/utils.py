"""Utility helpers used across the Streamlit dashboard."""
from __future__ import annotations
import json
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import LOGS_DIR, LOG_PATH

def setup_logging() -> logging.Logger:
    """Create a reusable dashboard logger."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("solar_flare_dashboard")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    file_handler = RotatingFileHandler(LOG_PATH, maxBytes=1_000_000, backupCount=5, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    logger.info("Logging initialized at %s", LOG_PATH)
    return logger

def utc_now() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc)

def format_utc(timestamp: datetime) -> str:
    """Format a timestamp for monitoring panels."""
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def find_first_existing(paths: Iterable[Path]) -> Optional[Path]:
    """Return the first existing path from a priority-ordered list."""
    for path in paths:
        if path.exists():
            return path
    return None


def file_cache_stamp(path: Path | str | None) -> str:
    """Return a cache-busting stamp from file modification time and size."""
    if path is None:
        return "missing"
    resolved = Path(path)
    if not resolved.exists():
        return f"missing:{resolved}"
    stat = resolved.stat()
    return f"{resolved}:{stat.st_mtime_ns}:{stat.st_size}"

def load_json(path: Optional[Path]) -> dict:
    """Load JSON from disk, returning an empty dictionary when absent."""
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)

def file_modified_utc(path: Optional[Path]) -> str:
    """Return a file's modification time in UTC."""
    if path is None or not path.exists():
        return "Not available"
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return format_utc(modified)

def format_flux(value: float) -> str:
    """Format X-ray flux in scientific notation with physical units."""
    return f"{value:.2e} W/m^2"

def compact_percent(value: float) -> str:
    """Format a probability as a percentage."""
    return f"{value * 100:.1f}%"
