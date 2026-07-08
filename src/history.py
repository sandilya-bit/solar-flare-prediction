"""Prediction history persistence."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import HISTORY_PATH, REFRESH_INTERVAL_SECONDS
from src.predict import PredictionResult
from src.utils import file_cache_stamp

HISTORY_COLUMNS = ["Timestamp", "Flux", "Prediction", "Confidence"]
LEGACY_COLUMN_MAP = {
    "timestamp": "Timestamp",
    "flux": "Flux",
    "predicted_class": "Prediction",
    "confidence": "Confidence",
}


def _normalize_history_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert legacy history files to the public CSV schema."""
    normalized = df.rename(columns=LEGACY_COLUMN_MAP).copy()
    for column in HISTORY_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = None
    return normalized[HISTORY_COLUMNS]


def append_prediction(result: PredictionResult, path: Path = HISTORY_PATH) -> None:
    """Append the latest prediction to prediction_history.csv.

    Rows are de-duplicated by Timestamp so Streamlit reruns do not repeatedly
    append the same NOAA update. The file keeps a bounded recent history to
    avoid unnecessary memory and disk growth.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame(
        [
            {
                "Timestamp": str(result.window_end),
                "Flux": result.current_xrsb_flux,
                "Prediction": result.predicted_class,
                "Confidence": result.confidence,
            }
        ]
    )

    if path.exists():
        existing = _normalize_history_columns(pd.read_csv(path))
        combined = pd.concat([existing, row], ignore_index=True)
        combined = combined.drop_duplicates(subset=["Timestamp"], keep="last")
    else:
        combined = row

    combined.tail(200).to_csv(path, index=False)
    load_history.clear()


@st.cache_data(ttl=REFRESH_INTERVAL_SECONDS, show_spinner=False)
def load_history(path: Path = HISTORY_PATH, file_stamp: str = "") -> pd.DataFrame:
    """Load the latest 50 predictions from prediction_history.csv."""
    if not path.exists():
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    df = _normalize_history_columns(pd.read_csv(path))
    return df.tail(50)


def cached_load_history(path: Path = HISTORY_PATH) -> pd.DataFrame:
    """Load history using a file stamp so edits invalidate the cache."""
    return load_history(path, file_cache_stamp(path))
