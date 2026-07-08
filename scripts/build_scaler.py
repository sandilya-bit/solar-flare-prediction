"""Build and save the StandardScaler used by the Solar Flare Dashboard.

Run this script once after cloning or after updating the training data:
    python scripts/build_scaler.py

The saved scaler is stored at models/scaler.pkl and loaded at dashboard
startup instead of the large X_data.npy training array.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Allow imports from the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_CANDIDATES, MODELS_DIR, HISTORY_MINUTES

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
LOGGER = logging.getLogger("build_scaler")

REQUIRED_COLUMNS = ["xrsa_flux", "xrsb_flux"]


def find_first_existing(candidates: list) -> Path | None:
    for p in candidates:
        if Path(p).exists():
            return Path(p)
    return None


def build_scaler_from_csv(csv_path: str, output_path: str) -> StandardScaler:
    """Read GOES CSV, build sliding windows, fit scaler, and save to pkl."""
    path = Path(csv_path)
    LOGGER.info("Loading CSV from %s ...", path)

    df = pd.read_csv(path, index_col="time", parse_dates=True)
    df["xrsa_flux"] = pd.to_numeric(df["xrsa_flux"], errors="coerce")
    df["xrsb_flux"] = pd.to_numeric(df["xrsb_flux"], errors="coerce")
    df = df[REQUIRED_COLUMNS].dropna()
    df = df.sort_index()

    LOGGER.info("Loaded %d clean rows. Building sliding windows...", len(df))

    values = df[["xrsa_flux", "xrsb_flux"]].to_numpy()
    windows = []
    for i in range(len(values) - HISTORY_MINUTES + 1):
        windows.append(values[i : i + HISTORY_MINUTES])

    x_all = np.array(windows)  # shape: (N, HISTORY_MINUTES, 2)
    train_size = int(len(x_all) * 0.70)
    x_train = x_all[:train_size]

    LOGGER.info(
        "Fitting StandardScaler on %d training windows (of %d total)...",
        train_size,
        len(x_all),
    )

    scaler = StandardScaler()
    scaler.fit(x_train.reshape(-1, x_train.shape[-1]))

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, out_path)

    LOGGER.info("Saved scaler to %s  (feature means: %s)", out_path, scaler.mean_)
    return scaler


def main() -> None:
    csv_path = find_first_existing(DATA_CANDIDATES)
    if csv_path is None:
        LOGGER.error(
            "No GOES CSV found. Checked: %s",
            [str(p) for p in DATA_CANDIDATES],
        )
        sys.exit(1)

    output_path = MODELS_DIR / "scaler.pkl"
    scaler = build_scaler_from_csv(str(csv_path), str(output_path))
    LOGGER.info(
        "Done! Scaler saved to %s  (mean shape: %s)",
        output_path,
        scaler.mean_.shape,
    )


if __name__ == "__main__":
    main()
