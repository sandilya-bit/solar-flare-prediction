"""Data access and preprocessing for GOES X-ray flux inputs."""
from __future__ import annotations

import glob
import logging
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import HISTORY_MINUTES

LOGGER = logging.getLogger("solar_flare_dashboard.preprocess")
REQUIRED_COLUMNS = ["xrsa_flux", "xrsb_flux"]


def clean_flux_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Clean GOES flux data and resample it to one-minute cadence."""
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required flux columns: {sorted(missing)}")
    clean = df[REQUIRED_COLUMNS].copy()
    clean["xrsa_flux"] = pd.to_numeric(clean["xrsa_flux"], errors="coerce")
    clean["xrsb_flux"] = pd.to_numeric(clean["xrsb_flux"], errors="coerce")
    clean[clean < 0] = np.nan
    clean = clean.sort_index()
    if isinstance(clean.index, pd.DatetimeIndex):
        clean = clean.ffill(limit=60).resample("1min").mean()
    else:
        clean = clean.ffill(limit=60)
    return clean.dropna()


@st.cache_data(show_spinner=False)
def load_goes_csv(csv_path: str, file_stamp: str) -> pd.DataFrame:
    """Load the local cleaned GOES CSV file."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"GOES data file not found: {path}")
    df = pd.read_csv(path, index_col="time", parse_dates=True)
    return clean_flux_frame(df)


@st.cache_data(show_spinner=False)
def load_latest_goes_netcdf(data_dir: str, cache_stamp: str) -> Tuple[pd.DataFrame, str]:
    """Load the latest local GOES NetCDF file.

    This is isolated so a future NOAA JSON loader can be added without changing
    dashboard or prediction code.
    """
    try:
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError(
            "xarray is required for NetCDF input. Use the CSV source or install xarray."
        ) from exc
    files = sorted(glob.glob(str(Path(data_dir) / "**" / "*.nc"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No GOES NetCDF files found under {data_dir}")
    latest_file = files[-1]
    ds = xr.open_dataset(latest_file)
    try:
        df = ds[["xrsa_flux", "xrsb_flux"]].to_dataframe().reset_index()
    finally:
        ds.close()
    if "energy" in df.columns:
        df = df.drop(columns=["energy"])
    df = df.set_index("time")
    return clean_flux_frame(df), latest_file


@st.cache_resource(show_spinner=False)
def load_scaler(scaler_path: str, file_stamp: str) -> StandardScaler:
    """Load the pre-fitted StandardScaler from a joblib pickle file.

    Generate this file once by running:
        python scripts/build_scaler.py
    """
    path = Path(scaler_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Scaler not found at {path}. "
            "Run 'python scripts/build_scaler.py' to generate it."
        )
    scaler = joblib.load(path)
    LOGGER.info("Loaded scaler from %s", path)
    return scaler


def build_and_save_scaler(csv_path: str, output_path: str) -> StandardScaler:
    """Build a StandardScaler from the GOES CSV and save it as a .pkl file.

    This is called by scripts/build_scaler.py — not needed at dashboard runtime.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"GOES CSV not found: {path}")

    df = pd.read_csv(path, index_col="time", parse_dates=True)
    df = clean_flux_frame(df)

    # Build sliding windows identical to training
    windows = []
    values = df[["xrsa_flux", "xrsb_flux"]].to_numpy()
    for i in range(len(values) - HISTORY_MINUTES + 1):
        windows.append(values[i : i + HISTORY_MINUTES])

    x_all = np.array(windows)  # shape: (N, HISTORY_MINUTES, 2)
    train_size = int(len(x_all) * 0.70)
    x_train = x_all[:train_size]

    scaler = StandardScaler()
    scaler.fit(x_train.reshape(-1, x_train.shape[-1]))

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, out_path)
    LOGGER.info("Saved scaler to %s (fitted on %d training windows)", out_path, train_size)
    return scaler


@st.cache_resource(show_spinner=False)
def build_training_scaler(training_array_path: str, file_stamp: str) -> StandardScaler:
    """Legacy: reconstruct scaler from X_data.npy. Use load_scaler() instead."""
    path = Path(training_array_path)
    if not path.exists():
        raise FileNotFoundError(f"Training array not found: {path}")
    x_values = np.load(path)
    train_size = int(len(x_values) * 0.70)
    x_train = x_values[:train_size]
    scaler = StandardScaler()
    scaler.fit(x_train.reshape(-1, x_train.shape[-1]))
    return scaler


def latest_window(df: pd.DataFrame) -> pd.DataFrame:
    """Return the latest 60-minute flux window expected by the CNN."""
    if len(df) < HISTORY_MINUTES:
        raise ValueError(
            f"Need at least {HISTORY_MINUTES} clean rows; only {len(df)} are available."
        )
    return df.tail(HISTORY_MINUTES)


def netcdf_cache_stamp(data_dir: str) -> str:
    """Build a cache stamp from the newest NetCDF file under a directory."""
    files = sorted(glob.glob(str(Path(data_dir) / "**" / "*.nc"), recursive=True))
    if not files:
        return f"empty:{data_dir}"
    latest_file = files[-1]
    stat = Path(latest_file).stat()
    return f"{latest_file}:{stat.st_mtime_ns}:{stat.st_size}"
