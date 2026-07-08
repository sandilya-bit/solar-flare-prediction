"""Central configuration for the Solar Flare Dashboard."""
from __future__ import annotations
from pathlib import Path

# Root directory (where app.py lives)
PROJECT_ROOT = Path(__file__).resolve().parent
APP_DIR = PROJECT_ROOT

# Subdirectories
ASSETS_DIR = PROJECT_ROOT / "assets"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"

# Application settings
APP_TITLE = "Solar Flare Dashboard"
MODEL_DISPLAY_NAME = "Best Solar Flare CNN"
GOES_VERSION = "GOES-R Series XRS, 1-minute X-ray flux"
SATELLITE_STATUS = "Operational"
SATELLITE_NAME = "NOAA GOES Primary"
PREDICTION_WINDOW_MINUTES = 60
HISTORY_MINUTES = 60
REFRESH_INTERVAL_SECONDS = 60
DATA_SOURCE = "live_noaa_json"
NOAA_GOES_XRAY_JSON_URL = "https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json"

# Model classes and candidates
CLASS_NAMES = ["A", "B", "C", "M", "X"]
CNN_OUTPUT_CLASSES = ["B", "C", "M", "X"]

# File path candidates (checked in order)
MODEL_CANDIDATES = [
    MODELS_DIR / "flare_model.pth",
    PROJECT_ROOT / "flare_model.pth",
    MODELS_DIR / "solar_flare_cnn.pth",
    PROJECT_ROOT / "solar_flare_cnn.pth"
]

METADATA_CANDIDATES = [
    MODELS_DIR / "flare_model_metadata.json",
    PROJECT_ROOT / "flare_model_metadata.json"
]

DATA_CANDIDATES = [
    DATA_DIR / "goes_xrs_1min_clean.csv",
    DATA_DIR / "goes_labeled.csv",
    PROJECT_ROOT / "goes_xrs_1min_clean.csv",
    PROJECT_ROOT / "goes_labeled.csv"
]

TRAINING_ARRAY_CANDIDATES = [
    DATA_DIR / "X_data.npy",
    PROJECT_ROOT / "X_data.npy"
]

SPLIT_DATASET_CANDIDATES = [
    DATA_DIR / "cnn_dataset_splits.npz",
    PROJECT_ROOT / "cnn_dataset_splits.npz"
]

# Persistent paths
HISTORY_PATH = DATA_DIR / "prediction_history.csv"
LAST_SUCCESS_PATH = DATA_DIR / "last_success_prediction.json"
LOG_PATH = LOGS_DIR / "dashboard.log"
