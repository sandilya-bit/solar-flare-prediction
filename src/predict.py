"""Prediction helpers for the Solar Flare Dashboard."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CNN_OUTPUT_CLASSES, HISTORY_MINUTES, PREDICTION_WINDOW_MINUTES
from src.model_loader import SolarFlareCNN

@dataclass(frozen=True)
class PredictionResult:
    """Dashboard-ready prediction payload."""
    timestamp: pd.Timestamp
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    current_xrsa_flux: float
    current_xrsb_flux: float
    current_flux_class: str
    predicted_class: str
    raw_cnn_class: str
    confidence: float
    probabilities: Dict[str, float]

def observed_flux_class(xrsb_flux: float) -> str:
    """Convert the latest XRS-B flux into a NOAA-style observed class."""
    if pd.isna(xrsb_flux):
        return "Unknown"
    if xrsb_flux >= 1e-4:
        return "X"
    if xrsb_flux >= 1e-5:
        return "M"
    if xrsb_flux >= 1e-6:
        return "C"
    if xrsb_flux >= 1e-7:
        return "B"
    return "A"

def prepare_model_input(window: pd.DataFrame, scaler: StandardScaler) -> torch.Tensor:
    """Scale and reshape a 60-minute flux window for Conv1D inference."""
    if len(window) < HISTORY_MINUTES:
        raise ValueError(f"Need {HISTORY_MINUTES} rows for prediction; got {len(window)}")
    values = window[["xrsa_flux", "xrsb_flux"]].tail(HISTORY_MINUTES).to_numpy()
    scaled = scaler.transform(values)
    model_input = np.transpose(scaled.reshape(1, HISTORY_MINUTES, 2), (0, 2, 1))
    return torch.FloatTensor(model_input)

def predict_flare(model: SolarFlareCNN, device: torch.device, window: pd.DataFrame, scaler: StandardScaler) -> PredictionResult:
    """Run the CNN and return class, confidence, and probabilities."""
    model_input = prepare_model_input(window, scaler)
    with torch.no_grad():
        logits = model(model_input.to(device))
        raw_probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    raw_index = int(np.argmax(raw_probs))
    raw_class = CNN_OUTPUT_CLASSES[raw_index]
    current_row = window.tail(1).iloc[0]
    current_class = observed_flux_class(float(current_row["xrsb_flux"]))
    predicted_class = "A" if raw_class == "B" and current_class == "A" else raw_class
    confidence = float(raw_probs[raw_index])
    probabilities = {"A": 0.0, "B": float(raw_probs[0]), "C": float(raw_probs[1]), "M": float(raw_probs[2]), "X": float(raw_probs[3])}
    if predicted_class == "A":
        probabilities["A"] = probabilities["B"]
        probabilities["B"] = 0.0
    window_start = window.index[0]
    window_end = window.index[-1]
    timestamp = window_end + pd.Timedelta(minutes=PREDICTION_WINDOW_MINUTES)
    return PredictionResult(timestamp=timestamp, window_start=window_start, window_end=window_end, current_xrsa_flux=float(current_row["xrsa_flux"]), current_xrsb_flux=float(current_row["xrsb_flux"]), current_flux_class=current_class, predicted_class=predicted_class, raw_cnn_class=raw_class, confidence=confidence, probabilities=probabilities)
