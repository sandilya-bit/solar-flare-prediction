"""CNN architecture, model loading, and model summary helpers."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
import json
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CNN_OUTPUT_CLASSES, MODEL_DISPLAY_NAME
from src.utils import file_modified_utc

@dataclass(frozen=True)
class ModelConfig:
    """Architecture knobs produced by the experiment runner."""
    batch_norm: bool = True
    extra_conv: bool = False
    dropout: float = 0.6

class SolarFlareCNN(nn.Module):
    """1D convolutional neural network for GOES XRS flare forecasting."""
    def __init__(self, config: ModelConfig, num_classes: int = 4):
        super().__init__()
        layers = []
        in_channels = 2
        for out_channels in (16, 32, 64):
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1))
            if config.batch_norm:
                layers.append(nn.BatchNorm1d(out_channels))
            layers.extend([nn.ReLU(), nn.MaxPool1d(kernel_size=2)])
            in_channels = out_channels
        final_channels = 64
        final_steps = 7
        if config.extra_conv:
            layers.append(nn.Conv1d(64, 128, kernel_size=3, padding=1))
            if config.batch_norm:
                layers.append(nn.BatchNorm1d(128))
            layers.append(nn.ReLU())
            final_channels = 128
        self.conv_layers = nn.Sequential(*layers)
        self.fc_layers = nn.Sequential(nn.Flatten(), nn.Linear(final_channels * final_steps, 128), nn.ReLU(), nn.Dropout(config.dropout), nn.Linear(128, num_classes))
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc_layers(self.conv_layers(inputs))

def config_from_metadata(metadata: Dict) -> ModelConfig:
    """Create a model config from experiment metadata."""
    config = metadata.get("config", {}) if metadata else {}
    return ModelConfig(batch_norm=config.get("batch_norm", True), extra_conv=config.get("extra_conv", False), dropout=config.get("dropout", 0.6))

@st.cache_resource(show_spinner=False)
def load_model(model_path: str, metadata_key: str, model_stamp: str) -> Tuple[SolarFlareCNN, torch.device]:
    """Load and cache the trained CNN weights."""
    path = Path(model_path)
    metadata = json.loads(metadata_key) if metadata_key else {}
    config = config_from_metadata(metadata)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SolarFlareCNN(config).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model, device

@st.cache_data(show_spinner=False)
def compute_split_accuracy(model_path: str, metadata_key: str, model_stamp: str, split_path: str, split_name: str) -> float | None:
    """Compute cached train/validation accuracy from saved dataset splits."""
    path = Path(split_path)
    if not path.exists():
        return None
    model, device = load_model(model_path, metadata_key, model_stamp)
    data = np.load(path)
    x_values = np.transpose(data[f"X_{split_name}"], (0, 2, 1))
    y_values = data[f"Y_{split_name}"]
    dataset = TensorDataset(torch.FloatTensor(x_values), torch.LongTensor(y_values))
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            logits = model(batch_x.to(device))
            preds = logits.argmax(dim=1).cpu()
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
    return correct / total if total else None

def build_model_summary(model_path: Path, metadata: Dict, split_path: Path | None, model_stamp: str) -> Dict[str, str]:
    """Create display-ready model summary values."""
    metadata_key = json.dumps(metadata, sort_keys=True)
    train_accuracy = None
    val_accuracy = None
    if split_path is not None and split_path.exists():
        try:
            train_accuracy = compute_split_accuracy(str(model_path), metadata_key, model_stamp, str(split_path), "train")
            val_accuracy = compute_split_accuracy(str(model_path), metadata_key, model_stamp, str(split_path), "val")
        except Exception:
            train_accuracy = None
            val_accuracy = None
    return {
        "model_name": metadata.get("selected_model", MODEL_DISPLAY_NAME),
        "model_type": "1D Convolutional Neural Network",
        "number_of_classes": f"5 displayed classes ({', '.join(['A'] + CNN_OUTPUT_CLASSES)}); 4 CNN outputs",
        "training_accuracy": f"{train_accuracy * 100:.2f}%" if train_accuracy is not None else "Not recorded in artifact",
        "validation_accuracy": f"{val_accuracy * 100:.2f}%" if val_accuracy is not None else "Not recorded in artifact",
        "date_trained": metadata.get("date_trained", file_modified_utc(model_path)),
        "checkpoint": str(model_path),
    }
