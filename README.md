# Solar Flare Prediction Dashboard

A real-time solar flare prediction system using deep learning and NOAA satellite data.
https://solar-flare-prediction-xrexehawlkmvueretyrj8t.streamlit.app/ 


---

## Table of Contents

1. [Installation](#installation)
2. [Running the Dashboard](#running-the-dashboard)
3. [Project Structure](#project-structure)
4. [Switching Data Sources](#switching-data-sources)
5. [Features](#features)
6. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Steps

1. Clone or navigate to the project directory:
```bash
cd solar_flare_project
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Ensure the pre-trained model is in the `models/` directory:
```
models/solar_flare_cnn.pth
```

---

## Running the Dashboard

### Start the application:
```bash
python app.py
```

The dashboard will be available at `http://localhost:8050`

### Features on startup:
- Automatically fetches the latest NOAA data
- Loads the pre-trained CNN model
- Displays real-time predictions
- Shows connection status and data source information

---

## Project Structure

```
solar_flare_project/
│
├── app.py                  # Main application entry point (Dash)
├── preprocess.py           # Data preprocessing and normalization
├── live_data.py            # NOAA data fetching and handling
├── predict.py              # Prediction logic and inference
├── model_loader.py         # Model loading and management
├── config.py               # Configuration settings
├── requirements.txt        # Project dependencies
│
├── models/                 # Pre-trained models directory
│   └── solar_flare_cnn.pth # CNN model (not modified)
│
├── data/                   # Downloaded and processed data
│   ├── raw/               # Raw NOAA data
│   └── processed/         # Preprocessed datasets
│
└── logs/                   # Application logs
    ├── app.log            # General application logs
    ├── data_download.log  # NOAA download logs
    └── predictions.log    # Prediction results
```

---

## Switching Data Sources

### Default: NOAA Live Data
The system fetches real-time data from NOAA's GOES satellites.

### Offline Mode
If NOAA becomes unavailable:

- The dashboard displays **"Live Data Unavailable"**
- Shows the **last successful prediction**
- Automatically retries connection
- Application continues running (never crashes)

To use cached data manually:
1. Place historical data in `data/processed/`
2. Modify `live_data.py` to use local data
3. Restart the application

---

## Features

### 1. Real-Time Predictions
- Uses CNN model trained on 2021-2024 GOES data
- Processes 1-minute resolution X-ray flux data
- Returns flare class probabilities (A, B, C, M, X)

### 2. Status Panel
Displays:
- **Connection Status** - Online/Offline
- **Current Data Source** - NOAA or Local
- **Satellite Name** - GOES-16
- **Last NOAA Update** - Timestamp
- **Model Status** - Loaded/Error
- **Refresh Countdown** - Seconds until next update

### 3. Offline Mode Handling
- Graceful fallback when NOAA is unavailable
- Continues serving last successful prediction
- Automatic retry mechanism
- No application crashes

### 4. Logging
All important events are logged to `logs/`:

**Log Events:**
- ✓ NOAA download success
- ✓ NOAA download failures
- ✓ Prediction results
- ✓ Errors and warnings
- ✓ Model inference times

**Log Format:**
```
[TIMESTAMP] [LEVEL] [MODULE] Message
```

### 5. Performance Optimizations
- **Memory Usage:** Efficient data buffering and cleanup
- **Dashboard Responsiveness:** Cached predictions and instant UI updates
- **Model Inference Speed:** GPU acceleration (if available)
- **Duplicate Processing:** Caching mechanism to prevent redundant predictions

---

## Configuration

Edit `config.py` to customize:

```python
# Data refresh interval (seconds)
REFRESH_INTERVAL = 60

# NOAA API timeout (seconds)
NOAA_TIMEOUT = 30

# Model path
MODEL_PATH = "models/solar_flare_cnn.pth"

# Logging level
LOG_LEVEL = "INFO"

# Dashboard port
DASHBOARD_PORT = 8050
```

---

## Troubleshooting

### Issue: "Model not found" error
**Solution:** Ensure `models/solar_flare_cnn.pth` exists in the models directory.

### Issue: NOAA data unavailable
**Solution:** 
- Check internet connection
- Verify NOAA servers are online (https://www.ncei.noaa.gov/)
- Check logs for detailed error messages
- Application will use last successful prediction

### Issue: Dashboard not loading
**Solution:**
- Verify port 8050 is not in use
- Check firewall settings
- Restart the application

### Issue: Memory usage increasing
**Solution:**
- Restart the application
- Reduce REFRESH_INTERVAL in config.py
- Check logs for memory leaks

### Issue: Slow predictions
**Solution:**
- Ensure GPU is available (if configured)
- Reduce batch size in config.py
- Close other applications

---

## Model Information

**Architecture:** Convolutional Neural Network (CNN)

**Training Data:**
- GOES-16 XRS X-ray flux data
- Date range: 2021-2024
- Resolution: 1-minute
- Labels: NOAA flare classes (A, B, C, M, X)

**Note:** The CNN model is NOT modified in this project. It serves predictions only.

---

## Support

For issues or questions:
1. Check the logs in `logs/`
2. Review configuration in `config.py`
3. Verify all dependencies are installed
4. Check NOAA data availability

---

## License

This project extends existing solar flare prediction models with real-time dashboard capabilities.
