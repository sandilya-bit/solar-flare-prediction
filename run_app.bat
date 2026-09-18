@echo off
cd /d "%~dp0"
echo Starting Solar Flare Prediction Dashboard...
if exist ".venv\Scripts\streamlit.exe" (
    start http://localhost:8501
    ".venv\Scripts\streamlit.exe" run app.py
) else (
    echo Virtual environment not found!
    pause
)
