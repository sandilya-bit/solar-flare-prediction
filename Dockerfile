# Portable image for any container host — Hugging Face Spaces (Docker SDK),
# Render, Fly.io, Railway, or a plain VM.
#
#   docker build -t solar-flare-dashboard .
#   docker run -p 7860:7860 solar-flare-dashboard
#
# The app listens on $PORT when the platform provides one (Render does), and on
# 7860 otherwise — the port Hugging Face Spaces expects.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# curl backs the healthcheck below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first, so editing code does not invalidate the wheel layer.
# requirements.txt points pip at the CPU-only PyTorch index, keeping this small.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# The dashboard appends to data/ and logs to logs/, so both must exist.
RUN mkdir -p data logs

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT:-7860}/_stcore/health" || exit 1

CMD streamlit run app.py --server.port=${PORT:-7860} --server.address=0.0.0.0
