---
title: Solar Flare Prediction Dashboard
emoji: 🛰
colorFrom: indigo
colorTo: orange
sdk: docker
app_port: 7860
pinned: false
---

# Solar Flare Prediction Dashboard

Live GOES X-ray flux monitoring with a 1D CNN forecasting the next hour's flare class.

This is the Hugging Face Space front-matter for the project at
<https://github.com/sandilya-bit/solar-flare-prediction>. A Space **requires** the YAML
block above — it tells Hugging Face which SDK to build and which port to expose — which is
why this copy lives in `docs/` instead of overwriting the project README.

Deployed automatically by `scripts/deploy_hf.sh`, or manually:

```bash
git clone https://huggingface.co/spaces/<user>/<space-name> space
cp -r app.py config.py requirements.txt src assets data models space/
cp docs/hf/README.md space/README.md
cd space && git add -A && git commit -m "Deploy dashboard" && git push
```

The image is built from the repository `Dockerfile`, so the Streamlit version comes from
`requirements.txt` and CPU-only PyTorch is installed automatically.

Once running, the Space answers on `https://<user>-<space-name>.hf.space`, and free CPU
Basic hardware sleeps only after 48 hours of inactivity.
