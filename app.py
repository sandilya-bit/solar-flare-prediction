"""Streamlit entry point for the NASA-style Solar Flare Dashboard."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

from src.alerts import alert_for_class, class_color, render_alert
from src.charts import flux_line_chart, probability_bar_chart
from config import (
    APP_DIR,
    APP_TITLE,
    DATA_CANDIDATES,
    DATA_SOURCE,
    GOES_VERSION,
    HISTORY_MINUTES,
    LAST_SUCCESS_PATH,
    METADATA_CANDIDATES,
    MODEL_CANDIDATES,
    MODEL_DISPLAY_NAME,
    NOAA_GOES_XRAY_JSON_URL,
    REFRESH_INTERVAL_SECONDS,
    SATELLITE_NAME,
    SATELLITE_STATUS,
    SCALER_CANDIDATES,
    SPLIT_DATASET_CANDIDATES,
)
from src.history import append_prediction, cached_load_history
from src.live_data import LiveDataResult, get_live_goes_dataframe
from src.model_loader import build_model_summary, load_model
from src.predict import PredictionResult, predict_flare
from src.preprocess import load_scaler, latest_window, load_goes_csv, load_latest_goes_netcdf, netcdf_cache_stamp
from src.utils import compact_percent, file_cache_stamp, file_modified_utc, find_first_existing, format_flux, format_utc, load_json, setup_logging, utc_now

# Resolve at module level — __file__ may not exist inside functions when Streamlit
# Cloud executes the script through its own loader.
_ASSETS_DIR = Path(__file__).resolve().parent / "assets" if "__file__" in dir() else APP_DIR / "assets"


def _supports_fragment_auto_refresh() -> bool:
    """Return True when native st.fragment(run_every=...) is available."""
    try:
        return "run_every" in inspect.signature(st.fragment).parameters
    except (TypeError, ValueError):
        return False


FRAGMENT_AUTO_REFRESH = _supports_fragment_auto_refresh()

st.set_page_config(page_title=APP_TITLE, page_icon="S", layout="wide", initial_sidebar_state="expanded")
LOGGER = setup_logging()
LOGGER.info("Solar Flare Dashboard module loaded")
SOURCE_LABELS = {
    "live_noaa_json": "Live NOAA JSON",
    "local_csv": "Local CSV",
    "local_netcdf": "Latest local NetCDF",
}


@st.cache_data(ttl=REFRESH_INTERVAL_SECONDS, show_spinner=False)
def cached_live_goes_data(url: str) -> LiveDataResult:
    """Cache the live NOAA request for the configured refresh interval."""
    return get_live_goes_dataframe(url=url)


def apply_theme(theme: str) -> None:
    """Apply dashboard CSS styling by reading static CSS files and injecting theme variables."""
    dark = theme == "Dark"
    background = "#08111f" if dark else "#f5f7fb"
    panel = "#101b2f" if dark else "#ffffff"
    text = "#f8f9fa" if dark else "#17202a"
    muted = "#7a90a4" if dark else "#4a5568"
    border = "#1f3047" if dark else "#cbd5e1"
    shadow = "rgba(0, 0, 0, 0.4)" if dark else "rgba(148, 163, 184, 0.15)"

    # Use the module-level _ASSETS_DIR (computed once with __file__ at import
    # time) so we never reference __file__ inside a function where it may be
    # undefined in Streamlit Cloud's execution model.
    assets_dir = _ASSETS_DIR

    def _read_css(filename: str) -> str:
        css_path = assets_dir / filename
        try:
            if css_path.exists():
                return css_path.read_text(encoding="utf-8")
        except Exception:
            LOGGER.exception("Failed reading CSS file %s", css_path)
        return ""

    style_content = _read_css("style.css")
    mobile_content = _read_css("mobile.css")

    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Inter:wght@300;400;500;600;700&display=swap');

:root {{
  --background-color: {background};
  --panel-color: {panel};
  --text-color: {text};
  --muted-color: {muted};
  --border-color: {border};
  --shadow-color: {shadow};
}}

{style_content}

{mobile_content}
</style>
""",
        unsafe_allow_html=True,
    )


def sidebar_controls() -> tuple[str, bool, str]:
    """Render sidebar controls and return user selections."""
    st.sidebar.title("Mission Controls")
    st.sidebar.caption("Solar flare nowcasting and next-hour CNN inference.")
    theme = st.sidebar.selectbox("Theme", ["Dark", "Light"], index=0)
    source_keys = list(SOURCE_LABELS)
    default_index = source_keys.index(DATA_SOURCE) if DATA_SOURCE in source_keys else 0
    selected_label = st.sidebar.radio("Data source", [SOURCE_LABELS[key] for key in source_keys], index=default_index)
    selected_source = next(key for key, label in SOURCE_LABELS.items() if label == selected_label)
    refresh_clicked = st.sidebar.button("Refresh now", type="primary")
    st.sidebar.caption(f"Auto-refresh every {REFRESH_INTERVAL_SECONDS} seconds (no manual action required).")
    st.sidebar.divider()
    st.sidebar.subheader("Project Information")
    st.sidebar.write("Model-driven monitoring dashboard for GOES X-ray flux.")
    st.sidebar.write(f"Model: {MODEL_DISPLAY_NAME}")
    st.sidebar.write("Dataset: GOES XRS 1-minute flux windows")
    st.sidebar.write(f"Satellite: {GOES_VERSION}")
    st.sidebar.write(f"History window: {HISTORY_MINUTES} minutes")
    return theme, refresh_clicked, selected_source


@st.cache_data(show_spinner=False)
def resolve_inputs_cached() -> tuple[str, str, str, str | None]:
    """Find and cache model, metadata, GOES fallback data, and scaler paths."""
    model_path = find_first_existing(MODEL_CANDIDATES)
    data_path = find_first_existing(DATA_CANDIDATES)
    scaler_path = find_first_existing(SCALER_CANDIDATES)
    metadata_path = find_first_existing(METADATA_CANDIDATES)
    if model_path is None:
        raise FileNotFoundError("No CNN model found. Expected flare_model.pth or solar_flare_cnn.pth.")
    if data_path is None:
        raise FileNotFoundError("No fallback GOES CSV data found. Expected goes_xrs_1min_clean.csv.")
    if scaler_path is None:
        raise FileNotFoundError(
            "Scaler not found (models/scaler.pkl). "
            "Run: python scripts/build_scaler.py"
        )
    return str(model_path), str(data_path), str(scaler_path), str(metadata_path) if metadata_path else None


@st.cache_data(show_spinner=False)
def load_metadata_cached(metadata_path: str | None, file_stamp: str) -> dict:
    """Load model metadata JSON with file-stamp cache invalidation."""
    if not metadata_path:
        return {}
    return load_json(Path(metadata_path))


@st.cache_resource(show_spinner=False)
def load_inference_bundle(
    model_path: str,
    scaler_path: str,
    metadata_path: str | None,
    model_stamp: str,
    scaler_stamp: str,
    metadata_stamp: str,
) -> tuple:
    """Load and cache the CNN, scaler, and metadata used for inference."""
    metadata = load_metadata_cached(metadata_path, metadata_stamp)
    metadata_key = json.dumps(metadata, sort_keys=True)
    model, device = load_model(model_path, metadata_key, model_stamp)
    scaler = load_scaler(scaler_path, scaler_stamp)
    LOGGER.info("Inference bundle ready; model=%s scaler=%s device=%s", model_path, scaler_path, device)
    return model, device, scaler, metadata, metadata_key


def resolve_inputs() -> tuple[Path, Path, Path, Path | None]:
    """Find model, metadata, GOES fallback data, and pre-fitted scaler."""
    model_path, data_path, scaler_path, metadata_path = resolve_inputs_cached()
    return Path(model_path), Path(data_path), Path(scaler_path), Path(metadata_path) if metadata_path else None


def load_selected_source(source: str, fallback_data_path: Path) -> tuple[pd.DataFrame, str, str, str | None]:
    """Load the selected data source and return frame, status, message, latest timestamp."""
    if source == "live_noaa_json":
        live_result = cached_live_goes_data(NOAA_GOES_XRAY_JSON_URL)
        return live_result.dataframe, live_result.connection_status, live_result.message, live_result.latest_timestamp
    if source == "local_netcdf":
        data_dir = str(fallback_data_path.parent)
        df, source_label = load_latest_goes_netcdf(data_dir, netcdf_cache_stamp(data_dir))
        return df, "offline-fallback", f"Using local NetCDF: {source_label}", str(df.index[-1]) if not df.empty else None
    csv_stamp = file_cache_stamp(fallback_data_path)
    df = load_goes_csv(str(fallback_data_path), csv_stamp)
    return df, "offline-fallback", f"Using local CSV: {fallback_data_path}", str(df.index[-1]) if not df.empty else None


def render_metric_card(label: str, value: str, detail: str = "") -> None:
    """Render a compact monitoring card."""
    st.markdown(
        f"""<div class="dashboard-card"><div class="small-label">{label}</div><div class="big-value">{value}</div><div style="margin-top:8px; opacity:0.78;">{detail}</div></div>""",
        unsafe_allow_html=True,
    )


def render_class_card(flare_class: str, confidence: float) -> None:
    """Render the predicted flare class in a color-coded card."""
    color = class_color(flare_class)
    st.markdown(
        f"""<div class="class-card" style="background:{color};"><div class="small-label" style="color:white; opacity:0.9;">Predicted Flare Class</div><div class="class-letter">{flare_class}</div><div class="class-caption">Confidence {compact_percent(confidence)}</div></div>""",
        unsafe_allow_html=True,
    )


def render_forecast_probability(flare_class: str, probabilities: dict[str, float], confidence: float) -> None:
    """Render the 6-hour forecast probability card."""
    m_x_prob = (probabilities.get("M", 0.0) + probabilities.get("X", 0.0)) * 100
    st.markdown(
        f"""<div class="forecast-card">
        <div class="forecast-title">6-Hour Forecast Probability</div>
        <div style="font-size: 0.95rem; color: #9fb3c8; margin-bottom: 8px;">Probability of M/X-class Flare</div>
        <div class="forecast-value">{m_x_prob:.0f}%</div>
        <div style="width: 100%; height: 8px; background: #263a59; border-radius: 4px; margin: 16px 0; overflow: hidden;">
            <div style="width: {min(m_x_prob, 100)}%; height: 100%; background: linear-gradient(90deg, #ff922b, #e03131); border-radius: 4px;"></div>
        </div>
        <div style="font-size: 0.85rem; color: #9fb3c8;">Based on current flux state: {flare_class}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_recent_events() -> None:
    """Render the recent flare events table."""
    history_df = cached_load_history()
    if history_df.empty:
        st.info("Recent events will appear after predictions are generated.")
        return
    
    display_df = history_df.copy()
    display_df["Timestamp"] = pd.to_datetime(display_df["Timestamp"], utc=True, errors="coerce")
    display_df = display_df.sort_values("Timestamp", ascending=False).head(10)
    
    html_table = '<div class="recent-events-table-wrapper">'
    html_table += '<table class="recent-events-table"><thead><tr>'
    html_table += '<th>Time (UTC)</th><th>Class</th><th>Peak Flux (W/m²)</th><th>Confidence</th></tr></thead><tbody>'
    
    for _, row in display_df.iterrows():
        time_str = pd.Timestamp(row["Timestamp"]).strftime("%Y-%m-%d %H:%M")
        flux_str = f"{float(row['Flux']):.2e}" if pd.notna(row["Flux"]) else "—"
        conf_str = f"{float(row['Confidence']) * 100:.1f}%" if pd.notna(row["Confidence"]) else "—"
        pred = str(row["Prediction"])
        color = class_color(pred)
        html_table += f'<tr><td>{time_str}</td><td><span style="background: {color}; color: white; padding: 2px 8px; border-radius: 4px; font-weight: bold;">{pred}</span></td><td>{flux_str}</td><td>{conf_str}</td></tr>'
    
    html_table += '</tbody></table>'
    html_table += '</div>'
    st.markdown(html_table, unsafe_allow_html=True)


def render_status_panel(
    connection_status: str,
    data_source: str,
    latest_timestamp: str | None,
    model_path: Path,
    model_loaded: bool,
) -> None:
    """Render the monitoring status panel requested by operations."""
    source_label = SOURCE_LABELS.get(data_source, data_source)
    last_noaa_update = latest_timestamp if data_source == "live_noaa_json" and latest_timestamp else "Not using live NOAA"
    model_status = f"Loaded: {model_path.name}" if model_loaded else "Unavailable"
    refresh_countdown = f"Next automatic check in up to {REFRESH_INTERVAL_SECONDS}s"
    st.subheader("Status Panel")
    c1, c2, c3 = st.columns(3)
    c4, c5, c6 = st.columns(3)
    c1.metric("Connection Status", connection_status)
    c2.metric("Current Data Source", source_label)
    c3.metric("Satellite Name", SATELLITE_NAME)
    c4.metric("Last NOAA Update", str(last_noaa_update))
    c5.metric("Model Status", model_status)
    c6.metric("Refresh Countdown", refresh_countdown)


def _prediction_to_payload(df: pd.DataFrame, result: PredictionResult, saved_at: str) -> dict:
    """Serialize the last successful prediction and chart window for offline use."""
    chart_df = df.tail(360).copy()
    chart_df.index = chart_df.index.astype(str)
    return {
        "saved_at": saved_at,
        "chart_data": chart_df.reset_index(names="time").to_dict(orient="records"),
        "prediction": {
            "timestamp": str(result.timestamp),
            "window_start": str(result.window_start),
            "window_end": str(result.window_end),
            "current_xrsa_flux": result.current_xrsa_flux,
            "current_xrsb_flux": result.current_xrsb_flux,
            "current_flux_class": result.current_flux_class,
            "predicted_class": result.predicted_class,
            "raw_cnn_class": result.raw_cnn_class,
            "confidence": result.confidence,
            "probabilities": result.probabilities,
        },
    }


def _payload_to_prediction(payload: dict) -> tuple[pd.DataFrame, PredictionResult, str]:
    """Restore the persisted last successful prediction from disk."""
    prediction = payload["prediction"]
    chart_df = pd.DataFrame(payload.get("chart_data", []))
    if chart_df.empty:
        chart_df = pd.DataFrame(
            [{"time": prediction["window_end"], "xrsa_flux": prediction["current_xrsa_flux"], "xrsb_flux": prediction["current_xrsb_flux"]}]
        )
    chart_df["time"] = pd.to_datetime(chart_df["time"], utc=True, errors="coerce")
    chart_df = chart_df.dropna(subset=["time"]).set_index("time")
    result = PredictionResult(
        timestamp=pd.Timestamp(prediction["timestamp"]),
        window_start=pd.Timestamp(prediction["window_start"]),
        window_end=pd.Timestamp(prediction["window_end"]),
        current_xrsa_flux=float(prediction["current_xrsa_flux"]),
        current_xrsb_flux=float(prediction["current_xrsb_flux"]),
        current_flux_class=str(prediction["current_flux_class"]),
        predicted_class=str(prediction["predicted_class"]),
        raw_cnn_class=str(prediction["raw_cnn_class"]),
        confidence=float(prediction["confidence"]),
        probabilities={key: float(value) for key, value in prediction["probabilities"].items()},
    )
    return chart_df, result, str(payload.get("saved_at", "unknown time"))


def save_last_success(df: pd.DataFrame, result: PredictionResult) -> None:
    """Keep the last successful prediction in memory and on disk for NOAA outages."""
    saved_at = format_utc(utc_now())
    st.session_state["last_success_df"] = df
    st.session_state["last_success_prediction"] = result
    st.session_state["last_success_time"] = saved_at
    try:
        LAST_SUCCESS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LAST_SUCCESS_PATH.open("w", encoding="utf-8") as file:
            json.dump(_prediction_to_payload(df, result, saved_at), file, indent=2)
        LOGGER.info("Persisted last successful prediction to %s", LAST_SUCCESS_PATH)
    except Exception:
        LOGGER.exception("Failed to persist last successful prediction")


def load_last_success() -> tuple[pd.DataFrame | None, PredictionResult | None, str | None]:
    """Read the last successful prediction from memory or disk."""
    memory_df = st.session_state.get("last_success_df")
    memory_result = st.session_state.get("last_success_prediction")
    memory_time = st.session_state.get("last_success_time")
    if memory_df is not None and memory_result is not None:
        return memory_df, memory_result, memory_time
    if not LAST_SUCCESS_PATH.exists():
        return None, None, None
    try:
        with LAST_SUCCESS_PATH.open("r", encoding="utf-8") as file:
            df, result, saved_at = _payload_to_prediction(json.load(file))
        st.session_state["last_success_df"] = df
        st.session_state["last_success_prediction"] = result
        st.session_state["last_success_time"] = saved_at
        return df, result, saved_at
    except Exception:
        LOGGER.exception("Failed to load persisted last successful prediction")
        return None, None, None


def render_live_dashboard(data_source: str) -> None:
    """Fetch live data, run inference, and render all monitoring panels."""
    top_a, top_b, top_c, top_d = st.columns(4)
    top_b.write(f"Current UTC Time: {format_utc(utc_now())}")

    try:
        model_path, fallback_data_path, scaler_path, metadata_path = resolve_inputs()
        model_stamp = file_cache_stamp(model_path)
        scaler_stamp = file_cache_stamp(scaler_path)
        metadata_stamp = file_cache_stamp(metadata_path)

        model, device, scaler, metadata, metadata_key = load_inference_bundle(
            str(model_path),
            str(scaler_path),
            str(metadata_path) if metadata_path else None,
            model_stamp,
            scaler_stamp,
            metadata_stamp,
        )

        df, connection_status, connection_message, latest_timestamp = load_selected_source(data_source, fallback_data_path)
        LOGGER.info("Connection status: %s | %s", connection_status, connection_message)

        using_cached_prediction = False
        if df.empty:
            last_df, last_result, last_success_time = load_last_success()
            if last_df is None or last_result is None:
                raise RuntimeError("Live data unavailable and no previous prediction is available yet.")
            df = last_df
            result = last_result
            using_cached_prediction = True
            LOGGER.warning("Live data unavailable; using last successful prediction from %s", last_success_time)
            st.error("Live Data Unavailable")
            st.warning(f"Showing last successful prediction from {last_success_time}. Retrying automatically.")
        else:
            prediction_key = f"{data_source}:{latest_timestamp or df.index[-1]}"
            cached_key = st.session_state.get("last_prediction_key")
            cached_result = st.session_state.get("last_success_prediction")
            if cached_key == prediction_key and cached_result is not None:
                result = cached_result
                if not LAST_SUCCESS_PATH.exists():
                    save_last_success(df, result)
                LOGGER.info("Reusing cached prediction for unchanged data (%s)", prediction_key)
            else:
                window = latest_window(df)
                result = predict_flare(model, device, window, scaler)
                append_prediction(result)
                save_last_success(df, result)
                st.session_state["last_prediction_key"] = prediction_key
                LOGGER.info(
                    "Prediction %s confidence %.4f flux %.6e",
                    result.predicted_class,
                    result.confidence,
                    result.current_xrsb_flux,
                )

        status_dot = "status-dot" if connection_status == "online" else "status-dot-offline"
        top_a.markdown(f"<span class='{status_dot}'></span><strong>Satellite Status:</strong> {SATELLITE_STATUS}", unsafe_allow_html=True)
        top_c.write(f"Last Updated: {format_utc(utc_now()) if connection_status == 'online' else 'Live data unavailable'}")
        top_d.write(f"Latest Timestamp: {latest_timestamp or result.window_end}")

        render_status_panel(connection_status, data_source, latest_timestamp, model_path, True)
        st.caption(f"{connection_message} | Source detail: {NOAA_GOES_XRAY_JSON_URL if data_source == 'live_noaa_json' else fallback_data_path}")

        flux_col, class_col, conf_col = st.columns([1.15, 1.0, 1.0])
        with flux_col:
            render_metric_card("Current Flux", format_flux(result.current_xrsb_flux), f"Observed class {result.current_flux_class}; Long X-ray Flux")
        with class_col:
            render_class_card(result.predicted_class, result.confidence)
        with conf_col:
            render_metric_card("Prediction Confidence", compact_percent(result.confidence), "CNN softmax confidence")
            st.progress(min(max(result.confidence, 0.0), 1.0))

        if using_cached_prediction:
            st.caption("Prediction, chart, and alert are based on the last successful live update.")

        # Two-column layout: Charts on left, Forecast + Alert on right
        chart_col, forecast_col = st.columns([1.5, 1.0])
        
        with chart_col:
            st.subheader("Live X-ray Flux (Last 24 Hours)")
            chart_window = df.tail(360)
            st.plotly_chart(flux_line_chart(chart_window), width="stretch")
            st.caption(f"Auto-updates every {REFRESH_INTERVAL_SECONDS} seconds when new NOAA data is available.")
        
        with forecast_col:
            render_forecast_probability(result.current_flux_class, result.probabilities, result.confidence)
            st.markdown("<br>", unsafe_allow_html=True)
            render_alert(alert_for_class(result.predicted_class))
        
        # Recent events section
        st.subheader("Recent Flare Events")
        render_recent_events()

        tab_live, tab_history, tab_model, tab_pipeline = st.tabs(["Probability Distribution", "Prediction History", "Model Summary", "Data Pipeline"])
        with tab_live:
            st.plotly_chart(probability_bar_chart(result.probabilities, result.predicted_class), width="stretch")
            st.caption("A-class display is derived from background B output when observed flux is below B threshold.")

        with tab_history:
            history_df = cached_load_history()
            if not history_df.empty:
                display_df = history_df.copy()
                display_df["Flux"] = display_df["Flux"].map(lambda value: f"{float(value):.3e}" if pd.notna(value) else "")
                display_df["Confidence"] = display_df["Confidence"].map(lambda value: f"{float(value) * 100:.1f}%" if pd.notna(value) else "")
                st.dataframe(display_df.sort_values("Timestamp", ascending=False).tail(50), width="stretch", hide_index=True)
            else:
                st.info("Prediction history will appear after the first successful dashboard run.")

        with tab_model:
            split_path = find_first_existing(SPLIT_DATASET_CANDIDATES)
            summary = build_model_summary(model_path, metadata, split_path, model_stamp)
            c1, c2, c3 = st.columns(3)
            c1.metric("Model Type", summary["model_type"])
            c2.metric("Training Accuracy", summary["training_accuracy"])
            c3.metric("Validation Accuracy", summary["validation_accuracy"])
            st.json({"model_name": summary["model_name"], "number_of_classes": summary["number_of_classes"], "date_trained": summary["date_trained"], "checkpoint": summary["checkpoint"], "data_source": SOURCE_LABELS.get(data_source, data_source), "fallback_data_last_modified": file_modified_utc(fallback_data_path)})
            with st.expander("Model metadata"):
                st.json(metadata if metadata else {"metadata": "No flare_model_metadata.json found; using default CNN architecture."})

        with tab_pipeline:
            st.status("NOAA Live JSON -> Download Latest Data -> Validate Response -> Extract Latest Flux -> Preprocess -> Normalize -> Load CNN -> Predict -> Dashboard -> Alert -> Save History -> Repeat", state="complete")
            st.write("1. Download NOAA GOES X-ray JSON with retries and HTTP error handling.")
            st.write("2. Validate JSON fields, timestamps, flux values, duplicates, and empty responses.")
            st.write("3. Convert short and long X-ray channels to the existing xrsa_flux/xrsb_flux schema.")
            st.write("4. Reuse the existing preprocessing and training scaler before inference.")
            st.write("5. Cache model loading, save prediction history, and refresh automatically.")

    except Exception as exc:
        LOGGER.exception("Dashboard failed")
        st.error(f"Dashboard could not load: {exc}")
        last_df, last_result, last_success_time = load_last_success()
        if last_result is not None:
            st.warning(f"Showing last successful prediction from {last_success_time}.")
            model_path = find_first_existing(MODEL_CANDIDATES) or Path("model unavailable")
            LOGGER.warning("Dashboard exception fallback using last successful prediction from %s", last_success_time)
            render_status_panel("unavailable", st.session_state.get("dashboard_data_source", DATA_SOURCE), str(last_result.window_end), model_path, model_path.exists())
            render_metric_card("Current Flux", format_flux(last_result.current_xrsb_flux), "Cached live result")
            render_class_card(last_result.predicted_class, last_result.confidence)
            render_alert(alert_for_class(last_result.predicted_class))
        else:
            LOGGER.error("Live data unavailable and no cached prediction exists")
            st.error("Live Data Unavailable")
            st.info("No previous prediction is available yet. The dashboard will retry automatically on the next refresh.")


def main() -> None:
    """Render the full live monitoring dashboard."""
    theme, refresh_clicked, data_source = sidebar_controls()
    apply_theme(theme)
    st.session_state["dashboard_data_source"] = data_source
    LOGGER.info("Dashboard render requested; data_source=%s", data_source)

    if refresh_clicked:
        st.cache_data.clear()
        LOGGER.info("Manual refresh requested; cache cleared.")

    st.markdown(f"# ☀️ {APP_TITLE}")
    st.markdown("<div class='subtitle'>Aditya-L1 SoLEXS</div>", unsafe_allow_html=True)

    active_source = st.session_state["dashboard_data_source"]

    if FRAGMENT_AUTO_REFRESH:

        @st.fragment(run_every=REFRESH_INTERVAL_SECONDS)
        def auto_refresh_live_monitor() -> None:
            LOGGER.debug("Auto-refresh fragment rerun started.")
            render_live_dashboard(st.session_state.get("dashboard_data_source", active_source))

        auto_refresh_live_monitor()
        return

    if st_autorefresh is not None:
        st_autorefresh(interval=REFRESH_INTERVAL_SECONDS * 1000, key="goes_dashboard_refresh")
    else:
        st.sidebar.warning(
            "Install streamlit>=1.37 or streamlit-autorefresh for automatic refresh. "
            "Use 'Refresh now' until then."
        )
    render_live_dashboard(active_source)


if __name__ == "__main__":
    main()
