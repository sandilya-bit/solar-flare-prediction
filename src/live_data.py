"""Live NOAA GOES X-ray JSON ingestion.

This module is intentionally responsible only for downloading, validating, and
parsing live NOAA JSON into the same DataFrame shape used by the existing
preprocessing and prediction pipeline: xrsa_flux and xrsb_flux indexed by time.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any
from pathlib import Path
import sys

import pandas as pd
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import NOAA_GOES_XRAY_JSON_URL, REFRESH_INTERVAL_SECONDS
from src.preprocess import clean_flux_frame

LOGGER = logging.getLogger("solar_flare_dashboard.live_data")
SHORT_ENERGY = "0.05-0.4nm"
LONG_ENERGY = "0.1-0.8nm"
REQUIRED_FIELDS = {"time_tag", "energy", "flux"}


@dataclass(frozen=True)
class LiveDataResult:
    """Result returned by the live NOAA data loader."""

    dataframe: pd.DataFrame
    connection_status: str
    message: str
    latest_timestamp: str | None
    source_url: str


class LiveDataError(RuntimeError):
    """Raised internally when NOAA data cannot be used safely."""


def _session_with_retries(retries: int, backoff_factor: float) -> requests.Session:
    """Create a requests session with retry behavior for transient failures."""
    retry_policy = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_policy)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def download_noaa_json(url: str = NOAA_GOES_XRAY_JSON_URL, timeout: int = 15, retries: int = 3) -> list[dict[str, Any]]:
    """Download NOAA GOES X-ray JSON with retries and HTTP error handling."""
    LOGGER.info("Downloading NOAA GOES X-ray JSON from %s", url)
    session = _session_with_retries(retries=retries, backoff_factor=0.6)
    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        LOGGER.error("NOAA JSON download failed: %s", exc)
        raise LiveDataError(f"NOAA request failed: {exc}") from exc
    except ValueError as exc:
        LOGGER.error("NOAA response was not valid JSON: %s", exc)
        raise LiveDataError("NOAA response was not valid JSON") from exc

    if not isinstance(payload, list) or not payload:
        LOGGER.error("NOAA response was empty or not a list")
        raise LiveDataError("NOAA response was empty or not a JSON list")

    LOGGER.info("Downloaded %d NOAA JSON records", len(payload))
    return payload


def _validate_record(record: Any, index: int) -> dict[str, Any] | None:
    """Validate a single NOAA record and return None when it must be skipped."""
    if not isinstance(record, dict):
        LOGGER.warning("Skipping NOAA record %s because it is not an object", index)
        return None
    missing = REQUIRED_FIELDS - set(record)
    if missing:
        LOGGER.warning("Skipping NOAA record %s with missing fields: %s", index, sorted(missing))
        return None
    timestamp = pd.to_datetime(record.get("time_tag"), utc=True, errors="coerce")
    if pd.isna(timestamp):
        LOGGER.warning("Skipping NOAA record %s with invalid timestamp: %s", index, record.get("time_tag"))
        return None
    try:
        flux = float(record.get("flux"))
    except (TypeError, ValueError):
        LOGGER.warning("Skipping NOAA record %s with invalid flux: %s", index, record.get("flux"))
        return None
    if flux < 0:
        LOGGER.warning("Skipping NOAA record %s with negative flux: %s", index, flux)
        return None
    energy = str(record.get("energy", "")).strip()
    if energy not in {SHORT_ENERGY, LONG_ENERGY}:
        return None
    return {
        "time": timestamp,
        "energy": energy,
        "flux": flux,
        "satellite": record.get("satellite", "GOES"),
    }


def parse_noaa_json(payload: list[dict[str, Any]]) -> pd.DataFrame:
    """Parse NOAA JSON into a clean GOES DataFrame with training-compatible columns."""
    valid_rows = []
    for index, record in enumerate(payload):
        parsed = _validate_record(record, index)
        if parsed is not None:
            valid_rows.append(parsed)

    if not valid_rows:
        LOGGER.error("NOAA JSON contained no usable X-ray flux records")
        raise LiveDataError("NOAA JSON contained no usable X-ray flux records")

    raw = pd.DataFrame(valid_rows)
    pivot = raw.pivot_table(index="time", columns="energy", values="flux", aggfunc="last")
    pivot = pivot.rename(columns={SHORT_ENERGY: "xrsa_flux", LONG_ENERGY: "xrsb_flux"})

    missing_flux_columns = {"xrsa_flux", "xrsb_flux"} - set(pivot.columns)
    if missing_flux_columns:
        LOGGER.error("NOAA JSON did not include required X-ray channels: %s", sorted(missing_flux_columns))
        raise LiveDataError(f"NOAA JSON missing X-ray channels: {sorted(missing_flux_columns)}")

    parsed_df = pivot[["xrsa_flux", "xrsb_flux"]].sort_index()
    parsed_df = parsed_df[~parsed_df.index.duplicated(keep="last")]
    clean_df = clean_flux_frame(parsed_df)
    if clean_df.empty:
        LOGGER.error("NOAA data became empty after preprocessing")
        raise LiveDataError("NOAA data became empty after preprocessing")

    LOGGER.info("Parsed NOAA data through %s", clean_df.index[-1])
    return clean_df


@st.cache_data(ttl=REFRESH_INTERVAL_SECONDS, show_spinner=False)
def cached_noaa_json_bytes(url: str) -> bytes:
    """Cache the raw NOAA JSON download for the configured refresh interval."""
    payload = download_noaa_json(url=url)
    return json.dumps(payload).encode("utf-8")


@st.cache_data(ttl=REFRESH_INTERVAL_SECONDS, show_spinner=False)
def cached_noaa_dataframe(url: str, payload_bytes: bytes) -> pd.DataFrame:
    """Cache parsed and preprocessed NOAA flux data keyed on the downloaded payload."""
    payload = json.loads(payload_bytes.decode("utf-8"))
    return parse_noaa_json(payload)


def get_live_goes_dataframe(url: str = NOAA_GOES_XRAY_JSON_URL) -> LiveDataResult:
    """Download and parse NOAA GOES JSON, returning a safe dashboard result.

    The dashboard can call this on every refresh. Failures are returned as an
    empty DataFrame with connection_status set to "unavailable" so the UI can
    keep showing the last successful prediction instead of crashing.
    """
    try:
        payload_bytes = cached_noaa_json_bytes(url)
        df = cached_noaa_dataframe(url, payload_bytes)
        LOGGER.info("NOAA download success; latest_timestamp=%s rows=%d", df.index[-1], len(df))
        return LiveDataResult(
            dataframe=df,
            connection_status="online",
            message="Live NOAA data received",
            latest_timestamp=str(df.index[-1]),
            source_url=url,
        )
    except Exception as exc:
        LOGGER.exception("Live data unavailable")
        return LiveDataResult(
            dataframe=pd.DataFrame(columns=["xrsa_flux", "xrsb_flux"]),
            connection_status="unavailable",
            message=f"Live data unavailable: {exc}",
            latest_timestamp=None,
            source_url=url,
        )
