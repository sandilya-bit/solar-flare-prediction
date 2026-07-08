"""Alert policy and Streamlit alert rendering."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st


@dataclass(frozen=True)
class Alert:
    """Dashboard alert payload for a predicted flare class."""

    level: str
    icon: str
    title: str
    message: str


def alert_for_class(flare_class: str) -> Alert:
    """Map flare classes to operational alert levels and messages."""
    if flare_class in {"A", "B"}:
        return Alert(
            "success",
            "[OK]",
            "Quiet Solar Conditions",
            "A/B-class activity predicted. No significant flare activity is expected in the next hour.",
        )
    if flare_class == "C":
        return Alert(
            "info",
            "[INFO]",
            "Elevated Solar Activity",
            "C-class activity is possible. Continue monitoring long X-ray flux trends.",
        )
    if flare_class == "M":
        return Alert(
            "warning",
            "[WARN]",
            "Strong Solar Flare Possible",
            "M-class flare conditions detected. Review operational readiness and continue close monitoring.",
        )
    if flare_class == "X":
        return Alert(
            "error",
            "[ALERT]",
            "Extreme Solar Flare Detected",
            "X-class flare risk detected. Escalate space weather monitoring immediately.",
        )
    return Alert(
        "info",
        "[INFO]",
        "Unknown Flare State",
        "Prediction class was not recognized. Continue monitoring the live NOAA feed.",
    )


def class_color(flare_class: str) -> str:
    """Return a dashboard color for a flare class."""
    return {"A": "#2fb344", "B": "#2fb344", "C": "#1c7ed6", "M": "#f08c00", "X": "#e03131"}.get(flare_class, "#6c757d")


def render_alert(alert: Alert) -> None:
    """Render the alert with Streamlit's native message components."""
    text = f"{alert.icon} {alert.title}: {alert.message}"
    if alert.level == "success":
        st.success(text)
    elif alert.level == "warning":
        st.warning(text)
    elif alert.level == "error":
        st.error(text)
    else:
        st.info(text)
