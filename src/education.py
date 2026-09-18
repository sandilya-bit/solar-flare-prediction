"""Solar flare education: GOES class scale, live flux decoding, and a plain-language
guide to reading every panel of the dashboard.

This module is read-only reference content plus a live read-out, so a user who has
never seen an X-ray flux plot can interpret the dashboard without leaving it.
No model, data, or prediction code is touched by this module.
"""

from __future__ import annotations

import math
from typing import Mapping

import streamlit as st

from src.alerts import class_color
from src.predict import PredictionResult, observed_flux_class
from src.utils import compact_percent, format_flux

# GOES XRS-B (0.1 - 0.8 nm) class thresholds in W/m^2. Every letter is exactly one
# decade, so X-class is 10x M-class, 100x C-class, and 1000x B-class.
CLASS_SCALE: tuple[dict[str, object], ...] = (
    {
        "class": "A",
        "low": 1e-8,
        "high": 1e-7,
        "meaning": "Quiet background",
        "impact": "No operational impact. The Sun is essentially resting.",
        "shape": "Flat line at the bottom of the chart.",
    },
    {
        "class": "B",
        "low": 1e-7,
        "high": 1e-6,
        "meaning": "Quiet to slightly active",
        "impact": "Background conditions; only subflare activity.",
        "shape": "Small, slow bumps that fade within minutes.",
    },
    {
        "class": "C",
        "low": 1e-6,
        "high": 1e-5,
        "meaning": "Small flare",
        "impact": "Minor; occasional weak shortwave radio fade on the sunlit side.",
        "shape": "Clear fast rise and slower decay, often 5-30 minutes.",
    },
    {
        "class": "M",
        "low": 1e-5,
        "high": 1e-4,
        "meaning": "Medium flare",
        "impact": "Brief HF radio blackouts (R1-R2) on the dayside; sometimes a radiation storm.",
        "shape": "Steep rise over minutes, decay over tens of minutes.",
    },
    {
        "class": "X",
        "low": 1e-4,
        "high": 1e-3,
        "meaning": "Large flare",
        "impact": "Long HF blackouts (R3+), radiation storm risk, and often a coronal mass ejection.",
        "shape": "Very steep rise, long decay, can last more than an hour.",
    },
)

# NOAA / SWPC space weather scales that a flare can trigger.
SPACE_WEATHER_SCALES: tuple[tuple[str, str, str], ...] = (
    ("R1-R5", "Radio blackout", "Shortwave/ HF radio and GPS degradation on the sunlit side."),
    ("S1-S5", "Solar radiation storm", "Energetic protons; risk to satellites, astronauts, and polar flights."),
    ("G1-G5", "Geomagnetic storm", "CME-driven disturbance; aurora, grid and satellite drag effects."),
)

GLOSSARY: tuple[tuple[str, str], ...] = (
    ("Solar flare", "A sudden release of magnetic energy in the solar corona. It heats plasma and accelerates particles, producing a burst of X-rays."),
    ("Active region", "A magnetically complex patch of the photosphere/corona, usually near sunspots, where flares are born."),
    ("Magnetic reconnection", "The process that snaps and re-joins magnetic field lines, converting stored magnetic energy into flare energy."),
    ("XRS-A / XRS-B", "The two GOES X-ray Sensor channels: XRS-A is 0.05-0.4 nm (short), XRS-B is 0.1-0.8 nm (long). Classification uses XRS-B."),
    ("X-ray flux", "Energy per unit area per second in W/m^2, plotted on a logarithmic axis because flares span decades."),
    ("Flare class", "The NOAA letter scale A < B < C < M < X. Each step is 10x the flux, and the number after the letter is the multiplier (M5.0 = 5e-5 W/m^2)."),
    ("Nowcast", "What is happening right now (here: the current flux class)."),
    ("Forecast", "What is expected next. This dashboard forecasts the class for the next hour."),
    ("CME", "Coronal mass ejection: a slower eruption of plasma that can arrive in 15 minutes to 3 days and drive geomagnetic storms."),
    ("Softmax output", "The model's probability distribution over B, C, M and X outputs. It adds to 100% but is not a calibrated error rate."),
    ("Confidence", "The softmax value of the winning class. High confidence means the model prefers that class, not that it is guaranteed correct."),
    ("Sliding window", "The last 60 one-minute flux samples, both XRS channels, presented to the CNN as a 2 x 60 input."),
    ("StandardScaler", "The per-channel normalisation fitted on the training split and reused at inference so live data matches training conditions."),
)


def class_from_flux(flux: float | None) -> str:
    """Return the GOES class letter for an XRS-B flux value (delegates to predict.py)."""
    if flux is None or (isinstance(flux, float) and math.isnan(flux)):
        return "Unknown"
    return observed_flux_class(float(flux))


def noaa_label(flux: float | None) -> str:
    """Format a flux value the way NOAA does, e.g. C2.4 or M5.0."""
    if flux is None or flux <= 0:
        return "n/a"
    letter = class_from_flux(flux)
    if letter == "Unknown":
        return "n/a"
    exponent = math.floor(math.log10(float(flux)))
    multiplier = float(flux) / (10.0**exponent)
    return f"{letter}{multiplier:.1f}"


def flux_position(flux: float | None) -> float:
    """Map a flux value onto 0-100 of the logarithmic A (1e-8) to X (1e-3) scale bar."""
    if flux is None or flux <= 0:
        return 0.0
    try:
        decade = math.log10(float(flux))
    except (TypeError, ValueError):
        return 0.0
    return min(max((decade + 8.0) / 5.0 * 100.0, 0.0), 100.0)


def _chips() -> str:
    """Build the inline A-to-X legend chips."""
    return "".join(
        f'<span class="flare-chip" style="--chip:{class_color(str(item["class"]))}">{item["class"]}</span>'
        for item in CLASS_SCALE
    )


def render_scale_bar(flux: float | None = None, caption: str | None = None) -> None:
    """Render the A-to-X logarithmic scale bar, with an optional live marker."""
    zones = "".join(
        f'<span class="flare-zone" style="--zone:{class_color(str(item["class"]))}">{item["class"]}</span>'
        for item in CLASS_SCALE
    )
    marker = ""
    if flux is not None and flux > 0:
        letter = class_from_flux(flux)
        marker = (
            f'<span class="flare-marker" style="left:{flux_position(flux):.1f}%;--marker:{class_color(letter)}">'
            f'<span class="flare-marker-dot"></span>'
            f'<span class="flare-marker-label">{noaa_label(flux)}</span></span>'
        )
    axis_parts = []
    for position, label in zip((0, 20, 40, 60, 80, 100), ("10⁻⁸", "10⁻⁷", "10⁻⁶", "10⁻⁵", "10⁻⁴", "10⁻³ W/m²")):
        extra = ' class="end"' if position == 100 else ""
        axis_parts.append(f'<span{extra} style="left:{position}%">{label}</span>')
    st.markdown(
        f'<div class="flare-scale">'
        f'<div class="flare-scale-track">{zones}{marker}</div>'
        f'<div class="flare-scale-axis">{"".join(axis_parts)}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )
    if caption:
        st.caption(caption)


def render_class_table() -> None:
    """Render the GOES class reference table using the existing themed table styles."""
    rows = "".join(
        f'<tr><td><span class="flare-chip" style="--chip:{class_color(str(item["class"]))}">{item["class"]}</span></td>'
        f'<td>{float(item["low"]):.0e} - {float(item["high"]):.0e}</td>'
        f'<td>{item["meaning"]}</td><td>{item["impact"]}</td><td>{item["shape"]}</td></tr>'
        for item in CLASS_SCALE
    )
    st.markdown(
        '<div class="recent-events-table-wrapper">'
        '<table class="recent-events-table flare-class-table"><thead><tr>'
        "<th>Class</th><th>XRS-B flux (W/m²)</th><th>What it means</th>"
        "<th>Typical effect</th><th>What it looks like on the chart</th>"
        "</tr></thead><tbody>" + rows + "</tbody></table></div>",
        unsafe_allow_html=True,
    )


def render_flare_basics() -> None:
    """Compact 'new here?' primer that sits directly under the page title."""
    with st.expander("New here? Solar flares in 60 seconds", expanded=False):
        st.markdown(
            "A **solar flare** is a sudden release of magnetic energy in the Sun's corona. "
            "Stored magnetic energy snaps into a lower state (magnetic reconnection) and heats plasma to "
            "millions of degrees, which shows up as a sharp spike in X-ray emission. Because X-rays travel "
            "at light speed, the spike reaches Earth about 8 minutes later — that is the flare we measure here."
        )
        st.markdown(
            "**Why one letter tells you a lot.** NOAA ranks flares by peak X-ray flux in the 0.1–0.8 nm band "
            "(GOES XRS-B, W/m²). Each letter is exactly one decade — 10× — stronger than the last:"
        )
        st.markdown(
            f'<div class="flare-legend">{_chips()}'
            '<span class="flare-legend-note">A &lt; B &lt; C &lt; M &lt; X &nbsp;·&nbsp; each step = 10× more X-ray output</span></div>',
            unsafe_allow_html=True,
        )
        render_scale_bar(
            caption="The number after the letter is the position inside the decade: M5.0 = 5.0×10⁻⁵ W/m², "
            "about half of the way to X-class.",
        )
        st.markdown(
            "- **C-class and below** — common, usually harmless.\n"
            "- **M-class** — brief shortwave radio blackouts on the sunlit side, possible radiation storm.\n"
            "- **X-class** — long radio blackouts, radiation storm risk, and often a coronal mass ejection "
            "(a slower plasma cloud that can reach Earth in 15 minutes to 3 days and drive geomagnetic storms)."
        )
        st.markdown(
            "**What this dashboard predicts.** The CNN reads the last 60 minutes of both GOES X-ray channels "
            "(XRS-A and XRS-B, one sample per minute) and forecasts the flare class for the **next hour**. "
            "The cards at the top separate what is *observed now* from what is *forecast*, so you can always tell "
            "facts from predictions."
        )
        st.caption(
            "Want the full walkthrough with a flux decoder, panel-by-panel guide and glossary? "
            "Open the **Solar Flare 101** tab at the bottom of the page."
        )


def _render_guide_cards() -> None:
    """Panel-by-panel reading guide."""
    cards = (
        (
            "Current Flux",
            "Latest observed XRS-B value",
            "The most recent one-minute GOES measurement. The letter here is derived from NOAA thresholds, "
            "not from the model — it is the ground truth for right now.",
        ),
        (
            "Predicted Flare Class",
            "CNN forecast, next hour",
            "The class the CNN expects for the coming hour, with its softmax confidence. An A display means the "
            "model's B output coincided with A-level background, so a forecast has to be judged together with the observed class.",
        ),
        (
            "M/X Forecast Watch",
            "M-class + X-class probability",
            "The model's combined M and X probability for its forecast window. Treat it as a watch level: "
            "anything above ~30% is worth operator attention.",
        ),
        (
            "Status Panel",
            "Feed and model health",
            "Connection status, which data source is in use, satellite, last NOAA update, model file, and the "
            "refresh countdown. If the feed drops, the dashboard falls back to the last successful prediction.",
        ),
        (
            "Live X-ray Flux chart",
            "Log-scale history",
            "Long X-ray flux on a logarithmic axis with dotted C, M and X thresholds. A flare is a sharp rise of "
            "at least one order of magnitude; a slow drift is just the solar cycle waking up.",
        ),
        (
            "Recent Flare Events",
            "The model's own log",
            "One row per prediction written to data/prediction_history.csv. These are model outputs, not confirmed "
            "NOAA flare detections — use it to audit how often the model changes its mind.",
        ),
    )
    for start in range(0, len(cards), 3):
        columns = st.columns(3)
        for column, (title, subtitle, body) in zip(columns, cards[start : start + 3]):
            with column:
                st.markdown(
                    f'<div class="dashboard-card guide-card"><div class="small-label">{subtitle}</div>'
                    f'<div class="guide-title">{title}</div>'
                    f'<div class="guide-body">{body}</div></div>',
                    unsafe_allow_html=True,
                )


def _render_glossary() -> None:
    """Render the terminology glossary as a themed two-column table."""
    rows = "".join(f"<tr><td><strong>{term}</strong></td><td>{meaning}</td></tr>" for term, meaning in GLOSSARY)
    st.markdown(
        '<div class="recent-events-table-wrapper">'
        '<table class="recent-events-table guide-glossary"><thead><tr>'
        "<th>Term</th><th>Plain-language meaning</th></tr></thead><tbody>" + rows + "</tbody></table></div>",
        unsafe_allow_html=True,
    )


def _render_space_weather_scales() -> None:
    """Explain what a flare can trigger downstream."""
    rows = "".join(f"<tr><td>{code}</td><td>{name}</td><td>{detail}</td></tr>" for code, name, detail in SPACE_WEATHER_SCALES)
    st.markdown(
        '<div class="recent-events-table-wrapper">'
        '<table class="recent-events-table"><thead><tr>'
        "<th>NOAA scale</th><th>Hazard</th><th>What it affects</th></tr></thead><tbody>" + rows + "</tbody></table></div>",
        unsafe_allow_html=True,
    )


def _render_flux_decoder(current_flux: float | None) -> None:
    """Interactive 'decode any flux value' slider."""
    default_log = -6.0
    if current_flux and current_flux > 0:
        default_log = min(max(round(math.log10(float(current_flux)), 2), -8.0), -3.0)
    log_value = st.slider(
        "Slide to any X-ray flux level (log scale)",
        min_value=-8.0,
        max_value=-3.0,
        value=float(default_log),
        step=0.01,
        key="learn_flux_slider",
        help="Each +1.0 on this slider is a 10× increase in X-ray output.",
    )
    flux = 10.0**log_value
    letter = class_from_flux(flux)
    entry = next((item for item in CLASS_SCALE if item["class"] == letter), CLASS_SCALE[0])
    left, right = st.columns([1.0, 1.4])
    with left:
        st.markdown(
            f'<div class="class-card" style="background:{class_color(letter)};">'
            f'<div class="small-label" style="color:white; opacity:0.9;">Decoded class</div>'
            f'<div class="class-letter">{letter}</div>'
            f'<div class="class-caption">NOAA label {noaa_label(flux)}</div></div>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f'<div class="dashboard-card"><div class="small-label">Reading of {format_flux(flux)}</div>'
            f'<div class="big-value">{noaa_label(flux)}</div>'
            f'<div class="guide-body"><strong>{entry["meaning"]}.</strong> {entry["impact"]}</div>'
            f'<div class="guide-body" style="opacity:0.85;">On the chart: {entry["shape"]}</div></div>',
            unsafe_allow_html=True,
        )
    render_scale_bar(flux=flux, caption="The marker shows where that value sits on the full A-to-X scale.")


def render_learn_tab(result: PredictionResult | None = None, probabilities: Mapping[str, float] | None = None) -> None:
    """Render the full 'Solar Flare 101' education tab, optionally wired to a live result."""
    current_flux = float(result.current_xrsb_flux) if result is not None else None
    current_class = result.current_flux_class if result is not None else class_from_flux(current_flux)
    predicted_class = result.predicted_class if result is not None else None
    confidence = result.confidence if result is not None else None

    st.markdown("#### Start here: the 30-second version")
    st.markdown(
        "Solar flares are ranked by peak X-ray brightness on a logarithmic A-to-X scale where every letter is "
        "10× stronger than the previous one. C-class is everyday weather, M-class blackouts are noticed on the "
        "sunlit side of Earth, and X-class events can disrupt radio, satellites and power infrastructure. "
        "This dashboard measures the two GOES X-ray channels and predicts the class for the next hour."
    )

    st.markdown("#### Where the Sun is right now")
    if current_flux is not None:
        render_scale_bar(
            flux=current_flux,
            caption=f"Observed {format_flux(current_flux)} → **{current_class}-class**"
            + (f", model forecast for the next hour: **{predicted_class}-class** ({compact_percent(confidence)} confidence)." if predicted_class and confidence is not None else "."),
        )
    else:
        render_scale_bar(caption="Live flux is unavailable, so no marker is shown right now.")
    render_class_table()

    st.markdown("#### Decode any flux value")
    _render_flux_decoder(current_flux)

    st.markdown("#### How to read every panel")
    _render_guide_cards()

    st.markdown("#### What this model cannot do")
    st.markdown(
        "- **Confidence is not precision.** It is the softmax value of the winning class, and neural networks "
        "trained on imbalanced data are frequently overconfident. Read 90% confidence as \"the model strongly "
        "prefers this class\", not as a 90% hit rate.\n"
        "- **Quiet classes dominate the training data**, so B/C recall is far better than X-class recall. The rare "
        "big events are exactly the ones hardest to catch.\n"
        "- **The only inputs are X-ray flux.** No magnetograms, no EUV images, no active-region complexity — so the "
        "model cannot see the magnetic configuration that ultimately decides how big a flare becomes.\n"
        "- **Flares only.** It says nothing about coronal mass ejections, solar energetic particles or geomagnetic "
        "storms, which is where most infrastructure risk actually comes from.\n"
        "- **Feed gaps matter.** Missing NOAA minutes are forward-filled for up to 60 samples, which flattens the "
        "window and can damp the model's reaction."
    )

    st.markdown("#### What a flare can trigger")
    _render_space_weather_scales()

    st.markdown("#### Glossary")
    _render_glossary()
