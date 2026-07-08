"""Plotly chart builders for the monitoring dashboard."""
from __future__ import annotations
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.alerts import class_color
from config import CLASS_NAMES

def flux_line_chart(df: pd.DataFrame) -> go.Figure:
    """Create a continuously updating long X-ray flux chart."""
    fig = go.Figure()
    if df.empty:
        fig.update_layout(
            height=360,
            template="plotly_dark",
            title="Live GOES Long X-ray Flux",
            annotations=[dict(text="No live flux data available", x=0.5, y=0.5, showarrow=False, xref="paper", yref="paper")],
        )
        return fig

    latest_time = df.index[-1]
    latest_flux = df["xrsb_flux"].iloc[-1]
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["xrsb_flux"],
            name="Long X-ray Flux (0.1-0.8 nm)",
            mode="lines",
            line=dict(color="#ff922b", width=2.5),
            hovertemplate="%{x}<br>Long flux %{y:.3e} W/m^2<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[latest_time],
            y=[latest_flux],
            name="Latest NOAA point",
            mode="markers",
            marker=dict(color="#ffffff", size=9, line=dict(color="#ff922b", width=3)),
            hovertemplate="Latest %{x}<br>%{y:.3e} W/m^2<extra></extra>",
        )
    )

    threshold_lines = [("C", 1e-6, "#1c7ed6"), ("M", 1e-5, "#f08c00"), ("X", 1e-4, "#e03131")]
    for label, value, color in threshold_lines:
        fig.add_hline(
            y=value,
            line_width=1,
            line_dash="dot",
            line_color=color,
            annotation_text=f"{label}-class",
            annotation_position="top left",
        )

    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=35, b=10),
        template="plotly_dark",
        title="Live GOES Long X-ray Flux",
        xaxis_title="Time",
        yaxis_title="Long X-ray Flux (W/m^2)",
        yaxis_type="log",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig

def probability_bar_chart(probabilities: dict[str, float], predicted_class: str) -> go.Figure:
    """Create a horizontal class probability chart and highlight the prediction."""
    values = [probabilities.get(name, 0.0) * 100 for name in CLASS_NAMES]
    colors = [class_color(name) if name == predicted_class else "#536579" for name in CLASS_NAMES]
    line_widths = [3 if name == predicted_class else 0 for name in CLASS_NAMES]
    labels = [f"{name} - {value:.1f}%" if name == predicted_class else f"{value:.1f}%" for name, value in zip(CLASS_NAMES, values)]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=CLASS_NAMES,
            orientation="h",
            marker=dict(color=colors, line=dict(color="#ffffff", width=line_widths)),
            text=labels,
            textposition="auto",
            hovertemplate="Class %{y}<br>Probability %{x:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=35, b=10),
        template="plotly_dark",
        title=f"Class Probability Distribution - Predicted {predicted_class}",
        xaxis_title="Probability (%)",
        yaxis_title="Flare Class",
        xaxis=dict(range=[0, 100]),
        showlegend=False,
    )
    return fig
