"""
Module 8: MD Visualization
Generates plots for trajectory analysis results.
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")


def rmsd_time_series(rmsd_df, compound_name="Compound"):
    """Plot RMSD time series (ligand + protein).

    Args:
        rmsd_df: DataFrame with 'time_ps', 'protein_rmsd_A', optionally 'ligand_rmsd_A'

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    time_ns = rmsd_df["time_ps"] / 1000  # Convert to ns

    if "protein_rmsd_A" in rmsd_df.columns:
        fig.add_trace(go.Scatter(
            x=time_ns, y=rmsd_df["protein_rmsd_A"],
            mode="lines", name="Protein Backbone",
            line=dict(color="#3498db", width=1.5),
        ))

    if "ligand_rmsd_A" in rmsd_df.columns:
        fig.add_trace(go.Scatter(
            x=time_ns, y=rmsd_df["ligand_rmsd_A"],
            mode="lines", name="Ligand",
            line=dict(color="#e74c3c", width=1.5),
        ))

    # Stability threshold
    fig.add_hline(y=3.0, line_dash="dash", line_color="gray",
                  annotation_text="Stability threshold (3.0 A)")

    fig.update_layout(
        title=f"RMSD Time Series: {compound_name}",
        xaxis_title="Time (ns)",
        yaxis_title="RMSD (A)",
        height=400,
        yaxis_range=[0, max(5.0, rmsd_df.get("ligand_rmsd_A", rmsd_df["protein_rmsd_A"]).max() * 1.2)],
    )

    return fig


def rmsf_bar_chart(rmsf_df, compound_name="Compound", binding_site_range=(100, 120)):
    """Per-residue RMSF bar chart with binding site annotation.

    Args:
        rmsf_df: DataFrame with 'residue' and 'rmsf_A'
        binding_site_range: Residue range for binding site highlighting

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    colors = []
    for _, row in rmsf_df.iterrows():
        if binding_site_range[0] <= row["residue"] <= binding_site_range[1]:
            colors.append("#e74c3c")  # Red for binding site
        else:
            colors.append("#3498db")  # Blue for other residues

    fig.add_trace(go.Bar(
        x=rmsf_df["residue"],
        y=rmsf_df["rmsf_A"],
        marker_color=colors,
        hovertemplate="Residue %{x}<br>RMSF: %{y:.2f} A<extra></extra>",
    ))

    # Highlight binding site region
    fig.add_vrect(
        x0=binding_site_range[0], x1=binding_site_range[1],
        fillcolor="rgba(231, 76, 60, 0.1)", line_width=0,
        annotation_text="Binding Site",
    )

    fig.update_layout(
        title=f"Per-Residue RMSF: {compound_name}",
        xaxis_title="Residue Number",
        yaxis_title="RMSF (A)",
        height=400,
    )

    return fig


def hbond_occupancy_chart(hbond_data, compound_name="Compound"):
    """H-bond occupancy bar chart.

    Args:
        hbond_data: dict or float with occupancy data

    Returns:
        Plotly figure
    """
    if isinstance(hbond_data, (int, float)):
        # Simple single-value occupancy
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Ligand-Protein"],
            y=[hbond_data * 100],
            marker_color="#2ecc71" if hbond_data > 0.5 else "#e74c3c",
            text=[f"{hbond_data*100:.1f}%"],
            textposition="outside",
        ))

        fig.add_hline(y=50, line_dash="dash", line_color="gray",
                      annotation_text="50% threshold")

        fig.update_layout(
            title=f"H-bond Occupancy: {compound_name}",
            yaxis_title="Occupancy (%)",
            yaxis_range=[0, 100],
            height=350,
        )
        return fig

    # Multiple H-bond data
    fig = go.Figure()
    names = list(hbond_data.keys())
    values = [v * 100 for v in hbond_data.values()]

    fig.add_trace(go.Bar(
        x=names, y=values,
        marker_color=["#2ecc71" if v > 50 else "#e74c3c" for v in values],
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
    ))

    fig.update_layout(
        title=f"H-bond Occupancy: {compound_name}",
        yaxis_title="Occupancy (%)",
        yaxis_range=[0, 100],
        height=350,
    )

    return fig


def binding_stability_gauge(stability, compound_name="Compound"):
    """Gauge chart showing binding stability percentage.

    Args:
        stability: float 0-1

    Returns:
        Plotly figure
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=stability * 100,
        title={"text": f"Binding Stability: {compound_name}"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#2ecc71" if stability > 0.8 else "#e74c3c"},
            "steps": [
                {"range": [0, 50], "color": "#ffcccc"},
                {"range": [50, 80], "color": "#ffffcc"},
                {"range": [80, 100], "color": "#ccffcc"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 2},
                "thickness": 0.75,
                "value": 80,
            },
        },
        number={"suffix": "%"},
    ))

    fig.update_layout(height=300)
    return fig


def save_md_figures(analysis_results, compound_name, output_dir=None):
    """Save all MD visualization figures."""
    if output_dir is None:
        output_dir = FIGURES_DIR
    os.makedirs(output_dir, exist_ok=True)

    safe_name = compound_name.replace(" ", "_")

    if analysis_results is None:
        return

    # RMSD
    if "rmsd" in analysis_results:
        fig = rmsd_time_series(analysis_results["rmsd"], compound_name)
        try:
            fig.write_image(os.path.join(output_dir, f"fig_md_rmsd_{safe_name}.png"), scale=2)
            print(f"    Saved RMSD plot")
        except Exception:
            pass

    # RMSF
    if "rmsf" in analysis_results:
        fig = rmsf_bar_chart(analysis_results["rmsf"], compound_name)
        try:
            fig.write_image(os.path.join(output_dir, f"fig_md_rmsf_{safe_name}.png"), scale=2)
            print(f"    Saved RMSF plot")
        except Exception:
            pass

    # H-bond
    if "hbond_occupancy" in analysis_results:
        fig = hbond_occupancy_chart(analysis_results["hbond_occupancy"], compound_name)
        try:
            fig.write_image(os.path.join(output_dir, f"fig_md_hbond_{safe_name}.png"), scale=2)
            print(f"    Saved H-bond plot")
        except Exception:
            pass
