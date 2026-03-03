"""
Shared visualization functions for the chemogenetic pipeline.
Generates publication-quality figures (300 DPI) for poster and dashboard.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import plotly.express as px

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")

# Color palette for compounds
COMPOUND_COLORS = {
    "CNO": "#e74c3c",
    "Clozapine": "#e67e22",
    "DCZ": "#2ecc71",
    "Compound 21": "#3498db",
    "Olanzapine": "#9b59b6",
    "Perlapine": "#1abc9c",
}


def chemical_space_plot(df, output_path=None, show=False):
    """Chemical space scatter: LogP vs TPSA with BBB permeability zones.

    Bubble size = MW, color = BBB status.
    Includes threshold lines at TPSA=90, LogP=1, LogP=3.
    """
    if output_path is None:
        output_path = os.path.join(FIGURES_DIR, "fig_chemical_space.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7))

    # BBB-penetrant zone shading
    ax.axhspan(0, 90, xmin=0, xmax=1, alpha=0.06, color="green")
    ax.axhline(y=90, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="TPSA = 90 Å²")
    ax.axvline(x=1.0, color="gray", linestyle=":", linewidth=1, alpha=0.7, label="LogP = 1.0")
    ax.axvline(x=3.0, color="gray", linestyle=":", linewidth=1, alpha=0.7, label="LogP = 3.0")

    # Add green zone label
    ax.text(2.0, 10, "BBB Penetrant Zone", fontsize=10, color="green",
            alpha=0.5, ha="center", fontstyle="italic")

    # Plot each compound
    for _, row in df.iterrows():
        color = "#2ecc71" if row["bbb_predicted"] == "Penetrant" else "#e74c3c"
        edge_color = COMPOUND_COLORS.get(row["name"], "#333333")
        size = row["mw"] * 1.5
        ax.scatter(row["logp"], row["tpsa"], s=size, c=color, edgecolors=edge_color,
                   linewidths=2, alpha=0.75, zorder=5)
        ax.annotate(row["name"], (row["logp"], row["tpsa"]),
                    textcoords="offset points", xytext=(8, 8),
                    fontsize=9, fontweight="bold")

    ax.set_xlabel("LogP (Lipophilicity)", fontsize=12)
    ax.set_ylabel("TPSA (Å²)", fontsize=12)
    ax.set_title("Chemical Space: DREADD Actuators\nLogP vs TPSA with BBB Permeability Zones",
                 fontsize=14, fontweight="bold")

    # Legend
    penetrant_patch = mpatches.Patch(color="#2ecc71", alpha=0.75, label="BBB Penetrant")
    non_penetrant_patch = mpatches.Patch(color="#e74c3c", alpha=0.75, label="Non-penetrant")
    ax.legend(handles=[penetrant_patch, non_penetrant_patch, ax.lines[0], ax.lines[1]],
              loc="upper left", fontsize=9)

    ax.set_xlim(-1, 6)
    ax.set_ylim(0, max(df["tpsa"].max() * 1.3, 120))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    print(f"Saved chemical space plot: {output_path}")
    return output_path


def chemical_space_plotly(df):
    """Interactive plotly version of the chemical space plot for the dashboard."""
    fig = go.Figure()

    # BBB zone shading
    fig.add_shape(type="rect", x0=1.0, x1=3.0, y0=0, y1=90,
                  fillcolor="green", opacity=0.08, line_width=0)

    # Threshold lines
    fig.add_hline(y=90, line_dash="dash", line_color="gray", opacity=0.5,
                  annotation_text="TPSA = 90 Å²")
    fig.add_vline(x=1.0, line_dash="dot", line_color="gray", opacity=0.5)
    fig.add_vline(x=3.0, line_dash="dot", line_color="gray", opacity=0.5)

    colors = ["#2ecc71" if b == "Penetrant" else "#e74c3c" for b in df["bbb_predicted"]]

    fig.add_trace(go.Scatter(
        x=df["logp"], y=df["tpsa"],
        mode="markers+text",
        text=df["name"],
        textposition="top center",
        marker=dict(
            size=df["mw"] / 15,
            color=colors,
            line=dict(width=2, color="white"),
            opacity=0.8,
        ),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "LogP: %{x:.2f}<br>"
            "TPSA: %{y:.1f} Å²<br>"
            "MW: %{marker.size:.0f} Da<br>"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        title="Chemical Space: DREADD Actuators",
        xaxis_title="LogP (Lipophilicity)",
        yaxis_title="TPSA (Å²)",
        height=500,
        showlegend=False,
    )
    return fig


def radar_chart(df, compounds=None, output_path=None, show=False):
    """Radar/spider chart comparing normalized properties across compounds.

    Properties: MW, LogP, TPSA, HBD, HBA, Rotatable Bonds.
    Normalized to 0-1 scale based on dataset range.
    """
    if output_path is None:
        output_path = os.path.join(FIGURES_DIR, "fig_radar_comparison.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if compounds is None:
        compounds = df["name"].tolist()
    subset = df[df["name"].isin(compounds)]

    properties = ["mw", "logp", "tpsa", "hbd", "hba", "rotatable_bonds"]
    labels = ["MW", "LogP", "TPSA", "HBD", "HBA", "Rot. Bonds"]

    # Normalize each property to 0-1
    normalized = subset[properties].copy()
    for col in properties:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_max > col_min:
            normalized[col] = (normalized[col] - col_min) / (col_max - col_min)
        else:
            normalized[col] = 0.5

    num_vars = len(properties)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for idx, (_, row) in enumerate(subset.iterrows()):
        name = row["name"]
        values = normalized.iloc[idx][properties].tolist()
        values += values[:1]
        color = COMPOUND_COLORS.get(name, f"C{idx}")
        ax.plot(angles, values, "o-", linewidth=2, label=name, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title("Property Comparison Radar Chart", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    print(f"Saved radar chart: {output_path}")
    return output_path


def radar_chart_plotly(df, compounds=None):
    """Interactive plotly radar chart for the dashboard."""
    if compounds is None:
        compounds = df["name"].tolist()
    subset = df[df["name"].isin(compounds)]

    properties = ["mw", "logp", "tpsa", "hbd", "hba", "rotatable_bonds"]
    labels = ["MW", "LogP", "TPSA", "HBD", "HBA", "Rot. Bonds"]

    # Normalize
    normalized = subset[properties].copy()
    for col in properties:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_max > col_min:
            normalized[col] = (normalized[col] - col_min) / (col_max - col_min)
        else:
            normalized[col] = 0.5

    fig = go.Figure()
    for idx, (_, row) in enumerate(subset.iterrows()):
        name = row["name"]
        values = normalized.iloc[idx][properties].tolist()
        color = COMPOUND_COLORS.get(name, None)
        fig.add_trace(go.Scatterpolar(
            r=values, theta=labels, fill="toself",
            name=name, line_color=color, opacity=0.7,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Property Comparison Radar",
        height=500,
    )
    return fig


def lipinski_bar_chart(df, output_path=None, show=False):
    """Stacked bar chart of Lipinski Rule of Five violations per compound."""
    if output_path is None:
        output_path = os.path.join(FIGURES_DIR, "fig_lipinski_violations.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Calculate individual rule violations
    violations_data = []
    for _, row in df.iterrows():
        violations_data.append({
            "name": row["name"],
            "MW ≥ 500": 1 if row["mw"] >= 500 else 0,
            "LogP ≥ 5": 1 if row["logp"] >= 5 else 0,
            "HBD > 5": 1 if row["hbd"] > 5 else 0,
            "HBA > 10": 1 if row["hba"] > 10 else 0,
        })
    vdf = pd.DataFrame(violations_data)

    fig, ax = plt.subplots(figsize=(10, 5))
    rules = ["MW ≥ 500", "LogP ≥ 5", "HBD > 5", "HBA > 10"]
    rule_colors = ["#e74c3c", "#e67e22", "#9b59b6", "#3498db"]
    bottom = np.zeros(len(vdf))

    for rule, color in zip(rules, rule_colors):
        ax.bar(vdf["name"], vdf[rule], bottom=bottom, label=rule, color=color, alpha=0.8)
        bottom += vdf[rule].values

    ax.set_ylabel("Number of Violations", fontsize=12)
    ax.set_title("Lipinski Rule of Five Violations", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 4.5)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.axhline(y=1.5, color="orange", linestyle="--", alpha=0.5, label="Drug-like threshold (≤1)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    print(f"Saved Lipinski bar chart: {output_path}")
    return output_path


def generate_all_figures(df):
    """Generate all publication-quality figures for Module 2."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    paths = []
    paths.append(chemical_space_plot(df))
    paths.append(radar_chart(df))
    paths.append(lipinski_bar_chart(df))

    # Individual radar charts per compound
    for name in df["name"]:
        path = os.path.join(FIGURES_DIR, f"fig_radar_{name.replace(' ', '_')}.png")
        radar_chart(df, compounds=[name], output_path=path)
        paths.append(path)

    return paths
