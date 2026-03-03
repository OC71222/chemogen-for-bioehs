"""
Module 4: Docking Analysis
Analyzes and visualizes docking results.
"""

import os
import sys
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

# Published experimental binding data for validation
PUBLISHED_DATA = {
    "DCZ": {"Ki_nM": 0.5, "ref": "Nagai et al. 2020"},
    "CNO": {"Ki_nM": 75.0, "ref": "Thompson et al. 2018"},
    "Clozapine": {"Ki_nM": 15.0, "ref": "Armbruster et al. 2007"},
    "Compound 21": {"Ki_nM": 50.0, "ref": "Chen et al. 2015"},
    "Olanzapine": {"Ki_nM": 100.0, "ref": "Weston et al. 2019"},
    "Perlapine": {"Ki_nM": 30.0, "ref": "Roth 2016"},
}

# Key binding site residues
BINDING_RESIDUES = {
    "Asp3.32": {"type": "salt_bridge", "importance": "Critical for amine recognition"},
    "Trp6.48": {"type": "aromatic_cage", "importance": "Toggle switch residue"},
    "Tyr3.33": {"type": "H-bond", "importance": "Orthosteric pocket lining"},
    "Phe6.51": {"type": "aromatic", "importance": "Aromatic cage member"},
    "Tyr6.51": {"type": "aromatic", "importance": "Aromatic stacking"},
    "Asn6.52": {"type": "H-bond", "importance": "Polar contact"},
}


def rank_compounds(docking_df):
    """Rank compounds by predicted binding affinity (more negative = better)."""
    ranked = docking_df.sort_values("affinity_kcal_mol", ascending=True).copy()
    ranked["rank"] = range(1, len(ranked) + 1)
    return ranked


def compare_with_published(docking_df):
    """Compare predicted affinities with published Ki values."""
    import numpy as np

    comparison = []
    for _, row in docking_df.iterrows():
        name = row["name"]
        pub = PUBLISHED_DATA.get(name, {})
        comparison.append({
            "name": name,
            "predicted_affinity": row["affinity_kcal_mol"],
            "published_Ki_nM": pub.get("Ki_nM"),
            "reference": pub.get("ref", "N/A"),
        })

    comp_df = pd.DataFrame(comparison)

    # Add log-transformed Ki for correlation
    if "published_Ki_nM" in comp_df.columns:
        comp_df["pKi"] = comp_df["published_Ki_nM"].apply(
            lambda x: -np.log10(x * 1e-9) if pd.notna(x) and x > 0 else None
        )

    return comp_df


def affinity_bar_chart(docking_df):
    """Generate binding affinity bar chart (Plotly)."""
    ranked = rank_compounds(docking_df)

    colors = []
    for _, row in ranked.iterrows():
        if row["affinity_kcal_mol"] is None:
            colors.append("#cccccc")
        elif row["affinity_kcal_mol"] <= -8.0:
            colors.append("#2ecc71")  # Strong binder
        elif row["affinity_kcal_mol"] <= -6.0:
            colors.append("#f39c12")  # Moderate binder
        else:
            colors.append("#e74c3c")  # Weak binder

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ranked["name"],
        y=ranked["affinity_kcal_mol"],
        marker_color=colors,
        text=[f"{a:.2f}" if a else "N/A" for a in ranked["affinity_kcal_mol"]],
        textposition="outside",
    ))

    fig.update_layout(
        title="Predicted Binding Affinity (AutoDock Vina)",
        xaxis_title="Compound",
        yaxis_title="Binding Affinity (kcal/mol)",
        yaxis=dict(autorange="reversed"),  # More negative = better
        height=500,
    )

    # Add reference line for strong binding threshold
    fig.add_hline(y=-7.0, line_dash="dash", line_color="gray",
                  annotation_text="Strong binding threshold (-7.0 kcal/mol)")

    return fig


def comparison_scatter(comp_df):
    """Scatter plot: predicted affinity vs published pKi."""
    valid = comp_df.dropna(subset=["predicted_affinity", "pKi"])
    if len(valid) < 2:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=valid["pKi"],
        y=valid["predicted_affinity"],
        mode="markers+text",
        text=valid["name"],
        textposition="top center",
        marker=dict(size=12, color="#3498db", line=dict(width=2, color="white")),
    ))

    fig.update_layout(
        title="Predicted Affinity vs Published pKi",
        xaxis_title="Published pKi (-log10(Ki))",
        yaxis_title="Predicted Affinity (kcal/mol)",
        height=500,
    )

    return fig


def interaction_table():
    """Generate key binding site interactions table."""
    rows = []
    for residue, info in BINDING_RESIDUES.items():
        rows.append({
            "Residue": residue,
            "Interaction Type": info["type"],
            "Importance": info["importance"],
        })
    return pd.DataFrame(rows)


def save_analysis(docking_df, output_dir=None):
    """Save all analysis outputs."""
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "figures")
    os.makedirs(output_dir, exist_ok=True)

    # Save ranked results
    ranked = rank_compounds(docking_df)
    comp = compare_with_published(docking_df)

    # Save affinity chart
    fig = affinity_bar_chart(docking_df)
    try:
        fig.write_image(os.path.join(output_dir, "fig_docking_affinity.png"), scale=2)
    except Exception:
        pass

    # Print summary
    print("\n--- Docking Results Summary ---")
    print(ranked[["rank", "name", "affinity_kcal_mol"]].to_string(index=False))

    print("\n--- Comparison with Published Data ---")
    print(comp[["name", "predicted_affinity", "published_Ki_nM", "reference"]].to_string(index=False))

    return ranked, comp
