"""
Module 5: Hit Analysis for Virtual Screening
Analyzes screening results, identifies novel scaffolds, generates reports.
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

# Known actuator SMILES for similarity comparison
from src.module2.compounds import COMPOUNDS
KNOWN_SMILES = {name: info["smiles"] for name, info in COMPOUNDS.items()}


def calculate_tanimoto(smiles1, smiles2, radius=2, n_bits=2048):
    """Calculate Tanimoto similarity between two SMILES using Morgan FP."""
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return 0.0

    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius, nBits=n_bits)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius, nBits=n_bits)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def max_tanimoto_to_known(smiles):
    """Calculate maximum Tanimoto similarity to any known actuator."""
    max_sim = 0.0
    closest_name = None

    for name, known_smi in KNOWN_SMILES.items():
        sim = calculate_tanimoto(smiles, known_smi)
        if sim > max_sim:
            max_sim = sim
            closest_name = name

    return max_sim, closest_name


def analyze_hits(screening_df, affinity_threshold=-7.0, novelty_threshold=0.4,
                 top_n=20, verbose=True):
    """Analyze virtual screening hits.

    Args:
        screening_df: DataFrame from screen_engine with 'affinity' column
        affinity_threshold: Keep compounds better than this (kcal/mol)
        novelty_threshold: Compounds with max Tanimoto < this are "novel"
        top_n: Number of top hits to report
        verbose: Print progress

    Returns:
        DataFrame of analyzed hits
    """
    # Filter by affinity
    valid = screening_df.dropna(subset=["affinity"]).copy()
    hits = valid[valid["affinity"] <= affinity_threshold].copy()

    if verbose:
        print(f"\n    Hits with affinity <= {affinity_threshold} kcal/mol: "
              f"{len(hits)}/{len(valid)} ({len(hits)/max(len(valid),1)*100:.1f}%)")

    if hits.empty:
        # Lower threshold if no hits
        relaxed = affinity_threshold + 1.0
        hits = valid.nsmallest(min(top_n, len(valid)), "affinity").copy()
        if verbose:
            print(f"    Relaxed threshold: reporting top {len(hits)} compounds")

    # Calculate Tanimoto similarity to known actuators
    if verbose:
        print("    Calculating Tanimoto similarities...")

    similarities = []
    closest_names = []
    for smi in hits["smiles"]:
        sim, closest = max_tanimoto_to_known(smi)
        similarities.append(round(sim, 4))
        closest_names.append(closest)

    hits["max_tanimoto"] = similarities
    hits["closest_actuator"] = closest_names
    hits["is_novel"] = hits["max_tanimoto"] < novelty_threshold

    # Sort by affinity
    hits = hits.sort_values("affinity", ascending=True)

    # Take top N
    top_hits = hits.head(top_n).copy()
    top_hits["hit_rank"] = range(1, len(top_hits) + 1)

    if verbose:
        n_novel = hits["is_novel"].sum()
        print(f"    Novel scaffolds (Tanimoto < {novelty_threshold}): {n_novel}/{len(hits)}")
        print(f"\n    Top {len(top_hits)} hits:")
        for _, row in top_hits.iterrows():
            novel_tag = " [NOVEL]" if row["is_novel"] else ""
            print(f"      #{row['hit_rank']:2d} {row['compound_id']:15s} "
                  f"Affinity: {row['affinity']:.2f} kcal/mol "
                  f"Tanimoto: {row['max_tanimoto']:.3f} (closest: {row['closest_actuator']})"
                  f"{novel_tag}")

    return top_hits


def affinity_vs_novelty_scatter(hits_df):
    """Plotly scatter: affinity vs Tanimoto similarity."""
    fig = go.Figure()

    # Novel scaffolds
    novel = hits_df[hits_df["is_novel"]]
    known_like = hits_df[~hits_df["is_novel"]]

    if not known_like.empty:
        fig.add_trace(go.Scatter(
            x=known_like["max_tanimoto"],
            y=known_like["affinity"],
            mode="markers",
            name="Known-like",
            marker=dict(size=10, color="#3498db", opacity=0.7),
            text=known_like["compound_id"],
            hovertemplate="<b>%{text}</b><br>Tanimoto: %{x:.3f}<br>Affinity: %{y:.2f}<extra></extra>",
        ))

    if not novel.empty:
        fig.add_trace(go.Scatter(
            x=novel["max_tanimoto"],
            y=novel["affinity"],
            mode="markers",
            name="Novel Scaffold",
            marker=dict(size=12, color="#e74c3c", symbol="star", opacity=0.8),
            text=novel["compound_id"],
            hovertemplate="<b>%{text}</b><br>Tanimoto: %{x:.3f}<br>Affinity: %{y:.2f}<extra></extra>",
        ))

    fig.update_layout(
        title="Screening Hits: Affinity vs Novelty",
        xaxis_title="Max Tanimoto Similarity to Known Actuators",
        yaxis_title="Binding Affinity (kcal/mol)",
        yaxis=dict(autorange="reversed"),
        height=500,
    )

    # Add novelty threshold line
    fig.add_vline(x=0.4, line_dash="dash", line_color="gray",
                  annotation_text="Novelty threshold")

    return fig


def save_hit_report(hits_df, output_dir=None):
    """Save hit analysis report."""
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "results")
    os.makedirs(output_dir, exist_ok=True)

    hits_df.to_csv(os.path.join(output_dir, "screening_hits.csv"), index=False)

    # Save figure
    fig = affinity_vs_novelty_scatter(hits_df)
    try:
        figures_dir = os.path.join(PROJECT_ROOT, "figures")
        os.makedirs(figures_dir, exist_ok=True)
        fig.write_image(os.path.join(figures_dir, "fig_screening_hits.png"), scale=2)
    except Exception:
        pass

    print(f"    Saved hit report to {output_dir}")
    return hits_df
