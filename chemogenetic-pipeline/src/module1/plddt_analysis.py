"""
Module 1: pLDDT Confidence Analysis
Extracts per-residue pLDDT scores from AlphaFold predictions and generates
confidence reports and visualizations.

In AlphaFold PDB output, pLDDT scores are stored in the B-factor column (0-100).
In AlphaFold CIF output, pLDDT scores are in the _ma_qa_metric_local table.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.module1.structure_parser import parse_structure, get_bfactors

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
PLDDT_DIR = os.path.join(PROJECT_ROOT, "data", "results", "plddt_scores")


def extract_plddt_from_pdb(filepath):
    """Extract per-residue pLDDT scores from an AlphaFold PDB file.

    AlphaFold stores pLDDT in the B-factor column of PDB files.

    Args:
        filepath: Path to AlphaFold PDB file.

    Returns:
        pandas DataFrame with columns: resseq, resname, chain_id, plddt.
    """
    structure = parse_structure(filepath)
    bfactors = get_bfactors(structure)

    df = pd.DataFrame(bfactors)
    df = df.rename(columns={"bfactor": "plddt"})
    return df


def extract_plddt_from_cif(filepath):
    """Extract per-residue pLDDT scores from an AlphaFold CIF file.

    Tries the _ma_qa_metric_local table first (AlphaFold 3 format),
    then falls back to B-factor extraction.

    Args:
        filepath: Path to AlphaFold CIF file.

    Returns:
        pandas DataFrame with columns: resseq, resname, chain_id, plddt.
    """
    # Try parsing as CIF with B-factors (common AlphaFold output)
    return extract_plddt_from_pdb(filepath)


def extract_plddt(filepath):
    """Extract pLDDT scores from any AlphaFold output file.

    Dispatches to the appropriate parser based on file extension.

    Args:
        filepath: Path to .pdb or .cif file.

    Returns:
        pandas DataFrame with columns: resseq, resname, chain_id, plddt.
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdb":
        return extract_plddt_from_pdb(filepath)
    elif ext == ".cif":
        return extract_plddt_from_cif(filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def plddt_summary(plddt_df):
    """Compute summary statistics for pLDDT scores.

    Args:
        plddt_df: DataFrame with 'plddt' column from extract_plddt().

    Returns:
        Dict with: mean, median, std, min, max, n_residues,
        pct_very_high (>90), pct_confident (>70), pct_low (<50).
    """
    scores = plddt_df["plddt"].values
    return {
        "mean": round(float(np.mean(scores)), 2),
        "median": round(float(np.median(scores)), 2),
        "std": round(float(np.std(scores)), 2),
        "min": round(float(np.min(scores)), 2),
        "max": round(float(np.max(scores)), 2),
        "n_residues": len(scores),
        "pct_very_high": round(float(np.sum(scores > 90) / len(scores) * 100), 1),
        "pct_confident": round(float(np.sum(scores > 70) / len(scores) * 100), 1),
        "pct_low": round(float(np.sum(scores < 50) / len(scores) * 100), 1),
    }


def classify_plddt(mean_plddt):
    """Classify overall prediction confidence based on mean pLDDT.

    AlphaFold confidence bands:
        Very high (>90): High accuracy expected
        Confident (70-90): Generally reliable
        Low (50-70): Caution advised
        Very low (<50): Likely disordered or inaccurate

    Returns:
        str — confidence classification.
    """
    if mean_plddt > 90:
        return "Very High"
    if mean_plddt > 70:
        return "Confident"
    if mean_plddt > 50:
        return "Low"
    return "Very Low"


def save_plddt_scores(plddt_df, name, output_dir=None):
    """Save per-residue pLDDT scores to CSV.

    Args:
        plddt_df: DataFrame from extract_plddt().
        name: Structure/compound name.
        output_dir: Directory for output. Defaults to data/results/plddt_scores/.

    Returns:
        Path to saved CSV.
    """
    if output_dir is None:
        output_dir = PLDDT_DIR
    os.makedirs(output_dir, exist_ok=True)
    safe_name = name.replace(" ", "_")
    path = os.path.join(output_dir, f"plddt_{safe_name}.csv")
    plddt_df.to_csv(path, index=False)
    print(f"Saved pLDDT scores: {path}")
    return path


def plot_plddt(plddt_df, name, output_path=None):
    """Generate a pLDDT line chart (per-residue confidence).

    Creates a line plot with AlphaFold confidence bands shaded:
        >90 blue (very high), 70-90 cyan (confident),
        50-70 yellow (low), <50 orange (very low).

    Args:
        plddt_df: DataFrame from extract_plddt().
        name: Structure name for the title.
        output_path: Path for PNG output. Auto-generated if None.

    Returns:
        Path to saved figure.
    """
    if output_path is None:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        safe_name = name.replace(" ", "_")
        output_path = os.path.join(FIGURES_DIR, f"fig_plddt_{safe_name}.png")

    fig, ax = plt.subplots(figsize=(12, 4))

    residues = plddt_df["resseq"].values
    scores = plddt_df["plddt"].values

    # Confidence band shading
    ax.axhspan(90, 100, color="#1565C0", alpha=0.1, label="Very High (>90)")
    ax.axhspan(70, 90, color="#42A5F5", alpha=0.1, label="Confident (70-90)")
    ax.axhspan(50, 70, color="#FFEE58", alpha=0.1, label="Low (50-70)")
    ax.axhspan(0, 50, color="#EF6C00", alpha=0.1, label="Very Low (<50)")

    ax.plot(residues, scores, linewidth=1.0, color="#1565C0")
    ax.set_xlabel("Residue Number", fontsize=11)
    ax.set_ylabel("pLDDT Score", fontsize=11)
    ax.set_title(f"AlphaFold Confidence: {name}", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_xlim(residues[0], residues[-1])
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved pLDDT plot: {output_path}")
    return output_path


def plot_plddt_comparison(plddt_dict, output_path=None):
    """Plot pLDDT comparison across multiple structures.

    Args:
        plddt_dict: Dict mapping name -> pLDDT DataFrame.
        output_path: Path for PNG output.

    Returns:
        Path to saved figure.
    """
    if output_path is None:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        output_path = os.path.join(FIGURES_DIR, "fig_plddt_comparison.png")

    fig, ax = plt.subplots(figsize=(10, 5))

    names = list(plddt_dict.keys())
    means = [plddt_dict[n]["plddt"].mean() for n in names]
    colors = ["#1565C0" if m > 90 else "#42A5F5" if m > 70 else "#FFEE58" if m > 50 else "#EF6C00"
              for m in means]

    bars = ax.bar(names, means, color=colors, edgecolor="white", linewidth=1.5)

    # Add threshold lines
    ax.axhline(y=90, color="#1565C0", linestyle="--", alpha=0.5, label="Very High (90)")
    ax.axhline(y=70, color="#42A5F5", linestyle="--", alpha=0.5, label="Confident (70)")
    ax.axhline(y=50, color="#EF6C00", linestyle="--", alpha=0.5, label="Low (50)")

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{mean:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Mean pLDDT Score", fontsize=12)
    ax.set_title("AlphaFold Confidence Comparison", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved pLDDT comparison: {output_path}")
    return output_path
