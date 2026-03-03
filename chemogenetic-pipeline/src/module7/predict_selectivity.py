"""
Module 7: Selectivity Prediction Engine
Predicts activity probability across multiple targets.
"""

import os
import sys
import numpy as np
import joblib
import pandas as pd
import plotly.graph_objects as go

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.module6.fingerprints import smiles_to_morgan_fp
from src.module7.chembl_data import TARGETS

MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "selectivity")

_model_cache = {}


def _load_target_model(target_key):
    """Load a trained selectivity model."""
    if target_key in _model_cache:
        return _model_cache[target_key]

    model_path = os.path.join(MODELS_DIR, f"{target_key}_model.joblib")
    if not os.path.exists(model_path):
        print(f"  Model for {target_key} not found. Training...")
        from src.module7.train_selectivity import train_target_model
        from src.module7.chembl_data import download_all_targets
        binding_data = download_all_targets(verbose=False)
        train_target_model(target_key, binding_data, verbose=True)

    model_data = joblib.load(model_path)
    _model_cache[target_key] = model_data
    return model_data


def predict_target_activity(smiles, target_key):
    """Predict probability of activity at a given target.

    Args:
        smiles: SMILES string
        target_key: Target identifier (e.g., "M1", "D2")

    Returns:
        dict with probability and metadata
    """
    model_data = _load_target_model(target_key)
    fp = smiles_to_morgan_fp(smiles, n_bits=model_data["n_bits"])
    if fp is None:
        return None

    X = fp.reshape(1, -1).astype(np.float64)

    rf_prob = model_data["rf"].predict_proba(X)[0, 1]
    gb_prob = model_data["gb"].predict_proba(X)[0, 1]
    ensemble_prob = (rf_prob + gb_prob) / 2

    return {
        "target": target_key,
        "target_name": TARGETS[target_key]["name"],
        "probability": round(float(ensemble_prob), 4),
        "is_active": ensemble_prob >= 0.5,
        "model_auroc": model_data["metrics"]["auroc"],
    }


def predict_selectivity_profile(smiles):
    """Predict activity across all targets for a compound.

    Returns:
        dict of target_key -> prediction result
    """
    profile = {}
    for target_key in TARGETS:
        try:
            result = predict_target_activity(smiles, target_key)
            if result:
                profile[target_key] = result
        except Exception as e:
            profile[target_key] = {"target": target_key, "probability": 0.0, "error": str(e)}

    return profile


def selectivity_score(smiles, primary_target="M3"):
    """Calculate selectivity score: P(active at M3) / max(P(active at off-targets)).

    Higher = more selective for the DREADD target.

    Args:
        smiles: SMILES string
        primary_target: Primary target (M3 for hM3Dq DREADD)

    Returns:
        float selectivity score
    """
    profile = predict_selectivity_profile(smiles)

    primary_prob = profile.get(primary_target, {}).get("probability", 0.0)

    off_target_probs = [
        v.get("probability", 0.0)
        for k, v in profile.items()
        if k != primary_target
    ]

    max_off_target = max(off_target_probs) if off_target_probs else 0.001

    # Avoid division by zero
    if max_off_target < 0.001:
        max_off_target = 0.001

    return round(primary_prob / max_off_target, 4)


def selectivity_radar(profile, compound_name="Compound"):
    """Generate Plotly radar chart of target activity profile.

    Args:
        profile: dict from predict_selectivity_profile()
        compound_name: Label for the compound

    Returns:
        Plotly figure
    """
    targets = list(profile.keys())
    probs = [profile[t].get("probability", 0.0) for t in targets]
    target_names = [TARGETS.get(t, {}).get("name", t) for t in targets]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=probs,
        theta=target_names,
        fill="toself",
        name=compound_name,
        line_color="#3498db",
        fillcolor="rgba(52, 152, 219, 0.2)",
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),
        ),
        title=f"Selectivity Profile: {compound_name}",
        height=500,
    )

    return fig


def multi_compound_radar(compounds_dict):
    """Generate radar chart comparing multiple compounds.

    Args:
        compounds_dict: dict of name -> profile

    Returns:
        Plotly figure
    """
    from src.utils.plotting import COMPOUND_COLORS

    fig = go.Figure()
    target_names = [TARGETS[t]["name"] for t in TARGETS]

    for name, profile in compounds_dict.items():
        probs = [profile.get(t, {}).get("probability", 0.0) for t in TARGETS]
        color = COMPOUND_COLORS.get(name, "#333333")

        fig.add_trace(go.Scatterpolar(
            r=probs,
            theta=target_names,
            fill="toself",
            name=name,
            line_color=color,
            opacity=0.7,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Selectivity Profiles: All Actuators",
        height=600,
    )

    return fig


def off_target_heatmap(compounds_profiles):
    """Generate heatmap of off-target activity (compounds x targets).

    Args:
        compounds_profiles: dict of compound_name -> profile dict

    Returns:
        Plotly figure
    """
    names = list(compounds_profiles.keys())
    target_names = [TARGETS[t]["name"] for t in TARGETS]

    z = []
    for name in names:
        profile = compounds_profiles[name]
        row = [profile.get(t, {}).get("probability", 0.0) for t in TARGETS]
        z.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=target_names,
        y=names,
        colorscale="RdYlGn_r",  # Red = high activity (bad for off-targets)
        zmin=0, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in z],
        texttemplate="%{text}",
        hovertemplate="Compound: %{y}<br>Target: %{x}<br>P(active): %{z:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title="Off-Target Activity Heatmap",
        xaxis_title="Target",
        yaxis_title="Compound",
        height=400,
    )

    return fig


def flag_off_targets(profile, threshold=0.5, primary_target="M3"):
    """Flag off-target liabilities for a compound.

    Args:
        profile: dict from predict_selectivity_profile()
        threshold: Probability threshold for flagging
        primary_target: Primary target (not flagged)

    Returns:
        list of flagged targets
    """
    flagged = []
    for target_key, result in profile.items():
        if target_key == primary_target:
            continue
        if result.get("probability", 0.0) >= threshold:
            flagged.append({
                "target": target_key,
                "target_name": TARGETS[target_key]["name"],
                "probability": result["probability"],
            })

    return sorted(flagged, key=lambda x: x["probability"], reverse=True)
