"""
Module 6: ML ADMET Prediction — Orchestrator
Train models (if not cached) → evaluate → predict for 6 actuators → save results.
"""

import os
import sys
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.module2.compounds import load_compounds
from src.module2.evaluate_actuators import predict_bbb


def ensure_models_trained(verbose=True):
    """Ensure all ADMET models are trained. Train if not cached."""
    from src.module6.train_models import MODELS_DIR, train_all_models

    all_models = ["bbb", "herg", "cyp2d6", "hia", "clearance"]
    missing = [
        k for k in all_models
        if not os.path.exists(os.path.join(MODELS_DIR, f"{k}_model.joblib"))
    ]

    if missing:
        if verbose:
            print(f"  Missing models: {missing}. Training all...")
        return train_all_models(verbose=verbose)
    else:
        if verbose:
            print("  All ADMET models found in cache.")
        return None


def predict_actuators(verbose=True):
    """Run ADMET predictions on all 6 known actuators."""
    from src.module6.predict_admet import predict_admet_batch

    compounds = load_compounds()
    smiles_list = compounds["smiles"].tolist()
    names = compounds["name"].tolist()

    if verbose:
        print("\n  Predicting ADMET for actuators...")

    results_df = predict_admet_batch(smiles_list, names=names)

    # Add rule-based BBB for comparison
    from src.module2.evaluate_actuators import calculate_properties, predict_bbb
    rule_bbb = []
    for smi in smiles_list:
        props = calculate_properties(smi)
        if props:
            rule_bbb.append(predict_bbb(props))
        else:
            rule_bbb.append("Unknown")
    results_df["bbb_rule_based"] = rule_bbb

    return results_df


def generate_admet_figure(results_df, output_path=None):
    """Generate grouped bar chart comparing ADMET predictions across compounds."""
    if output_path is None:
        output_path = os.path.join(PROJECT_ROOT, "figures", "fig_admet_comparison.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig = go.Figure()

    endpoints = []
    if "bbb_ml_prob" in results_df.columns:
        endpoints.append(("bbb_ml_prob", "BBB Permeability"))
    if "herg_prob" in results_df.columns:
        endpoints.append(("herg_prob", "hERG Inhibition"))
    if "cyp2d6_prob" in results_df.columns:
        endpoints.append(("cyp2d6_prob", "CYP2D6 Inhibition"))
    if "hia_prob" in results_df.columns:
        endpoints.append(("hia_prob", "HIA"))

    colors = ["#2ecc71", "#e74c3c", "#e67e22", "#3498db"]

    for i, (col, label) in enumerate(endpoints):
        fig.add_trace(go.Bar(
            name=label,
            x=results_df["name"],
            y=results_df[col],
            marker_color=colors[i % len(colors)],
        ))

    fig.update_layout(
        title="ADMET Predictions: DREADD Actuators",
        xaxis_title="Compound",
        yaxis_title="Predicted Probability",
        yaxis_range=[0, 1],
        barmode="group",
        height=500,
    )

    fig.write_image(output_path, scale=2)
    print(f"  Saved ADMET figure: {output_path}")
    return fig


def run_module6(verbose=True):
    """Full Module 6 pipeline."""
    from src.utils.progress import update_module_status

    print("\n" + "=" * 60)
    print("  MODULE 6: ML ADMET PREDICTION")
    print("=" * 60)
    start = time.time()

    update_module_status(6, "running", step="Training ADMET models",
                         detail="Downloading datasets and training RF+GB ensembles")

    # Step 1: Ensure models trained
    training_results = ensure_models_trained(verbose=verbose)

    update_module_status(6, "running", step="Predicting ADMET for actuators",
                         detail="Running BBB, hERG, CYP2D6, HIA, Clearance predictions",
                         progress=1, total=3)

    # Step 2: Predict for actuators
    results_df = predict_actuators(verbose=verbose)

    update_module_status(6, "running", step="Saving results",
                         progress=2, total=3)

    # Step 3: Save results
    output_path = os.path.join(PROJECT_ROOT, "data", "results", "admet_predictions.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\n  Saved ADMET predictions: {output_path}")

    # Step 4: Print results
    if verbose:
        print("\n--- ADMET Predictions ---")
        display_cols = ["name"]
        for col in ["bbb_ml_class", "bbb_rule_based", "herg_class", "herg_risk",
                     "cyp2d6_inhibitor", "hia_class", "clearance_cat"]:
            if col in results_df.columns:
                display_cols.append(col)
        print(results_df[display_cols].to_string(index=False))

        # BBB comparison
        if "bbb_ml_class" in results_df.columns and "bbb_rule_based" in results_df.columns:
            print("\n--- BBB: ML vs Rule-based ---")
            for _, row in results_df.iterrows():
                ml = row.get("bbb_ml_class", "N/A")
                rule = row.get("bbb_rule_based", "N/A")
                match = "AGREE" if ml == rule else "DISAGREE"
                prob = row.get("bbb_ml_prob", 0)
                print(f"  {row['name']:15s} ML: {ml:15s} (p={prob:.3f}) | Rule: {rule:15s} | {match}")

    # Step 5: Generate figure
    try:
        generate_admet_figure(results_df)
    except Exception as e:
        print(f"  Warning: Could not generate figure: {e}")

    elapsed = time.time() - start
    print(f"\n  Module 6 completed in {elapsed:.1f}s")

    n_compounds = len(results_df)
    update_module_status(6, "completed", step="Done",
                         detail=f"Completed in {elapsed:.1f}s",
                         metrics={"Compounds": n_compounds, "Time": f"{elapsed:.1f}s"})

    return results_df


if __name__ == "__main__":
    run_module6()
