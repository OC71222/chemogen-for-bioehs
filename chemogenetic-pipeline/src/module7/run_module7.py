"""
Module 7: Selectivity Prediction — Orchestrator
Download ChEMBL data → train models → predict for 6 actuators → save results.
"""

import os
import sys
import time
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)


def run_module7(verbose=True):
    """Full Module 7 pipeline."""
    from src.module7.chembl_data import download_all_targets, TARGETS
    from src.module7.train_selectivity import train_all_selectivity_models
    from src.module7.predict_selectivity import (
        predict_selectivity_profile,
        selectivity_score,
        flag_off_targets,
        multi_compound_radar,
        off_target_heatmap,
    )
    from src.module2.compounds import load_compounds

    from src.utils.progress import update_module_status

    print("\n" + "=" * 60)
    print("  MODULE 7: SELECTIVITY PREDICTION")
    print("=" * 60)
    start = time.time()

    # Step 1: Download ChEMBL data
    print("\n--- Step 1: ChEMBL Data ---")
    update_module_status(7, "running", step="Downloading ChEMBL binding data",
                         detail="Querying 8 receptor targets from ChEMBL API",
                         progress=0, total=4)
    binding_data = download_all_targets(verbose=verbose)

    # Step 2: Train models
    print("\n--- Step 2: Train Selectivity Models ---")
    update_module_status(7, "running", step="Training selectivity models",
                         detail=f"Training RF+GB on {len(binding_data)} total compounds across 8 targets",
                         progress=1, total=4)
    models = train_all_selectivity_models(binding_data, verbose=verbose)

    # Step 3: Predict for actuators
    print("\n--- Step 3: Predict Selectivity Profiles ---")
    update_module_status(7, "running", step="Predicting selectivity profiles",
                         detail="Running predictions for 6 DREADD actuators",
                         progress=2, total=4)
    compounds = load_compounds()

    profiles = {}
    rows = []
    for _, row in compounds.iterrows():
        name = row["name"]
        smiles = row["smiles"]

        if verbose:
            print(f"\n  {name}:")

        profile = predict_selectivity_profile(smiles)
        profiles[name] = profile

        sel_score = selectivity_score(smiles)
        flagged = flag_off_targets(profile)

        result_row = {"name": name, "selectivity_score": sel_score}
        for target_key in TARGETS:
            prob = profile.get(target_key, {}).get("probability", 0.0)
            result_row[f"p_{target_key}"] = prob

        n_offtargets = len(flagged)
        result_row["n_off_targets"] = n_offtargets
        if flagged:
            result_row["off_targets"] = ", ".join(f["target_name"] for f in flagged)
        else:
            result_row["off_targets"] = "None"

        rows.append(result_row)

        if verbose:
            print(f"    Selectivity score: {sel_score:.2f}")
            if flagged:
                parts = [f"{ft['target_name']} (p={ft['probability']:.2f})" for ft in flagged]
                print(f"    Off-target flags: {', '.join(parts)}")
            else:
                print(f"    No off-target flags")

    results_df = pd.DataFrame(rows)

    # Step 4: Save results
    output_path = os.path.join(PROJECT_ROOT, "data", "results", "selectivity_predictions.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\n  Saved selectivity predictions: {output_path}")

    # Step 5: Generate figures
    print("\n--- Step 4: Generate Figures ---")
    figures_dir = os.path.join(PROJECT_ROOT, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    try:
        fig_radar = multi_compound_radar(profiles)
        fig_radar.write_image(os.path.join(figures_dir, "fig_selectivity_radar.png"), scale=2)
        print("    Saved selectivity radar chart")
    except Exception as e:
        print(f"    Warning: Could not save radar chart: {e}")

    try:
        fig_heatmap = off_target_heatmap(profiles)
        fig_heatmap.write_image(os.path.join(figures_dir, "fig_selectivity_heatmap.png"), scale=2)
        print("    Saved off-target heatmap")
    except Exception as e:
        print(f"    Warning: Could not save heatmap: {e}")

    # Summary
    if verbose:
        print("\n--- Selectivity Summary ---")
        display_cols = ["name", "selectivity_score", "n_off_targets", "off_targets"]
        print(results_df[display_cols].to_string(index=False))

    elapsed = time.time() - start
    print(f"\n  Module 7 completed in {elapsed:.1f}s")

    n_targets = len(TARGETS)
    n_compounds = len(results_df)
    update_module_status(7, "completed", step="Done",
                         detail=f"Completed in {elapsed:.1f}s",
                         progress=4, total=4,
                         metrics={"Targets": n_targets, "Compounds": n_compounds,
                                  "Time": f"{elapsed:.1f}s"})

    return results_df, profiles


if __name__ == "__main__":
    run_module7()
