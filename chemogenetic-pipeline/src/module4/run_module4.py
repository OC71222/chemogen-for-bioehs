"""
Module 4: Molecular Docking — Orchestrator
Download structures → prep receptor → dock all → analyze → save.
"""

import os
import sys
import time
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)


def run_module4(compound=None, exhaustiveness=32, verbose=True):
    """Full Module 4 pipeline.

    Args:
        compound: Single compound name to dock (None = all)
        exhaustiveness: Vina exhaustiveness (32 = default, increase for production)
        verbose: Print progress
    """
    from src.module4.receptor_prep import prepare_receptor
    from src.module4.ligand_prep import prepare_all_ligands, prepare_ligand
    from src.module4.docking_engine import dock_all_compounds, save_docking_results
    from src.module4.docking_analysis import save_analysis
    from src.module2.compounds import load_compounds

    print("\n" + "=" * 60)
    print("  MODULE 4: MOLECULAR DOCKING")
    print("=" * 60)
    start = time.time()

    # Step 1: Prepare receptor (use 8E9W — DCZ-bound structure)
    print("\n--- Step 1: Receptor Preparation ---")
    receptor_info = prepare_receptor(pdb_id="8E9W", verbose=verbose)

    # Step 2: Prepare ligands
    print("\n--- Step 2: Ligand Preparation ---")
    if compound:
        compounds_df = load_compounds()
        row = compounds_df[compounds_df["name"] == compound]
        if row.empty:
            print(f"  ERROR: Compound '{compound}' not found.")
            return None
        smiles = row.iloc[0]["smiles"]
        ligand_dict = {compound: prepare_ligand(smiles, name=compound)}
    else:
        ligand_dict = prepare_all_ligands()

    # Remove failed preparations
    ligand_dict = {k: v for k, v in ligand_dict.items() if v is not None}
    print(f"\n  Prepared {len(ligand_dict)} ligands for docking")

    # Step 3: Dock
    print(f"\n--- Step 3: Docking (exhaustiveness={exhaustiveness}) ---")
    docking_df = dock_all_compounds(
        receptor_info, ligand_dict,
        exhaustiveness=exhaustiveness,
        verbose=verbose,
    )

    # Step 4: Save results
    print("\n--- Step 4: Save Results ---")
    save_docking_results(docking_df)

    # Step 5: Analyze
    print("\n--- Step 5: Analysis ---")
    ranked, comparison = save_analysis(docking_df)

    elapsed = time.time() - start
    print(f"\n  Module 4 completed in {elapsed:.1f}s")

    return docking_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 4: Molecular Docking")
    parser.add_argument("--compound", type=str, default=None,
                        help="Dock a single compound by name")
    parser.add_argument("--exhaustiveness", type=int, default=32,
                        help="Vina exhaustiveness (default: 32)")
    args = parser.parse_args()

    run_module4(compound=args.compound, exhaustiveness=args.exhaustiveness)
