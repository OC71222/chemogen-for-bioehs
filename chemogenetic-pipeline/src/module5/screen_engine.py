"""
Module 5: Virtual Screening Engine
Parallel docking of compound library against hM3Dq receptor.
"""

import os
import sys
import time
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)


def _dock_single(args):
    """Dock a single compound (for parallel execution).

    Args:
        args: tuple of (compound_id, smiles, receptor_pdbqt, center, box_size, exhaustiveness)

    Returns:
        dict with docking results
    """
    compound_id, smiles, receptor_pdbqt, center, box_size, exhaustiveness = args

    from src.module4.ligand_prep import prepare_ligand
    from src.module4.docking_engine import dock_compound

    # Prepare ligand
    ligand_info = prepare_ligand(smiles, name=compound_id)
    if ligand_info is None:
        return {"compound_id": compound_id, "smiles": smiles,
                "affinity": None, "success": False}

    # Dock
    pdbqt = ligand_info.get("pdbqt_string", "")
    result = dock_compound(
        receptor_pdbqt, pdbqt, center, box_size,
        exhaustiveness=exhaustiveness,
    )

    return {
        "compound_id": compound_id,
        "smiles": smiles,
        "affinity": result.get("best_affinity"),
        "n_poses": result.get("n_poses", 0),
        "success": result.get("success", False),
        "estimated": result.get("estimated", False),
    }


def screen_library(library_df, receptor_info, exhaustiveness=8, n_cpus=1, verbose=True):
    """Screen compound library against receptor.

    Args:
        library_df: DataFrame with 'compound_id' and 'smiles' columns
        receptor_info: dict from receptor_prep.prepare_receptor()
        exhaustiveness: Vina exhaustiveness (8 for screening speed)
        n_cpus: Number of parallel processes
        verbose: Print progress

    Returns:
        DataFrame with screening results
    """
    receptor_pdbqt = receptor_info["pdbqt"]
    bs = receptor_info["binding_site"]
    center = (bs["center_x"], bs["center_y"], bs["center_z"])
    box_size = (bs["size_x"], bs["size_y"], bs["size_z"])

    n_total = len(library_df)
    if verbose:
        print(f"\n    Screening {n_total} compounds (exhaustiveness={exhaustiveness}, cpus={n_cpus})")

    # Prepare arguments
    args_list = [
        (
            row.get("compound_id", f"CPD_{i}"),
            row["smiles"],
            receptor_pdbqt,
            center,
            box_size,
            exhaustiveness,
        )
        for i, (_, row) in enumerate(library_df.iterrows())
    ]

    results = []
    start_time = time.time()

    if n_cpus > 1:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=n_cpus) as executor:
            futures = {executor.submit(_dock_single, args): args[0] for args in args_list}

            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                results.append(result)

                if verbose and (i + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    remaining = (n_total - i - 1) / rate if rate > 0 else 0
                    print(f"      Progress: {i+1}/{n_total} "
                          f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")
    else:
        # Sequential execution
        for i, args in enumerate(args_list):
            result = _dock_single(args)
            results.append(result)

            if verbose and (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (n_total - i - 1) / rate if rate > 0 else 0
                print(f"      Progress: {i+1}/{n_total} "
                      f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

    results_df = pd.DataFrame(results)
    elapsed = time.time() - start_time

    if verbose:
        n_success = results_df["success"].sum()
        print(f"\n    Screening complete: {n_success}/{n_total} successful "
              f"in {elapsed:.1f}s ({n_total/elapsed:.1f} compounds/s)")

    return results_df


def save_screening_results(df, output_path=None):
    """Save screening results to CSV."""
    if output_path is None:
        output_path = os.path.join(PROJECT_ROOT, "data", "results", "screening_results.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"    Saved screening results: {output_path}")
    return output_path
