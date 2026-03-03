"""
Module 5: Virtual Screening — Orchestrator
Download library → filter → screen → analyze hits.
"""

import os
import sys
import time
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)


def run_module5(n_compounds=500, n_cpus=1, verbose=True):
    """Full Module 5 pipeline.

    Args:
        n_compounds: Number of compounds in screening library
        n_cpus: Number of CPUs for parallel docking
        verbose: Print progress
    """
    from src.module5.library_prep import prepare_screening_library
    from src.module5.screen_engine import screen_library, save_screening_results
    from src.module5.hit_analysis import analyze_hits, save_hit_report
    from src.module4.receptor_prep import prepare_receptor

    print("\n" + "=" * 60)
    print("  MODULE 5: VIRTUAL SCREENING")
    print("=" * 60)
    start = time.time()

    # Step 1: Prepare screening library
    print("\n--- Step 1: Library Preparation ---")
    library_df = prepare_screening_library(n_compounds=n_compounds, verbose=verbose)
    print(f"    Library size: {len(library_df)} compounds")

    # Step 2: Prepare receptor (reuse from Module 4 if available)
    print("\n--- Step 2: Receptor Preparation ---")
    receptor_info = prepare_receptor(pdb_id="8E9W", verbose=verbose)

    # Step 3: Screen
    print("\n--- Step 3: Virtual Screening ---")
    results_df = screen_library(
        library_df, receptor_info,
        exhaustiveness=8,
        n_cpus=n_cpus,
        verbose=verbose,
    )

    # Step 4: Save screening results
    save_screening_results(results_df)

    # Step 5: Analyze hits
    print("\n--- Step 4: Hit Analysis ---")
    hits_df = analyze_hits(results_df, verbose=verbose)
    save_hit_report(hits_df)

    elapsed = time.time() - start
    print(f"\n  Module 5 completed in {elapsed:.1f}s")

    return results_df, hits_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 5: Virtual Screening")
    parser.add_argument("--n_compounds", type=int, default=500,
                        help="Number of compounds in screening library")
    parser.add_argument("--n_cpus", type=int, default=1,
                        help="Number of CPUs for parallel docking")
    args = parser.parse_args()

    run_module5(n_compounds=args.n_compounds, n_cpus=args.n_cpus)
