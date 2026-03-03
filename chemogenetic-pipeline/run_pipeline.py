"""
AI-Accelerated Chemogenetic Actuator Design — Full Pipeline
Orchestrates all modules in sequence:
  Module 1: Structural Analysis (AlphaFold / demo mode)
  Module 2: Property Evaluation (RDKit)
  Module 3: Dashboard (Streamlit — launched separately)
  Module 4: Molecular Docking (AutoDock Vina)
  Module 5: Virtual Screening (ZINC library)
  Module 6: ML ADMET Prediction (TDC datasets)
  Module 7: Selectivity Prediction (ChEMBL data)
  Module 8: Molecular Dynamics (OpenMM)

Usage:
    python3 run_pipeline.py              # Run all modules (demo mode for Module 1)
    python3 run_pipeline.py --no-demo    # Skip demo mode for Module 1
    python3 run_pipeline.py --dashboard  # Launch Streamlit dashboard after
    python3 run_pipeline.py --modules 1 2 6   # Run specific modules only
    python3 run_pipeline.py --skip 8     # Skip specific modules
"""

import os
import sys
import time
import argparse
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def run_module1(demo_mode=True):
    """Run Module 1: Structural Analysis."""
    print("\n" + "=" * 60)
    print("  MODULE 1: STRUCTURAL ANALYSIS")
    print("=" * 60)
    start = time.time()

    from src.module1.run_module1 import run_analysis
    results = run_analysis(demo_mode=demo_mode)

    elapsed = time.time() - start
    print(f"\n  Module 1 completed in {elapsed:.1f}s")
    return results


def run_module2():
    """Run Module 2: Actuator Property Evaluation."""
    print("\n" + "=" * 60)
    print("  MODULE 2: ACTUATOR PROPERTY EVALUATION")
    print("=" * 60)
    start = time.time()

    from src.module2.evaluate_actuators import (
        evaluate_all_compounds,
        save_results,
        generate_2d_images,
    )
    from src.module2.compounds import load_compounds
    from src.utils.plotting import generate_all_figures

    compounds = load_compounds()
    print(f"\n  Loaded {len(compounds)} compounds")

    results = evaluate_all_compounds(compounds)
    print("\n--- Actuator Properties ---")
    print(results.to_string(index=False))

    save_results(results)
    generate_2d_images(compounds)

    print("\n--- Generating Publication Figures ---")
    figure_paths = generate_all_figures(results)
    print(f"  Generated {len(figure_paths)} figures")

    elapsed = time.time() - start
    print(f"\n  Module 2 completed in {elapsed:.1f}s")
    return results


def run_module4():
    """Run Module 4: Molecular Docking."""
    start = time.time()
    from src.module4.run_module4 import run_module4 as _run
    results = _run(exhaustiveness=32)
    elapsed = time.time() - start
    print(f"\n  Module 4 total time: {elapsed:.1f}s")
    return results


def run_module5():
    """Run Module 5: Virtual Screening."""
    start = time.time()
    from src.module5.run_module5 import run_module5 as _run
    results = _run(n_compounds=500, n_cpus=1)
    elapsed = time.time() - start
    print(f"\n  Module 5 total time: {elapsed:.1f}s")
    return results


def run_module6():
    """Run Module 6: ML ADMET Prediction."""
    start = time.time()
    from src.module6.run_module6 import run_module6 as _run
    results = _run()
    elapsed = time.time() - start
    print(f"\n  Module 6 total time: {elapsed:.1f}s")
    return results


def run_module7():
    """Run Module 7: Selectivity Prediction."""
    start = time.time()
    from src.module7.run_module7 import run_module7 as _run
    results, profiles = _run()
    elapsed = time.time() - start
    print(f"\n  Module 7 total time: {elapsed:.1f}s")
    return results


def run_module8():
    """Run Module 8: Molecular Dynamics."""
    start = time.time()
    from src.module8.run_module8 import run_module8 as _run
    results = _run(compound="DCZ", length_ns=50)
    elapsed = time.time() - start
    print(f"\n  Module 8 total time: {elapsed:.1f}s")
    return results


def print_summary(results):
    """Print a final pipeline summary."""
    print("\n" + "=" * 60)
    print("  PIPELINE SUMMARY")
    print("=" * 60)

    # Module 1 summary
    m1 = results.get(1)
    if m1:
        structures = m1.get("structures", {})
        plddt_data = m1.get("plddt_dict", {})
        print(f"\n  Module 1 — Structural Analysis:")
        print(f"    Structures: {len(structures)}")
        print(f"    pLDDT analyses: {len(plddt_data)}")

    # Module 2 summary
    m2 = results.get(2)
    if m2 is not None:
        print(f"\n  Module 2 — Property Evaluation:")
        print(f"    Compounds evaluated: {len(m2)}")
        penetrant = m2[m2["bbb_predicted"] == "Penetrant"]
        print(f"    BBB-penetrant (rule-based): {len(penetrant)}")

    # Module 4 summary
    m4 = results.get(4)
    if m4 is not None:
        print(f"\n  Module 4 — Molecular Docking:")
        print(f"    Compounds docked: {len(m4)}")
        if "affinity_kcal_mol" in m4.columns:
            best = m4.loc[m4["affinity_kcal_mol"].idxmin()]
            print(f"    Best binder: {best['name']} ({best['affinity_kcal_mol']:.2f} kcal/mol)")

    # Module 5 summary
    m5 = results.get(5)
    if m5 is not None:
        if isinstance(m5, tuple):
            screening_df, hits_df = m5
            print(f"\n  Module 5 — Virtual Screening:")
            print(f"    Compounds screened: {len(screening_df)}")
            print(f"    Top hits: {len(hits_df)}")

    # Module 6 summary
    m6 = results.get(6)
    if m6 is not None:
        print(f"\n  Module 6 — ADMET Prediction:")
        print(f"    Compounds predicted: {len(m6)}")
        if "bbb_ml_class" in m6.columns:
            bbb_pen = (m6["bbb_ml_class"] == "Penetrant").sum()
            print(f"    BBB-penetrant (ML): {bbb_pen}")

    # Module 7 summary
    m7 = results.get(7)
    if m7 is not None:
        print(f"\n  Module 7 — Selectivity:")
        print(f"    Compounds profiled: {len(m7)}")
        if "n_off_targets" in m7.columns:
            flagged = (m7["n_off_targets"] > 0).sum()
            print(f"    With off-target flags: {flagged}")

    # Module 8 summary
    m8 = results.get(8)
    if m8 is not None:
        print(f"\n  Module 8 — Molecular Dynamics:")
        if isinstance(m8, dict):
            stab = m8.get("binding_stability")
            if stab is not None:
                print(f"    DCZ binding stability: {stab*100:.1f}%")

    # Files generated
    results_dir = os.path.join(PROJECT_ROOT, "data", "results")
    figures_dir = os.path.join(PROJECT_ROOT, "figures")
    n_results = len([f for f in os.listdir(results_dir) if f.endswith(".csv")]) if os.path.isdir(results_dir) else 0
    n_figures = len([f for f in os.listdir(figures_dir) if f.endswith(".png")]) if os.path.isdir(figures_dir) else 0
    print(f"\n  Output files:")
    print(f"    CSV results: {n_results}")
    print(f"    Figures: {n_figures}")

    print(f"\n  Dashboard: streamlit run app.py")
    print("=" * 60)
    print("  Pipeline complete.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="AI-Accelerated Chemogenetic Actuator Design Pipeline"
    )
    parser.add_argument(
        "--no-demo", action="store_true",
        help="Skip demo mode for Module 1"
    )
    parser.add_argument(
        "--dashboard", action="store_true",
        help="Launch Streamlit dashboard after pipeline completes"
    )
    parser.add_argument(
        "--modules", nargs="*", type=int, default=None,
        help="Run specific modules only (e.g., --modules 4 6 7)"
    )
    parser.add_argument(
        "--skip", nargs="*", type=int, default=None,
        help="Skip specific modules (e.g., --skip 8)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  AI-ACCELERATED CHEMOGENETIC ACTUATOR DESIGN")
    print("  Full Pipeline Execution")
    print("=" * 60)

    total_start = time.time()

    # Determine which modules to run
    all_modules = [1, 2, 4, 5, 6, 7, 8]
    if args.modules:
        modules_to_run = args.modules
    elif args.skip:
        modules_to_run = [m for m in all_modules if m not in args.skip]
    else:
        modules_to_run = all_modules

    print(f"  Modules to run: {modules_to_run}")

    results = {}

    module_runners = {
        1: lambda: run_module1(demo_mode=not args.no_demo),
        2: run_module2,
        4: run_module4,
        5: run_module5,
        6: run_module6,
        7: run_module7,
        8: run_module8,
    }

    for mod_num in modules_to_run:
        if mod_num in module_runners:
            try:
                results[mod_num] = module_runners[mod_num]()
            except Exception as e:
                print(f"\n  ERROR in Module {mod_num}: {e}")
                print(f"  Continuing with remaining modules...")
                results[mod_num] = None

    # Summary
    total_elapsed = time.time() - total_start
    print_summary(results)
    print(f"\n  Total time: {total_elapsed:.1f}s")

    # Optional dashboard launch
    if args.dashboard:
        print("\n  Launching Streamlit dashboard...")
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "app.py",
             "--server.headless", "true"],
            cwd=PROJECT_ROOT,
        )


if __name__ == "__main__":
    main()
