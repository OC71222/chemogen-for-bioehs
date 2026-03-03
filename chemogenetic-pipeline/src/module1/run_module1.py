"""
Module 1: Structural Analysis — Main Orchestrator
Runs the full Module 1 pipeline:
  1. Prepare AlphaFold input JSON files
  2. Check for available AlphaFold output
  3. Parse predicted structures
  4. Calculate RMSD against experimental structures (if available)
  5. Extract and analyze pLDDT scores
  6. Generate demo structures (if no AlphaFold output)

Usage:
    python3 -m src.module1.run_module1           # Full run (or demo mode)
    python3 -m src.module1.run_module1 --prep    # Only generate AF input JSONs
    python3 -m src.module1.run_module1 --demo    # Force demo mode
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.module1.alphafold_prep import (
    generate_all_inputs,
    check_alphafold_outputs,
    get_submission_instructions,
    COMPOUND_SMILES,
)
from src.module1.structure_parser import parse_structure, get_structure_summary
from src.module1.rmsd_calculator import (
    calculate_rmsd_from_files,
    save_rmsd_results,
    classify_rmsd,
)
from src.module1.plddt_analysis import (
    extract_plddt,
    plddt_summary,
    classify_plddt,
    save_plddt_scores,
    plot_plddt,
    plot_plddt_comparison,
)

STRUCTURES_DIR = os.path.join(PROJECT_ROOT, "data", "structures")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "results")


def generate_demo_structures():
    """Generate RDKit 3D conformers as demo PDB files.

    Creates simple 3D conformers for each compound when AlphaFold
    output is not yet available. These are NOT AlphaFold predictions.

    Returns:
        List of paths to generated PDB files.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    demo_dir = os.path.join(STRUCTURES_DIR, "predicted")
    os.makedirs(demo_dir, exist_ok=True)

    paths = []
    for name, smiles in COMPOUND_SMILES.items():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"  WARNING: Could not parse SMILES for {name}")
            continue

        mol = Chem.AddHs(mol)
        result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        if result != 0:
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.MMFFOptimizeMolecule(mol)

        safe_name = name.replace(" ", "_")
        pdb_path = os.path.join(demo_dir, f"{safe_name}.pdb")
        Chem.MolToPDBFile(mol, pdb_path)
        paths.append(pdb_path)
        print(f"  Generated demo structure: {pdb_path}")

    return paths


def generate_demo_plddt_scores():
    """Generate synthetic pLDDT scores for demo mode.

    Creates realistic-looking pLDDT profiles that demonstrate
    the analysis pipeline without actual AlphaFold data.

    Returns:
        Dict mapping compound name -> pLDDT DataFrame.
    """
    np.random.seed(42)
    plddt_dict = {}

    # Simulate different confidence profiles per compound
    profiles = {
        "CNO": {"mean": 72, "std": 12},
        "Clozapine": {"mean": 68, "std": 15},
        "DCZ": {"mean": 85, "std": 8},
        "Compound 21": {"mean": 65, "std": 18},
        "Olanzapine": {"mean": 75, "std": 10},
        "Perlapine": {"mean": 80, "std": 9},
    }

    for name, profile in profiles.items():
        n_residues = 590  # hM3Dq receptor length
        scores = np.clip(
            np.random.normal(profile["mean"], profile["std"], n_residues),
            0, 100
        )
        # Add realistic features: lower confidence at termini
        taper = np.ones(n_residues)
        taper[:30] = np.linspace(0.5, 1.0, 30)
        taper[-30:] = np.linspace(1.0, 0.5, 30)
        scores = np.clip(scores * taper, 0, 100)

        df = pd.DataFrame({
            "resseq": range(1, n_residues + 1),
            "resname": ["ALA"] * n_residues,  # Placeholder
            "chain_id": ["A"] * n_residues,
            "plddt": np.round(scores, 2),
        })
        plddt_dict[name] = df

    return plddt_dict


def run_prep_only():
    """Step 1: Generate AlphaFold input JSON files."""
    print("\n--- Step 1: Generating AlphaFold Input Files ---")
    paths = generate_all_inputs()
    print(f"\nGenerated {len(paths)} input files.")
    print(get_submission_instructions())


def run_analysis(demo_mode=False):
    """Run full Module 1 analysis pipeline.

    Args:
        demo_mode: If True, generate demo structures and synthetic pLDDT data.
    """
    print("=" * 60)
    print("Module 1: Structural Analysis Pipeline")
    print("=" * 60)

    # Step 1: Generate AlphaFold inputs
    print("\n--- Step 1: AlphaFold Input Preparation ---")
    generate_all_inputs()

    # Step 2: Check for AlphaFold outputs
    print("\n--- Step 2: Checking AlphaFold Outputs ---")
    available = check_alphafold_outputs()
    found = {k: v for k, v in available.items() if v is not None}
    missing = {k: v for k, v in available.items() if v is None}

    print(f"  Found: {len(found)}/{len(available)} structures")
    for name, path in found.items():
        print(f"    {name}: {path}")

    if missing and not demo_mode:
        print(f"\n  Missing structures for: {list(missing.keys())}")
        print("  Use --demo flag to generate demo structures, or")
        print("  submit jobs to AlphaFold Server and place output in data/structures/predicted/")

    # Step 3: Generate demo structures if needed
    if demo_mode or (not found):
        print("\n--- Step 3: Generating Demo Structures (RDKit) ---")
        demo_paths = generate_demo_structures()
        # Refresh available structures
        available = check_alphafold_outputs()
        found = {k: v for k, v in available.items() if v is not None}

    # Step 4: Parse structures and show summaries
    print("\n--- Step 4: Structure Summaries ---")
    for name, path in found.items():
        if path is None:
            continue
        try:
            struct = parse_structure(path)
            summary = get_structure_summary(struct)
            print(f"  {name}: {summary['n_atoms']} atoms, "
                  f"{summary['n_residues']} residues, "
                  f"{summary['n_chains']} chain(s)")
        except Exception as e:
            print(f"  {name}: Error parsing — {e}")

    # Step 5: RMSD calculation (if experimental structures exist)
    print("\n--- Step 5: RMSD Calculation ---")
    experimental_dir = os.path.join(STRUCTURES_DIR, "experimental")
    rmsd_results = []

    # PDB ID → compound name mapping for hM3Dq DREADD cryo-EM structures
    # 8E9W: hM3Dq + DCZ, 8E9Y: hM3Dq + CNO — compatible residue numbering
    # 7WC7: hM3Dq + DCZ, 7WC8: hM3Dq + Compound 21 — fusion-construct residue numbering
    _PDB_TO_COMPOUND = {
        "8E9W": "DCZ",
        "8E9Y": "CNO",
        "7WC7": "DCZ",
        "7WC8": "Compound 21",
    }

    for name, pred_path in found.items():
        if pred_path is None:
            continue
        # Look for experimental structure by compound name or PDB ID mapping
        safe_name = name.replace(" ", "_")
        exp_paths = []

        # Direct name match (e.g. CNO.pdb)
        for ext in [".pdb", ".cif"]:
            candidate = os.path.join(experimental_dir, f"{safe_name}{ext}")
            if os.path.exists(candidate):
                exp_paths.append(candidate)

        # PDB ID match — collect ALL matching PDB IDs for this compound
        for pdb_id, compound in _PDB_TO_COMPOUND.items():
            if compound == name:
                for ext in [".pdb", ".cif"]:
                    candidate = os.path.join(experimental_dir, f"{pdb_id}{ext}")
                    if os.path.exists(candidate) and candidate not in exp_paths:
                        exp_paths.append(candidate)

        if exp_paths:
            for exp_path in exp_paths:
                exp_basename = os.path.splitext(os.path.basename(exp_path))[0]
                label = f"{name} vs {exp_basename}"
                # Use chain A for both (receptor chain in cryo-EM complexes)
                result = calculate_rmsd_from_files(exp_path, pred_path,
                                                   ref_chain="A", pred_chain="A")
                result["name"] = label
                quality = classify_rmsd(result["rmsd"])
                print(f"  {label}: RMSD = {result['rmsd']:.3f} Å ({quality}), "
                      f"{result['n_atoms']} aligned atoms")
                rmsd_results.append(result)
        else:
            print(f"  {name}: No experimental structure available for comparison")

    if rmsd_results:
        rmsd_df = pd.DataFrame(rmsd_results)
        save_rmsd_results(rmsd_df)
    else:
        print("  No RMSD comparisons possible (no experimental structures).")

    # Step 6: pLDDT analysis
    print("\n--- Step 6: pLDDT Confidence Analysis ---")
    if demo_mode or not found:
        print("  Using synthetic pLDDT scores (demo mode)")
        plddt_dict = generate_demo_plddt_scores()
    else:
        plddt_dict = {}
        for name, path in found.items():
            if path is None:
                continue
            try:
                plddt_df = extract_plddt(path)
                if len(plddt_df) > 0 and plddt_df["plddt"].sum() > 0:
                    plddt_dict[name] = plddt_df
                else:
                    print(f"  {name}: No pLDDT data in structure file")
            except Exception as e:
                print(f"  {name}: Error extracting pLDDT — {e}")

        # Fall back to demo if no real pLDDT data
        if not plddt_dict:
            print("  No pLDDT data found in structure files. Using synthetic scores.")
            plddt_dict = generate_demo_plddt_scores()

    # Save and visualize pLDDT
    for name, plddt_df in plddt_dict.items():
        save_plddt_scores(plddt_df, name)
        plot_plddt(plddt_df, name)
        summary = plddt_summary(plddt_df)
        confidence = classify_plddt(summary["mean"])
        print(f"  {name}: mean pLDDT = {summary['mean']:.1f} ({confidence}), "
              f"{summary['pct_confident']:.0f}% confident residues")

    if len(plddt_dict) > 1:
        plot_plddt_comparison(plddt_dict)

    # Summary
    print("\n" + "=" * 60)
    print("Module 1 Analysis Complete")
    print("=" * 60)
    print(f"  Structures processed: {len(found)}")
    print(f"  RMSD comparisons: {len(rmsd_results)}")
    print(f"  pLDDT analyses: {len(plddt_dict)}")

    if demo_mode:
        print("\n  NOTE: Demo mode was used. Results use RDKit conformers")
        print("  and synthetic pLDDT scores — NOT AlphaFold predictions.")

    return {
        "structures": found,
        "rmsd_results": rmsd_results,
        "plddt_dict": plddt_dict,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 1: Structural Analysis")
    parser.add_argument("--prep", action="store_true", help="Only generate AlphaFold input files")
    parser.add_argument("--demo", action="store_true", help="Force demo mode with synthetic data")
    args = parser.parse_args()

    if args.prep:
        run_prep_only()
    else:
        run_analysis(demo_mode=args.demo)
