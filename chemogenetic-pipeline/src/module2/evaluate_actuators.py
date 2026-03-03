"""
Module 2: Actuator Evaluation Engine
Calculates molecular properties, BBB permeability, and Lipinski compliance
for DREADD actuator compounds using RDKit.
"""

import os
import sys
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw

# Add project root to path for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.module2.compounds import load_compounds


def calculate_properties(smiles):
    """Calculate all 8 molecular descriptors from a SMILES string.

    Returns dict of properties or None if SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return {
        "mw": round(Descriptors.MolWt(mol), 2),
        "logp": round(Descriptors.MolLogP(mol), 2),
        "tpsa": round(Descriptors.TPSA(mol), 2),
        "hbd": Descriptors.NumHDonors(mol),
        "hba": Descriptors.NumHAcceptors(mol),
        "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "aromatic_rings": Descriptors.NumAromaticRings(mol),
        "fsp3": round(Descriptors.FractionCSP3(mol), 3),
    }


def predict_bbb(props):
    """Rule-based BBB permeability prediction.

    A compound is BBB-penetrant only if ALL four criteria are met:
      - MW < 450 Da
      - LogP between 1.0 and 3.0 (inclusive)
      - TPSA < 90 Å²
      - HBD <= 3
    """
    if (
        props["mw"] < 450
        and 1.0 <= props["logp"] <= 3.0
        and props["tpsa"] < 90
        and props["hbd"] <= 3
    ):
        return "Penetrant"
    return "Non-penetrant"


def bbb_criteria_detail(props):
    """Return per-criterion pass/fail for BBB prediction."""
    return {
        "mw_pass": props["mw"] < 450,
        "logp_pass": 1.0 <= props["logp"] <= 3.0,
        "tpsa_pass": props["tpsa"] < 90,
        "hbd_pass": props["hbd"] <= 3,
    }


def count_lipinski_violations(props):
    """Count Lipinski Rule of Five violations.

    Violations: MW >= 500, LogP >= 5, HBD > 5, HBA > 10.
    0-1 violations = drug-like; 2+ = flagged.
    """
    violations = 0
    if props["mw"] >= 500:
        violations += 1
    if props["logp"] >= 5:
        violations += 1
    if props["hbd"] > 5:
        violations += 1
    if props["hba"] > 10:
        violations += 1
    return violations


def evaluate_all_compounds(compounds_df=None):
    """Run full evaluation pipeline on all compounds.

    Returns DataFrame with all properties, BBB prediction, and Lipinski score.
    """
    if compounds_df is None:
        compounds_df = load_compounds()

    results = []
    for _, row in compounds_df.iterrows():
        name = row["name"]
        smiles = row["smiles"]

        props = calculate_properties(smiles)
        if props is None:
            print(f"WARNING: Failed to parse SMILES for {name}: {smiles}")
            continue

        props["name"] = name
        props["smiles"] = smiles
        props["bbb_predicted"] = predict_bbb(props)
        props["lipinski_violations"] = count_lipinski_violations(props)

        results.append(props)

    df = pd.DataFrame(results)

    # Reorder columns to match PRD schema
    column_order = [
        "name", "mw", "logp", "tpsa", "hbd", "hba",
        "rotatable_bonds", "aromatic_rings", "fsp3",
        "bbb_predicted", "lipinski_violations",
    ]
    df = df[column_order]
    return df


def save_results(df, output_path=None):
    """Save evaluation results to CSV."""
    if output_path is None:
        output_path = os.path.join(PROJECT_ROOT, "data", "results", "actuator_properties.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved actuator properties to {output_path}")
    return output_path


def generate_2d_images(compounds_df=None, output_dir=None):
    """Generate 2D molecular structure images for each compound."""
    if compounds_df is None:
        compounds_df = load_compounds()
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "figures")
    os.makedirs(output_dir, exist_ok=True)

    for _, row in compounds_df.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None:
            continue
        img = Draw.MolToImage(mol, size=(400, 300))
        path = os.path.join(output_dir, f"mol_{row['name'].replace(' ', '_')}.png")
        img.save(path)
        print(f"Saved 2D structure: {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("Module 2: Actuator Evaluation Engine")
    print("=" * 60)

    # Load compounds
    compounds = load_compounds()
    print(f"\nLoaded {len(compounds)} compounds from compounds.csv")

    # Run evaluation
    results = evaluate_all_compounds(compounds)

    # Display results
    print("\n--- Actuator Properties ---")
    print(results.to_string(index=False))

    # Show BBB criteria detail
    print("\n--- BBB Criteria Breakdown ---")
    for _, row in compounds.iterrows():
        props = calculate_properties(row["smiles"])
        if props is None:
            continue
        criteria = bbb_criteria_detail(props)
        status = predict_bbb(props)
        checks = " | ".join(
            f"{'✓' if v else '✗'} {k.replace('_pass', '').upper()}"
            for k, v in criteria.items()
        )
        print(f"  {row['name']:15s} → {status:15s} [{checks}]")

    # Save CSV
    save_results(results)

    # Generate 2D structure images
    generate_2d_images(compounds)

    print("\nModule 2 evaluation complete.")
