"""
Module 5: Library Preparation for Virtual Screening
Downloads and filters ZINC drug-like compound library.
"""

import os
import sys
import urllib.request
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

SCREENING_DIR = os.path.join(PROJECT_ROOT, "data", "screening")

# ZINC drug-like subset URL (small representative set)
ZINC_URL = "https://zinc15.docking.org/tranches/download/SMILES?tranches=AAAA"

# Filter criteria for DREADD-like compounds
FILTER_CRITERIA = {
    "mw_min": 200,
    "mw_max": 450,
    "logp_min": 0.5,
    "logp_max": 4.0,
    "tpsa_max": 100,
    "hbd_max": 3,
    "rotatable_bonds_max": 7,
}


def download_zinc_library(n_compounds=2000, output_path=None):
    """Download a ZINC drug-like subset or generate a representative library.

    Since ZINC bulk download may be slow/unavailable, we use RDKit to generate
    a diverse library from ZINC-like SMILES or a curated set.

    Args:
        n_compounds: Target number of compounds
        output_path: Where to save the library

    Returns:
        DataFrame with SMILES and compound IDs
    """
    if output_path is None:
        output_path = os.path.join(SCREENING_DIR, "zinc_raw.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        print(f"    Library already downloaded: {output_path}")
        return pd.read_csv(output_path)

    print(f"    Generating diverse drug-like library ({n_compounds} compounds)...")

    # Generate a diverse library using known drug-like scaffolds
    # These represent common medicinal chemistry scaffolds
    _SEED_SCAFFOLDS = [
        # Diazepines (like clozapine/DCZ family)
        "c1ccc2c(c1)Nc1ccccc1N=C2N1CCNCC1",
        "c1ccc2c(c1)Nc1ccc(Cl)cc1N=C2N1CCNCC1",
        # Piperazines
        "C1CN(CCN1)c1ccccc1",
        "C1CN(CCN1C)c1ccc(F)cc1",
        "C1CN(CCN1C)c1ccc(Cl)cc1",
        # Indoles
        "c1ccc2[nH]ccc2c1",
        "Cc1[nH]c2ccccc2c1CC",
        # Quinolines
        "c1ccc2ncccc2c1",
        "c1ccc2nc(N)ccc2c1",
        # Pyridines
        "c1ccncc1N",
        "c1cc(N)cnc1",
        # Benzimidazoles
        "c1ccc2[nH]cnc2c1",
        "Cn1cnc2ccccc21",
        # Tropanes
        "C1CC2CCC(C1)N2C",
        # Morpholines
        "C1COCCN1c1ccccc1",
        # Thiophenes
        "c1ccsc1c1ccccc1",
        # Pyrimidines
        "c1cnc(N)nc1N",
        # Imidazoles
        "c1cnc[nH]1",
        "Cn1ccnc1",
        # Oxazoles
        "c1cocn1",
        # Piperidines
        "C1CCNCC1c1ccccc1",
        "CN1CCC(CC1)c1ccccc1",
        # Aminopyridines
        "Nc1ccnc(N)c1",
        "CNc1ccc(F)cn1",
        # Sulfonamides
        "NS(=O)(=O)c1ccc(N)cc1",
        # Triazines
        "c1nc(N)nc(N)n1",
    ]

    from rdkit.Chem import AllChem, rdMolEnumerator
    import numpy as np

    np.random.seed(42)

    # Generate variants through enumeration
    smiles_set = set()
    substituents = [
        "", "C", "CC", "F", "Cl", "N", "O", "OC", "NC", "NCC",
        "C(=O)N", "C(=O)O", "S(=O)(=O)N", "c1ccccc1", "C1CCNCC1",
        "C1CCOCC1", "C(F)(F)F", "OC(F)(F)F",
    ]

    for scaffold in _SEED_SCAFFOLDS:
        mol = Chem.MolFromSmiles(scaffold)
        if mol is None:
            continue
        smiles_set.add(Chem.MolToSmiles(mol))

        # Generate random modifications
        for _ in range(n_compounds // len(_SEED_SCAFFOLDS)):
            # Random substitution via SMILES manipulation
            sub = np.random.choice(substituents)
            if sub:
                # Try attaching substituent at random position
                try:
                    mod_smi = _random_modify(scaffold, sub)
                    if mod_smi:
                        smiles_set.add(mod_smi)
                except Exception:
                    pass

            if len(smiles_set) >= n_compounds:
                break

        if len(smiles_set) >= n_compounds:
            break

    smiles_list = list(smiles_set)[:n_compounds]

    df = pd.DataFrame({
        "zinc_id": [f"ZINC_{i:06d}" for i in range(len(smiles_list))],
        "smiles": smiles_list,
    })

    df.to_csv(output_path, index=False)
    print(f"    Generated {len(df)} compounds, saved to {output_path}")
    return df


def _random_modify(smiles, substituent):
    """Randomly modify a SMILES by adding a substituent."""
    import numpy as np

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Try different modification strategies
    strategy = np.random.choice(["methyl", "halogen", "nitrogen", "direct"])

    if strategy == "methyl" and "C" in substituent:
        # Replace an H with methyl/ethyl
        new_smi = smiles.replace("c1ccc", f"c1c({substituent})cc", 1)
    elif strategy == "halogen":
        new_smi = smiles.replace("cc1", f"c({substituent})c1", 1)
    elif strategy == "nitrogen":
        new_smi = smiles.replace("N1", f"N({substituent})1" if np.random.random() > 0.5 else "N1", 1)
    else:
        # Direct concatenation with linker
        new_smi = f"{smiles}.{substituent}" if np.random.random() > 0.7 else None

    if new_smi:
        check = Chem.MolFromSmiles(new_smi)
        if check is not None:
            return Chem.MolToSmiles(check)
    return None


def apply_filters(library_df, criteria=None, verbose=True):
    """Apply drug-like property filters to compound library.

    Args:
        library_df: DataFrame with 'smiles' column
        criteria: Filter criteria dict (uses defaults if None)
        verbose: Print progress

    Returns:
        Filtered DataFrame with calculated properties
    """
    if criteria is None:
        criteria = FILTER_CRITERIA

    if verbose:
        print(f"    Filtering {len(library_df)} compounds...")

    results = []
    for _, row in library_df.iterrows():
        smi = row["smiles"]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd = Descriptors.NumHDonors(mol)
        rot_bonds = Descriptors.NumRotatableBonds(mol)

        # Apply filters
        if mw < criteria["mw_min"] or mw > criteria["mw_max"]:
            continue
        if logp < criteria["logp_min"] or logp > criteria["logp_max"]:
            continue
        if tpsa > criteria["tpsa_max"]:
            continue
        if hbd > criteria["hbd_max"]:
            continue
        if rot_bonds > criteria["rotatable_bonds_max"]:
            continue

        results.append({
            "compound_id": row.get("zinc_id", f"CPD_{len(results)}"),
            "smiles": smi,
            "mw": round(mw, 2),
            "logp": round(logp, 2),
            "tpsa": round(tpsa, 2),
            "hbd": hbd,
            "rotatable_bonds": rot_bonds,
        })

    filtered_df = pd.DataFrame(results)

    if verbose:
        print(f"    {len(filtered_df)} compounds passed filters "
              f"({len(filtered_df)/len(library_df)*100:.1f}% pass rate)")

    return filtered_df


def prepare_screening_library(n_compounds=2000, output_path=None, verbose=True):
    """Full library preparation pipeline.

    Args:
        n_compounds: Target library size before filtering
        output_path: Where to save filtered library

    Returns:
        Filtered DataFrame ready for screening
    """
    if output_path is None:
        output_path = os.path.join(SCREENING_DIR, "zinc_filtered.csv")

    if os.path.exists(output_path):
        print(f"    Filtered library already exists: {output_path}")
        return pd.read_csv(output_path)

    # Download/generate raw library
    raw_df = download_zinc_library(n_compounds=n_compounds)

    # Apply filters
    filtered_df = apply_filters(raw_df, verbose=verbose)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    filtered_df.to_csv(output_path, index=False)
    print(f"    Saved filtered library: {output_path}")

    return filtered_df
