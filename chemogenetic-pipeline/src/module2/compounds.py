"""
Compound definitions for the 6 known DREADD actuators.
SMILES sourced from PubChem (verified by name-based lookup Feb 2026).
"""

import os
import pandas as pd

# Canonical SMILES dictionary — fallback if compounds.csv is unavailable
COMPOUNDS = {
    "CNO": {
        "pubchem_cid": 135445691,
        "smiles": "C[N+]1(CCN(CC1)C2=NC3=C(C=CC(=C3)Cl)NC4=CC=CC=C42)[O-]",
        "role": "baseline",
        "description": "Clozapine-N-oxide — original DREADD actuator; converts to clozapine in vivo",
    },
    "Clozapine": {
        "pubchem_cid": 135398737,
        "smiles": "CN1CCN(CC1)C2=NC3=C(C=CC(=C3)Cl)NC4=CC=CC=C42",
        "role": "metabolite",
        "description": "Off-target metabolite of CNO; binds D2, 5-HT2A, H1, muscarinic receptors",
    },
    "DCZ": {
        "pubchem_cid": 16103,
        "smiles": "CN1CCN(CC1)C2=NC3=CC=CC=C3NC4=CC=CC=C42",
        "role": "preferred",
        "description": "Deschloroclozapine — current preferred actuator; high potency, crosses BBB",
    },
    "Compound 21": {
        "pubchem_cid": 135445020,
        "smiles": "CCN1C2=CC=CC=C2C(=C(C1=O)C=NC3=CC(=C(C=C3)O)C)O",
        "role": "alternative",
        "description": "Alternative actuator; lower potency, some off-target activity",
    },
    "Olanzapine": {
        "pubchem_cid": 135398745,
        "smiles": "CC1=CC2=C(S1)NC3=CC=CC=C3N=C2N4CCN(CC4)C",
        "role": "repurposed",
        "description": "Repurposed antipsychotic with DREADD activity; significant off-target effects",
    },
    "Perlapine": {
        "pubchem_cid": 16106,
        "smiles": "CN1CCN(CC1)C2=NC3=CC=CC=C3CC4=CC=CC=C42",
        "role": "experimental",
        "description": "Sedative with DREADD activity; limited characterization",
    },
}


def load_compounds(csv_path=None):
    """Load compound data from CSV, falling back to the hardcoded dictionary."""
    if csv_path is None:
        csv_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "compounds", "compounds.csv"
        )

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df

    # Fallback to hardcoded dictionary
    rows = []
    for name, info in COMPOUNDS.items():
        rows.append({
            "name": name,
            "pubchem_cid": info["pubchem_cid"],
            "smiles": info["smiles"],
            "role": info["role"],
        })
    return pd.DataFrame(rows)
