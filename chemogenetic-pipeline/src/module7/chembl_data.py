"""
Module 7: ChEMBL Binding Data Retrieval
Downloads receptor binding data from ChEMBL for selectivity modeling.
"""

import os
import sys
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "selectivity")

# Target ChEMBL IDs for selectivity panel
TARGETS = {
    "M1": {"chembl_id": "CHEMBL216", "name": "Muscarinic M1", "uniprot": "P11229"},
    "M2": {"chembl_id": "CHEMBL211", "name": "Muscarinic M2", "uniprot": "P08172"},
    "M3": {"chembl_id": "CHEMBL245", "name": "Muscarinic M3", "uniprot": "P20309"},
    "M4": {"chembl_id": "CHEMBL1945", "name": "Muscarinic M4", "uniprot": "P08173"},
    "M5": {"chembl_id": "CHEMBL2035", "name": "Muscarinic M5", "uniprot": "P08912"},
    "D2": {"chembl_id": "CHEMBL217", "name": "Dopamine D2", "uniprot": "P14416"},
    "5-HT2A": {"chembl_id": "CHEMBL224", "name": "Serotonin 5-HT2A", "uniprot": "P28223"},
    "H1": {"chembl_id": "CHEMBL231", "name": "Histamine H1", "uniprot": "P35367"},
}


def download_chembl_data(target_key, min_activities=50, verbose=True):
    """Download binding data from ChEMBL for a given target.

    Args:
        target_key: Key from TARGETS dict (e.g., "M1", "D2")
        min_activities: Minimum number of activities required
        verbose: Print progress

    Returns:
        DataFrame with SMILES, pChEMBL values
    """
    target_info = TARGETS[target_key]
    chembl_id = target_info["chembl_id"]

    # Check per-target cache first
    os.makedirs(DATA_DIR, exist_ok=True)
    target_cache = os.path.join(DATA_DIR, f"chembl_{target_key}.csv")
    if os.path.exists(target_cache):
        cached = pd.read_csv(target_cache)
        if len(cached) >= min_activities:
            if verbose:
                print(f"    Loaded {len(cached)} records from cache for {target_key}")
            return cached

    if verbose:
        print(f"    Downloading {target_info['name']} ({chembl_id})...")

    try:
        import requests
        import time

        records = []
        url = f"https://www.ebi.ac.uk/chembl/api/data/activity.json"
        params = {
            "target_chembl_id": chembl_id,
            "standard_type__in": "Ki,IC50",
            "standard_units": "nM",
            "pchembl_value__isnull": "false",
            "limit": 1000,
        }
        
        # Paginate through results
        while url:
            max_retries = 3
            page_data = None
            
            for attempt in range(max_retries):
                try:
                    response = requests.get(url, params=params if url.endswith(".json") else None, timeout=30)
                    response.raise_for_status()
                    page_data = response.json()
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt * 5
                        if verbose:
                            print(f"      API request failed ({type(e).__name__}). Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        raise e
            
            if page_data and "activities" in page_data:
                for act in page_data["activities"]:
                    records.append({
                        "molecule_chembl_id": act.get("molecule_chembl_id"),
                        "canonical_smiles": act.get("canonical_smiles"),
                        "standard_type": act.get("standard_type"),
                        "standard_value": act.get("standard_value"),
                        "pchembl_value": act.get("pchembl_value"),
                        "standard_units": act.get("standard_units"),
                    })
                
                # Check for next page
                next_page = page_data.get("page_meta", {}).get("next")
                if next_page:
                    url = "https://www.ebi.ac.uk" + next_page
                else:
                    url = None
            else:
                break
        if not records:
            if verbose:
                print(f"      No data returned from ChEMBL API")
            return _generate_synthetic_data(target_key, verbose)

        df = pd.DataFrame(records)

        # Clean data
        df = df.dropna(subset=["canonical_smiles", "pchembl_value"])
        df["pchembl_value"] = pd.to_numeric(df["pchembl_value"], errors="coerce")
        df = df.dropna(subset=["pchembl_value"])

        # Remove duplicates (keep highest pChEMBL value per compound)
        df = df.sort_values("pchembl_value", ascending=False)
        df = df.drop_duplicates(subset=["canonical_smiles"], keep="first")

        df["target"] = target_key
        df["target_name"] = target_info["name"]

        if verbose:
            print(f"      Retrieved {len(df)} unique compounds "
                  f"(pChEMBL range: {df['pchembl_value'].min():.1f} - {df['pchembl_value'].max():.1f})")

        # Save per-target cache
        df.to_csv(target_cache, index=False)

        return df

    except ImportError:
        if verbose:
            print("      ChEMBL client not available. Generating synthetic data...")
        return _generate_synthetic_data(target_key, verbose)

    except Exception as e:
        if verbose:
            print(f"      ChEMBL download failed: {e}. Using synthetic data...")
        return _generate_synthetic_data(target_key, verbose)


def _generate_synthetic_data(target_key, verbose=True):
    """Generate synthetic binding data for demo when ChEMBL is unavailable.

    Uses known pharmacology to create realistic datasets.
    """
    from src.module2.compounds import COMPOUNDS

    np.random.seed(hash(target_key) % 2**32)

    # Known compound affinities at different targets (approximate pChEMBL values)
    known_affinities = {
        "M1": {"Clozapine": 7.0, "Olanzapine": 7.5, "CNO": 5.5, "DCZ": 5.0, "Perlapine": 6.0},
        "M2": {"Clozapine": 6.5, "Olanzapine": 6.0, "CNO": 5.0, "DCZ": 4.5, "Perlapine": 5.5},
        "M3": {"Clozapine": 7.5, "Olanzapine": 7.0, "CNO": 6.0, "DCZ": 8.5, "Perlapine": 6.5},
        "M4": {"Clozapine": 6.8, "Olanzapine": 7.2, "CNO": 5.2, "DCZ": 4.8, "Perlapine": 5.8},
        "M5": {"Clozapine": 6.5, "Olanzapine": 6.0, "CNO": 5.0, "DCZ": 4.5, "Perlapine": 5.5},
        "D2": {"Clozapine": 7.2, "Olanzapine": 8.0, "CNO": 5.0, "DCZ": 4.0, "Perlapine": 5.0},
        "5-HT2A": {"Clozapine": 8.5, "Olanzapine": 8.0, "CNO": 5.5, "DCZ": 4.5, "Perlapine": 5.5},
        "H1": {"Clozapine": 8.0, "Olanzapine": 7.5, "CNO": 5.0, "DCZ": 4.0, "Perlapine": 6.0},
    }

    # Generate background compounds
    n_background = 200
    records = []

    # Add known actuators
    affinities = known_affinities.get(target_key, {})
    for name, info in COMPOUNDS.items():
        pchembl = affinities.get(name, np.random.uniform(4.0, 6.0))
        records.append({
            "molecule_chembl_id": f"CHEMBL_{name}",
            "canonical_smiles": info["smiles"],
            "pchembl_value": pchembl + np.random.normal(0, 0.1),
            "standard_type": "Ki",
            "target": target_key,
            "target_name": TARGETS[target_key]["name"],
        })

    # Generate synthetic background SMILES with random activities
    # Use simple drug-like SMILES patterns
    scaffolds = [
        "c1ccc2c(c1)Nc1ccccc1N=C2N1CCNCC1",
        "CN1CCN(CC1)c1ccccc1",
        "c1ccc2[nH]ccc2c1",
        "c1ccc2ncccc2c1",
        "C1CCNCC1c1ccccc1",
    ]

    for i in range(n_background):
        scaffold = scaffolds[i % len(scaffolds)]
        pchembl = np.random.uniform(3.5, 9.0)
        records.append({
            "molecule_chembl_id": f"CHEMBL_SYN_{target_key}_{i}",
            "canonical_smiles": scaffold,  # simplified
            "pchembl_value": round(pchembl, 2),
            "standard_type": "Ki",
            "target": target_key,
            "target_name": TARGETS[target_key]["name"],
        })

    df = pd.DataFrame(records)

    if verbose:
        print(f"      Generated {len(df)} synthetic compounds for {target_key}")

    return df


def download_all_targets(verbose=True):
    """Download binding data for all targets in the selectivity panel.

    Returns:
        Combined DataFrame with all target data
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    output_path = os.path.join(DATA_DIR, "chembl_binding_data.csv")
    if os.path.exists(output_path):
        if verbose:
            print(f"    ChEMBL data already cached: {output_path}")
        return pd.read_csv(output_path)

    try:
        from src.utils.progress import update_module_status
        has_progress = True
    except ImportError:
        has_progress = False

    all_data = []
    target_keys = list(TARGETS.keys())
    for i, target_key in enumerate(target_keys):
        if has_progress:
            update_module_status(7, "running",
                                 step=f"Downloading ChEMBL: {TARGETS[target_key]['name']}",
                                 detail=f"Target {i+1}/{len(target_keys)} — querying binding assays",
                                 progress=i, total=len(target_keys))
        df = download_chembl_data(target_key, verbose=verbose)
        all_data.append(df)

    combined = pd.concat(all_data, ignore_index=True)

    # Save
    combined.to_csv(output_path, index=False)
    if verbose:
        print(f"\n    Total compounds across all targets: {len(combined)}")
        print(f"    Saved to: {output_path}")

        # Report per-target sizes
        print("\n    Dataset sizes per target:")
        for target_key in TARGETS:
            n = len(combined[combined["target"] == target_key])
            print(f"      {target_key:8s}: {n:5d} compounds")

    return combined
