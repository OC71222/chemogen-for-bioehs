# SPEC-07: Replace Fake Compound Library with ChEMBL Query

**Priority:** IMPORTANT
**Module:** 5 (Virtual Screening)
**Depends on:** SPEC-01 (needs real Vina docking for screening scores)

## Problem Statement

`download_zinc_library()` in `src/module5/library_prep.py` does NOT query any real database. Instead it generates 151 compounds from 26 hardcoded seed SMILES by doing string manipulations like `smiles.replace("N", "O")`. This produces:

1. Chemically dubious molecules — some are disconnected fragments (e.g., `F.c1ccc2c(c1)...` which is fluoride ion + a separate molecule)
2. A tiny library (151 compounds) that has no real connection to DREADD pharmacology
3. All screening docking scores are fake (inherited from Module 4's `_estimate_docking()` fallback)

## Current State

### Fake Library
```
data/screening/zinc_raw.csv     — 151 fake compounds from string manipulation
data/screening/zinc_filtered.csv — filtered subset
```

### Library Generation in `src/module5/library_prep.py`
- `download_zinc_library()` — tries to fetch from `https://zinc15.docking.org/tranches/download/SMILES?tranches=AAAA` but this URL doesn't work, so it falls back to generating from 26 seed scaffolds + string modifications
- `_random_modify(smiles, substituent)` — applies `smiles.replace()` style substitutions
- `apply_filters()` — applies drug-likeness filters (MW 200-450, LogP 0.5-4.0, TPSA<100, etc.)

### User Decision
Use ChEMBL to query compounds with measured activity at muscarinic receptors (M1-M5). The `chembl_webresource_client` is already installed.

## Implementation Steps

### Step 1: Replace `download_zinc_library()` with ChEMBL query

**File:** `src/module5/library_prep.py`
**Function:** `download_zinc_library(n_compounds=2000, output_path=None)`

Replace the entire body of this function with a real ChEMBL query that:

1. Queries compounds with measured activity at muscarinic receptor targets (M1-M5)
2. Filters to drug-like compounds
3. Removes the 6 known actuators (don't screen what we already know about)
4. Returns a DataFrame with `zinc_id` (or `chembl_id`) and `smiles` columns

**New implementation:**

```python
def download_zinc_library(n_compounds=2000, output_path=None):
    """Download a drug-like compound library from ChEMBL.

    Queries compounds with measured binding activity at muscarinic receptors
    (M1-M5) to find molecules in the same pharmacological space as DREADD
    actuators.
    """
    if output_path is None:
        output_path = os.path.join(PROJECT_ROOT, "data", "screening", "zinc_raw.csv")

    # Return cached if exists and has enough compounds
    if os.path.exists(output_path):
        cached = pd.read_csv(output_path)
        if len(cached) >= n_compounds * 0.5:  # Allow some margin
            return cached

    from chembl_webresource_client.new_client import new_client

    # Muscarinic receptor ChEMBL IDs
    target_ids = [
        "CHEMBL216",   # M1
        "CHEMBL211",   # M2
        "CHEMBL245",   # M3
        "CHEMBL1945",  # M4
        "CHEMBL2035",  # M5
    ]

    all_compounds = []
    activity = new_client.activity

    for target_id in target_ids:
        try:
            results = activity.filter(
                target_chembl_id=target_id,
                standard_type__in=["Ki", "IC50", "EC50"],
                standard_units="nM",
                pchembl_value__isnull=False,
            ).only([
                "molecule_chembl_id",
                "canonical_smiles",
                "pchembl_value",
            ])

            for rec in results:
                smiles = rec.get("canonical_smiles")
                if smiles:
                    all_compounds.append({
                        "chembl_id": rec["molecule_chembl_id"],
                        "smiles": smiles,
                    })
        except Exception as e:
            print(f"    Warning: ChEMBL query failed for {target_id}: {e}")
            continue

    if not all_compounds:
        print("    ERROR: No compounds retrieved from ChEMBL")
        return _generate_fallback_library(n_compounds, output_path)

    df = pd.DataFrame(all_compounds).drop_duplicates(subset="smiles")

    # Remove known actuators
    from src.module2.compounds import load_compounds
    known_smiles = set(load_compounds()["smiles"].tolist())
    df = df[~df["smiles"].isin(known_smiles)]

    # Rename for compatibility with downstream code
    df = df.rename(columns={"chembl_id": "zinc_id"})

    # Limit to requested size
    if len(df) > n_compounds:
        df = df.sample(n=n_compounds, random_state=42)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
```

### Step 2: Add SMILES validation to reject disconnected fragments

**File:** `src/module5/library_prep.py`
**Function:** `apply_filters()` — add a validation step

Add this check at the beginning of the filter function:

```python
from rdkit import Chem

def _is_valid_smiles(smiles):
    """Reject disconnected fragments and invalid SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    # Reject disconnected fragments (SMILES containing ".")
    if "." in smiles:
        return False
    return True

# In apply_filters(), before computing properties:
library_df = library_df[library_df["smiles"].apply(_is_valid_smiles)]
```

### Step 3: Add a minimal fallback that generates valid compounds

If ChEMBL is unreachable, we still need a fallback — but one that generates valid, non-disconnected molecules. Replace `_random_modify()` with proper RDKit enumeration:

```python
def _generate_fallback_library(n_compounds, output_path):
    """Generate a small library using RDKit combinatorial enumeration.
    Only used if ChEMBL is unreachable.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    # Drug-like scaffolds relevant to DREADD pharmacology
    scaffolds = [
        "c1ccc2c(c1)NC1=CC=CC=C1N=C2N1CCNCC1",   # Clozapine core
        "c1ccc2c(c1)NC1=CC=CC=C1N=C2N1CCN(C)CC1",  # N-methyl piperazine variant
        "c1ccc2[nH]c(-c3ccccn3)nc2c1",              # Benzimidazole
        "O=c1[nH]c2ccccc2n1CC",                     # Benzoxazinone
        "c1ccc(-c2nc3ccccc3[nH]2)nc1",              # Pyridyl benzimidazole
    ]

    substituents = ["C", "CC", "F", "Cl", "OC", "N(C)C", "C(F)(F)F"]

    compounds = []
    for scaffold in scaffolds:
        mol = Chem.MolFromSmiles(scaffold)
        if mol is None:
            continue
        compounds.append({"zinc_id": f"FALLBACK_{len(compounds):05d}", "smiles": scaffold})

        # Generate R-group variants at aromatic positions
        # ... (RDKit enumeration logic)

    df = pd.DataFrame(compounds).head(n_compounds)
    df.to_csv(output_path, index=False)
    return df
```

### Step 4: Delete old fake library data

```bash
rm data/screening/zinc_raw.csv
rm data/screening/zinc_filtered.csv
rm data/results/screening_hits.csv
rm data/results/screening_results.csv
```

### Step 5: Re-run Module 5

```bash
python3 -m src.module5.run_module5 --n_compounds 500
```

This will:
1. Query ChEMBL for ~500 drug-like muscarinic-active compounds
2. Filter for drug-likeness (MW, LogP, TPSA, etc.)
3. Validate all SMILES (no disconnected fragments)
4. Dock each compound with real Vina (requires SPEC-01 complete)
5. Rank hits by affinity and similarity to known actuators

**Expected runtime:** With Vina installed, screening 500 compounds at exhaustiveness=8 takes 1-4 hours.

## Verification Checklist

- [ ] `data/screening/zinc_raw.csv` contains compounds with real ChEMBL IDs (not "ZINC_000xxx")
- [ ] No SMILES in the library contain "." (disconnected fragments)
- [ ] `data/results/screening_results.csv` has `estimated=False` for all rows
- [ ] `data/results/screening_hits.csv` contains top hits with real docking scores
- [ ] Library size is ≥200 unique compounds (after filtering)

## Validation Script

```python
import pandas as pd
from rdkit import Chem

# Check library
lib = pd.read_csv("data/screening/zinc_raw.csv")
print(f"Library size: {len(lib)}")

# Check for disconnected fragments
fragments = lib[lib["smiles"].str.contains(r"\.", regex=True)]
print(f"Disconnected fragments: {len(fragments)} (should be 0)")

# Check SMILES validity
invalid = 0
for smi in lib["smiles"]:
    if Chem.MolFromSmiles(smi) is None:
        invalid += 1
print(f"Invalid SMILES: {invalid} (should be 0)")

# Check screening results
if pd.io.common.file_exists("data/results/screening_results.csv"):
    results = pd.read_csv("data/results/screening_results.csv")
    estimated = results.get("estimated", pd.Series([True]))
    print(f"Estimated docking scores: {estimated.sum()} / {len(results)} (should be 0)")
```

## Files Changed

- **Modified:** `src/module5/library_prep.py` — replace `download_zinc_library()` body with ChEMBL query, add SMILES validation, add proper fallback
- **Deleted:** Old fake data files (`zinc_raw.csv`, `zinc_filtered.csv`, `screening_hits.csv`, `screening_results.csv`)
- **New:** Real library and screening results CSVs
