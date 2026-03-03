# SPEC-06: Complete Selectivity Models (Module 7)

**Priority:** IMPORTANT
**Module:** 7 (Selectivity Prediction)
**Depends on:** Nothing (but benefits from stable network)

## Problem Statement

Module 7 never finished running. The ChEMBL data download was interrupted at target 6/8 ("Serotonin 5-HT2A"). No selectivity model files were saved — `models/selectivity/` is empty. No output CSV exists.

If the download fails again, the code falls back to `_generate_synthetic_data()` which creates 206 records per target using 5 repeated scaffolds with random pChEMBL values — the same garbage data problem as Module 6.

## Current State

- `models/selectivity/` — **empty directory**
- `data/results/selectivity_predictions.csv` — **does not exist** (or may have partial data)
- `chembl_webresource_client` — **installed** (development version)
- ChEMBL download interrupted at target 6/8

### Target Configuration in `src/module7/chembl_data.py`
```python
TARGETS = {
    "M1":     {"chembl_id": "CHEMBL216",  "name": "Muscarinic M1"},
    "M2":     {"chembl_id": "CHEMBL211",  "name": "Muscarinic M2"},
    "M3":     {"chembl_id": "CHEMBL245",  "name": "Muscarinic M3"},
    "M4":     {"chembl_id": "CHEMBL1945", "name": "Muscarinic M4"},
    "M5":     {"chembl_id": "CHEMBL2035", "name": "Muscarinic M5"},
    "D2":     {"chembl_id": "CHEMBL217",  "name": "Dopamine D2"},
    "5-HT2A": {"chembl_id": "CHEMBL224",  "name": "Serotonin 5-HT2A"},
    "H1":     {"chembl_id": "CHEMBL231",  "name": "Histamine H1"},
}
```

### ChEMBL Query Parameters
- Activity types: Ki, IC50
- Units: nM
- Filter: pchembl_value not null
- Returns: canonical_smiles, pchembl_value, target, target_name

## Implementation Steps

### Step 1: Pre-check ChEMBL API connectivity

```python
from chembl_webresource_client.new_client import new_client
activity = new_client.activity
# Quick test: fetch a small batch from M3 (CHEMBL245)
test = activity.filter(
    target_chembl_id="CHEMBL245",
    standard_type__in=["Ki", "IC50"],
    standard_units="nM",
    pchembl_value__isnull=False
).only(["molecule_chembl_id", "canonical_smiles", "pchembl_value"])[:5]
print(f"ChEMBL test query returned {len(list(test))} results")
```

If this times out, the ChEMBL API may be down. Options:
- Wait and retry
- Download ChEMBL SQLite database directly (ChEMBL 34, ~4GB) for local queries

### Step 2: Check for cached data from previous partial run

```python
import os
cache_path = "data/selectivity/chembl_binding_data.csv"
if os.path.exists(cache_path):
    import pandas as pd
    df = pd.read_csv(cache_path)
    print(f"Cached data: {len(df)} rows")
    print(f"Targets present: {df['target'].unique().tolist()}")
    # If some targets are already cached, the download function
    # may skip them on re-run
```

### Step 3: Add retry/timeout logic to `download_chembl_data()` (code change)

**File:** `src/module7/chembl_data.py`
**Function:** `download_chembl_data(target_key, min_activities=50, verbose=True)`

The current code may hang or crash on large targets (5-HT2A and D2 have thousands of entries). Add:

1. A timeout on the ChEMBL query (the client supports `timeout` parameter)
2. Chunked fetching for large result sets
3. Caching per-target to avoid re-downloading successful targets

**Suggested code addition** at the top of `download_chembl_data()`:
```python
# Check per-target cache first
target_cache = os.path.join(DATA_DIR, f"chembl_{target_key}.csv")
if os.path.exists(target_cache):
    cached = pd.read_csv(target_cache)
    if len(cached) >= min_activities:
        if verbose:
            print(f"    Loaded {len(cached)} records from cache for {target_key}")
        return cached
```

And after successful download, save per-target:
```python
# Save per-target cache
target_df.to_csv(target_cache, index=False)
```

### Step 4: Run Module 7

```bash
python3 -m src.module7.run_module7
```

This will:
1. Download binding data for all 8 targets from ChEMBL
2. Train RF+GB ensemble per target (activity threshold: pChEMBL >= 6.0)
3. Predict selectivity profiles for 6 actuators
4. Save models to `models/selectivity/`
5. Save predictions to `data/results/selectivity_predictions.csv`

**Expected runtime:** 5-30 minutes (dominated by ChEMBL downloads)

### Step 5: Verify model quality

After training, check metrics for each target model:
```python
import joblib
import os

for target_key in ["M1", "M2", "M3", "M4", "M5", "D2", "5-HT2A", "H1"]:
    model_path = f"models/selectivity/{target_key}_model.joblib"
    if os.path.exists(model_path):
        data = joblib.load(model_path)
        metrics = data.get("metrics", {})
        n_train = data.get("train_size", "?")
        auroc = metrics.get("auroc", "N/A")
        print(f"{target_key}: AUROC={auroc}, train_size={n_train}")
    else:
        print(f"{target_key}: MODEL MISSING")
```

**Expected:**
- Each model should have AUROC > 0.6 (ideally > 0.7)
- Training sizes should be >100 compounds per target (from real ChEMBL data)
- Muscarinic targets (M1-M5) should have more data than D2/5-HT2A/H1

## Verification Checklist

- [ ] `models/selectivity/` contains 8 `.joblib` files (one per target)
- [ ] `data/results/selectivity_predictions.csv` exists with 6 rows (one per actuator)
- [ ] Selectivity scores are distinct per compound (not all identical)
- [ ] DCZ should show high selectivity for M3 (it's designed to be M3-selective)
- [ ] Clozapine and Olanzapine should show broad off-target activity (known pharmacology)

## Validation Script

```python
import pandas as pd

df = pd.read_csv("data/results/selectivity_predictions.csv")
print("Compounds:", df["name"].tolist())
print("\nSelectivity scores:")
for _, row in df.iterrows():
    print(f"  {row['name']}: score={row['selectivity_score']:.3f}, "
          f"off-targets={row.get('n_off_targets', 'N/A')}")

# Check per-target probabilities are different
target_cols = [c for c in df.columns if c.startswith("p_")]
print(f"\nTarget columns: {target_cols}")
for col in target_cols:
    print(f"  {col}: {df[col].describe()['mean']:.3f} (mean), "
          f"{df[col].nunique()} unique values")
```

## Files Changed

- **Modified:** `src/module7/chembl_data.py` — add per-target caching in `download_chembl_data()`
- **New:** 8 model files in `models/selectivity/`
- **New:** `data/results/selectivity_predictions.csv`
- **New:** `data/selectivity/chembl_*.csv` (per-target cache files)
