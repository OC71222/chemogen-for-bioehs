# SPEC-02: Install PyTDC + Retrain ADMET Models

**Priority:** CRITICAL
**Module:** 6 (ML ADMET Prediction)
**Depends on:** Nothing
**Blocks:** SPEC-10

## Problem Statement

PyTDC is not installed. When `src/module6/train_models.py` tries to load training data via `_load_tdc_dataset()`, it fails and falls back to `_generate_synthetic_dataset()` which creates 800 "compounds" by repeating 10 drug-like SMILES (positive) and 8 non-drug SMILES (negative), 40-50 copies each.

Result: hERG, CYP2D6, HIA, and clearance models are all trained on the same repeated scaffolds and produce **identical predictions** for every compound. The BBB model was trained on real BBBP data (2,050 compounds from DeepChem) but achieves only AUROC=0.5 on scaffold split.

## Current State

### Installed
- scikit-learn 1.8.0
- joblib 1.5.3
- rdkit 2025.09.5

### NOT Installed
- PyTDC (tdc): NOT INSTALLED

### Existing Model Files (all garbage except BBB)
```
models/admet/bbb_model.joblib       ← Real BBBP data, but AUROC=0.5
models/admet/herg_model.joblib      ← GARBAGE: 5 repeated SMILES × 40 copies
models/admet/cyp2d6_model.joblib    ← GARBAGE: identical to hERG
models/admet/hia_model.joblib       ← GARBAGE: identical to hERG
models/admet/clearance_model.joblib ← GARBAGE: synthetic random data, R²=-0.003
```

### Dataset Config in `src/module6/train_models.py`
```python
CLASSIFICATION_DATASETS = {
    "bbb":    {"tdc_name": "BBB_Martins",         "description": "BBB Permeability"},
    "herg":   {"tdc_name": "hERG",                "description": "hERG Channel Inhibition"},
    "cyp2d6": {"tdc_name": "CYP2D6_Veith",       "description": "CYP2D6 Inhibition"},
    "hia":    {"tdc_name": "HIA_Hou",             "description": "Human Intestinal Absorption"},
}
REGRESSION_DATASETS = {
    "clearance": {"tdc_name": "Clearance_Hepatocyte_AZ", "description": "Hepatic Clearance"},
}
```

### Data Loading Fallback Chain in `_load_tdc_dataset()`
1. Try `from tdc.single_pred import ADME` → **fails** (PyTDC not installed)
2. Try loading from cache `data/admet_cache/{tdc_name}.csv`
3. Try downloading from hardcoded URL (only BBB_Martins has one)
4. Fall back to `_generate_synthetic_dataset()` → **this is what happens for hERG/CYP2D6/HIA/clearance**

## Implementation Steps

### Step 1: Install PyTDC

```bash
pip install PyTDC
```

**Verification:**
```bash
python3 -c "from tdc.single_pred import ADME; print('PyTDC OK')"
```

### Step 2: Delete garbage model files

```bash
rm models/admet/herg_model.joblib
rm models/admet/cyp2d6_model.joblib
rm models/admet/hia_model.joblib
rm models/admet/clearance_model.joblib
```

**Do NOT delete `bbb_model.joblib` yet** — it's trained on real data (even if performance is poor). It will be retrained automatically along with the others.

### Step 3: Verify PyTDC can download each dataset

Before running the full retraining, verify data access for each endpoint:

```python
from tdc.single_pred import ADME

# These should each download and return a DataFrame
datasets = {
    "hERG": ADME(name="hERG"),
    "CYP2D6_Veith": ADME(name="CYP2D6_Veith"),
    "HIA_Hou": ADME(name="HIA_Hou"),
    "Clearance_Hepatocyte_AZ": ADME(name="Clearance_Hepatocyte_AZ"),
    "BBB_Martins": ADME(name="BBB_Martins"),
}
for name, ds in datasets.items():
    split = ds.get_split()
    total = len(split['train']) + len(split['valid']) + len(split['test'])
    print(f"{name}: {total} compounds")
```

**Expected approximate sizes from TDC:**
- hERG: ~600+ compounds (central hERG dataset)
- CYP2D6_Veith: ~13,130 compounds
- HIA_Hou: ~578 compounds
- Clearance_Hepatocyte_AZ: ~1,020 compounds
- BBB_Martins: ~2,050 compounds

If any dataset fails to download, check network connectivity. TDC downloads from Harvard Dataverse.

### Step 4: Verify `_load_tdc_dataset()` integration

The existing code in `train_models.py` already has the TDC loading path:
```python
def _load_tdc_dataset(tdc_name):
    try:
        from tdc.single_pred import ADME
        data = ADME(name=tdc_name)
        split = data.get_split()
        ...
    except ImportError:
        # Falls back to cache/download/synthetic
```

With PyTDC installed, this `try` block will succeed. **No code changes needed** in the loading logic itself.

### Step 5: Check if `_load_tdc_dataset()` correctly maps TDC column names

TDC datasets return DataFrames with columns: `Drug`, `Y` (and sometimes `Drug_ID`).
The existing code should map these to `smiles` and `label` columns.

**Verify** that `_load_tdc_dataset()` handles this mapping. If it expects `smiles`/`label` columns but TDC returns `Drug`/`Y`, a column rename is needed:
```python
df = df.rename(columns={"Drug": "smiles", "Y": "label"})
```

Read `_load_tdc_dataset()` carefully to confirm this mapping exists.

### Step 6: Re-run Module 6 to retrain all models

```bash
python3 -m src.module6.run_module6
```

This will:
1. Check for missing models (all 4 garbage ones were deleted)
2. Call `train_all_models()` which trains:
   - BBB (classification): on BBBP/BBB_Martins dataset
   - hERG (classification): on hERG dataset
   - CYP2D6 (classification): on CYP2D6_Veith dataset
   - HIA (classification): on HIA_Hou dataset
   - Clearance (regression): on Clearance_Hepatocyte_AZ dataset
3. Save trained models to `models/admet/`
4. Run predictions on 6 actuators
5. Save to `data/results/admet_predictions.csv`

**Expected runtime:** 2-10 minutes for training (depends on dataset sizes).

### Step 7: Validate retrained models

After retraining, check the training metrics printed to console:
- **hERG AUROC** should be >0.7 (ideally >0.8)
- **CYP2D6 AUROC** should be >0.7
- **HIA AUROC** should be >0.7
- **Clearance R²** should be >0.0 (anything above 0 is better than the current -0.003)
- **BBB AUROC** should be >0.5 (any improvement over current coin-flip)

## Verification Checklist

- [ ] `python3 -c "from tdc.single_pred import ADME"` succeeds
- [ ] All 5 model files exist in `models/admet/` with recent timestamps
- [ ] `data/results/admet_predictions.csv` has **DIFFERENT** values for `herg_prob`, `cyp2d6_prob`, and `hia_prob` columns
  - Previously these were all identical (e.g., DCZ got 0.8675 from all three)
  - Now they should be distinct values per endpoint
- [ ] Clearance predictions are not all the same value
- [ ] Training metrics (AUROC, R²) are printed during retraining and are reasonable

## Validation Script

```python
import pandas as pd
df = pd.read_csv("data/results/admet_predictions.csv")
print("hERG unique values:", df["herg_prob"].nunique())
print("CYP2D6 unique values:", df["cyp2d6_prob"].nunique())
print("HIA unique values:", df["hia_prob"].nunique())
# All should have >1 unique value, ideally 6 (one per compound)

# Check they're not identical across endpoints
print("\nPer-compound check:")
for _, row in df.iterrows():
    h, c, i = row.get("herg_prob"), row.get("cyp2d6_prob"), row.get("hia_prob")
    same = (h == c == i)
    print(f"  {row['name']}: hERG={h:.3f}, CYP2D6={c:.3f}, HIA={i:.3f} {'← IDENTICAL (BAD)' if same else '← DIFFERENT (GOOD)'}")
```

## Rollback

If PyTDC install fails or datasets are unavailable:
- The synthetic fallback still works (pipeline won't crash)
- But results will remain garbage — flag this as a blocking issue

## Files Changed

- No source code changes required (assuming column mapping is correct — verify in Step 5)
- Deleted: 4 garbage `.joblib` files from `models/admet/`
- New/updated: 5 retrained `.joblib` files in `models/admet/`
- Updated: `data/results/admet_predictions.csv`
