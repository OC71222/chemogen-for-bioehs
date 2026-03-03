# SPEC-10: Retrain BBB Model on Larger Dataset

**Priority:** NICE TO HAVE
**Module:** 6 (ML ADMET Prediction)
**Depends on:** SPEC-02 (PyTDC installed, ADMET infrastructure working)

## Problem Statement

The current BBB model was trained on the BBBP dataset (Martins et al., 2,050 compounds) from DeepChem. While it correctly scores known BBB+ compounds (caffeine: 0.907) and known BBB- compounds (glucuronic acid: 0.488), it achieves only **AUROC = 0.5 on scaffold-split test set** — meaning it doesn't generalize to novel scaffolds.

## Current State

- BBB model: `models/admet/bbb_model.joblib`
- Training data: `data/admet_cache/BBB_Martins.csv` (2,050 compounds)
- Architecture: RF(200 trees) + GB(200 trees) ensemble on 2048-bit Morgan fingerprints
- Scaffold-split test AUROC: 0.5 (coin flip)

## Root Cause Analysis

The poor generalization likely stems from:
1. **Small dataset** — 2,050 compounds may not cover enough chemical space
2. **Features too sparse** — 2048-bit Morgan FPs alone may not capture BBB-relevant properties
3. **No physicochemical descriptors** — BBB permeability strongly correlates with MW, LogP, TPSA, HBD which are not included as features

## Implementation Steps

### Step 1: Check if PyTDC provides a larger BBB dataset

```python
from tdc.single_pred import ADME

# B3DB is a larger BBB dataset with 7,807 compounds
# Check if TDC has it
try:
    data = ADME(name="BBB_Martins")  # Standard BBBP
    split = data.get_split()
    print(f"BBB_Martins: {sum(len(v) for v in split.values())} total")
except:
    print("BBB_Martins not available")

# Try B3DB if available in TDC
try:
    data = ADME(name="B3DB")
    split = data.get_split()
    print(f"B3DB: {sum(len(v) for v in split.values())} total")
except Exception as e:
    print(f"B3DB not available: {e}")
```

If B3DB is not in TDC, we can still improve the model with the existing dataset by adding features.

### Step 2: Add physicochemical descriptors to feature set

**File:** `src/module6/train_models.py`
**Function:** `_featurize_split(df, radius=2, n_bits=2048)`

Currently only uses Morgan fingerprints. Add BBB-relevant physicochemical descriptors:

```python
def _featurize_split(df, radius=2, n_bits=2048, include_descriptors=True):
    """Generate features from SMILES.

    Args:
        include_descriptors: If True, append physicochemical descriptors
                           to Morgan fingerprints (improves BBB/ADMET prediction)
    """
    from src.module6.fingerprints import smiles_to_morgan_fp, smiles_to_rdkit_descriptors

    features = []
    labels = []

    for _, row in df.iterrows():
        fp = smiles_to_morgan_fp(row["smiles"], radius=radius, n_bits=n_bits)
        if fp is None:
            continue

        if include_descriptors:
            desc = smiles_to_rdkit_descriptors(row["smiles"])
            if desc is None:
                continue
            # Select BBB-relevant descriptors
            selected = [
                desc.get("MolWt", 0),
                desc.get("LogP", 0),
                desc.get("TPSA", 0),
                desc.get("NumHDonors", 0),
                desc.get("NumHAcceptors", 0),
                desc.get("NumRotatableBonds", 0),
                desc.get("NumAromaticRings", 0),
                desc.get("FractionCSP3", 0),
                desc.get("NumHeavyAtoms", 0),
                desc.get("RingCount", 0),
                desc.get("MolMR", 0),
                desc.get("HallKierAlpha", 0),
                desc.get("LabuteASA", 0),
                desc.get("BertzCT", 0),
                desc.get("NumHeteroatoms", 0),
            ]
            feature_vec = np.concatenate([fp, np.array(selected, dtype=np.float32)])
        else:
            feature_vec = fp

        features.append(feature_vec)
        labels.append(row["label"])

    return np.array(features), np.array(labels)
```

Note: The `smiles_to_rdkit_descriptors()` function already exists in `src/module6/fingerprints.py` and computes 32 descriptors. We select the 15 most BBB-relevant ones.

### Step 3: Consider cross-validation instead of single scaffold split

The current code does a single 80/10/10 scaffold split. A single split can be misleading. Consider adding 5-fold cross-validation for more robust metric estimation:

```python
from sklearn.model_selection import cross_val_score

# During training, in addition to the single split evaluation:
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='roc_auc')
print(f"    5-fold CV AUROC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
```

### Step 4: Delete old BBB model and retrain

```bash
rm models/admet/bbb_model.joblib
python3 -c "
from src.module6.train_models import train_classification_model
result = train_classification_model('bbb', verbose=True)
print(f'New BBB AUROC: {result[\"metrics\"][\"auroc\"]:.3f}')
"
```

### Step 5: Compare old vs new predictions

```python
# Save old predictions first for comparison
import pandas as pd
old = pd.read_csv("data/results/admet_predictions.csv")
print("Old BBB predictions:")
print(old[["name", "bbb_ml_prob"]].to_string())

# After retraining and re-predicting:
# python3 -m src.module6.run_module6
new = pd.read_csv("data/results/admet_predictions.csv")
print("\nNew BBB predictions:")
print(new[["name", "bbb_ml_prob"]].to_string())
```

## Expected Improvement

With physicochemical descriptors added:
- BBB AUROC should improve from 0.5 to 0.65-0.80
- The model will have access to MW, LogP, TPSA, HBD — all known BBB predictors
- If the dataset is also upgraded to B3DB (7,807 compounds), further improvement expected

## Verification Checklist

- [ ] New BBB model AUROC > 0.6 on scaffold split (was 0.5)
- [ ] Cross-validation AUROC is reported and reasonable
- [ ] Known BBB+ compounds (caffeine, verapamil) still score high (>0.7)
- [ ] Known BBB- compounds (glucuronic acid) still score low (<0.5)
- [ ] The 6 DREADD actuator predictions are updated in `admet_predictions.csv`

## Files Changed

- **Modified:** `src/module6/train_models.py` — `_featurize_split()` to include physicochemical descriptors, add CV scoring
- **Regenerated:** `models/admet/bbb_model.joblib`
- **Updated:** `data/results/admet_predictions.csv`
