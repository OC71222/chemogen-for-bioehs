"""
Module 7: Train Selectivity Models
Per-target binary classifiers for activity prediction.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.module6.fingerprints import batch_smiles_to_morgan
from src.module7.chembl_data import TARGETS

MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "selectivity")


def _scaffold_split(smiles_list, labels, test_size=0.2, random_state=42):
    """Approximate scaffold split using Murcko scaffolds.

    Falls back to stratified random split if scaffold splitting fails.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold

        scaffolds = {}
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol:
                try:
                    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                        mol=mol, includeChirality=False
                    )
                except Exception:
                    scaffold = "unknown"
            else:
                scaffold = "unknown"

            if scaffold not in scaffolds:
                scaffolds[scaffold] = []
            scaffolds[scaffold].append(i)

        # Sort scaffolds by size (largest first) and split
        scaffold_sets = sorted(scaffolds.values(), key=len, reverse=True)

        train_idx = []
        test_idx = []
        n_total = len(smiles_list)
        n_test_target = int(n_total * test_size)

        for indices in scaffold_sets:
            if len(test_idx) < n_test_target:
                test_idx.extend(indices)
            else:
                train_idx.extend(indices)

        return train_idx, test_idx

    except Exception:
        # Fallback to random split
        indices = list(range(len(smiles_list)))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=random_state,
            stratify=labels,
        )
        return train_idx, test_idx


def train_target_model(target_key, binding_data, activity_threshold=6.0,
                       n_bits=2048, verbose=True):
    """Train a binary classifier for a single target.

    Active = pChEMBL > threshold (i.e., Ki < 1uM for threshold=6.0)

    Args:
        target_key: Target identifier (e.g., "M1", "D2")
        binding_data: DataFrame with 'canonical_smiles' and 'pchembl_value'
        activity_threshold: pChEMBL cutoff for active/inactive
        n_bits: Morgan FP bit count
        verbose: Print progress

    Returns:
        dict with models and metrics
    """
    target_data = binding_data[binding_data["target"] == target_key].copy()

    if verbose:
        print(f"\n    Training model for {TARGETS[target_key]['name']} ({target_key})...")
        print(f"      Data points: {len(target_data)}")

    # Create binary labels
    target_data["active"] = (target_data["pchembl_value"] >= activity_threshold).astype(int)
    smiles_list = target_data["canonical_smiles"].tolist()
    labels = target_data["active"].values

    if verbose:
        n_active = labels.sum()
        print(f"      Active: {n_active} ({n_active/len(labels)*100:.1f}%)")

    # Featurize
    features, valid_idx = batch_smiles_to_morgan(smiles_list, n_bits=n_bits)
    if len(valid_idx) == 0:
        print(f"      ERROR: No valid features generated for {target_key}")
        return None

    valid_labels = labels[valid_idx]

    # Check for class balance
    if valid_labels.sum() == 0 or valid_labels.sum() == len(valid_labels):
        print(f"      WARNING: Only one class present for {target_key}. Skipping.")
        return None

    # Scaffold split
    valid_smiles = [smiles_list[i] for i in valid_idx]
    train_idx, test_idx = _scaffold_split(valid_smiles, valid_labels)

    X_train = features[train_idx]
    y_train = valid_labels[train_idx]
    X_test = features[test_idx]
    y_test = valid_labels[test_idx]

    if verbose:
        print(f"      Train: {len(X_train)}, Test: {len(X_test)}")

    # Train RandomForest
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=None, min_samples_leaf=2,
        n_jobs=-1, random_state=42, class_weight="balanced",
    )
    rf.fit(X_train, y_train)

    # Train GradientBoosting
    gb = GradientBoostingClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        min_samples_leaf=3, random_state=42,
    )
    gb.fit(X_train, y_train)

    # Ensemble evaluation
    rf_probs = rf.predict_proba(X_test)[:, 1]
    gb_probs = gb.predict_proba(X_test)[:, 1]
    ensemble_probs = (rf_probs + gb_probs) / 2

    # Check if test set has both classes
    if len(np.unique(y_test)) < 2:
        auroc = 0.5
        auprc = y_test.mean()
    else:
        auroc = roc_auc_score(y_test, ensemble_probs)
        auprc = average_precision_score(y_test, ensemble_probs)

    acc = accuracy_score(y_test, (ensemble_probs >= 0.5).astype(int))

    if verbose:
        print(f"      AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}, Accuracy: {acc:.4f}")

    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_data = {
        "rf": rf,
        "gb": gb,
        "target_key": target_key,
        "n_bits": n_bits,
        "activity_threshold": activity_threshold,
        "metrics": {
            "auroc": round(auroc, 4),
            "auprc": round(auprc, 4),
            "accuracy": round(acc, 4),
        },
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    model_path = os.path.join(MODELS_DIR, f"{target_key}_model.joblib")
    joblib.dump(model_data, model_path)

    return model_data


def train_all_selectivity_models(binding_data=None, verbose=True):
    """Train models for all targets in the selectivity panel.

    Returns:
        dict of target_key -> model_data
    """
    if binding_data is None:
        from src.module7.chembl_data import download_all_targets
        binding_data = download_all_targets(verbose=verbose)

    results = {}
    for target_key in TARGETS:
        result = train_target_model(target_key, binding_data, verbose=verbose)
        if result:
            results[target_key] = result

    if verbose:
        print("\n--- Selectivity Model Summary ---")
        for key, data in results.items():
            metrics = data["metrics"]
            print(f"  {key:8s}: AUROC={metrics['auroc']:.4f}, "
                  f"Accuracy={metrics['accuracy']:.4f}")

    return results
