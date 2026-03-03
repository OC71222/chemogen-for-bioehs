"""
Module 6: ADMET Prediction Engine
Loads trained models and predicts ADMET properties for compounds.
"""

import os
import sys
import numpy as np
import joblib

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.module6.fingerprints import smiles_to_morgan_fp

MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "admet")

# Cache loaded models
_model_cache = {}


def _load_model(dataset_key):
    """Load a trained model from disk, with caching."""
    if dataset_key in _model_cache:
        return _model_cache[dataset_key]

    model_path = os.path.join(MODELS_DIR, f"{dataset_key}_model.joblib")
    if not os.path.exists(model_path):
        # Train on first call if missing
        print(f"  Model {dataset_key} not found. Training...")
        from src.module6.train_models import (
            train_classification_model,
            train_regression_model,
            CLASSIFICATION_DATASETS,
            REGRESSION_DATASETS,
        )
        if dataset_key in CLASSIFICATION_DATASETS:
            train_classification_model(dataset_key, verbose=True)
        elif dataset_key in REGRESSION_DATASETS:
            train_regression_model(dataset_key, verbose=True)
        else:
            raise ValueError(f"Unknown dataset key: {dataset_key}")

    model_data = joblib.load(model_path)
    _model_cache[dataset_key] = model_data
    return model_data


def _predict_classification(smiles, dataset_key):
    """Generic classification prediction using ensemble."""
    model_data = _load_model(dataset_key)
    fp = smiles_to_morgan_fp(smiles, n_bits=model_data["n_bits"])
    if fp is None:
        return None

    X = fp.reshape(1, -1).astype(np.float64)

    rf_prob = model_data["rf"].predict_proba(X)[0, 1]
    gb_prob = model_data["gb"].predict_proba(X)[0, 1]
    ensemble_prob = (rf_prob + gb_prob) / 2

    return {
        "probability": round(float(ensemble_prob), 4),
        "rf_probability": round(float(rf_prob), 4),
        "gb_probability": round(float(gb_prob), 4),
        "model_metrics": model_data["metrics"],
    }


def predict_bbb_ml(smiles):
    """Predict BBB permeability using ML model.

    Returns:
        dict with probability, classification, confidence
    """
    result = _predict_classification(smiles, "bbb")
    if result is None:
        return None

    prob = result["probability"]
    classification = "Penetrant" if prob >= 0.5 else "Non-penetrant"
    confidence = abs(prob - 0.5) * 2  # 0-1 scale, 1 = most confident

    return {
        "probability": prob,
        "classification": classification,
        "confidence": round(confidence, 4),
        "model_auroc": result["model_metrics"]["auroc"],
        **{k: v for k, v in result.items() if k != "model_metrics"},
    }


def predict_herg(smiles):
    """Predict hERG channel inhibition risk.

    Returns:
        dict with probability, classification, risk_level
    """
    result = _predict_classification(smiles, "herg")
    if result is None:
        return None

    prob = result["probability"]
    classification = "Inhibitor" if prob >= 0.5 else "Non-inhibitor"

    if prob >= 0.7:
        risk_level = "High"
    elif prob >= 0.4:
        risk_level = "Moderate"
    else:
        risk_level = "Low"

    return {
        "probability": prob,
        "classification": classification,
        "risk_level": risk_level,
        "model_auroc": result["model_metrics"]["auroc"],
    }


def predict_cyp2d6(smiles):
    """Predict CYP2D6 inhibition.

    Returns:
        dict with probability, is_inhibitor
    """
    result = _predict_classification(smiles, "cyp2d6")
    if result is None:
        return None

    prob = result["probability"]
    return {
        "probability": prob,
        "is_inhibitor": prob >= 0.5,
        "model_auroc": result["model_metrics"]["auroc"],
    }


def predict_hia(smiles):
    """Predict human intestinal absorption.

    Returns:
        dict with probability, classification
    """
    result = _predict_classification(smiles, "hia")
    if result is None:
        return None

    prob = result["probability"]
    classification = "High absorption" if prob >= 0.5 else "Low absorption"

    return {
        "probability": prob,
        "classification": classification,
        "model_auroc": result["model_metrics"]["auroc"],
    }


def predict_clearance(smiles):
    """Predict hepatic clearance.

    Returns:
        dict with predicted_clearance, category
    """
    model_data = _load_model("clearance")
    fp = smiles_to_morgan_fp(smiles, n_bits=model_data["n_bits"])
    if fp is None:
        return None

    X = fp.reshape(1, -1).astype(np.float64)

    rf_pred = model_data["rf"].predict(X)[0]
    gb_pred = model_data["gb"].predict(X)[0]
    ensemble_pred = (rf_pred + gb_pred) / 2

    # Categorize clearance
    if ensemble_pred < 10:
        category = "Low clearance"
    elif ensemble_pred < 50:
        category = "Moderate clearance"
    else:
        category = "High clearance"

    return {
        "predicted_clearance": round(float(ensemble_pred), 2),
        "category": category,
        "model_r2": model_data["metrics"]["r2"],
        "model_mae": model_data["metrics"]["mae"],
    }


def predict_all_admet(smiles):
    """Run all ADMET predictions for a single compound.

    Returns:
        dict with all prediction results
    """
    bbb = predict_bbb_ml(smiles)
    herg = predict_herg(smiles)
    cyp2d6 = predict_cyp2d6(smiles)
    hia = predict_hia(smiles)
    clearance = predict_clearance(smiles)

    return {
        "smiles": smiles,
        "bbb": bbb,
        "herg": herg,
        "cyp2d6": cyp2d6,
        "hia": hia,
        "clearance": clearance,
    }


def predict_admet_batch(smiles_list, names=None):
    """Run all ADMET predictions for a list of compounds.

    Returns:
        pandas DataFrame with all predictions
    """
    import pandas as pd

    if names is None:
        names = [f"Compound_{i}" for i in range(len(smiles_list))]

    rows = []
    for smi, name in zip(smiles_list, names):
        result = predict_all_admet(smi)
        if result is None:
            continue

        row = {"name": name, "smiles": smi}

        if result["bbb"]:
            row["bbb_ml_prob"] = result["bbb"]["probability"]
            row["bbb_ml_class"] = result["bbb"]["classification"]
            row["bbb_ml_confidence"] = result["bbb"]["confidence"]

        if result["herg"]:
            row["herg_prob"] = result["herg"]["probability"]
            row["herg_class"] = result["herg"]["classification"]
            row["herg_risk"] = result["herg"]["risk_level"]

        if result["cyp2d6"]:
            row["cyp2d6_prob"] = result["cyp2d6"]["probability"]
            row["cyp2d6_inhibitor"] = result["cyp2d6"]["is_inhibitor"]

        if result["hia"]:
            row["hia_prob"] = result["hia"]["probability"]
            row["hia_class"] = result["hia"]["classification"]

        if result["clearance"]:
            row["clearance_pred"] = result["clearance"]["predicted_clearance"]
            row["clearance_cat"] = result["clearance"]["category"]

        rows.append(row)

    return pd.DataFrame(rows)
