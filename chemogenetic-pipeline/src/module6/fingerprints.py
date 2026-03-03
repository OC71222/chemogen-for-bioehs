"""
Module 6: Molecular Fingerprints & Feature Generation
Converts SMILES to numerical feature vectors for ML models.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors


def smiles_to_morgan_fp(smiles, radius=2, n_bits=2048):
    """Convert SMILES to Morgan fingerprint (ECFP-like) as numpy array.

    Args:
        smiles: SMILES string
        radius: Morgan radius (2 = ECFP4, 3 = ECFP6)
        n_bits: Number of bits in fingerprint

    Returns:
        numpy array of shape (n_bits,) or None if invalid SMILES
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.int8)
    fp.ToBitString()
    for bit in fp.GetOnBits():
        arr[bit] = 1
    return arr


def smiles_to_rdkit_descriptors(smiles):
    """Calculate RDKit 2D molecular descriptors.

    Returns:
        dict of descriptor_name -> value, or None if invalid SMILES
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    desc = {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
        "NumAromaticRings": Descriptors.NumAromaticRings(mol),
        "FractionCSP3": Descriptors.FractionCSP3(mol),
        "NumHeavyAtoms": Descriptors.HeavyAtomCount(mol),
        "NumValenceElectrons": Descriptors.NumValenceElectrons(mol),
        "NumRadicalElectrons": Descriptors.NumRadicalElectrons(mol),
        "RingCount": Descriptors.RingCount(mol),
        "MolMR": Descriptors.MolMR(mol),
        "HallKierAlpha": Descriptors.HallKierAlpha(mol),
        "BertzCT": Descriptors.BertzCT(mol),
        "NumAliphaticRings": Descriptors.NumAliphaticRings(mol),
        "NumSaturatedRings": Descriptors.NumSaturatedRings(mol),
        "NumHeteroatoms": rdMolDescriptors.CalcNumHeteroatoms(mol),
        "NumAmideBonds": rdMolDescriptors.CalcNumAmideBonds(mol),
        "LabuteASA": Descriptors.LabuteASA(mol),
        "PEOE_VSA1": Descriptors.PEOE_VSA1(mol),
        "PEOE_VSA6": Descriptors.PEOE_VSA6(mol),
        "SMR_VSA1": Descriptors.SMR_VSA1(mol),
        "SMR_VSA5": Descriptors.SMR_VSA5(mol),
        "SlogP_VSA1": Descriptors.SlogP_VSA1(mol),
        "SlogP_VSA2": Descriptors.SlogP_VSA2(mol),
        "MaxAbsEStateIndex": Descriptors.MaxAbsEStateIndex(mol),
        "MinAbsEStateIndex": Descriptors.MinAbsEStateIndex(mol),
        "EState_VSA1": Descriptors.EState_VSA1(mol),
        "Chi0": Descriptors.Chi0(mol),
        "Chi1": Descriptors.Chi1(mol),
        "Kappa1": Descriptors.Kappa1(mol),
        "Kappa2": Descriptors.Kappa2(mol),
    }
    return desc


# Selected descriptors to combine with Morgan FP for ML models
_KEY_DESCRIPTORS = [
    "MolWt", "LogP", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "NumAromaticRings", "FractionCSP3",
    "NumHeavyAtoms", "RingCount", "MolMR", "HallKierAlpha",
    "LabuteASA", "BertzCT", "NumHeteroatoms",
]


def smiles_to_features(smiles, radius=2, n_bits=2048):
    """Combined feature vector: Morgan FP + key RDKit descriptors.

    Returns:
        numpy array of shape (n_bits + len(key_descriptors),) or None
    """
    fp = smiles_to_morgan_fp(smiles, radius=radius, n_bits=n_bits)
    if fp is None:
        return None

    desc = smiles_to_rdkit_descriptors(smiles)
    if desc is None:
        return None

    desc_values = np.array([desc.get(k, 0.0) for k in _KEY_DESCRIPTORS], dtype=np.float64)
    # Replace NaN/inf with 0
    desc_values = np.nan_to_num(desc_values, nan=0.0, posinf=0.0, neginf=0.0)

    return np.concatenate([fp.astype(np.float64), desc_values])


def batch_smiles_to_morgan(smiles_list, radius=2, n_bits=2048):
    """Batch convert SMILES list to Morgan FP matrix.

    Returns:
        features: numpy array of shape (n_valid, n_bits)
        valid_indices: list of indices that successfully converted
    """
    features = []
    valid_indices = []

    for i, smi in enumerate(smiles_list):
        fp = smiles_to_morgan_fp(smi, radius=radius, n_bits=n_bits)
        if fp is not None:
            features.append(fp)
            valid_indices.append(i)

    if not features:
        return np.array([]), []

    return np.vstack(features).astype(np.float64), valid_indices


def batch_smiles_to_features(smiles_list, radius=2, n_bits=2048):
    """Batch convert SMILES list to combined feature matrix.

    Returns:
        features: numpy array of shape (n_valid, n_features)
        valid_indices: list of indices that successfully converted
    """
    features = []
    valid_indices = []

    for i, smi in enumerate(smiles_list):
        feat = smiles_to_features(smi, radius=radius, n_bits=n_bits)
        if feat is not None:
            features.append(feat)
            valid_indices.append(i)

    if not features:
        return np.array([]), []

    return np.vstack(features), valid_indices
