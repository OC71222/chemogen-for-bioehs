"""
Module 4: Ligand Preparation
Converts SMILES to 3D conformers and PDBQT format for docking.
"""

import os
import sys
import tempfile
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolfiles

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)


def smiles_to_3d(smiles, n_conformers=1, optimize=True):
    """Convert SMILES to 3D molecule with conformer.

    Uses ETKDGv3 for conformer generation and MMFF for optimization.

    Args:
        smiles: SMILES string
        n_conformers: Number of conformers to generate
        optimize: Whether to MMFF optimize

    Returns:
        RDKit Mol with 3D coordinates, or None
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    # ETKDGv3 conformer generation
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.numThreads = 0  # use all cores

    cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params)
    if len(cids) == 0:
        # Fallback to ETKDG
        params = AllChem.ETKDG()
        params.randomSeed = 42
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params)

    if len(cids) == 0:
        return None

    # MMFF optimization
    if optimize:
        for cid in cids:
            try:
                AllChem.MMFFOptimizeMolecule(mol, confId=cid, maxIters=500)
            except Exception:
                pass  # If MMFF fails, use unoptimized conformer

    return mol


def mol_to_pdbqt_string(mol, confId=0):
    """Convert RDKit mol with 3D coords to PDBQT string via Meeko.

    Args:
        mol: RDKit Mol with 3D conformer
        confId: Conformer ID to use

    Returns:
        PDBQT string
    """
    try:
        from meeko import MoleculePreparation, RDKitMolCreate

        preparator = MoleculePreparation()
        mol_setup_list = preparator.prepare(mol)

        if mol_setup_list:
            pdbqt_string = preparator.write_pdbqt_string()
            return pdbqt_string
        return None

    except ImportError:
        # Fallback: generate PDB and manually convert to PDBQT-like format
        return _mol_to_pdbqt_fallback(mol, confId)

    except Exception as e:
        print(f"    Warning: Meeko prep failed: {e}. Using fallback.")
        return _mol_to_pdbqt_fallback(mol, confId)


def _mol_to_pdbqt_fallback(mol, confId=0):
    """Fallback PDBQT generation without Meeko.

    Generates a PDB-style file with AutoDock atom types and charges.
    """
    # Get Gasteiger charges
    AllChem.ComputeGasteigerCharges(mol)

    conf = mol.GetConformer(confId)
    lines = []

    for i, atom in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        element = atom.GetSymbol()
        charge = float(atom.GetDoubleProp("_GasteigerCharge"))
        if not (-10 < charge < 10):
            charge = 0.0

        # AD4 type mapping
        ad_type = element
        if element == "C" and atom.GetIsAromatic():
            ad_type = "A"

        atom_name = f"{element}{i+1}"
        line = (
            f"HETATM{i+1:>5d} {atom_name:<4s} LIG A   1    "
            f"{pos.x:>8.3f}{pos.y:>8.3f}{pos.z:>8.3f}"
            f"  1.00  0.00    {charge:>+6.3f} {ad_type:<2s}\n"
        )
        lines.append(line)

    lines.append("END\n")
    return "".join(lines)


def prepare_ligand(smiles, name=None, save_dir=None):
    """Full ligand preparation: SMILES → 3D → PDBQT.

    Args:
        smiles: SMILES string
        name: Compound name (for filenames)
        save_dir: Directory to save PDBQT file

    Returns:
        dict with mol, pdbqt_string, and optional file path
    """
    if name is None:
        name = "ligand"

    mol = smiles_to_3d(smiles)
    if mol is None:
        print(f"    Failed to generate 3D conformer for {name}")
        return None

    pdbqt_string = mol_to_pdbqt_string(mol)
    if pdbqt_string is None:
        print(f"    Failed to generate PDBQT for {name}")
        return None

    result = {
        "name": name,
        "smiles": smiles,
        "mol": mol,
        "pdbqt_string": pdbqt_string,
    }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        safe_name = name.replace(" ", "_")
        pdbqt_path = os.path.join(save_dir, f"{safe_name}.pdbqt")
        with open(pdbqt_path, "w") as f:
            f.write(pdbqt_string)
        result["pdbqt_path"] = pdbqt_path

    return result


def prepare_all_ligands(compounds_df=None, save_dir=None):
    """Prepare all 6 actuator ligands for docking.

    Args:
        compounds_df: DataFrame with 'name' and 'smiles' columns
        save_dir: Directory to save PDBQT files

    Returns:
        dict of name -> preparation result
    """
    if compounds_df is None:
        from src.module2.compounds import load_compounds
        compounds_df = load_compounds()

    if save_dir is None:
        save_dir = os.path.join(PROJECT_ROOT, "data", "docking", "ligands")

    results = {}
    for _, row in compounds_df.iterrows():
        name = row["name"]
        smiles = row["smiles"]
        print(f"    Preparing ligand: {name}")
        result = prepare_ligand(smiles, name=name, save_dir=save_dir)
        if result:
            results[name] = result

    return results
