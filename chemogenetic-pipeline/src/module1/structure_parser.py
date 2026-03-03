"""
Module 1: Structure Parser
Parses PDB and CIF files using BioPython to extract atom coordinates,
residue information, and chain data from AlphaFold predictions and
experimental structures.
"""

import os
import numpy as np
from Bio.PDB import PDBParser, MMCIFParser, Selection


def parse_structure(filepath, structure_id=None):
    """Parse a PDB or CIF file and return a BioPython Structure object.

    Args:
        filepath: Path to .pdb or .cif file.
        structure_id: Optional identifier. Defaults to filename stem.

    Returns:
        Bio.PDB.Structure.Structure object.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file extension is not .pdb or .cif.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Structure file not found: {filepath}")

    if structure_id is None:
        structure_id = os.path.splitext(os.path.basename(filepath))[0]

    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdb":
        parser = PDBParser(QUIET=True)
    elif ext == ".cif":
        parser = MMCIFParser(QUIET=True)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .pdb or .cif")

    return parser.get_structure(structure_id, filepath)


def get_ca_atoms(structure, chain_id=None):
    """Extract C-alpha atom coordinates from a structure.

    Args:
        structure: BioPython Structure object.
        chain_id: Optional chain ID to filter by (e.g. "A").

    Returns:
        List of Bio.PDB.Atom.Atom objects (CA atoms only).
    """
    ca_atoms = []
    for model in structure:
        for chain in model:
            if chain_id is not None and chain.get_id() != chain_id:
                continue
            for residue in chain:
                if residue.get_id()[0] == " " and "CA" in residue:
                    ca_atoms.append(residue["CA"])
        break  # Use first model only
    return ca_atoms


def get_ca_coordinates(structure):
    """Extract C-alpha coordinates as a numpy array.

    Args:
        structure: BioPython Structure object.

    Returns:
        numpy array of shape (N, 3) with CA coordinates.
    """
    atoms = get_ca_atoms(structure)
    return np.array([atom.get_vector().get_array() for atom in atoms])


def get_residue_info(structure):
    """Extract residue-level information from a structure.

    Args:
        structure: BioPython Structure object.

    Returns:
        List of dicts with keys: chain_id, resname, resseq, x, y, z (CA coords).
    """
    residues = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] != " ":
                    continue  # Skip heteroatoms/water
                if "CA" not in residue:
                    continue
                ca = residue["CA"]
                coord = ca.get_vector().get_array()
                residues.append({
                    "chain_id": chain.get_id(),
                    "resname": residue.get_resname(),
                    "resseq": residue.get_id()[1],
                    "x": float(coord[0]),
                    "y": float(coord[1]),
                    "z": float(coord[2]),
                })
        break  # First model only
    return residues


def get_structure_summary(structure):
    """Get a summary of the structure contents.

    Returns:
        Dict with keys: n_models, n_chains, n_residues, n_atoms, chain_ids.
    """
    model = structure[0]
    chains = list(model.get_chains())
    residues = list(model.get_residues())
    atoms = list(model.get_atoms())

    return {
        "n_models": len(list(structure.get_models())),
        "n_chains": len(chains),
        "n_residues": len(residues),
        "n_atoms": len(atoms),
        "chain_ids": [c.get_id() for c in chains],
    }


def get_bfactors(structure):
    """Extract B-factors (used as pLDDT in AlphaFold PDB output).

    In AlphaFold PDB files, the B-factor column stores pLDDT scores (0-100).

    Returns:
        List of dicts with keys: chain_id, resseq, resname, bfactor.
    """
    bfactors = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] != " ":
                    continue
                if "CA" not in residue:
                    continue
                ca = residue["CA"]
                bfactors.append({
                    "chain_id": chain.get_id(),
                    "resseq": residue.get_id()[1],
                    "resname": residue.get_resname(),
                    "bfactor": ca.get_bfactor(),
                })
        break
    return bfactors
