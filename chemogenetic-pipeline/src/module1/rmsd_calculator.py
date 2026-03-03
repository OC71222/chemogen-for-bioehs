"""
Module 1: RMSD Calculator
Calculates Root Mean Square Deviation between predicted and experimental
structures using BioPython's Superimposer for optimal alignment.
"""

import os
import numpy as np
import pandas as pd
from Bio.PDB import Superimposer

from src.module1.structure_parser import parse_structure, get_ca_atoms

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def calculate_rmsd(structure_a, structure_b, chain_a=None, chain_b=None):
    """Calculate RMSD between two structures after optimal superposition.

    Uses C-alpha atoms for alignment, matching by residue sequence number.
    Only residues present in both structures are compared. Structures are
    aligned using the Kabsch algorithm (BioPython Superimposer).

    Args:
        structure_a: BioPython Structure (reference/experimental).
        structure_b: BioPython Structure (predicted).
        chain_a: Optional chain ID for structure_a (e.g. "A").
        chain_b: Optional chain ID for structure_b.

    Returns:
        Dict with keys:
            rmsd: float — RMSD in Angstroms after superposition.
            n_atoms: int — Number of CA atoms used for alignment.
            aligned: bool — Whether alignment was successful.
    """
    from Bio import Align
    from Bio.SeqUtils import seq1

    ca_a = get_ca_atoms(structure_a, chain_id=chain_a)
    ca_b = get_ca_atoms(structure_b, chain_id=chain_b)

    if len(ca_a) == 0 or len(ca_b) == 0:
        return {"rmsd": None, "n_atoms": 0, "aligned": False}

    # Extract sequences
    seq_a = "".join([atom.get_parent().resname for atom in ca_a])
    seq_b = "".join([atom.get_parent().resname for atom in ca_b])
    
    # Convert to 1-letter codes, filling unknown with 'X'
    seq_a_1l = seq1(seq_a, custom_map={"UNK": "X"})
    seq_b_1l = seq1(seq_b, custom_map={"UNK": "X"})

    # Align sequences
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -5.0
    aligner.extend_gap_score = -0.5
    
    try:
        alignments = aligner.align(seq_a_1l, seq_b_1l)
        best_aln = next(alignments)
    except (StopIteration, OverflowError):
        return {"rmsd": None, "n_atoms": 0, "aligned": False}
    
    # Map aligned positions back to atoms using alignment coordinates
    matched_a = []
    matched_b = []
    
    # best_aln.aligned forms tuples of ((startA, endA), (startB, endB)) for matching blocks
    for (start_a, end_a), (start_b, end_b) in zip(best_aln.aligned[0], best_aln.aligned[1]):
        for offset in range(end_a - start_a):
            matched_a.append(ca_a[start_a + offset])
            matched_b.append(ca_b[start_b + offset])

    if len(matched_a) < 10:
        # Too few common residues for meaningful alignment
        return {"rmsd": None, "n_atoms": len(matched_a), "aligned": False}

    sup = Superimposer()
    sup.set_atoms(matched_a, matched_b)

    return {
        "rmsd": round(sup.rms, 3),
        "n_atoms": len(matched_a),
        "aligned": True,
    }


def calculate_rmsd_from_files(ref_path, pred_path, ref_chain=None, pred_chain=None):
    """Calculate RMSD between two structure files.

    Args:
        ref_path: Path to reference (experimental) PDB/CIF file.
        pred_path: Path to predicted PDB/CIF file.
        ref_chain: Optional chain ID for reference structure.
        pred_chain: Optional chain ID for predicted structure.

    Returns:
        Dict with rmsd, n_atoms, aligned, ref_file, pred_file.
    """
    ref = parse_structure(ref_path, "reference")
    pred = parse_structure(pred_path, "predicted")

    result = calculate_rmsd(ref, pred, chain_a=ref_chain, chain_b=pred_chain)
    result["ref_file"] = os.path.basename(ref_path)
    result["pred_file"] = os.path.basename(pred_path)
    return result


def batch_rmsd(pairs):
    """Calculate RMSD for multiple structure pairs.

    Args:
        pairs: List of dicts with keys:
            name: str — compound/structure name.
            ref_path: str — path to experimental structure.
            pred_path: str — path to predicted structure.

    Returns:
        pandas DataFrame with columns: name, rmsd, n_atoms, ref_file, pred_file.
    """
    results = []
    for pair in pairs:
        name = pair["name"]
        try:
            result = calculate_rmsd_from_files(pair["ref_path"], pair["pred_path"])
            result["name"] = name
            results.append(result)
        except (FileNotFoundError, ValueError) as e:
            results.append({
                "name": name,
                "rmsd": None,
                "n_atoms": 0,
                "aligned": False,
                "ref_file": os.path.basename(pair.get("ref_path", "")),
                "pred_file": os.path.basename(pair.get("pred_path", "")),
                "error": str(e),
            })

    return pd.DataFrame(results)


def save_rmsd_results(df, output_path=None):
    """Save RMSD results to CSV.

    Args:
        df: DataFrame from batch_rmsd().
        output_path: Output CSV path. Defaults to data/results/rmsd_results.csv.

    Returns:
        Path to saved CSV file.
    """
    if output_path is None:
        output_path = os.path.join(PROJECT_ROOT, "data", "results", "rmsd_results.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved RMSD results: {output_path}")
    return output_path


def classify_rmsd(rmsd_value):
    """Classify RMSD quality for structure prediction.

    Returns:
        str — "Excellent" (<1.0), "Good" (<2.0), "Acceptable" (<3.0), or "Poor".
    """
    if rmsd_value is None:
        return "N/A"
    if rmsd_value < 1.0:
        return "Excellent"
    if rmsd_value < 2.0:
        return "Good"
    if rmsd_value < 3.0:
        return "Acceptable"
    return "Poor"
