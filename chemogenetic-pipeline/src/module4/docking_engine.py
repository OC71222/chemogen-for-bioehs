"""
Module 4: Docking Engine
Wrapper around AutoDock Vina Python API for molecular docking.
"""

import os
import sys
import tempfile
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)


def dock_compound(receptor_pdbqt, ligand_pdbqt, center, box_size, exhaustiveness=32,
                  n_poses=9, energy_range=3.0):
    """Dock a single compound using Vina.

    Args:
        receptor_pdbqt: Path to receptor PDBQT file
        ligand_pdbqt: PDBQT string or path to ligand PDBQT
        center: tuple/list of (x, y, z) for box center
        box_size: tuple/list of (sx, sy, sz) for box dimensions
        exhaustiveness: Vina exhaustiveness (higher = more thorough)
        n_poses: Maximum number of poses to return
        energy_range: Energy range for pose clustering (kcal/mol)

    Returns:
        dict with affinity, poses, and metadata
    """
    try:
        from vina import Vina

        v = Vina(sf_name="vina")

        # Set receptor
        v.set_receptor(receptor_pdbqt)

        # Set ligand (from string or file)
        if os.path.isfile(ligand_pdbqt):
            v.set_ligand_from_file(ligand_pdbqt)
        else:
            # Write string to temp file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".pdbqt", delete=False) as f:
                f.write(ligand_pdbqt)
                tmp_path = f.name
            try:
                v.set_ligand_from_file(tmp_path)
            finally:
                os.unlink(tmp_path)

        # Configure search space
        v.compute_vina_maps(
            center=list(center),
            box_size=list(box_size),
        )

        # Dock
        v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)

        # Get results
        energies = v.energies()
        # energies is (n_poses, 3): [affinity, inter, intra]

        poses_pdbqt = v.poses()

        results = {
            "best_affinity": float(energies[0][0]),
            "n_poses": len(energies),
            "all_affinities": [float(e[0]) for e in energies],
            "poses_pdbqt": poses_pdbqt,
            "success": True,
        }

        return results

    except ImportError:
        print("    Warning: AutoDock Vina not available. Using estimated docking scores.")
        return _estimate_docking(ligand_pdbqt)

    except Exception as e:
        print(f"    Docking error: {e}")
        return {"best_affinity": None, "n_poses": 0, "success": False, "error": str(e)}


def _estimate_docking(ligand_pdbqt):
    """Estimate docking score when Vina is unavailable.

    Uses a simple heuristic based on molecular size and charge.
    For demo purposes only.
    """
    import numpy as np

    # Count heavy atoms from PDBQT
    if isinstance(ligand_pdbqt, str) and not os.path.isfile(ligand_pdbqt):
        n_atoms = ligand_pdbqt.count("HETATM") + ligand_pdbqt.count("ATOM")
    else:
        with open(ligand_pdbqt) as f:
            content = f.read()
            n_atoms = content.count("HETATM") + content.count("ATOM")

    # Simple estimator: larger molecules tend to bind more tightly
    # Typical range: -4 to -12 kcal/mol
    np.random.seed(hash(str(n_atoms)) % 2**32)
    base_affinity = -5.0 - (n_atoms / 10.0) + np.random.normal(0, 0.5)
    base_affinity = max(-12.0, min(-3.0, base_affinity))

    return {
        "best_affinity": round(base_affinity, 2),
        "n_poses": 1,
        "all_affinities": [round(base_affinity, 2)],
        "poses_pdbqt": None,
        "success": True,
        "estimated": True,
    }


def dock_all_compounds(receptor_info, ligand_dict, exhaustiveness=32, verbose=True):
    """Batch dock all prepared ligands against receptor.

    Args:
        receptor_info: dict from receptor_prep.prepare_receptor()
        ligand_dict: dict from ligand_prep.prepare_all_ligands()
        exhaustiveness: Vina exhaustiveness parameter
        verbose: Print progress

    Returns:
        pandas DataFrame with docking results
    """
    receptor_pdbqt = receptor_info["pdbqt"]
    bs = receptor_info["binding_site"]
    center = (bs["center_x"], bs["center_y"], bs["center_z"])
    box_size = (bs["size_x"], bs["size_y"], bs["size_z"])

    results = []
    for name, ligand_info in ligand_dict.items():
        if verbose:
            print(f"    Docking {name}...")

        pdbqt = ligand_info.get("pdbqt_path", ligand_info.get("pdbqt_string"))
        dock_result = dock_compound(
            receptor_pdbqt, pdbqt, center, box_size,
            exhaustiveness=exhaustiveness,
        )

        row = {
            "name": name,
            "affinity_kcal_mol": dock_result.get("best_affinity"),
            "n_poses": dock_result.get("n_poses", 0),
            "success": dock_result.get("success", False),
            "estimated": dock_result.get("estimated", False),
        }

        # Save top pose if available
        if dock_result.get("poses_pdbqt"):
            poses_dir = os.path.join(PROJECT_ROOT, "data", "docking", "poses")
            os.makedirs(poses_dir, exist_ok=True)
            safe_name = name.replace(" ", "_")
            pose_path = os.path.join(poses_dir, f"{safe_name}_pose.pdbqt")
            with open(pose_path, "w") as f:
                f.write(dock_result["poses_pdbqt"])
            row["pose_path"] = pose_path

        results.append(row)

        if verbose and dock_result.get("best_affinity"):
            est = " (estimated)" if dock_result.get("estimated") else ""
            print(f"      Affinity: {dock_result['best_affinity']:.2f} kcal/mol{est}")

    return pd.DataFrame(results)


def save_docking_results(df, output_path=None):
    """Save docking results to CSV."""
    if output_path is None:
        output_path = os.path.join(PROJECT_ROOT, "data", "results", "docking_results.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"    Saved docking results: {output_path}")
    return output_path
