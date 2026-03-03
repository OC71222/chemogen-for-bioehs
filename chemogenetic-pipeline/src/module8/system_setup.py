"""
Module 8: Molecular Dynamics System Setup
Builds OpenMM simulation systems for receptor-ligand complexes.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

SYSTEMS_DIR = os.path.join(PROJECT_ROOT, "data", "md", "systems")


def load_receptor(pdb_path):
    """Load and prepare receptor structure for MD.

    Args:
        pdb_path: Path to fixed receptor PDB from Module 4

    Returns:
        OpenMM Modeller or PDB object
    """
    try:
        from openmm.app import PDBFile
        pdb = PDBFile(pdb_path)
        print(f"    Loaded receptor: {pdb_path}")
        print(f"    Atoms: {pdb.topology.getNumAtoms()}")
        print(f"    Residues: {pdb.topology.getNumResidues()}")
        return pdb
    except ImportError:
        print("    Warning: OpenMM not available. Cannot load receptor for MD.")
        return None


def _pdbqt_to_pdb_block(pdbqt_path):
    """Convert PDBQT file to PDB block string (first model only).

    PDBQT files use AutoDock atom types (A, NA, OA, HD, etc.) in the
    element column, which RDKit cannot parse. This strips the extra
    PDBQT columns and infers proper element symbols from atom names.
    Also skips PDBQT-specific keywords (ROOT, BRANCH, TORSDOF).
    """
    lines = []
    in_first_model = False
    with open(pdbqt_path) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("MODEL"):
                if in_first_model:
                    break  # hit MODEL 2, stop
                in_first_model = True
                continue
            if stripped == "ENDMDL":
                break
            if stripped.startswith(("ROOT", "ENDROOT", "BRANCH", "ENDBRANCH", "TORSDOF", "REMARK")):
                continue
            if line.startswith(("ATOM", "HETATM")):
                atom_name = line[12:16].strip()
                element = atom_name.lstrip("0123456789")[0:1]
                if atom_name.startswith("Cl"):
                    element = "Cl"
                elif atom_name.startswith("Br"):
                    element = "Br"
                pdb_line = line[:66].ljust(76) + element.rjust(2) + "  "
                lines.append(pdb_line)
    lines.append("END")
    return "\n".join(lines) + "\n"


def load_docked_ligand(pose_path, smiles=None):
    """Load docked ligand pose from Module 4 output.

    Args:
        pose_path: Path to docked pose PDBQT
        smiles: SMILES string for parameterization

    Returns:
        RDKit Mol with 3D coordinates
    """
    from rdkit import Chem

    # Try to read PDBQT by converting to PDB format first
    if pose_path and os.path.exists(pose_path):
        if pose_path.endswith(".pdbqt"):
            pdb_block = _pdbqt_to_pdb_block(pose_path)
            mol = Chem.MolFromPDBBlock(pdb_block, removeHs=False, sanitize=False)
            if mol:
                try:
                    Chem.SanitizeMol(mol)
                except Exception:
                    pass  # Accept even if sanitization is imperfect
                print(f"    Loaded docked pose: {pose_path}")
                return mol
        else:
            mol = Chem.MolFromPDBFile(pose_path, removeHs=False)
            if mol:
                print(f"    Loaded docked pose: {pose_path}")
                return mol

    # Fallback: generate from SMILES
    if smiles:
        from src.module4.ligand_prep import smiles_to_3d
        mol = smiles_to_3d(smiles)
        if mol:
            print(f"    Generated 3D conformer from SMILES (no docked pose available)")
            return mol

    return None


def build_system(receptor_pdb, ligand_mol, compound_name, verbose=True):
    """Build complete MD simulation system.

    1. Combine receptor + ligand
    2. Parameterize with AMBER ff14SB + GAFF2
    3. Solvate with TIP3P
    4. Add ions (0.15M NaCl)

    Args:
        receptor_pdb: OpenMM PDB object
        ligand_mol: RDKit Mol with 3D coords
        compound_name: Name for output files
        verbose: Print progress

    Returns:
        dict with system, topology, positions, and paths
    """
    os.makedirs(SYSTEMS_DIR, exist_ok=True)

    try:
        import openmm
        from openmm import app, unit
        from openmm.app import ForceField, Modeller, PDBFile

        # Create modeller from receptor
        modeller = Modeller(receptor_pdb.topology, receptor_pdb.positions)

        if verbose:
            print(f"    Receptor atoms: {modeller.topology.getNumAtoms()}")

        # Parameterize protein with AMBER ff14SB
        forcefield = ForceField("amber14-all.xml", "amber14/tip3pfb.xml")

        # Try to add ligand with OpenFF SMIRNOFF
        ligand_added = False
        try:
            from openmmforcefields.generators import SMIRNOFFTemplateGenerator
            from openff.toolkit import Molecule as OFFMolecule
            from rdkit import Chem

            # Add explicit hydrogens for MD (PDBQT files are missing most H)
            ligand_mol = Chem.AddHs(ligand_mol, addCoords=True)

            # Convert RDKit Mol to OpenFF Molecule for SMIRNOFF parameterization
            off_mol = OFFMolecule.from_rdkit(ligand_mol, allow_undefined_stereo=True)

            # Pre-assign Gasteiger charges to avoid AM1BCC/NAGL dependency
            off_mol.assign_partial_charges(partial_charge_method="gasteiger")

            smirnoff = SMIRNOFFTemplateGenerator(
                molecules=off_mol,
                forcefield="openff-2.1.0",
            )
            forcefield.registerTemplateGenerator(smirnoff.generator)

            # Write ligand PDB with unique atom names so OpenMM doesn't
            # merge "duplicate" atoms (all share residue UNL)
            from io import StringIO
            ligand_pdb_string = Chem.MolToPDBBlock(ligand_mol)
            elem_counts = {}
            fixed_lines = []
            for line in ligand_pdb_string.splitlines():
                if line.startswith(("ATOM", "HETATM")):
                    elem = line[76:78].strip()
                    if not elem:
                        elem = line[12:16].strip()[0]
                    elem_counts[elem] = elem_counts.get(elem, 0) + 1
                    name = f"{elem}{elem_counts[elem]}"
                    name = name.ljust(4) if len(name) < 4 else name[:4]
                    line = line[:12] + name.ljust(4) + line[16:]
                fixed_lines.append(line)
            ligand_pdb_string = "\n".join(fixed_lines) + "\n"
            ligand_pdb = PDBFile(StringIO(ligand_pdb_string))
            modeller.add(ligand_pdb.topology, ligand_pdb.positions)
            ligand_added = True

            if verbose:
                print(f"    Added ligand with OpenFF SMIRNOFF parameterization")

        except ImportError:
            if verbose:
                print("    Warning: openmmforcefields not available. Proceeding without ligand.")
        except Exception as e:
            if verbose:
                print(f"    Warning: Ligand parameterization failed: {e}")

        # Solvate
        if verbose:
            print("    Solvating system (10A padding, TIP3P)...")

        modeller.addSolvent(
            forcefield,
            model="tip3p",
            padding=1.0 * unit.nanometers,  # 10A
            ionicStrength=0.15 * unit.molar,  # 0.15M NaCl
        )

        if verbose:
            print(f"    Total atoms after solvation: {modeller.topology.getNumAtoms()}")

        # Create system
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0 * unit.nanometers,
            constraints=app.HBonds,
        )

        # Save system
        safe_name = compound_name.replace(" ", "_")
        system_path = os.path.join(SYSTEMS_DIR, f"{safe_name}_system.xml")

        from openmm import XmlSerializer
        with open(system_path, "w") as f:
            f.write(XmlSerializer.serialize(system))

        # Save topology + positions
        pdb_path = os.path.join(SYSTEMS_DIR, f"{safe_name}_solvated.pdb")
        with open(pdb_path, "w") as f:
            PDBFile.writeFile(modeller.topology, modeller.positions, f)

        if verbose:
            print(f"    System saved: {system_path}")
            print(f"    Solvated PDB: {pdb_path}")

        return {
            "system": system,
            "topology": modeller.topology,
            "positions": modeller.positions,
            "system_path": system_path,
            "pdb_path": pdb_path,
            "n_atoms": modeller.topology.getNumAtoms(),
            "ligand_added": ligand_added,
        }

    except ImportError as e:
        print(f"    ERROR: OpenMM not available ({e})")
        print("    Install with: conda install -c conda-forge openmm openmmforcefields")
        return None

    except Exception as e:
        print(f"    ERROR during system setup: {e}")
        return None


def setup_for_compound(compound_name, verbose=True):
    """Set up MD system for a specific compound using Module 4 outputs.

    Args:
        compound_name: Name of compound (must have docking results)
        verbose: Print progress

    Returns:
        System info dict or None
    """
    from src.module4.receptor_prep import prepare_receptor
    from src.module2.compounds import load_compounds

    # Get receptor
    receptor_info = prepare_receptor(pdb_id="8E9W", verbose=False)
    receptor_pdb_path = receptor_info["fixed_pdb"]

    receptor_pdb = load_receptor(receptor_pdb_path)
    if receptor_pdb is None:
        return None

    # Get ligand
    compounds = load_compounds()
    row = compounds[compounds["name"] == compound_name]
    if row.empty:
        print(f"    ERROR: Compound '{compound_name}' not found")
        return None

    smiles = row.iloc[0]["smiles"]

    # Check for docked pose
    safe_name = compound_name.replace(" ", "_")
    pose_path = os.path.join(PROJECT_ROOT, "data", "docking", "poses", f"{safe_name}_pose.pdbqt")
    ligand_mol = load_docked_ligand(pose_path, smiles=smiles)

    if ligand_mol is None:
        print(f"    ERROR: Could not load ligand for {compound_name}")
        return None

    # Build system
    return build_system(receptor_pdb, ligand_mol, compound_name, verbose=verbose)
