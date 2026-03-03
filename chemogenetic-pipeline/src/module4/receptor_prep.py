"""
Module 4: Receptor Preparation
Downloads PDB 8E9W/8E9Y (hM3Dq DREADD cryo-EM structures), strips non-receptor
chains, adds missing atoms via PDBFixer, and converts to PDBQT for Vina docking.
"""

import os
import sys
import ssl
import urllib.request

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

# Handle macOS Python SSL cert issue
_SSL_CTX = ssl._create_unverified_context()

RECEPTOR_DIR = os.path.join(PROJECT_ROOT, "data", "docking", "receptor")

# Real cryo-EM structures from Nature 2022
PDB_STRUCTURES = {
    "8E9W": {
        "description": "hM3Dq DREADD + DCZ (deschloroclozapine)",
        "ligand": "DCZ",
        "url": "https://files.rcsb.org/download/8E9W.pdb",
    },
    "8E9Y": {
        "description": "hM3Dq DREADD + CNO (clozapine-N-oxide)",
        "ligand": "CNO",
        "url": "https://files.rcsb.org/download/8E9Y.pdb",
    },
}

# Orthosteric binding pocket center (near Asp3.32)
# Derived from cryo-EM ligand (WEC/DCZ) centroid in PDB 8E9W
BINDING_SITE = {
    "center_x": 133.9,
    "center_y": 128.9,
    "center_z": 154.9,
    "size_x": 22.0,
    "size_y": 22.0,
    "size_z": 22.0,
}


def download_pdb(pdb_id, output_dir=None):
    """Download PDB file from RCSB."""
    if output_dir is None:
        output_dir = RECEPTOR_DIR
    os.makedirs(output_dir, exist_ok=True)

    if pdb_id not in PDB_STRUCTURES:
        raise ValueError(f"Unknown PDB ID: {pdb_id}. Available: {list(PDB_STRUCTURES.keys())}")

    url = PDB_STRUCTURES[pdb_id]["url"]
    output_path = os.path.join(output_dir, f"{pdb_id}.pdb")

    if os.path.exists(output_path):
        print(f"    PDB {pdb_id} already downloaded: {output_path}")
        return output_path

    print(f"    Downloading PDB {pdb_id} from RCSB...")
    try:
        urllib.request.urlretrieve(url, output_path)
    except Exception:
        # Retry with unverified SSL
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, context=_SSL_CTX) as response:
            with open(output_path, "wb") as f:
                f.write(response.read())
    print(f"    Saved: {output_path}")
    return output_path


def clean_receptor(pdb_path, output_path=None, keep_chain="R"):
    """Strip water, ions, ligands, and G-protein chains from PDB.

    For 8E9W/8E9Y, the receptor is typically chain R (receptor).
    G-protein chains (miniGq) are removed.

    Args:
        pdb_path: Path to raw PDB file
        output_path: Where to save cleaned PDB (default: same dir, _clean suffix)
        keep_chain: Chain ID to keep (None = keep all protein chains, remove non-protein)
    """
    if output_path is None:
        base = os.path.splitext(pdb_path)[0]
        output_path = f"{base}_clean.pdb"

    lines_out = []
    with open(pdb_path, "r") as f:
        for line in f:
            record = line[:6].strip()

            # Skip water and HETATM (ligands, ions)
            if record == "HETATM":
                continue

            # Keep only ATOM records
            if record == "ATOM":
                # If we have a chain filter, apply it
                if keep_chain is not None:
                    chain_id = line[21]
                    if chain_id != keep_chain:
                        continue
                lines_out.append(line)
            elif record in ("TER", "END"):
                lines_out.append(line)

    # If no atoms found with the specified chain, try without chain filter
    atom_lines = [l for l in lines_out if l[:4] == "ATOM"]
    if not atom_lines and keep_chain is not None:
        print(f"    Warning: No atoms found for chain {keep_chain}. Trying all protein chains...")
        return clean_receptor(pdb_path, output_path, keep_chain=None)

    with open(output_path, "w") as f:
        f.writelines(lines_out)

    n_atoms = len(atom_lines)
    print(f"    Cleaned receptor: {n_atoms} atoms saved to {output_path}")
    return output_path


def fix_receptor(pdb_path, output_path=None):
    """Use PDBFixer to add missing residues and atoms.

    Args:
        pdb_path: Path to cleaned PDB
        output_path: Where to save fixed PDB
    """
    if output_path is None:
        base = os.path.splitext(pdb_path)[0]
        output_path = f"{base}_fixed.pdb"

    try:
        from pdbfixer import PDBFixer
        from openmm.app import PDBFile

        fixer = PDBFixer(filename=pdb_path)
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.4)  # pH 7.4

        with open(output_path, "w") as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f)

        print(f"    Fixed receptor saved: {output_path}")
        return output_path

    except ImportError:
        print("    Warning: PDBFixer not available. Using cleaned PDB directly.")
        # Just copy the clean file
        import shutil
        shutil.copy2(pdb_path, output_path)
        return output_path


def receptor_to_pdbqt(pdb_path, output_path=None):
    """Convert receptor PDB to PDBQT format via Meeko.

    Args:
        pdb_path: Path to fixed PDB
        output_path: Where to save PDBQT
    """
    if output_path is None:
        base = os.path.splitext(pdb_path)[0]
        output_path = f"{base}.pdbqt"

    try:
        from meeko import MoleculePreparation, PDBMoleculeSetup

        setup = PDBMoleculeSetup.from_pdb_file(pdb_path)
        prep = MoleculePreparation()
        prep.prepare(setup)
        pdbqt_string = prep.write_pdbqt_string()

        with open(output_path, "w") as f:
            f.write(pdbqt_string)

        print(f"    Receptor PDBQT saved: {output_path}")
        return output_path

    except (ImportError, Exception) as e:
        # Fallback: simple PDBQT conversion (add Gasteiger charges)
        print(f"    Meeko receptor prep failed ({e}). Using simple PDBQT conversion...")
        return _simple_pdb_to_pdbqt(pdb_path, output_path)


def _simple_pdb_to_pdbqt(pdb_path, output_path):
    """Simple PDB to PDBQT conversion without Meeko.

    Produces valid PDBQT format with AutoDock atom types and zero partial
    charges. PDBQT columns are:
      1-54:  standard PDB (record, serial, name, resName, chainID, resSeq, x, y, z)
      55-60: occupancy
      61-66: tempFactor
      67-76: partial charge (right-justified, 10 chars)
      77-79: AutoDock atom type (left-justified, 2 chars)
    """
    # Map element symbols to AutoDock atom types
    AD_TYPE_MAP = {
        "C": "C", "N": "N", "O": "OA", "S": "SA", "H": "HD",
        "F": "F", "P": "P", "I": "I", "BR": "Br", "CL": "Cl",
        "ZN": "Zn", "FE": "Fe", "MG": "Mg", "MN": "Mn", "CA": "Ca",
    }

    lines_out = []
    with open(pdb_path, "r") as f:
        for line in f:
            record = line[:6].strip()
            if record in ("ATOM", "HETATM"):
                # Extract element from columns 77-78 or infer from atom name
                element = line[76:78].strip().upper() if len(line) >= 78 else ""
                if not element:
                    atom_name = line[12:16].strip()
                    element = atom_name.lstrip("0123456789")[0:1].upper() if atom_name else "C"

                # Determine AD atom type
                ad_type = AD_TYPE_MAP.get(element, element)

                # Refine hydrogen types: HD = H that can donate (bonded to N/O)
                # For simplicity, label all H as "HD" (most are polar in proteins)
                if element == "H":
                    ad_type = "HD"

                # Refine N types: NA = H-bond acceptor N (e.g., His ring N without H)
                # For simplicity, keep all N as "N" (Vina handles this)

                # Refine O types: OA = H-bond acceptor oxygen
                if element == "O":
                    ad_type = "OA"

                # Build PDBQT line: cols 1-54 from PDB, then occupancy+bfactor+charge+type
                pdb_prefix = line[:54].rstrip()
                # Pad prefix to exactly 54 characters
                pdb_prefix = pdb_prefix.ljust(54)

                # Occupancy and B-factor from original PDB (cols 55-66)
                occ_bfac = line[54:66] if len(line) >= 66 else "  1.00  0.00"
                if len(occ_bfac.strip()) == 0:
                    occ_bfac = "  1.00  0.00"

                charge = 0.000
                # PDBQT: cols 1-54 PDB, 55-60 occ, 61-66 bfac, 67-76 charge(10), 77 space, 78-79 type
                pdbqt_line = f"{pdb_prefix}{occ_bfac}{charge:10.3f} {ad_type:<2s}\n"
                lines_out.append(pdbqt_line)
            elif record in ("TER", "END"):
                lines_out.append(line)

    with open(output_path, "w") as f:
        f.writelines(lines_out)

    print(f"    Simple PDBQT saved: {output_path}")
    return output_path


def get_binding_site():
    """Return binding site box coordinates for docking."""
    return BINDING_SITE.copy()


def prepare_receptor(pdb_id="8E9W", verbose=True):
    """Full receptor preparation pipeline.

    Returns:
        dict with paths and binding site info
    """
    if verbose:
        print(f"\n  Preparing receptor from PDB {pdb_id}...")

    # Download
    raw_pdb = download_pdb(pdb_id)

    # Clean
    clean_pdb = clean_receptor(raw_pdb)

    # Fix (add missing atoms)
    fixed_pdb = fix_receptor(clean_pdb)

    # Convert to PDBQT
    pdbqt_path = receptor_to_pdbqt(fixed_pdb)

    # Get binding site
    binding_site = get_binding_site()

    result = {
        "pdb_id": pdb_id,
        "raw_pdb": raw_pdb,
        "clean_pdb": clean_pdb,
        "fixed_pdb": fixed_pdb,
        "pdbqt": pdbqt_path,
        "binding_site": binding_site,
    }

    if verbose:
        print(f"    Receptor preparation complete for {pdb_id}")
        print(f"    Binding site center: ({binding_site['center_x']}, "
              f"{binding_site['center_y']}, {binding_site['center_z']})")
        print(f"    Box size: {binding_site['size_x']}A cube")

    return result


def prepare_all_receptors(verbose=True):
    """Prepare both receptor structures."""
    results = {}
    for pdb_id in PDB_STRUCTURES:
        results[pdb_id] = prepare_receptor(pdb_id, verbose=verbose)
    return results
