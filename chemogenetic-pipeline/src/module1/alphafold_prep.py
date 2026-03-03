"""
Module 1: AlphaFold Preparation
Prepares input data for AlphaFold Server submission. Defines DREADD receptor
sequences and generates input JSON for the AlphaFold 3 web interface.

AlphaFold Server: https://alphafoldserver.com
Note: AlphaFold Server does not have a public API — inputs must be
submitted through the web interface. This module prepares the input
data and provides instructions for manual submission.
"""

import os
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
STRUCTURES_DIR = os.path.join(PROJECT_ROOT, "data", "structures")

# hM3Dq DREADD receptor — human muscarinic M3 receptor with two key mutations
# (Y149C and A239G) that make it responsive to designer drugs instead of acetylcholine.
# UniProt: P20309 (CHRM3_HUMAN), modified with DREADD mutations.
# Sequence below is the full-length hM3Dq (590 residues).
HM3DQ_SEQUENCE = (
    "MTLHNNSTTSPLFPNISSSWIHSPSDAGLPPGTVTHFGSYNVSRAAGNFSSPDGTTDDPL"
    "GGHTVWQVVFIAFLTGILALVTIIGNILVIVSFKVNKQLKTVNNYFLLSLACADLIIGVI"
    "SMNLFTTYIIMNRWALGNLACDLWLAIDCVASNASVMNLLVISFDRCFSITRPLTYRAKR"
    "TTKRAGVMIGLAWVISFVLWAPAILFWQYFVGKRTVPPGECFIQFLSEPTITFGTAIAGF"
    "YMPVTIMTILYWRIYKETEKRTKELAGLQASGTEAETENFVHPTGSSRSCSSYELQQQSM"
    "KRSNRRKYGRCHFWFTTKSWKPSSEQMDQDHSSSDSWNNNDAAASLENSASSDEEDIGSET"
    "RAIYSIVLKLPGHSTILNSTKLPSSDNLQVPEEELGMVDLERKADKLQAQKSVDDGGSF"
    "PKSFSKLPIQLESAVDTAKTSDVNSSVGKSTATLPLSFKEATLAKRFALKTRSQITKRKRM"
    "SLVKEKKAAQTLSAILLAFIITWTPYNIMVLVNTFCDSCIPKTFWNLGYWLCYINSTVNP"
    "VCYALCNKTFRTTFKMLLLCQCDKKKRRKQQYQQRQSVIFHKRAPEQAL"
)

# Compound SMILES for AlphaFold 3 ligand input
# AlphaFold 3 can predict protein-ligand complexes
COMPOUND_SMILES = {
    "CNO": "C[N+]1(CCN(CC1)C2=NC3=C(C=CC(=C3)Cl)NC4=CC=CC=C42)[O-]",
    "Clozapine": "CN1CCN(CC1)C2=NC3=C(C=CC(=C3)Cl)NC4=CC=CC=C42",
    "DCZ": "CN1CCN(CC1)C2=NC3=CC=CC=C3NC4=CC=CC=C42",
    "Compound 21": "CCN1C2=CC=CC=C2C(=C(C1=O)C=NC3=CC(=C(C=C3)O)C)O",
    "Olanzapine": "CC1=CC2=C(S1)NC3=CC=CC=C3N=C2N4CCN(CC4)C",
    "Perlapine": "CN1CCN(CC1)C2=NC3=CC=CC=C3CC4=CC=CC=C42",
}


def generate_alphafold_input(compound_name, output_dir=None):
    """Generate AlphaFold 3 Server input JSON for a receptor-ligand complex.

    Creates a JSON file compatible with AlphaFold 3 Server input format,
    containing the hM3Dq receptor sequence and the compound SMILES.

    Args:
        compound_name: Name of the compound (must be in COMPOUND_SMILES).
        output_dir: Output directory. Defaults to data/structures/.

    Returns:
        Path to the generated JSON file.

    Raises:
        ValueError: If compound_name is not recognized.
    """
    if compound_name not in COMPOUND_SMILES:
        raise ValueError(
            f"Unknown compound: {compound_name}. "
            f"Available: {list(COMPOUND_SMILES.keys())}"
        )

    if output_dir is None:
        output_dir = STRUCTURES_DIR
    os.makedirs(output_dir, exist_ok=True)

    # AlphaFold Server JSON upload only supports CCD-code ligands, not
    # arbitrary SMILES.  We include the protein chain here; the user must
    # add the ligand SMILES manually through the web UI after uploading.
    job = {
        "name": f"hM3Dq-{compound_name}",
        "modelSeeds": [],
        "sequences": [
            {
                "proteinChain": {
                    "sequence": HM3DQ_SEQUENCE,
                    "count": 1,
                },
            },
        ],
        "dialect": "alphafoldserver",
        "version": 1,
    }

    # Wrap in an array — the server requires a list of jobs.
    input_data = [job]

    safe_name = compound_name.replace(" ", "_")
    output_path = os.path.join(output_dir, f"alphafold_input_{safe_name}.json")
    with open(output_path, "w") as f:
        json.dump(input_data, f, indent=2)

    print(f"Generated AlphaFold input: {output_path}")
    return output_path


def generate_all_inputs(output_dir=None):
    """Generate AlphaFold input JSON for all 6 compounds.

    Returns:
        List of paths to generated JSON files.
    """
    paths = []
    for name in COMPOUND_SMILES:
        path = generate_alphafold_input(name, output_dir)
        paths.append(path)
    return paths


def check_alphafold_outputs(structures_dir=None):
    """Check which AlphaFold predictions are available.

    Scans the predicted/ directory for PDB/CIF files matching compound names.

    Args:
        structures_dir: Path to structures directory.

    Returns:
        Dict mapping compound name -> filepath (or None if missing).
    """
    if structures_dir is None:
        structures_dir = STRUCTURES_DIR

    predicted_dir = os.path.join(structures_dir, "predicted")
    results = {}

    for name in COMPOUND_SMILES:
        safe_name = name.replace(" ", "_")
        found = None
        for ext in [".pdb", ".cif"]:
            # Check common naming patterns in predicted/
            for pattern in [
                f"{safe_name}{ext}",
                f"hM3Dq-{safe_name}{ext}",
                f"hM3Dq_{safe_name}{ext}",
                f"fold_{safe_name}{ext}",
            ]:
                candidate = os.path.join(predicted_dir, pattern)
                if os.path.exists(candidate):
                    found = candidate
                    break
            if found:
                break

        # If not found in predicted/, check fold_* directories
        if found is None:
            safe_lower = safe_name.lower()
            fold_dir = os.path.join(structures_dir, f"fold_hm3dq_{safe_lower}")
            if os.path.isdir(fold_dir):
                model_0 = os.path.join(fold_dir, f"fold_hm3dq_{safe_lower}_model_0.cif")
                if os.path.exists(model_0):
                    found = model_0

        results[name] = found

    return results


def get_submission_instructions():
    """Return human-readable instructions for AlphaFold Server submission.

    Returns:
        str — formatted instructions.
    """
    smiles_table = "\n".join(
        f"     {name:15s}  {smi}"
        for name, smi in COMPOUND_SMILES.items()
    )
    return f"""
=== AlphaFold 3 Server Submission Instructions ===

Option A — Upload JSON then add ligand manually:
  1. Go to https://alphafoldserver.com and sign in.
  2. Click "Upload JSON" and upload one of the files in data/structures/.
     This loads the hM3Dq protein chain automatically.
  3. In the web UI, click "+ Add entity" → choose Ligand → paste the
     SMILES string for that compound (see table below).
  4. Click "Submit".

Option B — Enter everything manually:
  1. Click "New Job".
  2. Add Protein entity → paste the hM3Dq sequence (590 residues).
  3. Add Ligand entity → paste the compound SMILES.
  4. Name the job (e.g. "hM3Dq-DCZ") and submit.

Compound SMILES for copy-paste:
{smiles_table}

After results are ready:
  5. Download the output .cif / .pdb files.
  6. Place them in data/structures/predicted/ as <CompoundName>.pdb
     (e.g. DCZ.pdb, CNO.pdb, Clozapine.pdb).
  7. Re-run Module 1:  python3 -m src.module1.run_module1

Note: The server allows ~10 jobs/day.  Custom SMILES ligands are only
supported through the web UI, NOT via JSON upload (JSON only supports
a fixed set of CCD-code ligands like ATP, HEM, etc.).
"""
