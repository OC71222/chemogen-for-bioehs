# SPEC-08: Auto-detect Docking Box from 8E9W Co-crystallized Ligand

**Priority:** NICE TO HAVE
**Module:** 4 (Molecular Docking)
**Depends on:** SPEC-01 (Vina installed and working)

## Problem Statement

The binding site coordinates in `src/module4/receptor_prep.py` are hardcoded:
```python
BINDING_SITE = {
    "center_x": -10.0, "center_y": 5.0, "center_z": -15.0,
    "size_x": 28.0, "size_y": 28.0, "size_z": 28.0,
}
```

These may or may not match the actual orthosteric binding pocket of hM3Dq in PDB 8E9W. If wrong, Vina will dock into the wrong region and produce meaningless scores even when running for real.

## Current State

- PDB 8E9W is already downloaded at `data/docking/receptor/8E9W.pdb`
- 8E9W contains a co-crystallized DCZ ligand — its coordinates define the true binding site
- The receptor is cleaned by `clean_receptor()` which removes HETATM records (including the ligand) before docking

## Implementation Steps

### Step 1: Extract ligand coordinates from 8E9W

**File:** `src/module4/receptor_prep.py`
**New function:** `detect_binding_site(pdb_path, ligand_resname=None)`

```python
def detect_binding_site(pdb_path, ligand_resname=None, padding=5.0):
    """Auto-detect binding site from co-crystallized ligand in PDB structure.

    Args:
        pdb_path: Path to the raw (uncleaned) PDB file
        ligand_resname: 3-letter residue name of the ligand (e.g., "DCZ").
                        If None, uses any non-standard residue that isn't HOH/common ions.
        padding: Extra space (Angstroms) around ligand bounding box

    Returns:
        Dict with center_x, center_y, center_z, size_x, size_y, size_z
    """
    from Bio.PDB import PDBParser
    import numpy as np

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("receptor", pdb_path)

    # Common non-ligand HETATM residues to ignore
    ignore = {"HOH", "WAT", "NA", "CL", "MG", "ZN", "CA", "K", "MN",
              "FE", "SO4", "PO4", "GOL", "EDO", "ACE", "NMA", "DMS"}

    ligand_atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                resname = residue.get_resname().strip()
                het_flag = residue.get_id()[0]

                # HETATM residues (het_flag starts with 'H_' or 'W')
                if het_flag.startswith("H_") or (ligand_resname and resname == ligand_resname):
                    if resname in ignore:
                        continue
                    if ligand_resname and resname != ligand_resname:
                        continue
                    for atom in residue:
                        ligand_atoms.append(atom.get_vector().get_array())

    if not ligand_atoms:
        print(f"    WARNING: No ligand atoms found in {pdb_path}")
        print(f"    Falling back to hardcoded binding site")
        return BINDING_SITE.copy()

    coords = np.array(ligand_atoms)
    center = coords.mean(axis=0)
    box_min = coords.min(axis=0)
    box_max = coords.max(axis=0)
    box_size = (box_max - box_min) + 2 * padding

    # Ensure minimum box size of 20 Å
    box_size = np.maximum(box_size, 20.0)

    return {
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "center_z": float(center[2]),
        "size_x": float(box_size[0]),
        "size_y": float(box_size[1]),
        "size_z": float(box_size[2]),
    }
```

### Step 2: Integrate into `prepare_receptor()`

**File:** `src/module4/receptor_prep.py`
**Function:** `prepare_receptor(pdb_id="8E9W", verbose=True)`

After downloading and before cleaning the receptor, detect the binding site from the raw PDB:

```python
# In prepare_receptor(), after download_pdb() but before clean_receptor():
raw_pdb_path = download_pdb(pdb_id)

# Auto-detect binding site from co-crystallized ligand
ligand_resname = PDB_STRUCTURES.get(pdb_id, {}).get("ligand")
detected_site = detect_binding_site(raw_pdb_path, ligand_resname=ligand_resname)

if verbose:
    print(f"    Detected binding site center: "
          f"({detected_site['center_x']:.1f}, {detected_site['center_y']:.1f}, {detected_site['center_z']:.1f})")
    print(f"    Box size: "
          f"({detected_site['size_x']:.1f}, {detected_site['size_y']:.1f}, {detected_site['size_z']:.1f})")

    # Compare with hardcoded values
    dx = abs(detected_site['center_x'] - BINDING_SITE['center_x'])
    dy = abs(detected_site['center_y'] - BINDING_SITE['center_y'])
    dz = abs(detected_site['center_z'] - BINDING_SITE['center_z'])
    dist = (dx**2 + dy**2 + dz**2)**0.5
    print(f"    Distance from hardcoded center: {dist:.1f} Å")
    if dist > 5.0:
        print(f"    WARNING: Detected center differs significantly from hardcoded values!")

# Use detected site instead of hardcoded
result["binding_site"] = detected_site
```

### Step 3: Update `get_binding_site()` to use detection

```python
def get_binding_site(pdb_path=None, ligand_resname=None):
    """Get binding site coordinates.

    If pdb_path provided, auto-detects from co-crystallized ligand.
    Otherwise returns hardcoded fallback.
    """
    if pdb_path and os.path.exists(pdb_path):
        return detect_binding_site(pdb_path, ligand_resname)
    return BINDING_SITE.copy()
```

### Step 4: Verify detected vs hardcoded coordinates

Run comparison without changing any docking results yet:

```python
from src.module4.receptor_prep import detect_binding_site, BINDING_SITE

detected = detect_binding_site("data/docking/receptor/8E9W.pdb")
print(f"Detected:  center=({detected['center_x']:.1f}, {detected['center_y']:.1f}, {detected['center_z']:.1f})")
print(f"           size=({detected['size_x']:.1f}, {detected['size_y']:.1f}, {detected['size_z']:.1f})")
print(f"Hardcoded: center=({BINDING_SITE['center_x']:.1f}, {BINDING_SITE['center_y']:.1f}, {BINDING_SITE['center_z']:.1f})")
print(f"           size=({BINDING_SITE['size_x']:.1f}, {BINDING_SITE['size_y']:.1f}, {BINDING_SITE['size_z']:.1f})")
```

If the centers are within 3-5 Å of each other, the hardcoded values are reasonable and both should produce similar docking results.

### Step 5: Re-run docking if coordinates differ significantly

If the detected center is >5 Å from the hardcoded center:
```bash
python3 -m src.module4.run_module4
```

## Verification Checklist

- [ ] `detect_binding_site()` returns coordinates derived from the co-crystallized ligand
- [ ] Detected center is compared to hardcoded center with distance printed
- [ ] If coordinates changed significantly, docking is re-run with correct box
- [ ] Docking results are physically reasonable (DCZ should have good affinity in 8E9W)

## Files Changed

- **Modified:** `src/module4/receptor_prep.py` — add `detect_binding_site()`, update `prepare_receptor()` and `get_binding_site()`
- **Possibly regenerated:** `data/results/docking_results.csv` if coordinates changed
