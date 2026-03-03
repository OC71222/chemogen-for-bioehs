# SPEC-04: Download Experimental Structures + RMSD Validation

**Priority:** CRITICAL
**Module:** 1 (Structural Analysis)
**Depends on:** SPEC-03 (real AlphaFold CIFs must be in `predicted/`)

## Problem Statement

The `data/structures/experimental/` directory is empty. The RMSD comparison code (`src/module1/rmsd_calculator.py`) needs cryo-EM structures from RCSB PDB to compute Kabsch RMSD between AlphaFold predictions and experimental ground truth. Without these, there is no validation of structural prediction quality.

## Current State

- `data/structures/experimental/` — **empty directory**
- `data/results/rmsd_results.csv` — **does not exist** (never generated)
- `rmsd_calculator.py` is fully functional — uses BioPython Superimposer for Kabsch alignment on C-alpha atoms
- The code expects `.pdb` files in `experimental/`

### Relevant PDB structures for hM3Dq DREADD

These are cryo-EM structures of the hM3Dq DREADD receptor:
- **7WC6** — hM3Dq DREADD bound to CNO (3.3 Å resolution)
- **7WC7** — hM3Dq DREADD bound to DCZ (3.2 Å resolution)
- **7WC8** — hM3Dq DREADD bound to Compound 21 (3.4 Å resolution)

Source: RCSB PDB (https://www.rcsb.org)

## Implementation Steps

### Step 1: Download experimental structures from RCSB PDB

```bash
cd data/structures/experimental/
curl -O https://files.rcsb.org/download/7WC6.pdb
curl -O https://files.rcsb.org/download/7WC7.pdb
curl -O https://files.rcsb.org/download/7WC8.pdb
cd ../../..
```

**Verification:**
```bash
wc -l data/structures/experimental/*.pdb
```

Each file should be several thousand lines. If downloads fail, check network and try:
```bash
wget https://files.rcsb.org/download/7WC6.pdb -O data/structures/experimental/7WC6.pdb
```

### Step 2: Understand the RMSD comparison pairs

The AlphaFold predictions model the hM3Dq receptor complexed with each ligand. The experimental structures are:
- 7WC6 → CNO complex → compare against `predicted/CNO.cif`
- 7WC7 → DCZ complex → compare against `predicted/DCZ.cif`
- 7WC8 → Compound 21 complex → compare against `predicted/Compound_21.cif`

Note: There are no experimental structures for Clozapine, Olanzapine, or Perlapine complexes with hM3Dq. RMSD can only be computed for 3 of the 6 compounds.

### Step 3: Verify `calculate_rmsd_from_files()` handles CIF + PDB mixed input

**File:** `src/module1/rmsd_calculator.py`
**Function:** `calculate_rmsd_from_files(ref_path, pred_path)`

The reference files are `.pdb` and the predicted files are now `.cif`. The function uses `src/module1/structure_parser.parse_structure()` which should handle both formats via BioPython. Verify:

```python
from src.module1.structure_parser import parse_structure
# Test PDB parsing
exp = parse_structure("data/structures/experimental/7WC7.pdb")
print(f"Experimental 7WC7: {exp}")

# Test CIF parsing
pred = parse_structure("data/structures/predicted/DCZ.cif")
print(f"Predicted DCZ: {pred}")
```

If CIF parsing fails, BioPython's MMCIFParser may need to be used instead of PDBParser. Check that `parse_structure()` dispatches by file extension.

### Step 4: Handle chain ID differences

Cryo-EM structures from RCSB may have different chain IDs than AlphaFold output. The RMSD code aligns on C-alpha atoms, but if chain selection is involved, the chain IDs must match or be mapped.

**Check:**
```python
from Bio.PDB import PDBParser, MMCIFParser

# Experimental
parser = PDBParser(QUIET=True)
exp = parser.get_structure("7WC7", "data/structures/experimental/7WC7.pdb")
print("Experimental chains:", [c.id for c in exp[0].get_chains()])

# Predicted
cif_parser = MMCIFParser(QUIET=True)
pred = cif_parser.get_structure("DCZ", "data/structures/predicted/DCZ.cif")
print("Predicted chains:", [c.id for c in pred[0].get_chains()])
```

If chains differ, the RMSD code may need to be told which chains to compare, or it should extract all CA atoms regardless of chain.

### Step 5: Define RMSD comparison pairs

In `run_module1.py`, the RMSD pairs need to be defined. Check how `run_analysis()` constructs the comparison list. It should be:

```python
rmsd_pairs = [
    {"ref_path": "data/structures/experimental/7WC6.pdb",
     "pred_path": "data/structures/predicted/CNO.cif",
     "name": "CNO vs 7WC6"},
    {"ref_path": "data/structures/experimental/7WC7.pdb",
     "pred_path": "data/structures/predicted/DCZ.cif",
     "name": "DCZ vs 7WC7"},
    {"ref_path": "data/structures/experimental/7WC8.pdb",
     "pred_path": "data/structures/predicted/Compound_21.cif",
     "name": "Compound_21 vs 7WC8"},
]
```

If `run_module1.py` doesn't automatically detect these pairs, the code needs to be updated to:
1. Scan `experimental/` for PDB files
2. Match them to predicted structures by compound name
3. A mapping dict may be needed: `{"7WC6": "CNO", "7WC7": "DCZ", "7WC8": "Compound_21"}`

### Step 6: Re-run Module 1 (RMSD portion)

```bash
python3 -m src.module1.run_module1
```

Or if you only want the RMSD calculation (not re-running pLDDT extraction):
```python
from src.module1.rmsd_calculator import calculate_rmsd_from_files, batch_rmsd, save_rmsd_results

pairs = [
    {"ref_path": "data/structures/experimental/7WC6.pdb",
     "pred_path": "data/structures/predicted/CNO.cif",
     "name": "CNO vs 7WC6"},
    {"ref_path": "data/structures/experimental/7WC7.pdb",
     "pred_path": "data/structures/predicted/DCZ.cif",
     "name": "DCZ vs 7WC7"},
    {"ref_path": "data/structures/experimental/7WC8.pdb",
     "pred_path": "data/structures/predicted/Compound_21.cif",
     "name": "Compound_21 vs 7WC8"},
]
results = batch_rmsd(pairs)
save_rmsd_results(results)
print(results)
```

## Verification Checklist

- [ ] `data/structures/experimental/` contains `7WC6.pdb`, `7WC7.pdb`, `7WC8.pdb`
- [ ] Each PDB file is >1000 lines (real cryo-EM structures)
- [ ] `data/results/rmsd_results.csv` exists with 3 rows
- [ ] RMSD values are physically reasonable:
  - GPCR structures typically have RMSD 1-4 Å between cryo-EM and predicted
  - Values <1 Å = excellent, 1-2 Å = good, 2-3 Å = acceptable, >3 Å = poor
- [ ] All rows have `aligned=True`

## Expected RMSD Ranges

For a well-predicted GPCR structure vs cryo-EM:
- **Transmembrane core (TM helices):** 1.0-2.5 Å
- **Full structure including loops:** 2.0-4.0 Å
- **If loop regions are disordered:** up to 5-6 Å

Very high RMSD (>6 Å) would indicate a parsing/alignment problem, not a prediction failure.

## Files Changed

- **Added:** `data/structures/experimental/7WC6.pdb`, `7WC7.pdb`, `7WC8.pdb`
- **Generated:** `data/results/rmsd_results.csv`
- **Possibly modified:** `src/module1/run_module1.py` — if RMSD pair matching logic needs PDB-to-compound mapping
- **Possibly modified:** `src/module1/structure_parser.py` — if CIF parsing dispatch is missing
