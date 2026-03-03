# SPEC-09: Document MD Membrane Limitation

**Priority:** NICE TO HAVE
**Module:** 8 (Molecular Dynamics)
**Depends on:** Nothing

## Problem Statement

The hM3Dq receptor is a G protein-coupled receptor (GPCR), a membrane protein with 7 transmembrane helices that normally sit within a lipid bilayer. The current MD setup (`src/module8/system_setup.py`) solvates the protein in a water box but does NOT embed it in a lipid membrane. This means:

1. Transmembrane helices are exposed to water instead of lipid
2. This can cause unrealistic conformational changes (unfolding of TM domains)
3. The simulation may not accurately represent the in vivo binding environment

This is a known limitation that should be documented, not fixed at this time (membrane embedding is a major architectural change).

## Implementation Steps

### Step 1: Add limitation note to Module 8 orchestrator

**File:** `src/module8/run_module8.py`

Update the module docstring at the top of the file:

```python
"""
Module 8: Molecular Dynamics — Orchestrator
Setup → minimize → equilibrate → production → analyze → visualize.

KNOWN LIMITATION - No Lipid Membrane:
    The hM3Dq receptor is a GPCR (7-transmembrane membrane protein) but is
    simulated here in a water box without a lipid bilayer. Transmembrane
    helices are exposed to aqueous solvent, which may cause non-physiological
    conformational changes over long simulation timescales. For publication-
    quality GPCR simulations, the receptor should be embedded in a POPC/POPE
    lipid membrane using tools such as:
    - CHARMM-GUI Membrane Builder (https://www.charmm-gui.org/)
    - OpenMM Membrane Builder
    - packmol-memgen (AmberTools)

    The current water-only setup is appropriate for:
    - Comparing relative ligand binding stability across actuators
    - Short (< 100 ns) binding pocket dynamics
    - Initial screening of compound behavior

    It is NOT appropriate for:
    - Studying receptor activation/conformational changes
    - Free energy calculations requiring accurate membrane environment
    - Simulations > 500 ns where TM unfolding may occur
"""
```

### Step 2: Add runtime warning when simulation starts

**File:** `src/module8/system_setup.py`
**Function:** `build_system()`

Add a printed warning during system setup:

```python
if verbose:
    print("    NOTE: Receptor is solvated in water only (no lipid membrane).")
    print("    This is a known limitation for GPCR simulations.")
```

### Step 3: Add limitation flag to analysis output

**File:** `src/module8/trajectory_analysis.py`

In the analysis results dictionary, include a `limitations` field:

```python
results["limitations"] = ["water_only_no_membrane"]
```

This allows downstream code (dashboard, reports) to display appropriate caveats.

## Verification Checklist

- [ ] Module 8 docstring contains the membrane limitation note
- [ ] Runtime warning is printed during system setup
- [ ] Analysis results include limitations field

## Files Changed

- **Modified:** `src/module8/run_module8.py` — update docstring
- **Modified:** `src/module8/system_setup.py` — add runtime warning in `build_system()`
- **Modified:** `src/module8/trajectory_analysis.py` — add limitations field to results
