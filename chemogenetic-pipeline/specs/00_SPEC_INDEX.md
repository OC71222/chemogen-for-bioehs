# Implementation Specs Index

All specs derived from PIPELINE_AUDIT.md dated 2026-02-25.

## Verified Environment State (as of spec creation)

| Package | Status |
|---------|--------|
| rdkit | 2025.09.5 |
| biopython | 1.86 |
| openmm | 8.4 |
| MDAnalysis | 2.10.0 |
| meeko | 0.7.1 |
| pdbfixer | installed |
| chembl_webresource_client | installed (dev) |
| scikit-learn | 1.8.0 |
| joblib | 1.5.3 |
| **vina** | **NOT INSTALLED** |
| **PyTDC (tdc)** | **NOT INSTALLED** |
| **openmmforcefields** | **NOT INSTALLED** |

## Spec List (Priority Order)

### Critical — Pipeline produces fake results without these

| # | Spec | Module | What it fixes |
|---|------|--------|---------------|
| 01 | [SPEC-01: Install Vina + Fix Docking](./SPEC-01_install_vina_fix_docking.md) | 4 | All docking scores are fake heuristics |
| 02 | [SPEC-02: Install PyTDC + Retrain ADMET Models](./SPEC-02_install_pytdc_retrain_admet.md) | 6 | hERG/CYP2D6/HIA/clearance models trained on garbage |
| 03 | [SPEC-03: Fix AlphaFold File Paths + pLDDT](./SPEC-03_fix_alphafold_paths.md) | 1 | Code can't find real AlphaFold CIFs, pLDDT scores are fake |
| 04 | [SPEC-04: Download Experimental Structures + RMSD](./SPEC-04_download_experimental_rmsd.md) | 1 | No RMSD validation against cryo-EM ground truth |

### Important — Incomplete results without these

| # | Spec | Module | What it fixes |
|---|------|--------|---------------|
| 05 | [SPEC-05: Install openmmforcefields + MD for remaining compounds](./SPEC-05_install_openmmforcefields_md.md) | 8 | Only DCZ has MD data, 5 compounds missing |
| 06 | [SPEC-06: Complete Selectivity Models (Module 7)](./SPEC-06_complete_selectivity.md) | 7 | Selectivity models never trained, no output CSV |
| 07 | [SPEC-07: Replace Fake Compound Library with ChEMBL](./SPEC-07_replace_fake_library.md) | 5 | Screening library is 151 fake string-manipulated compounds |

### Nice to Have — Scientific rigor improvements

| # | Spec | Module | What it fixes |
|---|------|--------|---------------|
| 08 | [SPEC-08: Auto-detect Docking Box from 8E9W Ligand](./SPEC-08_autodetect_docking_box.md) | 4 | Binding site coordinates may be wrong |
| 09 | [SPEC-09: Document MD Membrane Limitation](./SPEC-09_document_membrane_limitation.md) | 8 | GPCR simulated in water-only (no lipid bilayer) |
| 10 | [SPEC-10: Retrain BBB on Larger Dataset](./SPEC-10_retrain_bbb.md) | 6 | BBB model AUROC=0.5 on scaffold split |

## Execution Order

Specs should be executed in numerical order. Dependencies:
- SPEC-01 must complete before SPEC-07 (screening needs real docking)
- SPEC-01 must complete before SPEC-08 (box coordinates for docking)
- SPEC-02 must complete before SPEC-10 (BBB retraining builds on ADMET infra)
- SPEC-03 must complete before SPEC-04 (need real structures for RMSD)
- SPEC-05 depends on SPEC-01 (MD needs docked poses from Vina)
