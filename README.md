# AI-Accelerated Chemogenetic Actuator Design

A computational pipeline for designing next-generation chemogenetic actuators (DREADDs), using AlphaFold 3 for structure prediction alongside molecular docking, ADMET modeling, and molecular dynamics simulations.

Built for the 13th Annual Bioengineering High School Competition (BIOEHSC 2026) at UC Berkeley.

## What We Did with AlphaFold

### The Problem

DREADDs (Designer Receptors Exclusively Activated by Designer Drugs) are engineered GPCRs used in neuroscience to remotely control neural activity. The hM3Dq receptor is widely used, but its current actuator ligands (CNO, DCZ, etc.) have known pharmacological limitations — off-target binding, poor blood-brain barrier penetration, or back-metabolism into active compounds like clozapine.

We used **AlphaFold 3** to predict protein-ligand complex structures for hM3Dq bound to six different actuator compounds, then fed those structures into a multi-stage computational pipeline to evaluate and rank the ligands.

### AlphaFold 3 Server Workflow

We submitted jobs to [AlphaFold Server](https://alphafoldserver.com) for each receptor-ligand pair:

| Compound | Role | AlphaFold Job |
|---|---|---|
| CNO | Baseline actuator | `fold_hm3dq_cno` |
| Clozapine | CNO metabolite | `fold_hm3dq_clozapine` |
| DCZ | Preferred actuator | `fold_hm3dq_dcz` |
| Compound 21 | Alternative actuator | `fold_hm3dq_compound_21` |
| Olanzapine | Repurposed antipsychotic | `fold_hm3dq_olanzapine` |
| Perlapine | Experimental actuator | `fold_hm3dq_perlapine` |

Each job predicted the hM3Dq DREADD receptor (590 residues, based on human muscarinic M3 with Y149C/A239G mutations) complexed with a small-molecule ligand. AlphaFold 3 generated 5 ranked models per job (model_0 through model_4) with confidence metrics.

**Input preparation:** `src/module1/alphafold_prep.py` generates the JSON input files for the AlphaFold Server web interface. Since the server doesn't support arbitrary SMILES via JSON upload, the protein chain is uploaded via JSON and the ligand SMILES is added manually through the UI.

### What We Learned

**Structure confidence (pLDDT & pTM scores):**
- AlphaFold produced reasonable receptor folds with pTM scores around 0.6 and ranking scores around 0.84
- ~48% of residues were predicted as disordered (expected for GPCR intracellular loops and termini)
- Per-residue pLDDT analysis (`src/module1/plddt_analysis.py`) identified high-confidence transmembrane helices vs. low-confidence loop regions

**Validation against experimental cryo-EM structures:**
- We compared predictions to published cryo-EM structures (PDB: 8E9W, 8E9Y, 7WC7, 7WC8)
- **CNO vs 8E9Y:** RMSD = 2.18 Å (279 aligned C-alpha atoms) — Good
- **DCZ vs 8E9W:** RMSD = 2.35 Å (280 aligned C-alpha atoms) — Good
- The 7WC7/7WC8 comparisons showed high RMSD (~19-22 Å) due to fusion-construct numbering differences in those cryo-EM structures, not actual prediction failures

**Key takeaway:** AlphaFold 3 predictions for the hM3Dq-ligand complexes were structurally consistent with experimental data where comparable structures existed (~2.2 Å RMSD), making them suitable starting points for downstream docking and MD simulations.

## Pipeline Overview

The predicted structures feed into a multi-module pipeline:

1. **Module 1 — Structure Prediction** (AlphaFold 3): Predict hM3Dq-ligand complexes, extract pLDDT scores, compute RMSD vs experimental structures
2. **Module 2 — Actuator Evaluation** (RDKit): Compute physicochemical properties, Lipinski violations, drug-likeness scores
3. **Module 4 — Molecular Docking** (AutoDock Vina): Dock ligands into the receptor binding pocket, score binding affinities
4. **Module 5 — Virtual Screening**: Screen compound libraries (ZINC) for novel actuator candidates
5. **Module 6 — ADMET Prediction**: Predict BBB permeability, clearance, hERG liability, CYP2D6 inhibition
6. **Module 7 — Selectivity Profiling**: Predict off-target binding at 5-HT2A, D2, H1, and muscarinic M1-M5 receptors
7. **Module 8 — Molecular Dynamics** (OpenMM): Run MD simulations to assess binding stability (RMSD, RMSF, H-bonds)

A **Streamlit dashboard** (`app.py`) ties it all together with interactive visualizations.

## Repository Structure

```
chemogenetic-pipeline/
├── src/module1/          # AlphaFold prep, pLDDT analysis, RMSD calculation
├── src/module2/          # Compound properties and actuator evaluation
├── src/module4/          # Molecular docking (AutoDock Vina)
├── src/module5/          # Virtual screening
├── src/module6/          # ADMET prediction models
├── src/module7/          # Selectivity profiling (ChEMBL data)
├── src/module8/          # Molecular dynamics (OpenMM)
├── data/structures/      # AlphaFold inputs, outputs, and experimental PDBs
├── data/results/         # Pipeline output CSVs and pLDDT scores
├── data/docking/         # Receptor/ligand files and docking poses
├── figures/              # Generated plots (pLDDT, docking, ADMET, MD)
├── models/               # Trained ADMET and selectivity ML models
├── tabs/                 # Streamlit dashboard tabs
├── app.py                # Streamlit entry point
└── run_pipeline.py       # Run all modules end-to-end
```

## Quick Start

```bash
pip install -r chemogenetic-pipeline/requirements.txt

# Generate AlphaFold input files
python3 -m src.module1.run_module1 --prep

# Run full pipeline (demo mode if no AlphaFold outputs yet)
python3 -m src.module1.run_module1 --demo

# Launch dashboard
streamlit run chemogenetic-pipeline/app.py
```

## Data Sources

- **Receptor sequence:** hM3Dq DREADD (UniProt P20309 / CHRM3_HUMAN + Y149C, A239G mutations)
- **Experimental structures:** PDB 8E9W, 8E9Y (hM3Dq cryo-EM), 7WC6, 7WC7, 7WC8
- **Compound data:** PubChem CIDs for CNO, Clozapine, DCZ, Compound 21, Olanzapine, Perlapine
- **Selectivity data:** ChEMBL binding assays for off-target receptors
- **Screening library:** ZINC database (filtered subset)
