# AI-Accelerated Design of Next-Generation Chemogenetic Actuators

## BIOEHSC 2026 — Complete Research & Execution Master Document

**Competition:** 13th Annual Bioengineering High School Competition, UC Berkeley
**Date:** April 4, 2026
**Deliverables:** Research poster (30"×40"), 12-min academic presentation, 2-min industry pitch, midpoint video
**Timeline:** 7 weeks total (~5 weeks remaining as of Feb 24, 2026)

---

# TABLE OF CONTENTS

1. [Project Overview](#1-project-overview)
2. [Background: Chemogenetics Fundamentals](#2-background-chemogenetics-fundamentals)
3. [The Problem: Why Current Actuators Fail](#3-the-problem-why-current-actuators-fail)
4. [The Solution: AI-Powered Pipeline](#4-the-solution-ai-powered-pipeline)
5. [Deep Research Area 1: Structural Biology of DREADDs](#5-deep-research-area-1-structural-biology-of-dreadds)
6. [Deep Research Area 2: Ligand/Actuator Optimization with AI](#6-deep-research-area-2-ligandactuator-optimization-with-ai)
7. [Deep Research Area 3: Neural Data Analysis](#7-deep-research-area-3-neural-data-analysis)
8. [Feasibility & Hardware Analysis](#8-feasibility--hardware-analysis)
9. [Execution Plan: Week-by-Week](#9-execution-plan-week-by-week)
10. [Module 1: Structure Prediction (AlphaFold 3)](#10-module-1-structure-prediction-alphafold-3)
11. [Module 2: Actuator Evaluation (RDKit + SwissADME)](#11-module-2-actuator-evaluation-rdkit--swissadme)
12. [Module 3: Streamlit Dashboard](#12-module-3-streamlit-dashboard)
13. [Competition Deliverables Strategy](#13-competition-deliverables-strategy)
14. [Complete Reference Library](#14-complete-reference-library)
15. [Tools & Resources Inventory](#15-tools--resources-inventory)
16. [Key Sequences, IDs, and Data](#16-key-sequences-ids-and-data)
17. [Judging Preparation: Anticipated Q&A](#17-judging-preparation-anticipated-qa)

---

# 1. PROJECT OVERVIEW

## Problem Statement

Chemogenetics — the use of engineered receptors (DREADDs) to remotely control specific neurons with designer drugs — is one of the most promising tools for understanding and eventually treating neurological and psychiatric disorders. Over 600 million people worldwide suffer from conditions like Parkinson's disease, epilepsy, chronic pain, and treatment-resistant depression, many of which could benefit from precise, circuit-level neural control.

However, a critical bottleneck is blocking clinical translation: **the actuator drugs used to activate DREADDs are fundamentally flawed.**

- **CNO (clozapine-N-oxide)**, the original DREADD actuator, does not cross the blood-brain barrier on its own. Instead, it converts in vivo to **clozapine** — an antipsychotic that binds dopamine D2, serotonin 5-HT2A, histamine H1, and muscarinic receptors, causing serious off-target side effects (Gomez et al., *Science*, 2017).
- **DCZ (deschloroclozapine)** is more selective and potent, but still has limitations — it crosses the BBB indiscriminately, making it unsuitable for peripheral-only applications.
- **No actuator exists** that is simultaneously highly selective for DREADDs, has controllable BBB permeability, is non-toxic, and is orally bioavailable.

**The root cause:** Designing better actuators currently requires solving the receptor's atomic structure via cryo-EM (months to years), screening millions of compounds in a lab (months), and testing candidates in animals (months). The HCAD project (Roth & Scherrer labs, Cell, 2024) took **seven years** to complete for a single new chemogenetic system.

## Solution Statement

We propose an **AI-powered computational pipeline** that compresses the chemogenetic actuator design cycle from years to days by integrating three modules:

- **Module 1 — Rapid Receptor Structure Prediction:** AlphaFold 3 (Google DeepMind) to predict DREADD 3D structures from amino acid sequence, validated against published cryo-EM experimental structures.
- **Module 2 — Automated Actuator Drug Screening:** RDKit and SwissADME to evaluate candidate actuator molecules across selectivity, BBB permeability, drug-likeness, and safety.
- **Module 3 — Integrated Analysis Dashboard:** A working Streamlit web application chaining Modules 1 and 2 into a single researcher-facing interface.

## Specific Aims

**Aim 1:** Validate AlphaFold 3 for DREADD structure prediction by comparing predicted structures of hM4Di and hM3Dq against experimental cryo-EM structures (PDB: 7WC6, 7WC7, 7WC8), quantifying accuracy via backbone and binding pocket RMSD.

**Aim 2:** Build a computational actuator evaluation platform characterizing 6 known DREADD actuators (CNO, clozapine, DCZ, Compound 21, olanzapine, perlapine) across molecular descriptors, BBB prediction, and Lipinski compliance.

**Aim 3:** Integrate Modules 1 and 2 into a functional Streamlit prototype demonstrating end-to-end workflow: sequence → structure → drug evaluation → ranking.

## What We Are NOT Claiming

- We are **not** claiming to have designed a novel actuator drug — we are building the computational infrastructure to enable faster design.
- We are **not** claiming AlphaFold replaces experimental validation — we are demonstrating it as a rapid first-pass tool.
- We are **not** doing wet lab experiments — this is a computational/bioengineering project.
- We **are** claiming that our pipeline, if extended with generative AI models (future work), could dramatically accelerate the actuator design cycle.

---

# 2. BACKGROUND: CHEMOGENETICS FUNDAMENTALS

## 2.1 What Is Chemogenetics?

Chemogenetics is a method for remotely controlling specific cells (usually neurons) using engineered protein receptors that respond only to synthetic drugs. The most widely used chemogenetic system is **DREADDs** — Designer Receptors Exclusively Activated by Designer Drugs.

**Core concept:** A genetically engineered GPCR (G protein-coupled receptor) is inserted into target neurons via viral vectors (AAV). This receptor is "deaf" to the body's natural signaling molecules but responds to a specific synthetic compound (the "actuator" drug). When the actuator is administered (orally, IP injection, etc.), it activates only the cells expressing the DREADD, allowing researchers to turn specific neural circuits on or off.

## 2.2 How DREADDs Were Created

DREADDs were first developed by **Bryan L. Roth** at UNC Chapel Hill in 2005–2007, published in *PNAS* (Armbruster et al., 2007).

**Engineering method:** Starting from human muscarinic acetylcholine receptors (mAChRs), Roth's team used directed evolution in yeast to identify two key point mutations:

- **Y3.33C** (tyrosine → cysteine at Ballesteros-Weinstein position 3.33)
- **A5.46G** (alanine → glycine at position 5.46)

These mutations reshape the orthosteric binding pocket: the receptor loses its ability to bind acetylcholine (the natural ligand) and gains the ability to bind CNO (the synthetic actuator) at nanomolar concentrations. The positions Y3.33 and A5.46 are conserved across all muscarinic receptor subtypes (M1–M5), so the same mutations create DREADDs from any subtype.

## 2.3 Types of DREADDs

| DREADD | Base Receptor | G-protein Coupling | Effect on Neurons | Primary Use |
|--------|--------------|-------------------|-------------------|-------------|
| **hM3Dq** | Human M3 (hM3) | Gq | **Excitatory** — activates PLC/IP3/Ca²⁺ pathway, depolarizes neurons | Enhance neuronal firing |
| **hM4Di** | Human M4 (hM4) | Gi/o | **Inhibitory** — activates GIRK channels, hyperpolarizes neurons; also inhibits neurotransmitter release | Silence neurons |
| **hM1Dq** | Human M1 | Gq | Excitatory | Less commonly used |
| **hM5Dq** | Human M5 | Gq | Excitatory | Less commonly used |
| **rM3Ds** | Rat M3 chimera | Gs | **Stimulatory** — activates cAMP/PKA pathway | Study Gs signaling |
| **KORD** | Kappa-opioid receptor | Gi | Inhibitory (activated by SalB, not CNO) | Multiplexed experiments with hM3Dq |

**Most widely used:** hM3Dq (excitatory) and hM4Di (inhibitory).

## 2.4 Actuator Drugs (Ligands)

| Actuator | Status | Key Issue |
|----------|--------|-----------|
| **CNO** (clozapine-N-oxide) | Original, widely used | Converts to clozapine in vivo → off-target effects at D2, 5-HT2A, H1 receptors (Gomez et al., Science, 2017) |
| **Clozapine** | Metabolite of CNO | Active antipsychotic — causes sedation, metabolic effects, agranulocytosis at high doses |
| **DCZ** (deschloroclozapine) | Preferred current actuator | More selective and potent than CNO; crosses BBB readily; validated in mice and monkeys (Nagai et al., Nature Neuroscience, 2020) |
| **Compound 21 (C21)** | Alternative to CNO | Binds DREADDs but has lower potency; some off-target activity at serotonin receptors |
| **Olanzapine** | Repurposed antipsychotic | Low-dose DREADD activation possible but still has significant off-target effects |
| **Perlapine** | Sedative/hypnotic | Can activate DREADDs; limited characterization as actuator |
| **FCH-2296413** | Newest (HCAD system, Cell 2024) | Peripherally restricted — does NOT cross BBB; activates HCAD only |

## 2.5 How DREADDs Are Delivered

1. **Viral vector construction:** The DREADD gene is packaged into an adeno-associated virus (AAV) with a cell-type-specific promoter (e.g., CaMKIIα for excitatory neurons, GFAP for astrocytes).
2. **Stereotaxic injection:** The AAV is injected into the target brain region. Over 2–4 weeks, the virus infects neurons and they begin expressing the DREADD on their surface.
3. **Cre-lox targeting:** For cell-type specificity, Cre-dependent DREADD constructs (DIO/FLEX) are injected into transgenic Cre-driver mice — only cells expressing Cre recombinase will express the DREADD.
4. **Drug administration:** The actuator is given systemically (IP injection, oral gavage, or drinking water). It crosses the BBB (for CNS applications) or remains peripheral, binds the DREADD, and activates the engineered signaling pathway.

## 2.6 Applications of Chemogenetics

### Neuroscience Research
- **Neural circuit mapping:** Identify which circuits drive specific behaviors (feeding, fear, reward, sleep)
- **Psychiatric disorder modeling:** Depression (silencing habenula), anxiety (modulating amygdala), addiction (nucleus accumbens circuits), OCD (corticostriatal pathways)
- **Memory and cognition:** Hippocampal DREADDs can enhance or impair memory consolidation

### Preclinical Therapeutics
- **Parkinson's disease:** Gs-DREADD (rM3Ds) in striatal neurons rescues motor symptoms in dopamine-depleted mice
- **Epilepsy:** hM4Di in seizure-focus neurons suppresses seizure activity
- **Chronic pain:** HCAD system (Cell, 2024) — peripherally restricted DREADD in dorsal root ganglion nociceptors reduces inflammatory pain without CNS side effects
- **Obesity/metabolism:** DREADD modulation of hypothalamic feeding circuits

### Next-Generation Platforms
- **PSAMs (Pharmacologically Selective Actuator Modules):** Ligand-gated ion channels engineered for chemogenetic control — faster kinetics than GPCRs
- **PAGERs (Programmable Antigen-gated Engineered Receptors):** Published in *Nature* (2025) — receptors activated by extracellular antigens rather than small molecules, enabling cell-type-specific activation without genetic targeting

---

# 3. THE PROBLEM: WHY CURRENT ACTUATORS FAIL

## 3.1 The CNO Problem (Gomez et al., Science, 2017)

This landmark paper proved that CNO itself does not cross the blood-brain barrier. Instead, it is metabolized in vivo into **clozapine**, which is the actual compound reaching the brain and activating DREADDs.

**Implications:**
- All CNO-based DREADD experiments actually used clozapine as the actuator
- Clozapine binds dozens of native receptors (D1–D4, 5-HT2A/2C, H1, α1-adrenergic, M1–M5)
- This creates confounds: observed effects may be due to clozapine's off-target activity, not DREADD activation
- Thousands of published DREADD studies potentially affected

## 3.2 Why Designing Better Actuators Takes Years

**Case study: HCAD (Cell, 2024) — 7 years from start to publication**

The Roth and Scherrer labs spent seven years developing the HCAD peripherally restricted chemogenetic system. Here's where the time went:

### Step 1: Receptor Selection & Mutagenesis (~1–2 years)
- Identified HCA2 (hydroxycarboxylic acid receptor 2) as a viable GPCR scaffold — minimally expressed in the brain, primarily in peripheral immune cells
- Engineered mutations by trial and error: cloning, cell-based assays (BRET, TANGO), iterating through dozens of mutants
- Each round of mutagenesis → expression → functional assay takes weeks

### Step 2: Actuator Discovery via Ultra-Large Library Screening (~1–2 years)
- Screened Enamine REAL library (~37 billion virtual compounds) using computational docking
- Hits physically synthesized by Enamine (weeks per batch), shipped, tested in cell assays
- Multiple rounds of "screen → synthesize → test → redesign"

### Step 3: Cryo-EM Structure Determination (~6–12 months)
- Protein expression and purification (weeks)
- Forming stable receptor–G protein complexes (weeks of optimization)
- Grid preparation and screening (days to weeks)
- Data collection on a $5–10M cryo-EM microscope (days of beam time)
- Computational reconstruction and model building (weeks to months)
- GPCR complexes are particularly challenging due to their small size (~35–45 kDa inactive, ~130 kDa with G protein) and conformational flexibility

### Step 4: Animal Model Validation (~1–2 years)
- AAV virus production and quality control
- Injection into mice, waiting 3–4 weeks for expression
- Behavioral testing (von Frey filaments, hot plate, Hargreaves test, open field, rotarod)
- Pharmacokinetic studies: BBB penetration, microsomal stability, plasma half-life
- Off-target profiling across hundreds of receptors

### Step 5: Iteration
- If any step fails (wrong mutations, actuator doesn't cross/block BBB, off-target effects in animals), the process loops back
- Multiple loops = multiple years added

---

# 4. THE SOLUTION: AI-POWERED PIPELINE

## 4.1 What AI Replaces in the Pipeline

| Traditional Step | Time | Cost | AI Replacement | Time | Cost |
|-----------------|------|------|---------------|------|------|
| Cryo-EM structure determination | 6–12 months | $100K–$1M (microscope time, personnel) | AlphaFold 3 Server | ~5 min/prediction | Free |
| Wet lab drug property screening | Months | $50K–$500K | RDKit + SwissADME | Seconds | Free |
| Animal BBB permeability testing | Months | $100K+ | Computational TPSA/LogP prediction | Instant | Free |
| Manual cross-referencing | Weeks | Personnel time | Streamlit dashboard | Automated | Free |
| **Total** | **~7 years** | **$1M+** | **AI Pipeline** | **Days to weeks** | **$0 in compute** |

## 4.2 What AI Does NOT Replace

- **Experimental wet lab validation** — AI predictions must still be confirmed
- **Animal safety testing** — required for any compound heading toward clinical use
- **Clinical trials** — no shortcut for human testing
- **Novelty in receptor engineering** — AI can predict known scaffolds, but true protein design requires experimental iteration

**Our claim is honest:** We are not replacing the full drug development pipeline. We are compressing the *computational front end* — the part that identifies which structures to solve, which compounds to test, and which properties to optimize — from years to days.

## 4.3 Pipeline Architecture

```
INPUT: DREADD amino acid sequence (from UniProt, with mutations applied)
        ↓
MODULE 1: AlphaFold 3 Server
        → Predicted 3D structure (.cif file)
        → Confidence scores (pLDDT per residue, PAE)
        → Validated against experimental cryo-EM structures (PDB)
        → Output: RMSD comparison, binding pocket analysis
        ↓
MODULE 2: Actuator Evaluation Engine (RDKit + SwissADME)
        → Input: SMILES strings for candidate actuators
        → Calculate: MW, LogP, TPSA, HBD, HBA, rotatable bonds
        → Predict: BBB permeability, Lipinski compliance, drug-likeness
        → Visualize: Chemical space plots, property radar charts
        → Output: Ranked compound table with safety flags
        ↓
MODULE 3: Integration Dashboard (Streamlit)
        → 3D structure viewer (py3Dmol)
        → Side-by-side compound comparison
        → Interactive chemical space explorer
        → SMILES input for evaluating new candidate compounds
        → Pipeline visualization showing data flow
        ↓
OUTPUT: Ranked list of actuator candidates with predicted properties
        → Ready for experimental validation
```

---

# 5. DEEP RESEARCH AREA 1: STRUCTURAL BIOLOGY OF DREADDs

## 5.1 Experimental Structures — The Cryo-EM Breakthrough

**Key paper: Shao et al., "Molecular basis for selective activation of DREADD-based chemogenetics," *Nature*, 612, 354–362 (2022).**

This paper from Bryan Roth's lab (first author: Shicheng Zhang) solved **four high-resolution cryo-EM structures** of DREADD receptors:

| Structure | Complex | Ligand | PDB Code | Resolution |
|-----------|---------|--------|----------|------------|
| hM3Dq–miniGq | Excitatory DREADD + G protein | DCZ | **7WC7** | ~2.7 Å |
| hM4Di–miniGo | Inhibitory DREADD + G protein | DCZ | **7WC6** | ~2.6 Å |
| hM3Dq–miniGq | Excitatory DREADD + G protein | CNO | **7WC8** | ~2.8 Å |
| hM3R–miniGq | Wild-type M3 + G protein | Iperoxo | (control) | ~2.8 Å |

**Key structural findings:**
- The Y149C (hM3Dq) and Y113C (hM4Di) mutations remove a bulky tyrosine from the binding pocket, creating space for CNO/DCZ
- The A239G (hM3Dq) and A203G (hM4Di) mutations remove a methyl group, subtly reshaping the pocket
- Together, these two mutations shift ligand preference from acetylcholine (bulky, charged) to CNO/DCZ (planar, uncharged)
- DCZ makes extensive hydrophobic contacts with residues in TM3, TM5, TM6, and TM7

## 5.2 AlphaFold for GPCR Structure Prediction

### AlphaFold 2 (2021)
- Predicts protein backbone structure with remarkable accuracy
- Limitations for GPCRs: tends to favor inactive conformations, struggles with binding pocket side-chain orientations
- A study in *Int. J. Mol. Sci.* (2024) found that AF2 models of Class A GPCRs showed comparable docking results to experimental structures for certain receptors, but with notable limitations for novel or understudied targets

### AlphaFold 3 (May 2024)
- Major advance: predicts **protein–ligand complexes** (protein + DNA + RNA + ligands + ions) using diffusion-based architecture
- 50%+ improvement over previous methods for protein–ligand interaction prediction
- Published in *Nature* (Abramson et al., 2024)
- Limitation: a study on GPR139 reported "limited capability" for understudied GPCRs — prediction quality depends on available homologous structures in training data
- For DREADDs specifically: strong advantage because muscarinic receptors are well-characterized (many experimental structures available)

### AlphaFold Server (alphafoldserver.com)
- Free web interface for non-commercial academic use
- Runs on Google DeepMind's A100 GPU clusters — no local hardware needed
- Up to 20 jobs per day, 5,000 tokens per job
- A single GPCR (~350 amino acids) = ~350 tokens — well within limits
- Can model protein–ligand complexes (protein + small molecule)
- Outputs: predicted structure (.cif), confidence metrics (pLDDT, PAE), per-residue confidence

## 5.3 AI in Cryo-EM Processing (Context for Presentation)

Modern cryo-EM relies heavily on AI at every step:

| Step | AI Tool | What It Does |
|------|---------|-------------|
| Particle picking | Topaz, crYOLO, CryoTransformer | Identifies protein particles in noisy micrographs (F1 scores: 0.65–0.85) |
| 3D reconstruction | CryoDRGN-AI (*Nature Methods*, 2025) | Ab initio reconstruction from raw images, handles conformational heterogeneity |
| Model building | DeepTracer, ModelAngelo | Automated tracing of protein backbone into density maps |
| Refinement | AlphaFold-guided refinement | Uses AF predictions as starting models for cryo-EM fitting |

## 5.4 GPCR-Specific AI Tools

- **GPCRdb:** Database of GPCR structures including state-specific AlphaFold models, residue numbering schemes
- **GPCR-BERT:** Protein language model pre-trained on GPCR sequences for predicting function and ligand binding
- **GaMD+GLOW:** Gaussian accelerated molecular dynamics combined with generative models — produces dynamic free energy landscapes for 44 GPCR systems, capturing conformational dynamics that static structures miss

---

# 6. DEEP RESEARCH AREA 2: LIGAND/ACTUATOR OPTIMIZATION WITH AI

## 6.1 Case Study: HCAD Structure-Guided Design (Cell, 2024)

The Roth lab's HCAD paper is the gold standard for what our pipeline aims to accelerate:

1. **Selected HCA2 receptor** as scaffold (minimally expressed in brain)
2. **Engineered mutations** to create mHCAD — loses response to native ligand (niacin), gains response to synthetic compounds
3. **Screened Enamine REAL library** (~37 billion virtual compounds) using computational docking against the designed binding pocket
4. **Solved cryo-EM structure** of mHCAD–Gαi1–Gβ1γ2 complex bound to FCH-2296413 (resolution: 2.62 Å)
5. **Validated** FCH-2296413 as peripherally restricted: excellent drug-like properties, does NOT cross BBB, clean off-target profile
6. **Demonstrated** pain reduction in mouse models of acute and inflammatory pain

**Key computational steps we can partially replicate:**
- Structure prediction (AlphaFold → replacing cryo-EM for initial modeling)
- Property calculation (RDKit → molecular descriptors)
- BBB prediction (TPSA/LogP rules → replacing animal PK studies for initial screening)

## 6.2 Deep Reinforcement Learning for BBB Optimization

**Key paper: Pereira et al., "Optimizing blood–brain barrier permeation through deep reinforcement learning for de novo drug design," *Bioinformatics*, 37, i84–i92 (2021).**

**Architecture:**
- Generator: RNN with two LSTM layers (256 units each), learns molecular syntax via SMILES notation
- Training: Teacher Forcing on ChEMBL database to learn valid molecular structures
- Reinforcement Learning: Policy-based RL with multi-objective reward function
- Reward components: (1) binding affinity to target receptor, (2) BBB permeability prediction
- Exploration/exploitation strategy balances novelty vs. known drug-like space

**Results:**
- ~85% of generated molecules had both desired binding affinity AND BBB permeability
- Directly applicable to DREADD actuator design: optimize for DREADD binding + controllable BBB penetration

**Relevance to our project:** This demonstrates the "future work" direction — our current pipeline evaluates existing compounds, but a generative RL model could design entirely new ones. This is a strong talking point for judges.

## 6.3 BBB Prediction Methods

**Rule-based (what we implement):**
- **Lipinski's Rule of Five:** MW < 500, LogP < 5, HBD ≤ 5, HBA ≤ 10
- **BBB-specific rules:** MW < 450, LogP 1–3, TPSA < 90 Å², HBD ≤ 3
- **Key descriptor: TPSA (Topological Polar Surface Area)** — lower TPSA = better BBB penetration. Threshold ~90 Å² separates BBB-penetrant from non-penetrant compounds

**ML-based (context for presentation):**
- XGBoost, GNNs, DNNs trained on molecular fingerprints
- Merged molecular representations (SMILES + molecular graphs) achieve highest accuracy
- Comprehensive review: *Molecular Informatics* (2025)

**SwissADME BOILED-Egg model:**
- Plots compounds in LogP vs. TPSA space
- White region = GI absorption predicted; yellow region = BBB penetrant
- Free web tool at swissadme.ch — visual output ideal for poster

## 6.4 Generative AI for Drug Design (Context)

Recent generative models relevant to actuator design:

| Model | Publication | What It Does |
|-------|-----------|-------------|
| **DRAGONFLY** | *Nature Communications*, 2025 | Zero-shot interactome-based drug generation — designs molecules for target proteins without training on that specific target |
| **DeepBlock** | *Nature Computational Science*, 2024 | Toxicity-controlled molecule generation — constrains outputs to avoid toxic substructures |
| **FRAME** | *ACS Central Science*, 2024 | SE(3)-equivariant geometric deep learning for 3D-aware molecule generation |
| **Multi-target generators** | *Int. J. Mol. Sci.*, 2025 | Generate molecules with target embeddings and activity profiles — design for selectivity |

## 6.5 Ultra-Large Virtual Screening

The GPR139 case study demonstrates the power of computational screening at scale:
- 235 million compound virtual screen against GPR139
- Identified 5 nanomolar agonists from 68 tested hits
- Structure-guided optimization improved potency further
- Confirmed binding mode via cryo-EM
- This is exactly the approach HCAD used with Enamine REAL (37 billion compounds)

---

# 7. DEEP RESEARCH AREA 3: NEURAL DATA ANALYSIS

*Note: This area provides context for why better actuators matter and how DREADD effects are measured. It is NOT a module we build, but is important background for the presentation and Q&A.*

## 7.1 LFADS — Latent Factor Analysis via Dynamical Systems

**Key paper: Pandarinath et al., *Nature Methods*, 2018.**

**Architecture:**
- Variational autoencoder (VAE) + Recurrent Neural Network (RNN)
- **Encoder:** Compresses multi-neuron spiking activity → low-dimensional initial conditions
- **Generator:** Produces smoothed dynamics from initial conditions via GRU cells
- **Controller:** Infers external perturbations ("inferred inputs") at each timestep

**Why it matters for chemogenetics:**
- LFADS explicitly models "inferred inputs" — external perturbations to the neural system
- DREADD activation IS an external perturbation — LFADS can detect and characterize the effect of DREADD activation on neural dynamics at single-trial resolution
- Cursor-jump task performance: decoded hand velocities with R² ≈ 0.9 from just 25 neurons

## 7.2 RADICaL — For Calcium Imaging

**Key paper: Zhu et al., *Nature Neuroscience*, 2022.**
- Extends LFADS for calcium imaging data
- Accounts for nonlinear dynamics of GCaMP calcium indicators
- Achieves sub-frame temporal resolution
- Outperformed all previous neural analysis methods
- Critical for modern chemogenetics experiments that use two-photon calcium imaging

## 7.3 CaImAn — Open-Source Preprocessing

**Key paper: Giovannucci et al., *eLife*, 2019.**
- Complete pipeline: motion correction → source extraction → spike deconvolution → quality control
- Handles thousands of neurons over hours of recording
- Standard preprocessing step before LFADS/RADICaL analysis

## 7.4 Real Integration Example

**Zhao et al., *Science Advances*, 2024:**
- Gi-DREADD expressed in microglia + two-photon calcium imaging + computational analysis
- Quantified how DREADD activation inhibited microglial dynamics, reduced neuronal activity, impaired synchronization
- Demonstrates the full experimental pipeline our computational tools aim to support

## 7.5 MICrONS Dataset (Context)

- 75,000 neurons calcium imaging + 200,000 cells / 500 million synapses electron microscopy reconstruction
- AI-reconstructed wiring maps published in *Nature* (2025)
- Enables precision DREADD targeting: if you know the wiring diagram, you can design DREADDs to target specific circuit nodes

---

# 8. FEASIBILITY & HARDWARE ANALYSIS

## 8.1 Why the Traditional Approach Takes 7 Years

Detailed timeline breakdown for the HCAD project (Cell, 2024):

| Phase | Duration | Key Bottleneck | Hardware/Resources |
|-------|----------|---------------|-------------------|
| Receptor selection & mutagenesis | 1–2 years | Trial-and-error protein engineering, each cycle takes weeks | Molecular biology lab, cell culture facility |
| Ultra-large library screening | 1–2 years | 37 billion compounds; synthesis + shipping + testing per batch | HPC cluster for docking, Enamine synthesis, wet lab assays |
| Cryo-EM structure determination | 6–12 months | Sample prep optimization, microscope access ($5–10M instrument) | Cryo-EM facility (Titan Krios or Glacios), GPU cluster for reconstruction |
| Animal model validation | 1–2 years | AAV production, behavioral testing cohorts, PK studies | Vivarium, behavioral testing rigs, LC-MS for PK |
| Iteration (failures → restart) | Variable | Any failed step means looping back | All of the above, repeated |
| **Total** | **~7 years** | | **>$1M in total costs** |

## 8.2 Can Our Pipeline Run Without a Supercomputer?

### Module 1: AlphaFold 3 — NO Local Hardware Needed

**Running AlphaFold 3 locally requires:**
- NVIDIA A100 80GB or H100 80GB GPU (~$10,000–$30,000)
- 64GB+ system RAM
- Up to 1TB SSD for genetic databases
- Linux operating system
- The original AF3 paper used 16 NVIDIA A100s (40GB each)

**But we use AlphaFold Server (alphafoldserver.com):**
- Free web interface — Google's data center handles all computation
- We paste in a protein sequence, add ligand, click "predict"
- Google's A100 cluster does the work; results return in minutes
- Limit: ~20 jobs/day, 5,000 tokens/job
- We need ~6–10 jobs total — easily within free tier
- **Hardware needed: a web browser**

### Module 2: RDKit + SwissADME — Runs on Any Laptop

**RDKit performance:**
- Molecular property calculation (MW, LogP, TPSA, etc.) from SMILES: milliseconds per molecule
- RDKit benchmark: loading 699 drug-like molecules from SD file takes ~10 seconds on a standard laptop
- We are analyzing 6 compounds — essentially instant
- RDKit runs on Windows, Mac, or Linux via pip/conda

**SwissADME:**
- Free web tool (swissadme.ch), Swiss Institute of Bioinformatics
- Paste SMILES → full pharmacokinetic profile + BOILED-Egg BBB plot
- Zero local computation
- **Hardware needed: Python + web browser**

### Module 3: Streamlit Dashboard — Runs on Any Laptop

- Streamlit is a lightweight Python web framework
- BioPython RMSD calculation: loads two PDB files, computes backbone atom distances — sub-second
- Py3Dmol: JavaScript/WebGL molecular viewer, renders in browser
- **Hardware needed: same laptop running Python**

## 8.3 Summary: Hardware Requirements

| Component | Traditional Approach | Our AI Pipeline | What You Need |
|-----------|---------------------|----------------|--------------|
| Structure determination | $5–10M cryo-EM microscope | AlphaFold Server (free) | Web browser |
| Drug screening | Wet lab + HPC cluster | RDKit (free, local) | Any laptop with Python |
| BBB prediction | Animal PK studies | SwissADME (free web tool) + TPSA rules | Web browser |
| Structure validation | Additional cryo-EM | BioPython RMSD | Any laptop |
| Dashboard | N/A | Streamlit | Any laptop |
| **Total compute cost** | **>$1M** | **$0** | **Standard laptop + internet** |

## 8.4 Key Talking Point for Judges

*"Every tool in our pipeline is either a free cloud service or an open-source library that runs on a standard laptop. AlphaFold Server runs on Google DeepMind's infrastructure. RDKit processes molecules in milliseconds. SwissADME is a free web service. The entire pipeline costs zero dollars in compute. We're showing that the AI revolution in structural biology has democratized drug design — a researcher without access to a cryo-EM facility or supercomputer can now do meaningful structure-guided actuator screening."*

---

# 9. EXECUTION PLAN: WEEK-BY-WEEK

## Overview

| Week | Dates (Approx.) | Focus | Deliverable |
|------|-----------------|-------|-------------|
| 3 | Feb 24 – Mar 2 | AlphaFold predictions + PDB downloads + RMSD | Predicted structures + experimental comparisons |
| 4 | Mar 3 – Mar 9 | Actuator evaluation code (RDKit + SwissADME) | Property tables + chemical space plots |
| 5 | Mar 10 – Mar 16 | Streamlit dashboard + integration | Working prototype |
| 6 | Mar 17 – Mar 23 | Poster + presentation + video | All competition materials |
| 7 | Mar 24 – Apr 3 | Polish + rehearse + Q&A prep | Competition-ready |

---

# 10. MODULE 1: STRUCTURE PREDICTION (AlphaFold 3)

## 10.1 Step-by-Step Protocol

### Get Wild-Type Sequences from UniProt

| Receptor | UniProt ID | Gene Name | Length | URL |
|----------|-----------|-----------|--------|-----|
| Human M4 (base for hM4Di) | **P08173** | CHRM4 | 479 aa | uniprot.org/uniprotkb/P08173 |
| Human M3 (base for hM3Dq) | **P20309** | CHRM3 | 590 aa | uniprot.org/uniprotkb/P20309 |

### Apply DREADD Mutations

**hM4Di (from M4, P08173):**
- Y113C (Y3.33C → Ballesteros-Weinstein numbering)
- A203G (A5.46G)

**hM3Dq (from M3, P20309):**
- Y149C (Y3.33C)
- A239G (A5.46G)

### AlphaFold Server Jobs to Submit

| Job # | Input | Purpose |
|-------|-------|---------|
| 1 | Wild-type M4 sequence | Baseline — how well does AF3 predict a known GPCR? |
| 2 | Wild-type M3 sequence | Baseline |
| 3 | hM4Di sequence (Y113C + A203G) | Core prediction — does AF3 capture mutation effects? |
| 4 | hM3Dq sequence (Y149C + A239G) | Core prediction |
| 5 | hM4Di + DCZ ligand | Protein–ligand complex prediction |
| 6 | hM3Dq + DCZ ligand | Protein–ligand complex prediction |
| 7 | hM3Dq + CNO ligand | Compare to PDB 7WC8 |

### Download Experimental Structures for Comparison

| PDB Code | Complex | Ligand | Resolution | Download |
|----------|---------|--------|------------|----------|
| **7WC6** | hM4Di–miniGo | DCZ | ~2.6 Å | rcsb.org/structure/7WC6 |
| **7WC7** | hM3Dq–miniGq | DCZ | ~2.7 Å | rcsb.org/structure/7WC7 |
| **7WC8** | hM3Dq–miniGq | CNO | ~2.8 Å | rcsb.org/structure/7WC8 |

## 10.2 Validation: RMSD Calculation

**Using BioPython's Superimposer:**

```python
from Bio.PDB import PDBParser, Superimposer
import numpy as np

def calculate_rmsd(structure1_path, structure2_path, chain_id='A'):
    parser = PDBParser(QUIET=True)
    s1 = parser.get_structure('predicted', structure1_path)
    s2 = parser.get_structure('experimental', structure2_path)

    # Extract CA atoms (backbone)
    atoms1 = [atom for atom in s1[0][chain_id].get_atoms() if atom.get_name() == 'CA']
    atoms2 = [atom for atom in s2[0][chain_id].get_atoms() if atom.get_name() == 'CA']

    # Ensure same number of atoms (may need alignment first)
    min_len = min(len(atoms1), len(atoms2))
    atoms1 = atoms1[:min_len]
    atoms2 = atoms2[:min_len]

    sup = Superimposer()
    sup.set_atoms(atoms1, atoms2)
    sup.apply(s1[0].get_atoms())

    return sup.rms  # RMSD in Angstroms
```

**Expected results (based on literature):**
- Backbone RMSD: likely 1.5–3.0 Å (good for GPCRs)
- Binding pocket RMSD: may be higher (3–5 Å) due to side-chain flexibility
- These numbers are your data — plot them as bar charts for the poster

## 10.3 Confidence Analysis

AlphaFold outputs **pLDDT** (predicted Local Distance Difference Test) per residue:
- > 90: Very high confidence (backbone well-predicted)
- 70–90: Confident
- 50–70: Low confidence (flexible loops, disordered regions)
- < 50: Very low confidence

**What to look for:**
- Transmembrane helices should have high pLDDT (> 80)
- Extracellular loops may have lower pLDDT (flexible regions)
- **Binding pocket residues** (positions 3.33, 5.46, and surrounding) — check if pLDDT is high enough to trust the pocket prediction

---

# 11. MODULE 2: ACTUATOR EVALUATION (RDKit + SwissADME)

## 11.1 Target Compounds

| Compound | PubChem CID | SMILES | Role |
|----------|------------|--------|------|
| **CNO** (clozapine-N-oxide) | 135398508 | Retrieve from PubChem | Original actuator (problematic) |
| **Clozapine** | 2818 | Retrieve from PubChem | Metabolite of CNO (off-target) |
| **DCZ** (deschloroclozapine) | 44601286 | Retrieve from PubChem | Current preferred actuator |
| **Compound 21** | 135445020 | Retrieve from PubChem | Alternative actuator |
| **Olanzapine** | 4585 | Retrieve from PubChem | Repurposed antipsychotic |
| **Perlapine** | 4748 | Retrieve from PubChem | Sedative with DREADD activity |

## 11.2 Properties to Calculate

**With RDKit:**
- Molecular Weight (MW)
- LogP (Crippen method)
- TPSA (Topological Polar Surface Area)
- Number of H-bond donors (HBD)
- Number of H-bond acceptors (HBA)
- Number of rotatable bonds
- Number of aromatic rings
- Fraction Csp3

**BBB Prediction Rules (from literature):**
- MW < 450 Da
- LogP between 1 and 3
- TPSA < 90 Å²
- HBD ≤ 3
- Predicted BBB: "Penetrant" if ALL criteria met, "Non-penetrant" if any fail

**Lipinski Rule of Five compliance:**
- MW < 500, LogP < 5, HBD ≤ 5, HBA ≤ 10
- Violations count

## 11.3 Code Template

```python
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
import pandas as pd

def evaluate_compound(name, smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    rotatable = Descriptors.NumRotatableBonds(mol)
    aromatic_rings = Descriptors.NumAromaticRings(mol)
    fsp3 = Descriptors.FractionCSP3(mol)

    # BBB prediction
    bbb_mw = mw < 450
    bbb_logp = 1 <= logp <= 3
    bbb_tpsa = tpsa < 90
    bbb_hbd = hbd <= 3
    bbb_penetrant = all([bbb_mw, bbb_logp, bbb_tpsa, bbb_hbd])

    # Lipinski violations
    lipinski_violations = sum([
        mw >= 500,
        logp >= 5,
        hbd > 5,
        hba > 10
    ])

    return {
        'Name': name,
        'MW': round(mw, 1),
        'LogP': round(logp, 2),
        'TPSA': round(tpsa, 1),
        'HBD': hbd,
        'HBA': hba,
        'Rotatable Bonds': rotatable,
        'Aromatic Rings': aromatic_rings,
        'Fsp3': round(fsp3, 2),
        'BBB Predicted': 'Penetrant' if bbb_penetrant else 'Non-penetrant',
        'Lipinski Violations': lipinski_violations
    }
```

## 11.4 SwissADME Integration

For each compound:
1. Go to swissadme.ch
2. Paste SMILES string
3. Click "Run"
4. Screenshot the BOILED-Egg plot (BBB permeability visualization)
5. Export physicochemical properties for cross-validation against RDKit results

## 11.5 Visualization: Chemical Space Plot

Plot all 6 compounds in LogP vs. TPSA space with BBB boundary lines:

```python
import matplotlib.pyplot as plt

def plot_chemical_space(df):
    fig, ax = plt.subplots(figsize=(10, 8))

    # BBB boundary
    ax.axhline(y=90, color='red', linestyle='--', label='TPSA = 90 (BBB threshold)')
    ax.axvline(x=1, color='blue', linestyle=':', alpha=0.5)
    ax.axvline(x=3, color='blue', linestyle=':', alpha=0.5, label='LogP 1–3 (BBB optimal)')

    # Shade BBB-penetrant region
    ax.fill_between([1, 3], 0, 90, alpha=0.1, color='green', label='BBB-penetrant zone')

    # Plot compounds
    for _, row in df.iterrows():
        color = 'green' if row['BBB Predicted'] == 'Penetrant' else 'red'
        ax.scatter(row['LogP'], row['TPSA'], s=row['MW'], c=color,
                  edgecolors='black', alpha=0.7)
        ax.annotate(row['Name'], (row['LogP'], row['TPSA']),
                   textcoords="offset points", xytext=(5, 5), fontsize=9)

    ax.set_xlabel('LogP (Lipophilicity)', fontsize=12)
    ax.set_ylabel('TPSA (Å²)', fontsize=12)
    ax.set_title('Chemical Space: DREADD Actuators', fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig('chemical_space.png', dpi=300)
```

---

# 12. MODULE 3: STREAMLIT DASHBOARD

## 12.1 Dashboard Layout

```
┌──────────────────────────────────────────────────────┐
│                    HEADER / TITLE                     │
│  "AI-Accelerated Chemogenetic Actuator Design"       │
├──────────────┬───────────────────────────────────────┤
│  SIDEBAR     │  MAIN AREA                            │
│              │                                        │
│  Navigation: │  Tab 1: Structure Viewer               │
│  • Structure │    - 3D model (py3Dmol)               │
│  • Actuators │    - RMSD comparison table             │
│  • Analysis  │    - pLDDT confidence plot             │
│  • Pipeline  │                                        │
│              │  Tab 2: Actuator Comparison             │
│  Settings:   │    - Property table                    │
│  • DREADD    │    - Radar chart per compound          │
│    type      │    - Chemical space plot               │
│  • Ligand    │                                        │
│              │  Tab 3: Evaluate New Compound           │
│              │    - SMILES input box                  │
│              │    - Instant property calculation       │
│              │    - Compare to known actuators         │
│              │                                        │
│              │  Tab 4: Pipeline Overview               │
│              │    - Flow diagram                      │
│              │    - Methodology explanation            │
└──────────────┴───────────────────────────────────────┘
```

## 12.2 Key Libraries

```
streamlit
rdkit
biopython
py3Dmol
stmol (Streamlit component for py3Dmol)
pandas
matplotlib
plotly
```

## 12.3 3D Structure Viewer (py3Dmol in Streamlit)

```python
import streamlit as st
import py3Dmol
from stmol import showmol

def show_structure(pdb_file, style='cartoon'):
    with open(pdb_file) as f:
        pdb_data = f.read()

    view = py3Dmol.view(width=700, height=500)
    view.addModel(pdb_data, 'pdb')

    if style == 'cartoon':
        view.setStyle({'cartoon': {'color': 'spectrum'}})
    elif style == 'surface':
        view.addSurface(py3Dmol.VDW, {'opacity': 0.7})

    view.zoomTo()
    showmol(view, height=500, width=700)
```

---

# 13. COMPETITION DELIVERABLES STRATEGY

## 13.1 Research Poster (30" × 40" Landscape)

**Layout:**

```
┌────────────────────────────────────────────────────────────────┐
│                         TITLE BAR                              │
│  AI-Accelerated Design of Next-Generation Chemogenetic         │
│  Actuators                                                     │
│  [Your Name] | BIOEHSC 2026 | UC Berkeley                     │
├──────────┬──────────┬──────────┬──────────┬────────────────────┤
│          │          │          │          │                    │
│ PROBLEM  │ APPROACH │ MODULE 1 │ MODULE 2 │    RESULTS &       │
│          │          │          │          │    CONCLUSIONS      │
│ • CNO→   │ • 3-mod  │ • AF3    │ • RDKit  │                    │
│   cloz   │   AI     │   preds  │   props  │ • RMSD figure      │
│ • 7 yr   │   pipe   │ • RMSD   │ • BBB    │ • Chemical space   │
│   cycle  │ • Free   │   bar    │   pred   │ • Property table   │
│ • Off-   │   tools  │   chart  │ • Chem   │ • Pipeline diagram │
│   target │ • Zero   │ • pLDDT  │   space  │ • QR code → demo   │
│          │   cost   │   plot   │   plot   │                    │
│          │          │          │          │ FUTURE WORK         │
│ [brain   │ [pipe-   │ [struct  │ [BOILED  │ • Generative AI    │
│  fig]    │  line    │  fig]    │  Egg]    │ • Wet lab valid.   │
│          │  dia]    │          │          │ • Clinical path    │
├──────────┴──────────┴──────────┴──────────┴────────────────────┤
│                    REFERENCES (small font)                      │
└────────────────────────────────────────────────────────────────┘
```

**Key figures to include:**
1. DREADD mechanism schematic (receptor → G protein → effect)
2. Pipeline architecture diagram
3. AlphaFold predicted vs. experimental structure overlay
4. RMSD bar chart (backbone and binding pocket)
5. Chemical space plot (LogP vs. TPSA)
6. Compound property comparison table
7. SwissADME BOILED-Egg plot
8. QR code linking to GitHub repository / live Streamlit app

## 13.2 Academic Presentation (12 Minutes)

**Structure:**

| Time | Section | Content |
|------|---------|---------|
| 0:00–1:30 | Hook + Problem | "600 million people suffer from neurological disorders... chemogenetics offers precise control... but the drugs are broken" |
| 1:30–3:00 | Background | What are DREADDs, CNO problem, 7-year design cycle |
| 3:00–4:30 | Our Solution | Three-module AI pipeline overview |
| 4:30–6:30 | Module 1 Results | AlphaFold predictions, RMSD data, confidence analysis |
| 6:30–8:30 | Module 2 Results | Actuator properties, BBB predictions, chemical space |
| 8:30–9:30 | Module 3 Demo | Live Streamlit dashboard walkthrough |
| 9:30–10:30 | Validation | Cross-check with published experimental data |
| 10:30–11:30 | Future Work | Generative AI, wet lab validation, clinical path |
| 11:30–12:00 | Conclusion | Impact statement, democratizing drug design |

## 13.3 Industry Pitch (2 Minutes)

**Angle: Untapped $2B+ market opportunity**

"The chemogenetics market is projected to exceed $2 billion as the technology moves toward clinical trials. Pharma giants — Sanofi, Roche, Eli Lilly — are spending billions on AI drug discovery. But no one has built an AI platform specifically for engineered receptor–drug co-design.

Our pipeline demonstrates that AlphaFold + computational chemistry can compress the actuator design cycle from 7 years to days, using only free tools and a standard laptop. This is the Isomorphic Labs approach applied to a niche that Isomorphic Labs itself hasn't touched.

The market opportunity: license the platform to academic labs ($500/year SaaS), partner with pharma for proprietary actuator development, and expand to all engineered receptor systems — not just DREADDs, but PSAMs, PAGERs, and future chemogenetic platforms."

## 13.4 Midpoint Video

- 60–120 seconds
- Show: problem statement, pipeline concept, preliminary AlphaFold output
- Tone: professional but enthusiastic
- End with: "Our pipeline will be a working prototype by competition day"

---

# 14. COMPLETE REFERENCE LIBRARY

## Core Papers (Must-Cite)

1. **Gomez et al.** "Chemogenetics revealed: DREADD occupancy and activation via converted clozapine." *Science* 357, 503–507 (2017). — *The CNO problem*
2. **Shao/Zhang et al.** "Molecular basis for selective activation of DREADD-based chemogenetics." *Nature* 612, 354–362 (2022). — *Cryo-EM DREADD structures (PDB: 7WC6, 7WC7, 7WC8)*
3. **Nagai et al.** "Deschloroclozapine, a potent and selective chemogenetic actuator." *Nature Neuroscience* 23, 1157–1167 (2020). — *DCZ as improved actuator*
4. **Hu/Kang et al.** "Structure-guided design of a peripherally restricted chemogenetic system." *Cell* 187, 7265–7281 (2024). — *HCAD system, 7-year project*
5. **Abramson et al.** "Accurate structure prediction of biomolecular interactions with AlphaFold 3." *Nature* 630, 493–500 (2024). — *AlphaFold 3*
6. **Armbruster et al.** "Evolving the lock to fit the key to create a family of GPCRs potently activated by an inert ligand." *PNAS* 104, 5163–5168 (2007). — *Original DREADD invention*

## Supporting Papers

7. "Advances in chemogenetics: DREADDs in psychiatric disorders." *Molecular Psychiatry* 30, 2297–2308 (2025).
8. "AI meets physics in computational structure-based drug discovery for GPCRs." *npj Drug Discovery* (2025).
9. "Reliability of AlphaFold2 Models in Virtual Drug Screening: Class A GPCRs." *Int. J. Mol. Sci.* 25, 10139 (2024).
10. **Pereira et al.** "Optimizing BBB permeation through deep reinforcement learning for de novo drug design." *Bioinformatics* 37, i84–i92 (2021).
11. **Pandarinath et al.** "Inferring single-trial neural population dynamics using sequential auto-encoders." *Nature Methods* 15, 805–815 (2018). — *LFADS*
12. **Zhu et al.** "Deep inference of latent dynamics with time-series variational autoencoders for calcium imaging." *Nature Neuroscience* 25, 783–794 (2022). — *RADICaL*
13. **Giovannucci et al.** "CaImAn: An open source tool for scalable calcium imaging data analysis." *eLife* 8, e38173 (2019).
14. **Zhao et al.** "Chemogenetic inhibition of microglia..." *Science Advances* (2024).
15. "Machine Learning in Drug Development: BBB Permeability Prediction Models." *Molecular Informatics* (2025).

## Next-Generation Chemogenetics

16. **PAGERs** — Programmable antigen-gated engineered receptors. *Nature* (2025).
17. **DRAGONFLY** — Zero-shot interactome-based drug generation. *Nature Communications* (2025).
18. **DeepBlock** — Toxicity-controlled molecule generation. *Nature Computational Science* (2024).
19. **MICrONS** — Functional connectomics at cellular resolution. *Nature* (2025).

---

# 15. TOOLS & RESOURCES INVENTORY

## Free Cloud Services (No Local Hardware)

| Tool | URL | What It Does | Account Needed? |
|------|-----|-------------|----------------|
| AlphaFold Server | alphafoldserver.com | Protein structure prediction (AF3) | Google account |
| SwissADME | swissadme.ch | Drug property prediction + BOILED-Egg BBB plot | No |
| RCSB PDB | rcsb.org | Download experimental crystal/cryo-EM structures | No |
| UniProt | uniprot.org | Protein sequences | No |
| PubChem | pubchem.ncbi.nlm.nih.gov | Compound SMILES, properties, structures | No |
| AlphaFold DB | alphafold.ebi.ac.uk | Pre-computed structure predictions for known proteins | No |
| GPCRdb | gpcrdb.org | GPCR-specific structural and sequence data | No |

## Python Libraries (Install Locally)

| Library | Install Command | Purpose |
|---------|----------------|---------|
| RDKit | `conda install -c conda-forge rdkit` | Molecular property calculation |
| BioPython | `pip install biopython` | PDB file parsing, RMSD calculation |
| Streamlit | `pip install streamlit` | Web dashboard framework |
| py3Dmol | `pip install py3Dmol` | 3D molecular visualization |
| stmol | `pip install stmol` | Streamlit wrapper for py3Dmol |
| pandas | `pip install pandas` | Data manipulation |
| matplotlib | `pip install matplotlib` | Static plots |
| plotly | `pip install plotly` | Interactive plots |
| numpy | `pip install numpy` | Numerical computation |

## Visualization/Design Tools

| Tool | Purpose |
|------|---------|
| PyMOL (free educational license) or ChimeraX (free) | High-quality 3D structure rendering for poster figures |
| BioRender (free trial) or Inkscape (free) | Schematic diagrams (DREADD mechanism, pipeline architecture) |
| Canva or PowerPoint | Poster layout |
| Google Slides or PowerPoint | Presentation |

---

# 16. KEY SEQUENCES, IDs, AND DATA

## Protein Sequences

### Human Muscarinic M4 (P08173) — Base for hM4Di
- UniProt: P08173
- Gene: CHRM4
- Length: 479 amino acids
- **DREADD mutations for hM4Di:** Y113C, A203G

### Human Muscarinic M3 (P20309) — Base for hM3Dq
- UniProt: P20309
- Gene: CHRM3
- Length: 590 amino acids
- **DREADD mutations for hM3Dq:** Y149C, A239G

## Experimental Structure PDB Codes

| Code | Content | Resolution | Use |
|------|---------|------------|-----|
| 7WC6 | hM4Di–miniGo–DCZ | ~2.6 Å | Validate AF3 hM4Di prediction |
| 7WC7 | hM3Dq–miniGq–DCZ | ~2.7 Å | Validate AF3 hM3Dq prediction |
| 7WC8 | hM3Dq–miniGq–CNO | ~2.8 Å | Compare DCZ vs CNO binding |

## Compound PubChem CIDs

| Compound | CID | Use in Project |
|----------|-----|---------------|
| CNO | 135398508 | Problematic actuator — baseline for comparison |
| Clozapine | 2818 | Off-target metabolite |
| DCZ | 44601286 | Current best actuator |
| Compound 21 | 135445020 | Alternative actuator |
| Olanzapine | 4585 | Repurposed antipsychotic |
| Perlapine | 4748 | Sedative with DREADD activity |

## Key Binding Pocket Residues (for RMSD Subset)

**hM3Dq binding pocket (from Shao et al., Nature 2022):**
- Position 3.33: C149 (mutated from Y149)
- Position 5.46: G239 (mutated from A239)
- TM3: D148, C149, S152
- TM5: G239, T235
- TM6: F222, W525
- TM7: Y529, N507

**hM4Di binding pocket:**
- Position 3.33: C113 (mutated from Y113)
- Position 5.46: G203 (mutated from A203)
- Corresponding residues shifted by alignment

---

# 17. JUDGING PREPARATION: ANTICIPATED Q&A

## Technical Questions

**Q: "How accurate is AlphaFold 3 for GPCRs?"**
A: "AlphaFold 3 shows 50%+ improvement over previous methods for protein–ligand interactions. For well-characterized GPCR families like muscarinic receptors — which have dozens of experimental structures available — the predictions are particularly strong. Our project directly measures this accuracy by computing RMSD against four published cryo-EM structures."

**Q: "Can your pipeline actually design new drugs?"**
A: "Our current pipeline evaluates known compounds — it's a screening and ranking tool, not a generative one. However, the architecture is designed to be extended. Deep reinforcement learning models like Pereira et al. (2021) have shown that generative models can optimize for both receptor binding and BBB permeability simultaneously. That's our proposed future work."

**Q: "Why not just use the existing experimental structures?"**
A: "For hM4Di and hM3Dq, experimental structures exist and we use them for validation. But the value of our pipeline is for the next DREADD system — one that hasn't been crystallized yet. AlphaFold lets you predict the structure in minutes instead of waiting 6–12 months for cryo-EM. The HCAD team spent seven years; our pipeline could cut the computational front-end to days."

**Q: "What about molecular dynamics simulations?"**
A: "MD simulations would improve our predictions by modeling receptor flexibility and drug binding dynamics. However, they require GPU clusters and weeks of compute time, which is beyond our current scope. Tools like GaMD+GLOW are starting to make dynamics more accessible, and that's another future direction."

## Scope & Honesty Questions

**Q: "You didn't actually make a new drug, right?"**
A: "Correct — and we explicitly scoped that out. This is a computational infrastructure project. We're building the pipeline that enables faster design, not claiming to have completed a drug discovery cycle. The analogy is building the telescope, not discovering the planet."

**Q: "How is this bioengineering?"**
A: "We're engineering a computational system to solve a biological problem. Bioengineering isn't just wet lab work — it includes computational biology, bioinformatics, and systems integration. Our pipeline integrates AI structure prediction, computational chemistry, and pharmacokinetic modeling into a unified tool."

**Q: "What would you do with more time?"**
A: "Three things: (1) Add a generative AI module that designs novel actuator candidates rather than just screening known ones, (2) Implement molecular docking to predict binding affinity between predicted structures and candidate actuators, (3) Validate predictions experimentally by collaborating with a university DREADD lab."

## Impact & Market Questions

**Q: "Who would use this?"**
A: "Any lab doing chemogenetics research — over 10,000 published DREADD studies, growing rapidly. Also pharmaceutical companies developing chemogenetic therapeutics for pain (like the HCAD system), Parkinson's, and epilepsy. And the emerging gene therapy sector, where DREADDs are being combined with AAV delivery for clinical applications."

**Q: "What's the market opportunity?"**
A: "The AI drug discovery market is projected at $50B+ by 2030. Chemogenetics is a $2B+ niche growing as the technology approaches clinical translation. No existing AI platform targets engineered receptor–drug co-design specifically. We're identifying an unoccupied niche at the intersection of two high-growth markets."

---

# APPENDIX: PROJECT DIFFERENTIATORS

What sets this project apart from typical BIOEHSC entries:

1. **Working software demo** — not just a PowerPoint; judges can interact with the Streamlit dashboard
2. **Original computational data** — our own AlphaFold predictions + RMSD calculations, not just literature review
3. **Three integrated modules** demonstrating systems-level thinking
4. **Validated against Nature/Science/Cell publications** — we compare our predictions to gold-standard experimental data
5. **Industry-ready framing** — market analysis, commercialization pathway, Isomorphic Labs analogy
6. **Honest scope** — "what we are not claiming" section shows scientific maturity
7. **Reproducible** — everything uses free, open-source tools; any researcher can replicate our pipeline
8. **QR code** linking to GitHub repository and/or live deployed app

---

*Document compiled: February 24, 2026*
*Competition date: April 4, 2026*
*Status: Research complete, entering execution phase (Week 3)*
