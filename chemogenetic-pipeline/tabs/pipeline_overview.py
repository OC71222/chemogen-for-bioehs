"""Tab 4: Pipeline Overview — architecture diagram, module descriptions, cost comparison."""

import streamlit as st


def render(props_df, compounds_df):
    st.header("Pipeline Overview")
    st.markdown("End-to-end AI-accelerated pipeline for chemogenetic actuator design and evaluation.")

    # --- Mermaid flow diagram ---
    st.subheader("Pipeline Architecture")
    st.markdown("""
```mermaid
graph LR
    A[Input: 6 DREADD Actuators<br/>SMILES from PubChem] --> B[Module 1:<br/>Structural Analysis<br/>AlphaFold + RMSD]
    A --> C[Module 2:<br/>Property Evaluation<br/>RDKit + BBB Prediction]
    B --> D[Module 3:<br/>Interactive Dashboard<br/>Streamlit + Plotly]
    C --> D
    D --> E[Output:<br/>Ranked Actuator<br/>Recommendations]
```
""")

    # --- Module descriptions ---
    st.subheader("Module Descriptions")

    with st.expander("Module 1: Structural Analysis (AlphaFold)", expanded=False):
        st.markdown("""
**Status:** In progress

**Purpose:** Predict 3D protein-ligand structures for DREADD actuators bound to hM3Dq receptor.

**Tools:**
- **AlphaFold 3** — Structure prediction (Google DeepMind)
- **BioPython** — PDB/CIF parsing and RMSD calculation
- **py3Dmol** — Interactive 3D molecular visualization

**Process:**
1. Submit DREADD receptor + actuator sequences to AlphaFold
2. Parse predicted structures (PDB/CIF format)
3. Calculate RMSD against experimental structures
4. Evaluate prediction confidence via pLDDT scores

**Performance:** ~5 minutes per structure prediction
""")

    with st.expander("Module 2: Actuator Property Evaluation (RDKit)", expanded=True):
        st.markdown("""
**Status:** Complete

**Purpose:** Calculate molecular descriptors and predict BBB permeability for 6 DREADD actuators.

**Tools:**
- **RDKit** — Molecular descriptor calculation (MW, LogP, TPSA, HBD, HBA, etc.)
- **PubChem** — Canonical SMILES retrieval
- **Rule-based models** — BBB permeability (4-criterion) and Lipinski Rule of Five

**Process:**
1. Load compound SMILES from PubChem
2. Calculate 8 molecular descriptors per compound
3. Apply BBB permeability prediction (MW < 450, 1 \u2264 LogP \u2264 3, TPSA < 90, HBD \u2264 3)
4. Evaluate Lipinski Rule of Five compliance
5. Generate publication-quality figures

**Performance:** < 1 second per compound
""")

    with st.expander("Module 3: Interactive Dashboard (Streamlit)", expanded=False):
        st.markdown("""
**Status:** Complete

**Purpose:** Provide an interactive interface for exploring actuator evaluation results.

**Tools:**
- **Streamlit** — Web application framework
- **Plotly** — Interactive charts (radar, chemical space, bar charts)
- **py3Dmol / stmol** — 3D structure viewer
- **RDKit** — On-the-fly compound evaluation

**Tabs:**
1. **Structure Viewer** — 3D molecular visualization (demo mode until Module 1 completes)
2. **Actuator Comparison** — Side-by-side property comparison with interactive charts
3. **New Compound Evaluator** — Enter any SMILES for instant evaluation
4. **Pipeline Overview** — Architecture and methodology documentation
""")

    # --- Tool links ---
    st.subheader("Tools & Resources")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
| Tool | Purpose |
|------|---------|
| AlphaFold 3 | Structure prediction |
| PubChem | Compound data source |
| RDKit | Molecular descriptors |
| BioPython | Structure parsing |
""")

    with col2:
        st.markdown("""
| Tool | Purpose |
|------|---------|
| Streamlit | Dashboard framework |
| Plotly | Interactive visualizations |
| py3Dmol | 3D structure viewer |
| SwissADME | ADME property reference |
""")

    # --- Performance stats ---
    st.subheader("Performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Structure Prediction", "~5 min", delta="per structure")
    c2.metric("Property Calculation", "< 1 sec", delta="per compound")
    c3.metric("BBB Prediction", "< 1 sec", delta="per compound")
    c4.metric("Dashboard Load", "< 3 sec", delta="full dataset")

    # --- Cost comparison ---
    st.subheader("Cost Comparison: Traditional vs AI Pipeline")
    st.markdown("""
| Aspect | Traditional Approach | AI-Accelerated Pipeline |
|--------|---------------------|------------------------|
| **Structure Determination** | X-ray crystallography (months, $50K+) | AlphaFold prediction (minutes, free) |
| **Property Screening** | Wet-lab assays (weeks, $10K+/compound) | RDKit computation (seconds, free) |
| **BBB Assessment** | In vivo studies (months, $100K+) | Rule-based prediction (instant, free) |
| **Candidate Comparison** | Manual literature review (days) | Interactive dashboard (seconds) |
| **Total Timeline** | 6-12 months | 1-2 days |
| **Total Cost** | $200K+ | ~$0 (open-source tools) |
""")
