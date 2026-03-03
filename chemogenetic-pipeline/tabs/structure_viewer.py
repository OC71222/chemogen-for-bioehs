"""Tab 1: Structure Viewer — 3D conformer visualization with graceful degradation."""

import os
import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STRUCTURES_DIR = os.path.join(PROJECT_ROOT, "data", "structures")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
PLDDT_DIR = os.path.join(PROJECT_ROOT, "data", "results", "plddt_scores")


def _has_structure_files():
    """Check if AlphaFold PDB/CIF files exist."""
    for subdir in ["predicted", "experimental"]:
        dirpath = os.path.join(STRUCTURES_DIR, subdir)
        if os.path.isdir(dirpath):
            for f in os.listdir(dirpath):
                if f.endswith((".pdb", ".cif")):
                    return True
    return False


def _is_demo_pdb(compound_name):
    """Check if the PDB file is a small-molecule RDKit conformer (not a protein)."""
    safe_name = compound_name.replace(" ", "_")
    pdb_path = os.path.join(STRUCTURES_DIR, "predicted", f"{safe_name}.pdb")
    if not os.path.exists(pdb_path):
        return False
    with open(pdb_path, "r") as f:
        content = f.read(2000)
    # RDKit-generated PDBs for small molecules have very few ATOM lines
    atom_count = content.count("\nATOM") + content.count("\nHETATM")
    return atom_count < 100


def _generate_3d_conformer(smiles):
    """Generate a 3D conformer from SMILES and return PDB block."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if result != 0:
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.MMFFOptimizeMolecule(mol)
    return Chem.MolToPDBBlock(mol)


def render(props_df, compounds_df):
    st.header("Structure Viewer")

    live_mode = _has_structure_files()

    if live_mode:
        st.info("AlphaFold predicted structures detected. Showing predicted structures.")
    else:
        st.warning(
            "Module 1 (AlphaFold structural analysis) is not yet complete. "
            "Showing RDKit-generated 3D conformers as a demo. "
            "These are NOT AlphaFold predictions."
        )

    # Compound selector
    compound_names = compounds_df["name"].tolist()
    selected = st.selectbox("Select compound", compound_names)
    row = compounds_df[compounds_df["name"] == selected].iloc[0]
    smiles = row["smiles"]

    # --- 2D structure (always shown) ---
    col2d, col3d = st.columns([1, 2])

    with col2d:
        st.subheader("2D Structure")
        img_path = os.path.join(FIGURES_DIR, f"mol_{selected.replace(' ', '_')}.png")
        if os.path.exists(img_path):
            st.image(img_path, caption=selected, use_container_width=True)
        else:
            mol_2d = Chem.MolFromSmiles(smiles)
            if mol_2d:
                img = Draw.MolToImage(mol_2d, size=(400, 300))
                st.image(img, caption=selected, use_container_width=True)

    # --- 3D viewer ---
    with col3d:
        st.subheader("3D Structure")

        if live_mode:
            _render_live_mode(selected)
        else:
            _render_demo_mode(smiles, selected)

    # --- pLDDT scores (if available) ---
    safe_name = selected.replace(" ", "_")
    plddt_csv = os.path.join(PLDDT_DIR, f"plddt_{safe_name}.csv")
    plddt_fig = os.path.join(FIGURES_DIR, f"fig_plddt_{safe_name}.png")

    if os.path.exists(plddt_csv):
        st.divider()
        st.subheader("pLDDT Confidence Scores")
        plddt_df = pd.read_csv(plddt_csv)
        mean_plddt = plddt_df["plddt"].mean()
        pct_confident = (plddt_df["plddt"] > 70).sum() / len(plddt_df) * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("Mean pLDDT", f"{mean_plddt:.1f}")
        c2.metric("Confident Residues (>70)", f"{pct_confident:.0f}%")
        c3.metric("Total Residues", len(plddt_df))

        if os.path.exists(plddt_fig):
            st.image(plddt_fig, caption=f"pLDDT profile — {selected}", use_container_width=True)

        if live_mode and not _is_demo_pdb(selected):
            st.caption("pLDDT scores extracted from AlphaFold output.")
        else:
            st.caption("Synthetic pLDDT scores (demo mode). Not real AlphaFold confidence data.")
    else:
        st.divider()
        st.warning("pLDDT confidence scores are not available. Run the pipeline to generate them.")

    # --- RMSD data ---
    rmsd_csv = os.path.join(PROJECT_ROOT, "data", "results", "rmsd_results.csv")
    if os.path.exists(rmsd_csv):
        rmsd_df = pd.read_csv(rmsd_csv)
        compound_rmsd = rmsd_df[rmsd_df["name"] == selected]
        if not compound_rmsd.empty:
            st.subheader("RMSD Comparison")
            st.dataframe(compound_rmsd, use_container_width=True)
        else:
            st.warning("No RMSD data available for this compound. No experimental structure for comparison.")
    else:
        st.warning("RMSD comparison data is not available. No experimental structures found.")


def _show_py3dmol(pdb_data, fmt="pdb", style="stick", height=500, width=700):
    """Render a 3D structure using py3Dmol embedded directly via st.components.v1.html."""
    import py3Dmol
    import streamlit.components.v1 as components

    viewer = py3Dmol.view(width=width, height=height)
    viewer.addModel(pdb_data, fmt)
    if style == "stick":
        viewer.setStyle({"stick": {"colorscheme": "default"}})
    else:
        viewer.setStyle({"cartoon": {"color": "spectrum"}})
    viewer.setBackgroundColor("white")
    viewer.zoomTo()
    html = viewer._make_html()
    components.html(html, height=height + 20, width=width)


def _render_demo_mode(smiles, compound_name):
    """Generate and display RDKit 3D conformer via py3Dmol."""
    with st.spinner(f"Generating 3D conformer for {compound_name}..."):
        pdb_block = _generate_3d_conformer(smiles)

    if pdb_block is None:
        st.error(f"Could not generate 3D conformer for {compound_name}.")
        return

    _show_py3dmol(pdb_block, fmt="pdb", style="stick")
    st.caption("RDKit-generated 3D conformer (MMFF optimized). Not an AlphaFold prediction.")


def _render_live_mode(compound_name):
    """Load and display PDB/CIF structure."""
    # Search for structure files
    structure_file = None
    for subdir in ["predicted", "experimental"]:
        for ext in [".pdb", ".cif"]:
            candidate = os.path.join(STRUCTURES_DIR, subdir, f"{compound_name.replace(' ', '_')}{ext}")
            if os.path.exists(candidate):
                structure_file = candidate
                break
        if structure_file:
            break

    if structure_file is None:
        st.warning(f"No structure file found for {compound_name}.")
        return

    with open(structure_file, "r") as f:
        structure_data = f.read()

    fmt = "pdb" if structure_file.endswith(".pdb") else "cif"
    is_small_molecule = _is_demo_pdb(compound_name)
    style = "stick" if is_small_molecule else "cartoon"
    label = "RDKit 3D conformer (MMFF optimized)" if is_small_molecule else "AlphaFold predicted structure"

    _show_py3dmol(structure_data, fmt=fmt, style=style)
    st.caption(f"{label} — {os.path.basename(structure_file)}")
