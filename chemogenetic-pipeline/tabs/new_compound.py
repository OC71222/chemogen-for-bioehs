"""Tab 3: New Compound Evaluator — SMILES input, property calculation, BBB prediction."""

import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from src.module2.evaluate_actuators import (
    calculate_properties,
    predict_bbb,
    bbb_criteria_detail,
    count_lipinski_violations,
)
from src.utils.plotting import chemical_space_plotly, radar_chart_plotly, COMPOUND_COLORS
import plotly.graph_objects as go


def render(props_df, compounds_df):
    st.header("New Compound Evaluator")
    st.markdown("Enter a SMILES string to instantly evaluate a compound against the known DREADD actuators.")

    smiles = st.text_input("SMILES", placeholder="e.g. CN1CCN(CC1)C2=NC3=CC=CC=C3NC4=CC=CC=C42")

    if not smiles:
        return

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("Invalid SMILES string. Please check the input and try again.")
        return

    props = calculate_properties(smiles)
    bbb = predict_bbb(props)
    criteria = bbb_criteria_detail(props)
    lipinski = count_lipinski_violations(props)

    # --- 2D structure ---
    col_struct, col_props = st.columns([1, 2])
    with col_struct:
        st.subheader("2D Structure")
        img = Draw.MolToImage(mol, size=(400, 300))
        st.image(img, use_container_width=True)

    # --- Property card ---
    with col_props:
        st.subheader("Calculated Properties")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MW (Da)", f"{props['mw']:.2f}")
        c2.metric("LogP", f"{props['logp']:.2f}")
        c3.metric("TPSA (A\u00b2)", f"{props['tpsa']:.2f}")
        c4.metric("Fsp3", f"{props['fsp3']:.3f}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("HBD", props["hbd"])
        c6.metric("HBA", props["hba"])
        c7.metric("Rot. Bonds", props["rotatable_bonds"])
        c8.metric("Arom. Rings", props["aromatic_rings"])

    # --- BBB prediction ---
    st.subheader("BBB Permeability Prediction")
    if bbb == "Penetrant":
        st.success(f"**{bbb}** — All criteria met")
    else:
        st.warning(f"**{bbb}** — One or more criteria not met")

    cols = st.columns(4)
    for col, (key, label) in zip(cols, [
        ("mw_pass", "MW < 450"),
        ("logp_pass", "LogP 1-3"),
        ("tpsa_pass", "TPSA < 90"),
        ("hbd_pass", "HBD \u2264 3"),
    ]):
        passed = criteria[key]
        col.metric(label, "Pass" if passed else "Fail", delta="Pass" if passed else "Fail", delta_color="normal" if passed else "inverse")

    st.metric("Lipinski Violations", lipinski)

    # --- Chemical space overlay ---
    st.subheader("Chemical Space (with new compound)")
    new_row = {
        "name": "New Compound",
        "mw": props["mw"],
        "logp": props["logp"],
        "tpsa": props["tpsa"],
        "hbd": props["hbd"],
        "hba": props["hba"],
        "rotatable_bonds": props["rotatable_bonds"],
        "aromatic_rings": props["aromatic_rings"],
        "fsp3": props["fsp3"],
        "bbb_predicted": bbb,
        "lipinski_violations": lipinski,
    }
    combined_df = pd.concat([props_df, pd.DataFrame([new_row])], ignore_index=True)

    fig_cs = chemical_space_plotly(combined_df)
    # Add gold star for the new compound
    fig_cs.add_trace(go.Scatter(
        x=[props["logp"]],
        y=[props["tpsa"]],
        mode="markers+text",
        text=["New Compound"],
        textposition="bottom center",
        marker=dict(symbol="star", size=18, color="gold", line=dict(width=2, color="black")),
        name="New Compound",
        showlegend=False,
    ))
    st.plotly_chart(fig_cs, use_container_width=True)

    # --- Radar comparison ---
    st.subheader("Radar Comparison")
    fig_radar = radar_chart_plotly(combined_df, compounds=props_df["name"].tolist() + ["New Compound"])
    st.plotly_chart(fig_radar, use_container_width=True)
