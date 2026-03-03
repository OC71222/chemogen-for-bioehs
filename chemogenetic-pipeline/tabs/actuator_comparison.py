"""Tab 2: Actuator Comparison — property table, radar chart, chemical space, BBB cards, structures."""

import os
import streamlit as st
from src.utils.plotting import chemical_space_plotly, radar_chart_plotly, COMPOUND_COLORS

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")


def render(props_df, compounds_df):
    st.header("Actuator Comparison")

    # Compound filter
    all_names = props_df["name"].tolist()
    selected = st.multiselect(
        "Select compounds to compare",
        all_names,
        default=all_names,
    )

    if not selected:
        st.warning("Select at least one compound.")
        return

    filtered = props_df[props_df["name"].isin(selected)].copy()

    # --- Property table ---
    st.subheader("Molecular Properties")
    display_df = filtered.rename(columns={
        "name": "Compound",
        "mw": "MW (Da)",
        "logp": "LogP",
        "tpsa": "TPSA (A\u00b2)",
        "hbd": "HBD",
        "hba": "HBA",
        "rotatable_bonds": "Rot. Bonds",
        "aromatic_rings": "Arom. Rings",
        "fsp3": "Fsp3",
        "bbb_predicted": "BBB Prediction",
        "lipinski_violations": "Lipinski Violations",
    })
    st.dataframe(display_df, height=240, use_container_width=True)

    # --- Charts: radar + chemical space ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Radar Chart")
        fig_radar = radar_chart_plotly(props_df, compounds=selected)
        st.plotly_chart(fig_radar, use_container_width=True)

    with col2:
        st.subheader("Chemical Space")
        fig_cs = chemical_space_plotly(filtered)
        st.plotly_chart(fig_cs, use_container_width=True)

    # --- BBB metric cards ---
    st.subheader("BBB Permeability Criteria")
    criteria_labels = {
        "mw": ("MW < 450 Da", 450),
        "logp": ("1.0 \u2264 LogP \u2264 3.0", None),
        "tpsa": ("TPSA < 90 A\u00b2", 90),
        "hbd": ("HBD \u2264 3", 3),
    }

    for _, row in filtered.iterrows():
        name = row["name"]
        cols = st.columns([2, 1, 1, 1, 1])
        cols[0].markdown(f"**{name}** — _{row['bbb_predicted']}_")

        mw_pass = row["mw"] < 450
        logp_pass = 1.0 <= row["logp"] <= 3.0
        tpsa_pass = row["tpsa"] < 90
        hbd_pass = row["hbd"] <= 3

        cols[1].metric("MW < 450", f"{row['mw']:.1f}", delta="Pass" if mw_pass else "Fail", delta_color="normal" if mw_pass else "inverse")
        cols[2].metric("LogP 1-3", f"{row['logp']:.2f}", delta="Pass" if logp_pass else "Fail", delta_color="normal" if logp_pass else "inverse")
        cols[3].metric("TPSA < 90", f"{row['tpsa']:.1f}", delta="Pass" if tpsa_pass else "Fail", delta_color="normal" if tpsa_pass else "inverse")
        cols[4].metric("HBD \u2264 3", str(row["hbd"]), delta="Pass" if hbd_pass else "Fail", delta_color="normal" if hbd_pass else "inverse")

    # --- 2D structure gallery ---
    st.subheader("2D Structures")
    img_cols = st.columns(min(len(selected), 3))
    for i, name in enumerate(selected):
        fname = f"mol_{name.replace(' ', '_')}.png"
        img_path = os.path.join(FIGURES_DIR, fname)
        col = img_cols[i % len(img_cols)]
        if os.path.exists(img_path):
            col.image(img_path, caption=name, use_container_width=True)
        else:
            col.info(f"No image for {name}")
