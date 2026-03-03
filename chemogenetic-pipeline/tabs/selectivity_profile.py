"""Tab: Selectivity Profile — radar charts, off-target heatmap, selectivity scores."""

import os
import streamlit as st
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def render(props_df, compounds_df):
    st.header("Selectivity")

    # Load selectivity results
    sel_path = os.path.join(PROJECT_ROOT, "data", "results", "selectivity_predictions.csv")
    if not os.path.exists(sel_path):
        st.warning("Selectivity predictions not found. Run Module 7 first: `python3 -m src.module7.run_module7`")
        st.info("Module 7 trains per-target ML models using ChEMBL binding data "
                "for muscarinic (M1-M5), dopamine (D2), serotonin (5-HT2A), and histamine (H1) receptors.")
        return

    sel_df = pd.read_csv(sel_path)

    # --- Selectivity score ranking ---
    st.subheader("Selectivity Score Ranking")
    st.markdown("Higher score = more selective for hM3Dq DREADD vs off-targets.")

    ranked = sel_df.sort_values("selectivity_score", ascending=False)
    display_cols = ["name", "selectivity_score", "n_off_targets", "off_targets"]
    available_cols = [c for c in display_cols if c in ranked.columns]
    display_df = ranked[available_cols].rename(columns={
        "name": "Compound",
        "selectivity_score": "Selectivity Score",
        "n_off_targets": "Off-Target Flags",
        "off_targets": "Flagged Targets",
    })
    st.dataframe(display_df, use_container_width=True, height=250)

    # --- Selectivity radar chart ---
    st.subheader("Selectivity Radar Charts")

    # Build profiles from CSV columns
    from src.module7.chembl_data import TARGETS

    target_cols = [c for c in sel_df.columns if c.startswith("p_")]

    if target_cols:
        # Per-compound radar
        selected_compound = st.selectbox(
            "Select compound for radar",
            sel_df["name"].tolist(),
            index=0,
        )

        row = sel_df[sel_df["name"] == selected_compound].iloc[0]
        import plotly.graph_objects as go

        target_names = [TARGETS.get(c.replace("p_", ""), {}).get("name", c) for c in target_cols]
        probs = [row[c] for c in target_cols]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=probs,
            theta=target_names,
            fill="toself",
            name=selected_compound,
            line_color="#3498db",
        ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title=f"Target Activity Profile: {selected_compound}",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Multi-compound radar ---
        st.subheader("All Compounds Comparison")

        selected_compounds = st.multiselect(
            "Select compounds to compare",
            sel_df["name"].tolist(),
            default=sel_df["name"].tolist(),
        )

        if selected_compounds:
            from src.utils.plotting import COMPOUND_COLORS

            fig_multi = go.Figure()
            for compound in selected_compounds:
                row = sel_df[sel_df["name"] == compound].iloc[0]
                probs = [row[c] for c in target_cols]
                color = COMPOUND_COLORS.get(compound, "#333333")

                fig_multi.add_trace(go.Scatterpolar(
                    r=probs,
                    theta=target_names,
                    fill="toself",
                    name=compound,
                    line_color=color,
                    opacity=0.7,
                ))

            fig_multi.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title="Selectivity Profiles: All Actuators",
                height=600,
            )
            st.plotly_chart(fig_multi, use_container_width=True)

    # --- Off-target risk heatmap ---
    st.subheader("Off-Target Risk Heatmap")

    if target_cols:
        import plotly.graph_objects as go

        names = sel_df["name"].tolist()
        z = sel_df[target_cols].values.tolist()

        fig_heatmap = go.Figure(data=go.Heatmap(
            z=z,
            x=target_names,
            y=names,
            colorscale="RdYlGn_r",
            zmin=0, zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in z],
            texttemplate="%{text}",
            hovertemplate="Compound: %{y}<br>Target: %{x}<br>P(active): %{z:.3f}<extra></extra>",
        ))

        fig_heatmap.update_layout(
            title="Off-Target Activity Heatmap",
            xaxis_title="Target",
            yaxis_title="Compound",
            height=400,
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

    # --- Known vs predicted comparison ---
    st.subheader("Known Off-Target Comparison: Clozapine")
    st.markdown("""
    **Clozapine** is known to bind multiple off-targets:
    - **D2 (Dopamine)**: Moderate affinity — antipsychotic mechanism
    - **5-HT2A (Serotonin)**: High affinity — key pharmacological target
    - **H1 (Histamine)**: High affinity — causes sedation
    - **M1-M5 (Muscarinic)**: Variable — causes anticholinergic side effects

    **DCZ** was designed to minimize these off-target interactions while maintaining
    DREADD selectivity.
    """)

    if "Clozapine" in sel_df["name"].values and target_cols:
        cloz_row = sel_df[sel_df["name"] == "Clozapine"].iloc[0]
        st.markdown("**Clozapine predicted off-target probabilities:**")
        for col in target_cols:
            target = col.replace("p_", "")
            target_name = TARGETS.get(target, {}).get("name", target)
            prob = cloz_row[col]
            bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
            st.text(f"  {target_name:20s} {bar} {prob:.2f}")
