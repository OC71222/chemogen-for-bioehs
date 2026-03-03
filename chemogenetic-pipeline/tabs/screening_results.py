"""Tab: Virtual Screening — hit table, affinity vs novelty scatter, chemical space."""

import os
import streamlit as st
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def render(props_df, compounds_df):
    st.header("Virtual Screening")

    # Load screening results
    hits_path = os.path.join(PROJECT_ROOT, "data", "results", "screening_hits.csv")
    results_path = os.path.join(PROJECT_ROOT, "data", "results", "screening_results.csv")

    if not os.path.exists(hits_path) and not os.path.exists(results_path):
        st.warning("Screening results not found. Run Module 5 first: `python3 -m src.module5.run_module5`")
        st.info("Module 5 screens a drug-like compound library against hM3Dq to discover novel actuator scaffolds.")
        return

    # --- Screening summary ---
    if os.path.exists(results_path):
        all_results = pd.read_csv(results_path)
        col1, col2, col3 = st.columns(3)
        n_total = len(all_results)
        n_success = all_results["success"].sum() if "success" in all_results.columns else n_total
        col1.metric("Compounds Screened", n_total)
        col2.metric("Successful Docking", n_success)

        if "affinity" in all_results.columns:
            n_hits = (all_results["affinity"].dropna() <= -7.0).sum()
            col3.metric("Hits (< -7.0 kcal/mol)", n_hits)

    # --- Hit table ---
    if os.path.exists(hits_path):
        hits_df = pd.read_csv(hits_path)

        st.subheader("Top Screening Hits")

        display_cols = ["hit_rank", "compound_id", "affinity", "max_tanimoto",
                        "closest_actuator", "is_novel"]
        available_cols = [c for c in display_cols if c in hits_df.columns]

        display_df = hits_df[available_cols].copy()
        rename_map = {
            "hit_rank": "Rank",
            "compound_id": "Compound",
            "affinity": "Affinity (kcal/mol)",
            "max_tanimoto": "Tanimoto Similarity",
            "closest_actuator": "Closest Known",
            "is_novel": "Novel Scaffold",
        }
        display_df = display_df.rename(columns={k: v for k, v in rename_map.items() if k in display_df.columns})
        st.dataframe(display_df, use_container_width=True, height=400)

        # --- Affinity vs Novelty scatter ---
        st.subheader("Affinity vs Novelty")
        from src.module5.hit_analysis import affinity_vs_novelty_scatter

        fig = affinity_vs_novelty_scatter(hits_df)
        st.plotly_chart(fig, use_container_width=True)

        # --- Novel scaffolds highlight ---
        if "is_novel" in hits_df.columns:
            novel = hits_df[hits_df["is_novel"] == True]
            if not novel.empty:
                st.subheader(f"Novel Scaffolds ({len(novel)} found)")
                st.markdown("These compounds have **Tanimoto similarity < 0.4** to all known actuators, "
                            "representing potentially new chemical scaffolds for DREADD activation.")

                for _, row in novel.head(5).iterrows():
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        try:
                            from rdkit import Chem
                            from rdkit.Chem import Draw
                            mol = Chem.MolFromSmiles(row["smiles"])
                            if mol:
                                img = Draw.MolToImage(mol, size=(300, 200))
                                st.image(img)
                        except Exception:
                            st.text(row.get("smiles", "N/A"))
                    with col2:
                        st.markdown(f"**{row.get('compound_id', 'Unknown')}**")
                        st.markdown(f"Affinity: {row.get('affinity', 'N/A'):.2f} kcal/mol")
                        st.markdown(f"Tanimoto: {row.get('max_tanimoto', 'N/A'):.3f}")
                        st.markdown(f"Closest known: {row.get('closest_actuator', 'N/A')}")

        # --- ADMET filter summary ---
        st.subheader("ADMET Filter Summary")
        admet_path = os.path.join(PROJECT_ROOT, "data", "results", "admet_predictions.csv")
        if os.path.exists(admet_path):
            st.info("ADMET predictions available — see ADMET Predictions tab for detailed profiles.")
        else:
            st.info("Run Module 6 for ADMET predictions on screening hits.")
