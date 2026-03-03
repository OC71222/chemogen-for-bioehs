"""Tab: Docking Results — binding affinity chart, pose viewer, interaction table."""

import os
import streamlit as st
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def render(props_df, compounds_df):
    st.header("Docking Results")

    # Load docking results
    results_path = os.path.join(PROJECT_ROOT, "data", "results", "docking_results.csv")
    if not os.path.exists(results_path):
        st.warning("Docking results not found. Run Module 4 first: `python3 -m src.module4.run_module4`")
        st.info("Module 4 docks all 6 DREADD actuators against the hM3Dq receptor (PDB 8E9W) using AutoDock Vina.")
        return

    docking_df = pd.read_csv(results_path)

    # --- Binding affinity bar chart ---
    st.subheader("Predicted Binding Affinity")

    from src.module4.docking_analysis import affinity_bar_chart, compare_with_published, interaction_table
    fig = affinity_bar_chart(docking_df)
    st.plotly_chart(fig, use_container_width=True)

    # --- Results table ---
    st.subheader("Docking Scores")
    display_df = docking_df[["name", "affinity_kcal_mol", "n_poses", "success"]].copy()
    display_df.columns = ["Compound", "Affinity (kcal/mol)", "Poses", "Success"]
    display_df = display_df.sort_values("Affinity (kcal/mol)")
    st.dataframe(display_df, use_container_width=True, height=250)

    # --- Comparison with published data ---
    st.subheader("Comparison with Published Ki Values")
    comp_df = compare_with_published(docking_df)
    display_comp = comp_df[["name", "predicted_affinity", "published_Ki_nM", "reference"]].copy()
    display_comp.columns = ["Compound", "Predicted (kcal/mol)", "Published Ki (nM)", "Reference"]
    st.dataframe(display_comp, use_container_width=True, height=250)

    # Correlation plot
    from src.module4.docking_analysis import comparison_scatter
    fig_scatter = comparison_scatter(comp_df)
    if fig_scatter:
        st.plotly_chart(fig_scatter, use_container_width=True)

    # --- Docking pose 3D viewer ---
    st.subheader("Docking Pose Viewer")

    pose_dir = os.path.join(PROJECT_ROOT, "data", "docking", "poses")
    available_poses = []
    if os.path.isdir(pose_dir):
        available_poses = [
            f.replace("_pose.pdbqt", "").replace("_", " ")
            for f in os.listdir(pose_dir) if f.endswith("_pose.pdbqt")
        ]

    if available_poses:
        selected_pose = st.selectbox("Select compound pose", available_poses)
        safe_name = selected_pose.replace(" ", "_")
        pose_path = os.path.join(pose_dir, f"{safe_name}_pose.pdbqt")

        if os.path.exists(pose_path):
            try:
                import py3Dmol
                from stmol import showmol

                with open(pose_path) as f:
                    pose_data = f.read()

                viewer = py3Dmol.view(width=700, height=500)
                viewer.addModel(pose_data, "pdb")
                viewer.setStyle({"stick": {"colorscheme": "greenCarbon"}})
                viewer.zoomTo()
                showmol(viewer, height=500, width=700)
            except ImportError:
                st.info("Install py3Dmol and stmol for 3D visualization.")
    else:
        st.info("No docking poses available. Run Module 4 with AutoDock Vina installed.")

    # --- Key interactions ---
    st.subheader("Key Binding Site Interactions")
    interactions_df = interaction_table()
    st.dataframe(interactions_df, use_container_width=True, height=250)
