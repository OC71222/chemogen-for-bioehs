"""Tab: Molecular Dynamics — RMSD plots, RMSF, H-bonds, binding stability."""

import os
import streamlit as st
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def render(props_df, compounds_df):
    st.header("Molecular Dynamics")

    results_dir = os.path.join(PROJECT_ROOT, "data", "results")

    # Find available MD analysis files
    md_files = []
    if os.path.isdir(results_dir):
        md_files = [
            f.replace("md_analysis_", "").replace(".csv", "").replace("_", " ")
            for f in os.listdir(results_dir)
            if f.startswith("md_analysis_") and f.endswith(".csv")
        ]

    if not md_files:
        st.warning("MD analysis not found. Run Module 8 first: `python3 -m src.module8.run_module8`")
        st.info("Module 8 runs molecular dynamics simulations (50ns) of DREADD actuator-receptor complexes "
                "using OpenMM, then analyzes binding stability via RMSD, RMSF, and H-bond analysis.")
        return

    # Compound selector
    selected = st.selectbox("Select compound", md_files, index=0)
    safe_name = selected.replace(" ", "_")

    # Load analysis data
    rmsd_path = os.path.join(results_dir, f"md_analysis_{safe_name}.csv")
    rmsf_path = os.path.join(results_dir, f"md_rmsf_{safe_name}.csv")
    summary_path = os.path.join(results_dir, f"md_summary_{safe_name}.csv")

    # --- Summary metrics ---
    st.subheader("Simulation Summary")

    if os.path.exists(summary_path):
        summary_df = pd.read_csv(summary_path)
        row = summary_df.iloc[0]

        cols = st.columns(4)

        binding_stab = row.get("binding_stability", None)
        if binding_stab is not None:
            status = "STABLE" if binding_stab > 0.8 else "MARGINAL" if binding_stab > 0.5 else "UNSTABLE"
            color = "normal" if binding_stab > 0.8 else ("off" if binding_stab > 0.5 else "inverse")
            cols[0].metric("Binding Stability", f"{binding_stab*100:.1f}%",
                          delta=status, delta_color=color)

        hbond = row.get("hbond_occupancy", None)
        if hbond is not None:
            cols[1].metric("H-bond Occupancy", f"{hbond*100:.1f}%")

        prot_rmsd = row.get("mean_protein_rmsd", None)
        if prot_rmsd is not None:
            cols[2].metric("Mean Protein RMSD", f"{prot_rmsd:.2f} A")

        lig_rmsd = row.get("mean_ligand_rmsd", None)
        if lig_rmsd is not None:
            cols[3].metric("Mean Ligand RMSD", f"{lig_rmsd:.2f} A")

    # --- RMSD time series ---
    if os.path.exists(rmsd_path):
        rmsd_df = pd.read_csv(rmsd_path)

        st.subheader("RMSD Time Series")
        from src.module8.md_visualization import rmsd_time_series
        fig_rmsd = rmsd_time_series(rmsd_df, compound_name=selected)
        st.plotly_chart(fig_rmsd, use_container_width=True)

        # Binding stability gauge
        if "ligand_rmsd_A" in rmsd_df.columns:
            stability = (rmsd_df["ligand_rmsd_A"] < 3.0).mean()
            from src.module8.md_visualization import binding_stability_gauge
            fig_gauge = binding_stability_gauge(stability, compound_name=selected)
            st.plotly_chart(fig_gauge, use_container_width=True)

    # --- RMSF per-residue ---
    if os.path.exists(rmsf_path):
        rmsf_df = pd.read_csv(rmsf_path)

        st.subheader("Per-Residue RMSF")
        from src.module8.md_visualization import rmsf_bar_chart
        fig_rmsf = rmsf_bar_chart(rmsf_df, compound_name=selected)
        st.plotly_chart(fig_rmsf, use_container_width=True)

        # Top flexible residues
        top_flex = rmsf_df.nlargest(10, "rmsf_A")
        st.markdown("**Most Flexible Residues:**")
        st.dataframe(top_flex[["residue", "resname", "rmsf_A"]].rename(columns={
            "residue": "Residue #",
            "resname": "Residue Type",
            "rmsf_A": "RMSF (A)",
        }), use_container_width=True, height=300)

    # --- H-bond occupancy ---
    if os.path.exists(summary_path):
        summary_df = pd.read_csv(summary_path)
        hbond = summary_df.iloc[0].get("hbond_occupancy", None)
        if hbond is not None:
            st.subheader("Hydrogen Bond Occupancy")
            from src.module8.md_visualization import hbond_occupancy_chart
            fig_hb = hbond_occupancy_chart(hbond, compound_name=selected)
            st.plotly_chart(fig_hb, use_container_width=True)

    # --- Interpretation ---
    st.subheader("Interpretation Guide")
    st.markdown("""
    **Key Metrics:**
    - **Binding Stability >80%**: Ligand remains well-bound throughout simulation
    - **Protein RMSD <3.0 A**: Receptor maintains stable fold
    - **H-bond Occupancy >50%**: Key hydrogen bonds persist during dynamics
    - **RMSF**: Low values at binding site residues indicate a rigid, well-defined pocket

    **For DREADD actuators:**
    - DCZ should show stable binding — it is the most potent known actuator
    - The Asp3.32 salt bridge is critical for amine recognition
    - High protein RMSD may indicate that the cryo-EM structure relaxes during simulation
    """)
