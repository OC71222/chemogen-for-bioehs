"""Tab: ADMET Predictions — ML model results, comparison with rule-based."""

import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def render(props_df, compounds_df):
    st.header("ADMET Predictions")

    # Load ADMET results
    admet_path = os.path.join(PROJECT_ROOT, "data", "results", "admet_predictions.csv")
    if not os.path.exists(admet_path):
        st.warning("ADMET predictions not found. Run Module 6 first: `python3 -m src.module6.run_module6`")
        st.info("Module 6 trains ML models on TDC datasets for BBB, hERG, CYP2D6, HIA, and clearance prediction.")
        return

    admet_df = pd.read_csv(admet_path)

    # --- ADMET Profile Cards ---
    st.subheader("ADMET Profiles")

    for _, row in admet_df.iterrows():
        name = row["name"]

        with st.expander(f"**{name}**", expanded=True):
            cols = st.columns(5)

            # BBB
            bbb_class = row.get("bbb_ml_class", "N/A")
            bbb_prob = row.get("bbb_ml_prob", 0)
            bbb_color = "normal" if bbb_class == "Penetrant" else "inverse"
            cols[0].metric("BBB", bbb_class, delta=f"p={bbb_prob:.2f}", delta_color=bbb_color)

            # hERG
            herg_class = row.get("herg_class", "N/A")
            herg_risk = row.get("herg_risk", "N/A")
            herg_color = "normal" if herg_class == "Non-inhibitor" else "inverse"
            cols[1].metric("hERG", herg_risk, delta=herg_class, delta_color=herg_color)

            # CYP2D6
            cyp = row.get("cyp2d6_inhibitor", None)
            cyp_prob = row.get("cyp2d6_prob", 0)
            cyp_label = "Inhibitor" if cyp else "Non-inhibitor"
            cyp_color = "inverse" if cyp else "normal"
            cols[2].metric("CYP2D6", cyp_label, delta=f"p={cyp_prob:.2f}", delta_color=cyp_color)

            # HIA
            hia_class = row.get("hia_class", "N/A")
            hia_prob = row.get("hia_prob", 0)
            hia_color = "normal" if hia_class == "High absorption" else "inverse"
            cols[3].metric("HIA", hia_class, delta=f"p={hia_prob:.2f}", delta_color=hia_color)

            # Clearance
            cl_cat = row.get("clearance_cat", "N/A")
            cl_val = row.get("clearance_pred", 0)
            cols[4].metric("Clearance", cl_cat, delta=f"{cl_val:.1f} mL/min/kg")

    # --- Grouped bar chart ---
    st.subheader("ADMET Comparison")

    fig = go.Figure()
    endpoints = [
        ("bbb_ml_prob", "BBB Permeability", "#2ecc71"),
        ("herg_prob", "hERG Inhibition", "#e74c3c"),
        ("cyp2d6_prob", "CYP2D6 Inhibition", "#e67e22"),
        ("hia_prob", "HIA", "#3498db"),
    ]

    for col, label, color in endpoints:
        if col in admet_df.columns:
            fig.add_trace(go.Bar(
                name=label,
                x=admet_df["name"],
                y=admet_df[col],
                marker_color=color,
            ))

    fig.update_layout(
        title="ADMET Predictions: All Endpoints",
        xaxis_title="Compound",
        yaxis_title="Predicted Probability",
        yaxis_range=[0, 1],
        barmode="group",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- BBB: ML vs Rule-based comparison ---
    if "bbb_ml_class" in admet_df.columns and "bbb_rule_based" in admet_df.columns:
        st.subheader("BBB Prediction: ML vs Rule-based")

        comp_data = []
        for _, row in admet_df.iterrows():
            ml = row.get("bbb_ml_class", "N/A")
            rule = row.get("bbb_rule_based", "N/A")
            agree = ml == rule
            comp_data.append({
                "Compound": row["name"],
                "ML Prediction": ml,
                "ML Probability": round(row.get("bbb_ml_prob", 0), 3),
                "Rule-based": rule,
                "Agreement": "Yes" if agree else "No",
            })

        comp_df = pd.DataFrame(comp_data)
        st.dataframe(comp_df, use_container_width=True)

        n_agree = sum(1 for d in comp_data if d["Agreement"] == "Yes")
        n_total = len(comp_data)
        st.metric("Agreement Rate", f"{n_agree}/{n_total} ({n_agree/n_total*100:.0f}%)")

        if n_agree < n_total:
            st.markdown("**Disagreements** may indicate compounds where the simple 4-rule heuristic "
                        "is insufficient. ML models capture more complex structure-activity relationships "
                        "learned from ~2000 training compounds.")

    # --- Model Performance ---
    st.subheader("Model Performance Metrics")

    try:
        import joblib
        models_dir = os.path.join(PROJECT_ROOT, "models", "admet")

        perf_data = []
        for model_file in sorted(os.listdir(models_dir)):
            if model_file.endswith("_model.joblib"):
                model_data = joblib.load(os.path.join(models_dir, model_file))
                endpoint = model_file.replace("_model.joblib", "").upper()
                metrics = model_data.get("metrics", {})
                perf_data.append({
                    "Endpoint": endpoint,
                    "Train Size": model_data.get("train_size", "N/A"),
                    "Test Size": model_data.get("test_size", "N/A"),
                    **{k.upper(): v for k, v in metrics.items()},
                })

        if perf_data:
            perf_df = pd.DataFrame(perf_data)
            st.dataframe(perf_df, use_container_width=True)
    except Exception:
        st.info("Model performance data not available.")
