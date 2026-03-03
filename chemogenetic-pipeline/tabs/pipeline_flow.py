"""Single-page pipeline flow — replaces all sidebar tabs."""

import os
import json
import time
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "results")
PROGRESS_DIR = os.path.join(PROJECT_ROOT, "data", "progress")

STAGES = [
    ("Compounds", "compounds"),
    ("Properties", "properties"),
    ("Docking", "docking"),
    ("Screening", "screening"),
    ("ADMET", "admet"),
    ("Selectivity", "selectivity"),
    ("MD", "md"),
]

# ── LED pulse animation CSS ──────────────────────────────────────────

LED_PULSE_CSS = """
<style>
@keyframes led-pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 4px #3b82f6; }
    50% { opacity: 0.4; box-shadow: 0 0 8px #3b82f6; }
}
.led {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin: 4px auto 0 auto;
}
.led-green {
    background: #00ff88;
    box-shadow: 0 0 6px rgba(0,255,136,0.6);
}
.led-blue {
    background: #3b82f6;
    animation: led-pulse 1.2s ease-in-out infinite;
}
.led-gray {
    background: #2a3040;
}
</style>
"""

# ── helpers ──────────────────────────────────────────────────────────


def _dark_fig(fig):
    """Apply dark instrument-panel styling to a Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(family="JetBrains Mono, Fira Code, monospace", color="#c0c8d4"),
        margin=dict(t=30),
    )
    # Standard cartesian axes
    fig.update_xaxes(
        gridcolor="#1a2332", zerolinecolor="#1a2332",
        tickfont=dict(color="#607080"),
    )
    fig.update_yaxes(
        gridcolor="#1a2332", zerolinecolor="#1a2332",
        tickfont=dict(color="#607080"),
    )
    # Polar / radar axes (if present)
    if hasattr(fig, "layout") and hasattr(fig.layout, "polar") and fig.layout.polar is not None:
        fig.update_layout(
            polar=dict(
                bgcolor="#0d1117",
                radialaxis=dict(gridcolor="#1a2332", tickfont=dict(color="#607080")),
                angularaxis=dict(gridcolor="#1a2332", tickfont=dict(color="#607080")),
            )
        )
    return fig


def _stage_status(key):
    """Return 'completed', 'running', or 'idle' for a pipeline stage."""
    file_map = {
        "compounds": os.path.join(PROJECT_ROOT, "data", "compounds", "compounds.csv"),
        "properties": os.path.join(RESULTS_DIR, "actuator_properties.csv"),
        "docking": os.path.join(RESULTS_DIR, "docking_results.csv"),
        "screening": os.path.join(RESULTS_DIR, "screening_hits.csv"),
        "admet": os.path.join(RESULTS_DIR, "admet_predictions.csv"),
        "selectivity": os.path.join(RESULTS_DIR, "selectivity_predictions.csv"),
        "md": None,
    }
    module_map = {
        "docking": 4,
        "screening": 5,
        "admet": 6,
        "selectivity": 7,
        "md": 8,
    }

    # Check if module is currently running
    mod_num = module_map.get(key)
    if mod_num:
        prog_path = os.path.join(PROGRESS_DIR, f"module{mod_num}.json")
        if os.path.exists(prog_path):
            try:
                with open(prog_path) as f:
                    data = json.load(f)
                if data.get("status") == "running":
                    return "running"
            except (json.JSONDecodeError, IOError):
                pass

    # Check for MD results (glob pattern)
    if key == "md":
        if os.path.isdir(RESULTS_DIR):
            for fname in os.listdir(RESULTS_DIR):
                if fname.startswith("md_analysis_") and fname.endswith(".csv"):
                    return "completed"
        return "idle"

    result_file = file_map.get(key)
    if result_file and os.path.exists(result_file):
        return "completed"
    return "idle"


def _led_html(status):
    """Return an HTML LED span for a pipeline status."""
    css_class = {
        "completed": "led-green",
        "running": "led-blue",
        "idle": "led-gray",
    }.get(status, "led-gray")
    return f'<div style="text-align:center"><span class="led {css_class}"></span></div>'


# ── main render ──────────────────────────────────────────────────────


def render(props_df, compounds_df):
    st.markdown(LED_PULSE_CSS, unsafe_allow_html=True)

    st.markdown(
        '<p style="font-family: \'JetBrains Mono\', monospace; font-size: 1.4rem; '
        'color: #00ff88; letter-spacing: 0.15em; margin-bottom: 0.2rem; '
        'text-shadow: 0 0 12px rgba(0,255,136,0.3);">'
        'CHEMOGENETIC PIPELINE <span style="color:#607080; font-size:0.8rem;">v1.0</span></p>',
        unsafe_allow_html=True,
    )

    # ── Pipeline step buttons ────────────────────────────────────────
    cols = st.columns(len(STAGES))
    if "active_stage" not in st.session_state:
        st.session_state.active_stage = "compounds"

    for i, (label, key) in enumerate(STAGES):
        status = _stage_status(key)
        is_active = st.session_state.active_stage == key
        with cols[i]:
            btn_type = "primary" if is_active else "secondary"
            if st.button(label, key=f"btn_{key}", use_container_width=True, type=btn_type):
                st.session_state.active_stage = key
            st.markdown(_led_html(status), unsafe_allow_html=True)

    st.divider()

    # ── Stage detail area ────────────────────────────────────────────
    active = st.session_state.active_stage
    stage_renderers = {
        "compounds": _render_stage_compounds,
        "properties": _render_stage_properties,
        "docking": _render_stage_docking,
        "screening": _render_stage_screening,
        "admet": _render_stage_admet,
        "selectivity": _render_stage_selectivity,
        "md": _render_stage_md,
    }
    renderer = stage_renderers.get(active)
    if renderer:
        renderer(props_df, compounds_df)

    # ── SMILES evaluator ─────────────────────────────────────────────
    st.divider()
    _render_smiles_eval(props_df)

    # ── Live progress ────────────────────────────────────────────────
    st.divider()
    any_running = _render_progress()

    if any_running:
        time.sleep(5)
        st.rerun()


# ── Stage renderers ──────────────────────────────────────────────────


def _render_stage_compounds(props_df, compounds_df):
    st.subheader("Compounds")
    display = compounds_df.rename(columns={
        "name": "Name",
        "smiles": "SMILES",
        "role": "Role",
    })
    show_cols = [c for c in ["Name", "SMILES", "Role", "pubchem_cid"] if c in display.columns]
    st.dataframe(display[show_cols], use_container_width=True, height=260)


def _render_stage_properties(props_df, compounds_df):
    st.subheader("Properties")

    col_table, col_radar = st.columns([1, 1])

    with col_table:
        display_df = props_df.rename(columns={
            "name": "Compound",
            "mw": "MW",
            "logp": "LogP",
            "tpsa": "TPSA",
            "hbd": "HBD",
            "hba": "HBA",
            "rotatable_bonds": "Rot. Bonds",
            "bbb_predicted": "BBB",
        })
        show = ["Compound", "MW", "LogP", "TPSA", "HBD", "HBA", "Rot. Bonds", "BBB"]
        show = [c for c in show if c in display_df.columns]
        st.dataframe(display_df[show], use_container_width=True, height=280)

    with col_radar:
        from src.utils.plotting import radar_chart_plotly
        fig = radar_chart_plotly(props_df, compounds=props_df["name"].tolist())
        st.plotly_chart(_dark_fig(fig), use_container_width=True)


def _render_stage_docking(props_df, compounds_df):
    st.subheader("Docking")

    results_path = os.path.join(RESULTS_DIR, "docking_results.csv")
    if not os.path.exists(results_path):
        st.info("Run Module 4 first: `python3 -m src.module4.run_module4`")
        return

    docking_df = pd.read_csv(results_path)

    col_chart, col_table = st.columns([1, 1])

    with col_chart:
        from src.module4.docking_analysis import affinity_bar_chart
        fig = affinity_bar_chart(docking_df)
        st.plotly_chart(_dark_fig(fig), use_container_width=True)

    with col_table:
        display = docking_df[["name", "affinity_kcal_mol", "n_poses", "success"]].copy()
        display.columns = ["Compound", "Affinity (kcal/mol)", "Poses", "Success"]
        display = display.sort_values("Affinity (kcal/mol)")
        st.dataframe(display, use_container_width=True, height=280)


def _render_stage_screening(props_df, compounds_df):
    st.subheader("Screening")

    hits_path = os.path.join(RESULTS_DIR, "screening_hits.csv")
    if not os.path.exists(hits_path):
        st.info("Run Module 5 first: `python3 -m src.module5.run_module5`")
        return

    hits_df = pd.read_csv(hits_path)

    col_table, col_scatter = st.columns([1, 1])

    with col_table:
        display_cols = ["hit_rank", "compound_id", "affinity", "max_tanimoto",
                        "closest_actuator", "is_novel"]
        available = [c for c in display_cols if c in hits_df.columns]
        display = hits_df[available].rename(columns={
            "hit_rank": "Rank",
            "compound_id": "Compound",
            "affinity": "Affinity",
            "max_tanimoto": "Tanimoto",
            "closest_actuator": "Closest",
            "is_novel": "Novel",
        })
        st.dataframe(display, use_container_width=True, height=380)

    with col_scatter:
        from src.module5.hit_analysis import affinity_vs_novelty_scatter
        fig = affinity_vs_novelty_scatter(hits_df)
        st.plotly_chart(_dark_fig(fig), use_container_width=True)


def _render_stage_admet(props_df, compounds_df):
    st.subheader("ADMET")

    admet_path = os.path.join(RESULTS_DIR, "admet_predictions.csv")
    if not os.path.exists(admet_path):
        st.info("Run Module 6 first: `python3 -m src.module6.run_module6`")
        return

    admet_df = pd.read_csv(admet_path)

    # Grouped bar chart
    fig = go.Figure()
    endpoints = [
        ("bbb_ml_prob", "BBB Permeability", "#2ecc71"),
        ("herg_prob", "hERG Inhibition", "#e74c3c"),
        ("cyp2d6_prob", "CYP2D6 Inhibition", "#e67e22"),
        ("hia_prob", "HIA", "#3498db"),
    ]
    for col, label, color in endpoints:
        if col in admet_df.columns:
            fig.add_trace(go.Bar(name=label, x=admet_df["name"], y=admet_df[col], marker_color=color))

    fig.update_layout(
        xaxis_title="Compound",
        yaxis_title="Probability",
        yaxis_range=[0, 1],
        barmode="group",
        height=400,
    )
    st.plotly_chart(_dark_fig(fig), use_container_width=True)

    # ML vs rule-based table
    if "bbb_ml_class" in admet_df.columns and "bbb_rule_based" in admet_df.columns:
        rows = []
        for _, row in admet_df.iterrows():
            ml = row.get("bbb_ml_class", "N/A")
            rule = row.get("bbb_rule_based", "N/A")
            rows.append({
                "Compound": row["name"],
                "ML": ml,
                "ML Prob": round(row.get("bbb_ml_prob", 0), 3),
                "Rule-based": rule,
                "Agree": "Yes" if ml == rule else "No",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=260)


def _render_stage_selectivity(props_df, compounds_df):
    st.subheader("Selectivity")

    sel_path = os.path.join(RESULTS_DIR, "selectivity_predictions.csv")
    if not os.path.exists(sel_path):
        st.info("Run Module 7 first: `python3 -m src.module7.run_module7`")
        return

    sel_df = pd.read_csv(sel_path)
    target_cols = [c for c in sel_df.columns if c.startswith("p_")]

    if not target_cols:
        st.warning("No target probability columns found.")
        return

    try:
        from src.module7.chembl_data import TARGETS
        target_names = [TARGETS.get(c.replace("p_", ""), {}).get("name", c) for c in target_cols]
    except ImportError:
        target_names = [c.replace("p_", "") for c in target_cols]

    col_radar, col_heatmap = st.columns([1, 1])

    with col_radar:
        from src.utils.plotting import COMPOUND_COLORS
        fig = go.Figure()
        for _, row in sel_df.iterrows():
            name = row["name"]
            probs = [row[c] for c in target_cols]
            color = COMPOUND_COLORS.get(name, "#333333")
            fig.add_trace(go.Scatterpolar(
                r=probs, theta=target_names, fill="toself",
                name=name, line_color=color, opacity=0.7,
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=450,
        )
        st.plotly_chart(_dark_fig(fig), use_container_width=True)

    with col_heatmap:
        names = sel_df["name"].tolist()
        z = sel_df[target_cols].values.tolist()
        fig_hm = go.Figure(data=go.Heatmap(
            z=z, x=target_names, y=names,
            colorscale="RdYlGn_r", zmin=0, zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in z],
            texttemplate="%{text}",
        ))
        fig_hm.update_layout(height=450)
        st.plotly_chart(_dark_fig(fig_hm), use_container_width=True)


def _render_stage_md(props_df, compounds_df):
    st.subheader("Molecular Dynamics")

    md_files = []
    if os.path.isdir(RESULTS_DIR):
        md_files = [
            f.replace("md_analysis_", "").replace(".csv", "").replace("_", " ")
            for f in os.listdir(RESULTS_DIR)
            if f.startswith("md_analysis_") and f.endswith(".csv")
        ]

    if not md_files:
        st.info("Run Module 8 first: `python3 -m src.module8.run_module8`")
        return

    selected = st.selectbox("Compound", md_files, index=0, key="md_compound")
    safe_name = selected.replace(" ", "_")

    rmsd_path = os.path.join(RESULTS_DIR, f"md_analysis_{safe_name}.csv")
    summary_path = os.path.join(RESULTS_DIR, f"md_summary_{safe_name}.csv")

    # Summary metrics
    if os.path.exists(summary_path):
        summary = pd.read_csv(summary_path).iloc[0]
        mcols = st.columns(4)
        bs = summary.get("binding_stability")
        if bs is not None:
            label = "STABLE" if bs > 0.8 else "MARGINAL" if bs > 0.5 else "UNSTABLE"
            mcols[0].metric("Binding Stability", f"{bs*100:.1f}%", delta=label)
        hb = summary.get("hbond_occupancy")
        if hb is not None:
            mcols[1].metric("H-bond Occupancy", f"{hb*100:.1f}%")
        pr = summary.get("mean_protein_rmsd")
        if pr is not None:
            mcols[2].metric("Protein RMSD", f"{pr:.2f} A")
        lr = summary.get("mean_ligand_rmsd")
        if lr is not None:
            mcols[3].metric("Ligand RMSD", f"{lr:.2f} A")

    # RMSD time series
    if os.path.exists(rmsd_path):
        rmsd_df = pd.read_csv(rmsd_path)
        from src.module8.md_visualization import rmsd_time_series
        fig = rmsd_time_series(rmsd_df, compound_name=selected)
        st.plotly_chart(_dark_fig(fig), use_container_width=True)


# ── SMILES evaluator ─────────────────────────────────────────────────


def _render_smiles_eval(props_df):
    with st.expander("SMILES Evaluator", expanded=False):
        smiles = st.text_input("Enter SMILES", placeholder="e.g. CN1CCN(CC1)C2=NC3=CC=CC=C3NC4=CC=CC=C42", key="smiles_input")
        if not smiles:
            return

        try:
            from rdkit import Chem
            from rdkit.Chem import Draw
        except ImportError:
            st.error("RDKit not installed.")
            return

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            st.error("Invalid SMILES.")
            return

        from src.module2.evaluate_actuators import calculate_properties, predict_bbb

        props = calculate_properties(smiles)
        bbb = predict_bbb(props)

        col_img, col_props = st.columns([1, 2])
        with col_img:
            img = Draw.MolToImage(mol, size=(300, 200))
            st.image(img, use_container_width=True)

        with col_props:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("MW", f"{props['mw']:.1f}")
            c2.metric("LogP", f"{props['logp']:.2f}")
            c3.metric("TPSA", f"{props['tpsa']:.1f}")
            c4.metric("BBB", bbb)


# ── Live progress ────────────────────────────────────────────────────


def _time_ago(iso_str):
    try:
        dt = datetime.fromisoformat(iso_str)
        secs = (datetime.now() - dt).total_seconds()
        if secs < 60:
            return f"{int(secs)}s ago"
        if secs < 3600:
            return f"{int(secs // 60)}m ago"
        return f"{int(secs // 3600)}h ago"
    except Exception:
        return ""


MODULE_NAMES = {
    4: "Molecular Docking",
    5: "Virtual Screening",
    6: "ADMET Prediction",
    7: "Selectivity Prediction",
    8: "Molecular Dynamics",
}


def _render_progress():
    """Render live progress bars. Returns True if any module is running."""
    any_running = False
    bars = []

    for mod_num in [4, 5, 6, 7, 8]:
        prog_path = os.path.join(PROGRESS_DIR, f"module{mod_num}.json")
        if not os.path.exists(prog_path):
            continue
        try:
            with open(prog_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        if data.get("status") == "running":
            any_running = True
            bars.append(data)

    if not bars:
        st.markdown(
            '<p style="font-family:monospace; text-transform:uppercase; '
            'letter-spacing:0.1em; color:#4a5568; font-size:0.75rem;">'
            'STATUS: ALL MODULES IDLE</p>',
            unsafe_allow_html=True,
        )
        return False

    st.markdown(
        '<p style="font-family:monospace; text-transform:uppercase; '
        'letter-spacing:0.1em; color:#00ff88; font-size:0.75rem;">'
        'LIVE PROGRESS</p>',
        unsafe_allow_html=True,
    )
    for data in bars:
        mod = data.get("module", "?")
        step = data.get("step", "Processing...")
        progress = data.get("progress", 0)
        total = data.get("total", 1)
        pct = progress / total if total else 0
        updated = data.get("updated_at", "")

        label = f"Module {mod} — {MODULE_NAMES.get(mod, '')} — {step}"
        if updated:
            label += f"  ({_time_ago(updated)})"

        st.progress(pct, text=label)

    return any_running
