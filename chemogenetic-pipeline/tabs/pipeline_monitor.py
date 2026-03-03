"""
Pipeline Monitor Tab — Live progress tracking for running modules.
Auto-refreshes to show training status, download progress, and simulation steps.
"""

import os
import time
import json
import streamlit as st
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROGRESS_DIR = os.path.join(PROJECT_ROOT, "data", "progress")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "results")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

MODULE_NAMES = {
    1: "Structural Analysis",
    2: "Property Evaluation",
    4: "Molecular Docking",
    5: "Virtual Screening",
    6: "ADMET Prediction",
    7: "Selectivity Prediction",
    8: "Molecular Dynamics",
}

MODULE_ICONS = {
    1: ":material/science:",
    2: ":material/analytics:",
    4: ":material/hub:",
    5: ":material/search:",
    6: ":material/medication:",
    7: ":material/target:",
    8: ":material/speed:",
}

STATUS_COLORS = {
    "idle": "gray",
    "running": "blue",
    "completed": "green",
    "failed": "red",
}


def _get_module_status(module_num):
    """Read status from progress JSON file."""
    path = os.path.join(PROGRESS_DIR, f"module{module_num}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def _check_results_exist(module_num):
    """Check if result files exist for a module."""
    checks = {
        4: os.path.exists(os.path.join(RESULTS_DIR, "docking_results.csv")),
        5: os.path.exists(os.path.join(RESULTS_DIR, "screening_hits.csv")),
        6: os.path.exists(os.path.join(MODELS_DIR, "admet", "bbb_model.joblib")),
        7: os.path.exists(os.path.join(RESULTS_DIR, "selectivity_predictions.csv")),
        8: os.path.exists(os.path.join(RESULTS_DIR, "md_analysis_DCZ.csv")),
    }
    return checks.get(module_num, False)


def _time_ago(iso_str):
    """Convert ISO timestamp to 'X ago' string."""
    try:
        dt = datetime.fromisoformat(iso_str)
        diff = datetime.now() - dt
        secs = diff.total_seconds()
        if secs < 60:
            return f"{int(secs)}s ago"
        elif secs < 3600:
            return f"{int(secs / 60)}m ago"
        elif secs < 86400:
            return f"{int(secs / 3600)}h ago"
        else:
            return f"{int(secs / 86400)}d ago"
    except Exception:
        return ""


def render(props_df, compounds_df):
    st.header("Pipeline Monitor")
    st.caption("Live progress tracking for all computational modules")

    # Auto-refresh toggle
    col_refresh, col_interval = st.columns([1, 1])
    with col_refresh:
        auto_refresh = st.toggle("Auto-refresh", value=True)
    with col_interval:
        refresh_secs = st.select_slider(
            "Refresh interval",
            options=[2, 5, 10, 30],
            value=5,
        )

    if auto_refresh:
        time.sleep(0.1)  # Small delay to prevent flicker
        st.empty()

    st.divider()

    # Count running/completed
    any_running = False
    n_completed = 0

    for mod_num in [4, 5, 6, 7, 8]:
        status_data = _get_module_status(mod_num)
        has_results = _check_results_exist(mod_num)

        # Determine display status
        if status_data and status_data.get("status") == "running":
            display_status = "running"
            any_running = True
        elif status_data and status_data.get("status") == "completed":
            display_status = "completed"
            n_completed += 1
        elif status_data and status_data.get("status") == "failed":
            display_status = "failed"
        elif has_results:
            display_status = "completed"
            n_completed += 1
        else:
            display_status = "idle"

        # Module card
        with st.container(border=True):
            header_col, status_col = st.columns([3, 1])

            with header_col:
                st.subheader(f"Module {mod_num}: {MODULE_NAMES[mod_num]}")

            with status_col:
                if display_status == "running":
                    st.markdown("**:blue[RUNNING]**")
                elif display_status == "completed":
                    st.markdown("**:green[COMPLETE]**")
                elif display_status == "failed":
                    st.markdown("**:red[FAILED]**")
                else:
                    st.markdown("**:gray[IDLE]**")

            if display_status == "running" and status_data:
                # Show live progress
                step = status_data.get("step", "Processing...")
                detail = status_data.get("detail", "")
                progress = status_data.get("progress")
                total = status_data.get("total")
                updated = status_data.get("updated_at", "")

                st.info(f"**{step}**")

                if detail:
                    st.caption(detail)

                if progress is not None and total is not None and total > 0:
                    pct = progress / total
                    st.progress(pct, text=f"{progress}/{total} ({pct*100:.0f}%)")
                elif progress is not None:
                    st.caption(f"Step {progress}...")

                # Show metrics if available
                metrics = status_data.get("metrics", {})
                if metrics:
                    metric_cols = st.columns(min(len(metrics), 4))
                    for i, (k, v) in enumerate(metrics.items()):
                        with metric_cols[i % len(metric_cols)]:
                            st.metric(k, v)

                if updated:
                    st.caption(f"Last update: {_time_ago(updated)}")

            elif display_status == "completed":
                # Show completion info
                if status_data and status_data.get("metrics"):
                    metrics = status_data["metrics"]
                    metric_cols = st.columns(min(len(metrics), 4))
                    for i, (k, v) in enumerate(metrics.items()):
                        with metric_cols[i % len(metric_cols)]:
                            st.metric(k, v)

                if status_data and status_data.get("updated_at"):
                    st.caption(f"Completed {_time_ago(status_data['updated_at'])}")
                elif has_results:
                    st.caption("Results available")

            elif display_status == "failed" and status_data:
                st.error(status_data.get("detail", "Module failed"))

            else:
                st.caption("Not yet started. Run from terminal or use the pipeline runner.")

    # Summary bar
    st.divider()
    sum_cols = st.columns(3)
    with sum_cols[0]:
        st.metric("Modules Complete", f"{n_completed}/5")
    with sum_cols[1]:
        st.metric("Currently Running", "Yes" if any_running else "No")
    with sum_cols[2]:
        # Check for result files
        result_files = []
        if os.path.isdir(RESULTS_DIR):
            result_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".csv")]
        st.metric("Result Files", len(result_files))

    # Terminal-style log viewer
    with st.expander("Recent Log Output", expanded=any_running):
        log_text = ""
        for mod_num in [4, 5, 6, 7, 8]:
            status_data = _get_module_status(mod_num)
            if status_data and status_data.get("status") == "running":
                step = status_data.get("step", "")
                detail = status_data.get("detail", "")
                log_text += f"[Module {mod_num}] {step}\n"
                if detail:
                    log_text += f"  {detail}\n"
        if log_text:
            st.code(log_text, language=None)
        else:
            st.caption("No active modules.")

    # Auto-refresh via rerun
    if auto_refresh and any_running:
        time.sleep(refresh_secs)
        st.rerun()
