"""
AI-Accelerated Chemogenetic Actuator Design — Streamlit Dashboard
Entry point: streamlit run app.py
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
from src.utils.data_loader import load_actuator_properties, load_compounds_csv

st.set_page_config(
    page_title="AI-Accelerated Chemogenetic Actuator Design",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Lab-instrument dark CSS ──────────────────────────────────────────
LAB_CSS = """
<style>
/* ── Monospace data values on metrics ── */
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
    color: #00ff88 !important;
    font-size: 1.6rem;
    text-shadow: 0 0 8px rgba(0,255,136,0.3);
}
[data-testid="stMetricLabel"] {
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 0.7rem;
    color: #607080 !important;
}
[data-testid="stMetricDelta"] {
    font-family: monospace;
}

/* ── Instrument-panel borders on containers ── */
[data-testid="stVerticalBlock"] > div > [data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid #1a2332;
    background-color: #0d1117;
    border-radius: 4px;
}

/* ── LED-style button styling ── */
.stButton > button {
    background-color: #131820 !important;
    border: 1px solid #1a2332 !important;
    color: #c0c8d4 !important;
    font-family: monospace !important;
    font-size: 0.8rem !important;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    border-color: #00ff88 !important;
    box-shadow: 0 0 12px rgba(0,255,136,0.25);
    color: #00ff88 !important;
}
.stButton > button[kind="primary"],
.stButton > button.st-emotion-cache-primary {
    background-color: #0d2818 !important;
    border-color: #00ff88 !important;
    color: #00ff88 !important;
    box-shadow: 0 0 8px rgba(0,255,136,0.2);
}

/* ── Dark dataframe headers ── */
[data-testid="stDataFrame"] th {
    background-color: #131820 !important;
    color: #00ff88 !important;
    font-family: monospace !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Progress bars: green gradient + glow ── */
[data-testid="stProgress"] > div > div > div {
    background: linear-gradient(90deg, #00cc6a, #00ff88) !important;
    box-shadow: 0 0 10px rgba(0,255,136,0.4);
    border-radius: 2px;
}
[data-testid="stProgress"] > div > div {
    background-color: #1a2332 !important;
}

/* ── Dark inputs, selectboxes ── */
input, textarea, [data-baseweb="select"] {
    font-family: monospace !important;
}
input:focus, textarea:focus {
    caret-color: #00ff88 !important;
    border-color: #00ff88 !important;
}
[data-baseweb="select"] > div {
    background-color: #131820 !important;
    border-color: #1a2332 !important;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    border: 1px solid #1a2332 !important;
    background-color: #0d1117;
    border-radius: 4px;
}
[data-testid="stExpander"] summary {
    font-family: monospace;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.85rem;
}

/* ── Custom scrollbars ── */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #0a0e14;
}
::-webkit-scrollbar-thumb {
    background: #1a2332;
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: #2a3a4a;
}

/* ── Dividers ── */
hr {
    border-color: #1a2332 !important;
}

/* ── Caption / small text ── */
[data-testid="stCaption"] {
    font-family: monospace;
    color: #4a5568 !important;
}
</style>
"""

st.markdown(LAB_CSS, unsafe_allow_html=True)


@st.cache_data
def load_properties():
    return load_actuator_properties()


@st.cache_data
def load_compounds():
    return load_compounds_csv()


def main():
    props_df = load_properties()
    compounds_df = load_compounds()

    from tabs.pipeline_flow import render
    render(props_df, compounds_df)


if __name__ == "__main__":
    main()
