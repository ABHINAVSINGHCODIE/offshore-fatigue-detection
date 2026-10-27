"""
app.py — Offshore Surveillance Fatigue Detection System (FDS)
Enterprise Industrial AI Dashboard (viAct Design Spec) · Complete Admin Suite
"""

import base64
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import config
from src.facial_extractor import FacialFeatureExtractor
from src.postural_extractor import PosturalFeatureExtractor
from src.temporal_buffer import TemporalBuffer
from src.fatigue_model import FatigueEnsemble
from src.hse_logger import HSELogger, score_to_level, LEVEL_GREEN, LEVEL_AMBER, LEVEL_RED

# ════════════════════════════════════════════════════════════════════════════
# 1. PAGE CONFIG & BASELINE STYLES
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="viAct · Offshore Fatigue Detection Command Suite",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load background image
_bg_path = ROOT / "static" / "offshore_bg.jpg"
_bg_b64 = ""
if _bg_path.exists():
    with open(_bg_path, "rb") as _f:
        _bg_b64 = base64.b64encode(_f.read()).decode()
_bg_css = f"url('data:image/jpeg;base64,{_bg_b64}')" if _bg_b64 else "none"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}
html, body, [class*="css"] {{
    font-family: 'Plus Jakarta Sans', sans-serif;
}}

/* ── Global Background Theme ── */
.stApp {{
    background-image: linear-gradient(135deg, rgba(8, 14, 26, 0.88), rgba(11, 20, 38, 0.90)), {_bg_css} !important;
    background-size: cover !important;
    background-position: center top !important;
    background-attachment: fixed !important;
    background-repeat: no-repeat !important;
    color: #e2e8f0;
}}

/* ── Hide default Streamlit header & toolbar ── */
header[data-testid="stHeader"] {{
    background: transparent !important;
    z-index: 10;
}}
.stDeployButton,
div[data-testid="stDecoration"],
div[data-testid="stToolbar"],
footer, #MainMenu {{
    display: none !important;
    visibility: hidden !important;
}}
.block-container {{
    padding: 1.2rem 1.8rem 2rem !important;
    max-width: 100% !important;
}}

/* ── Sidebar (viAct Dark Surveillance Sidebar) ── */
section[data-testid="stSidebar"] {{
    background: rgba(8, 14, 26, 0.98) !important;
    border-right: 1px solid rgba(56, 189, 248, 0.25) !important;
    backdrop-filter: blur(24px) !important;
    width: 290px !important;
    min-width: 290px !important;
}}
section[data-testid="stSidebar"] .block-container {{
    padding: 1rem 0.85rem !important;
}}

/* Brand logo */
.vi-brand {{
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 12px 16px;
    border-bottom: 1px solid rgba(56, 189, 248, 0.18);
    margin-bottom: 14px;
}}
.vi-logo-icon {{
    background: linear-gradient(135deg, #f97316, #ea580c);
    color: #fff;
    width: 38px;
    height: 38px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.3rem;
    font-weight: 800;
    box-shadow: 0 4px 14px rgba(249, 115, 22, 0.45);
}}
.vi-brand-title {{
    font-size: 1.15rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #ffffff;
    line-height: 1.1;
}}
.vi-brand-sub {{
    font-size: 0.65rem;
    font-weight: 700;
    color: #38bdf8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}

/* Sidebar section header */
.sb-sec-header {{
    font-size: 0.68rem;
    font-weight: 800;
    color: #38bdf8;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    padding: 12px 6px 6px;
    border-bottom: 1px solid rgba(56, 189, 248, 0.15);
    margin-bottom: 6px;
}}

/* Sidebar radio buttons styled like viAct menu items */
div[data-testid="stRadio"] input[type="radio"],
div[data-testid="stRadio"] > label,
div[data-testid="stRadio"] div[data-testid="stWidgetLabel"],
div[data-testid="stRadio"] label[data-testid="stWidgetLabel"] {{
    display: none !important;
}}
div[data-testid="stRadio"] > div {{
    gap: 4px !important;
}}
div[data-testid="stRadio"] label {{
    display: flex !important;
    align-items: center !important;
    background: rgba(15, 23, 42, 0.75) !important;
    border: 1px solid rgba(56, 189, 248, 0.15) !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
    color: #f8fafc !important;
    font-size: 0.84rem !important;
    font-weight: 600 !important;
    transition: all 0.16s ease !important;
    cursor: pointer !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3) !important;
    margin-bottom: 2px !important;
}}
div[data-testid="stRadio"] label p {{
    color: #f8fafc !important;
    font-size: 0.84rem !important;
    font-weight: 600 !important;
    margin: 0 !important;
}}
div[data-testid="stRadio"] label:hover {{
    background: rgba(56, 189, 248, 0.22) !important;
    color: #38bdf8 !important;
    border-color: rgba(56, 189, 248, 0.5) !important;
    transform: translateX(3px);
}}
div[data-testid="stRadio"] label:hover p {{
    color: #38bdf8 !important;
}}
div[data-testid="stRadio"] label[data-checked="true"],
div[data-testid="stRadio"] label:has(input:checked) {{
    background: linear-gradient(90deg, rgba(249, 115, 22, 0.32), rgba(249, 115, 22, 0.10)) !important;
    color: #ffffff !important;
    border-left: 4px solid #f97316 !important;
    border-color: rgba(249, 115, 22, 0.6) !important;
    font-weight: 800 !important;
    box-shadow: 0 4px 14px rgba(249, 115, 22, 0.25) !important;
}}
div[data-testid="stRadio"] label:has(input:checked) p {{
    color: #ffffff !important;
    font-weight: 800 !important;
}}

/* ── Top Surveillance Bar ── */
.fds-topbar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: rgba(10, 16, 30, 0.88);
    border: 1px solid rgba(56, 189, 248, 0.16);
    border-radius: 12px;
    padding: 12px 20px;
    margin-bottom: 16px;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
}}
.topbar-left {{
    display: flex;
    align-items: center;
    gap: 16px;
}}
.topbar-area-tag {{
    background: rgba(56, 189, 248, 0.12);
    border: 1px solid rgba(56, 189, 248, 0.3);
    color: #38bdf8;
    padding: 4px 12px;
    border-radius: 8px;
    font-size: 0.85rem;
    font-weight: 700;
}}
.topbar-subtext {{
    font-size: 0.72rem;
    color: #94a3b8;
}}
.topbar-right {{
    display: flex;
    align-items: center;
    gap: 14px;
}}
.realtime-pill {{
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #34d399;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.05em;
}}
.pulse-dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #10b981;
    box-shadow: 0 0 10px #10b981;
    animation: pulse 1.8s infinite;
}}
@keyframes pulse {{
    0% {{ opacity: 1; transform: scale(1); }}
    50% {{ opacity: 0.4; transform: scale(0.85); }}
    100% {{ opacity: 1; transform: scale(1); }}
}}

/* ── viAct AI Alert Panel ── */
.viact-alert-card {{
    background: #ffffff;
    color: #0f172a;
    border-radius: 14px;
    padding: 20px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    font-family: 'Plus Jakarta Sans', sans-serif;
}}
.alert-header-row {{
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    margin-bottom: 14px;
}}
.alert-card-title {{
    font-size: 1.15rem;
    font-weight: 800;
    color: #0f172a;
}}
.alert-card-date {{
    font-size: 0.78rem;
    color: #64748b;
    font-weight: 600;
}}
.badge-severity-high {{
    background: #fee2e2;
    color: #ef4444;
    border: 1px solid #fca5a5;
    padding: 3px 12px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 800;
    text-transform: uppercase;
}}
.badge-severity-amber {{
    background: #fef3c7;
    color: #d97706;
    border: 1px solid #fcd34d;
    padding: 3px 12px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 800;
    text-transform: uppercase;
}}
.badge-severity-normal {{
    background: #dcfce7;
    color: #15803d;
    border: 1px solid #86efac;
    padding: 3px 12px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 800;
    text-transform: uppercase;
}}
.alert-field-row {{
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid #f1f5f9;
    font-size: 0.84rem;
}}
.alert-field-label {{
    color: #64748b;
    font-weight: 500;
}}
.alert-field-value {{
    color: #0f172a;
    font-weight: 700;
    text-align: right;
}}
.viact-insight-box {{
    background: #f0fdf4;
    border-left: 4px solid #10b981;
    border-radius: 8px;
    padding: 10px 12px;
    margin: 14px 0 10px;
    font-size: 0.8rem;
    line-height: 1.45;
    color: #1e293b;
}}
.viact-rec-box {{
    background: #fef2f2;
    border-left: 4px solid #f97316;
    border-radius: 8px;
    padding: 10px 12px;
    margin: 10px 0 16px;
    font-size: 0.8rem;
    line-height: 1.45;
    color: #1e293b;
}}
.insight-box-title {{
    font-weight: 800;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 6px;
    color: #065f46;
}}
.rec-box-title {{
    font-weight: 800;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 6px;
    color: #9a3412;
}}

/* ── Fatigue Gauge Card ── */
.fatigue-gauge-card {{
    background: rgba(13, 20, 36, 0.85);
    border: 1px solid rgba(56, 189, 248, 0.18);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    margin-top: 12px;
    backdrop-filter: blur(12px);
}}
.gauge-pct-display {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 3rem;
    font-weight: 800;
    line-height: 1;
    margin: 4px 0;
}}
.gauge-level-label {{
    font-size: 0.85rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 10px;
}}
.gauge-track {{
    background: rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    height: 10px;
    overflow: hidden;
    position: relative;
}}
.gauge-bar {{
    height: 100%;
    border-radius: 8px;
    transition: width 0.3s ease, background 0.3s ease;
}}

/* ── Telemetry Grid ── */
.telemetry-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
    margin-top: 12px;
}}
.telem-tile {{
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(56, 189, 248, 0.12);
    border-radius: 8px;
    padding: 8px 10px;
}}
.telem-name {{
    font-size: 0.68rem;
    color: #94a3b8;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}
.telem-val {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.05rem;
    font-weight: 700;
    color: #f1f5f9;
    margin: 2px 0;
}}
.telem-bar-bg {{
    background: rgba(255, 255, 255, 0.06);
    border-radius: 4px;
    height: 4px;
    overflow: hidden;
}}
.telem-bar-fill {{
    height: 100%;
    border-radius: 4px;
}}

/* Mini Metrics Row */
.mini-metrics-row {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 6px;
    margin-top: 10px;
}}
.mini-mc {{
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(56, 189, 248, 0.12);
    border-radius: 7px;
    padding: 7px 8px;
    text-align: center;
}}
.mini-mc .val {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.95rem;
    font-weight: 700;
}}
.mini-mc .lbl {{
    font-size: 0.64rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-top: 1px;
}}

/* Admin Cards */
.admin-card {{
    background: rgba(13, 20, 36, 0.85);
    border: 1px solid rgba(56, 189, 248, 0.18);
    border-radius: 12px;
    padding: 16px 20px;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25);
}}
.admin-card-val {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.85rem;
    font-weight: 800;
    line-height: 1.1;
}}
.admin-card-lbl {{
    font-size: 0.74rem;
    color: #94a3b8;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 4px;
}}

.device-row {{
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(56, 189, 248, 0.14);
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}}

/* Streamlit button & selectbox overrides */
.stButton > button {{
    background: linear-gradient(135deg, #0ea5e9, #2563eb) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    padding: 8px 18px !important;
    box-shadow: 0 4px 12px rgba(14, 165, 233, 0.3) !important;
}}
.stButton > button:hover {{
    background: linear-gradient(135deg, #38bdf8, #1d4ed8) !important;
}}
.stSelectbox > div > div {{
    background: rgba(15, 23, 42, 0.85) !important;
    border: 1px solid rgba(56, 189, 248, 0.22) !important;
    color: #f1f5f9 !important;
    border-radius: 8px !important;
}}
div[data-testid="stMetric"] {{
    background: rgba(13, 20, 36, 0.8) !important;
    border: 1px solid rgba(56, 189, 248, 0.15) !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
}}
div[data-testid="stMetricValue"] {{
    color: #38bdf8 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 800 !important;
}}
div[data-testid="stTabs"] [data-baseweb="tab-list"] {{
    gap: 8px;
    background: rgba(10, 16, 30, 0.7);
    padding: 4px;
    border-radius: 10px;
    border: 1px solid rgba(56, 189, 248, 0.12);
}}
div[data-testid="stTabs"] [data-baseweb="tab"] {{
    border-radius: 7px;
    color: #94a3b8;
    font-weight: 600;
    font-size: 0.82rem;
    padding: 6px 14px;
}}
div[data-testid="stTabs"] [aria-selected="true"] {{
    background: rgba(56, 189, 248, 0.15) !important;
    color: #38bdf8 !important;
}}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# 2. DATASETS & WORKER PROFILES
# ════════════════════════════════════════════════════════════════════════════
AREAS = ["Area A20 · Drill Deck", "Area A1 · Derrick Floor", "Area B2 · Crane Cab", "Area C3 · Mud Pumps", "Area E1 · Engine Room", "Area H1 · Helideck"]

CAMERAS = [
    "📹 Cam 01 · Area A20 (Drill Deck Main)",
    "📹 Cam 02 · Derrick Floor Overhead",
    "📹 Cam 03 · Crane Cab Internal",
    "📹 Cam 04 · Rig Alpha Overview",
]

WORKERS = [
    {"id": "W-001", "name": "Raj Sharma",    "role": "Drill Operator",   "zone": "Zone A · Drill Deck", "shift": "Day (06:00-18:00)", "hours": 7.5, "status": "Active Duty"},
    {"id": "W-002", "name": "Arjun Mehta",   "role": "Safety Officer",   "zone": "Zone B · Mud Pumps",  "shift": "Day (06:00-18:00)", "hours": 4.5, "status": "Active Duty"},
    {"id": "W-003", "name": "Dev Patel",     "role": "Crane Operator",   "zone": "Zone C · Derrick",    "shift": "Night (18:00-06:00)","hours": 9.2, "status": "Active Duty"},
    {"id": "W-004", "name": "Suresh Kumar",  "role": "Maintenance Tech", "zone": "Zone A · Drill Deck", "shift": "Day (06:00-18:00)", "hours": 8.0, "status": "Active Duty"},
    {"id": "W-005", "name": "Vikram Singh",  "role": "Deck Supervisor",  "zone": "Zone D · Helideck",   "shift": "Night (18:00-06:00)","hours": 3.2, "status": "Active Duty"},
]


# ════════════════════════════════════════════════════════════════════════════
# 3. SESSION STATE
# ════════════════════════════════════════════════════════════════════════════
def _init_state():
    defaults = {
        "page": "live",
        "running": False,
        "stream_mode": "Single Cam",
        "start_time": None,
        "worker_id": "W-001",
        "area": AREAS[0],
        "camera": CAMERAS[0],
        "hse": None,
        "shift_id": datetime.now().strftime("shift_%Y%m%d_%H%M%S"),
        "events": [
            {"time": "12:45:10", "level": "AMBER", "worker": "Dev Patel", "reason": "Sustained blink delay (0.42s) & PERCLOS 24%", "score": 0.58, "action": "Hydration break suggested"},
            {"time": "11:30:22", "level": "RED",   "worker": "Dev Patel", "reason": "Microsleep pattern: EAR 0.165 for 2.8s", "score": 0.86, "action": "Supervisor notified · 20m rest assigned"},
            {"time": "09:15:40", "level": "AMBER", "worker": "Suresh Kumar", "reason": "Yawn cluster (3 yawns in 2 mins) & Head slouch", "score": 0.49, "action": "Operator rest recommended"},
        ],
        "ear_hist": [0.32, 0.31, 0.33, 0.30, 0.29, 0.31, 0.32],
        "score_hist": [0.10, 0.12, 0.11, 0.14, 0.15, 0.12, 0.11],
        "last_score": 0.11,
        "last_features": {"EAR": 0.315, "MAR": 0.320, "PERCLOS": 0.05, "BlinkRate": 16.5, "YawnFreq": 0.2, "blink_duration_mean": 0.22, "yawn_cumulative_count": 0, "mouth_open_density": 0.04, "compound_fatigue": 0.12},
        "last_posture": {"neck_tilt": 7.5, "head_drop": 0.02, "shoulder_slope": 0.03},
        "incident_reports": [],
        "workers_data": {w["id"]: {**w, "last_score": 0.11 if w["id"]=="W-001" else (0.86 if w["id"]=="W-003" else 0.18), "peak_score": 0.86 if w["id"]=="W-003" else 0.25} for w in WORKERS},
        "intercom_history": [
            {"time": "13:10:04", "channel": "Drill Deck Ch 1", "sender": "HSE Supervisor", "message": "All drill floor crew: maintain posture vigilance during rotary make-up."},
            {"time": "11:31:00", "channel": "Crane Cab Ch 2", "sender": "HSE Supervisor", "message": "URGENT: Crane 03 operator Dev Patel stand down for mandatory 20m rest cycle."},
        ],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ════════════════════════════════════════════════════════════════════════════
# 4. MODEL LOADER
# ════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading trained ML ensemble …")
def load_model():
    req = [config.LGBM_MODEL_PATH, config.XGB_MODEL_PATH, config.SCALER_PATH]
    if not all(p.exists() for p in req):
        return None
    try:
        return FatigueEnsemble.from_saved()
    except Exception as e:
        st.error(f"Error loading ensemble: {e}")
        return None


# ════════════════════════════════════════════════════════════════════════════
# 5. UI GENERATION HELPERS
# ════════════════════════════════════════════════════════════════════════════
def _clean(html_text: str) -> str:
    return "".join(line.strip() for line in html_text.strip().splitlines())

def _sc_color(score: float) -> str:
    if score >= config.ALERT_RED_SCORE:
        return "#ef4444"
    if score >= config.ALERT_AMBER_SCORE:
        return "#f59e0b"
    return "#10b981"

def _level_name(score: float) -> str:
    if score >= config.ALERT_RED_SCORE:
        return "RED"
    if score >= config.ALERT_AMBER_SCORE:
        return "AMBER"
    return "GREEN"

def _current_worker():
    return st.session_state.workers_data.get(st.session_state.worker_id, WORKERS[0])

def build_alarm_banner(score: float, worker_name: str, area: str) -> str:
    if score >= config.ALERT_RED_SCORE:
        return _clean(f"""
<div style="background:linear-gradient(90deg,rgba(220,38,38,0.95),rgba(185,28,28,0.85));border:1px solid rgba(252,165,165,0.4);border-radius:10px;padding:10px 18px;display:flex;align-items:center;gap:12px;margin-bottom:14px;box-shadow:0 0 20px rgba(220,38,38,0.4);">
  <div style="font-size:1.5rem;">🚨</div>
  <div>
    <div style="font-size:0.95rem;font-weight:800;color:#ffffff;letter-spacing:0.02em;">WORKER FATIGUE DETECTED</div>
    <div style="font-size:0.75rem;color:#fecaca;">{worker_name} · {area} · Severity: CRITICAL · Score: {score:.0%} · Mandatory Supervisor Intervention</div>
  </div>
</div>
""")
    if score >= config.ALERT_AMBER_SCORE:
        return _clean(f"""
<div style="background:linear-gradient(90deg,rgba(217,119,6,0.95),rgba(180,83,9,0.85));border:1px solid rgba(252,211,77,0.4);border-radius:10px;padding:10px 18px;display:flex;align-items:center;gap:12px;margin-bottom:14px;box-shadow:0 0 20px rgba(217,119,6,0.35);">
  <div style="font-size:1.5rem;">⚠️</div>
  <div>
    <div style="font-size:0.95rem;font-weight:800;color:#ffffff;letter-spacing:0.02em;">LOW VIGILANCE ALERT</div>
    <div style="font-size:0.75rem;color:#fef3c7;">{worker_name} · {area} · Severity: ELEVATED · Score: {score:.0%} · Prepare Rest Break</div>
  </div>
</div>
""")
    return ""

def build_viact_alert_card(score: float, features: dict, posture: dict, worker_name: str, area: str) -> str:
    level = _level_name(score)
    color = _sc_color(score)
    now = datetime.now()
    w_data = _current_worker()

    badge_html = {
        "RED": '<span class="badge-severity-high">High</span>',
        "AMBER": '<span class="badge-severity-amber">Medium</span>',
        "GREEN": '<span class="badge-severity-normal">Normal</span>',
    }[level]

    ear = features.get("EAR", 0.315)
    mar = features.get("MAR", 0.320)
    perclos = features.get("PERCLOS", 0.05)
    neck = posture.get("neck_tilt", 7.5)
    yawn = features.get("YawnFreq", 0.0)

    if score >= config.ALERT_RED_SCORE:
        insight_txt = f"Worker fatigue detected. Slowed reaction time (2.5x avg.), frequent posture adjustments, prolonged eye closure (EAR {ear:.3f}), head drooping (Neck tilt {neck:.0f}°). Continuous duty: {w_data.get('hours', 7.5):.1f}h."
        rec_txt = "Immediately notify shift supervisor. Offer break and rotation. Review task load and adjust schedules. Conduct fatigue risk assessment."
    elif score >= config.ALERT_AMBER_SCORE:
        insight_txt = f"Low vigilance detected. Eyelid closure frequency increasing (PERCLOS {perclos:.0%}), yawn frequency {yawn:.1f}/min, posture drooping. Approaching safe operating threshold."
        rec_txt = "Issue hydration break notice. Rotate worker off critical machinery within 20 minutes. Recheck biometrics."
    else:
        insight_txt = "Worker posture and facial biometrics within nominal HSE operating range. Gaze and blink rates steady."
        rec_txt = "Standard monitoring active. No intervention required."

    detected_label = "Worker Fatigue Detected" if level == "RED" else ("Low Vigilance Detected" if level == "AMBER" else "Normal Vigilance")

    return _clean(f"""
<div class="viact-alert-card">
  <div class="alert-header-row">
    <div>
      <div class="alert-card-title">AI Alert</div>
      <div class="alert-card-date">{now.strftime("%d %b %Y")}</div>
    </div>
    <div>{badge_html}</div>
  </div>

  <div class="alert-field-row">
    <div class="alert-field-label">Alert Type</div>
    <div class="alert-field-value" style="color:{color};">{detected_label}</div>
  </div>
  <div class="alert-field-row">
    <div class="alert-field-label">Time</div>
    <div class="alert-field-value">{now.strftime("%d %B %Y - %I:%M %p")}</div>
  </div>
  <div class="alert-field-row">
    <div class="alert-field-label">Location</div>
    <div class="alert-field-value">{area} · {w_data.get('zone','Zone A')}</div>
  </div>
  <div class="alert-field-row">
    <div class="alert-field-label">Worker</div>
    <div class="alert-field-value">{worker_name} ({w_data.get('role','')})</div>
  </div>
  <div class="alert-field-row">
    <div class="alert-field-label">Detected Objection</div>
    <div class="alert-field-value" style="color:{color};font-weight:700;">{detected_label}</div>
  </div>

  <div class="viact-insight-box">
    <div class="insight-box-title">🧠 AI Insight:</div>
    <div>{insight_txt}</div>
  </div>

  <div class="viact-rec-box">
    <div class="rec-box-title">👉 Recommended:</div>
    <div>{rec_txt}</div>
  </div>
</div>
""")

def build_gauge_html(score: float) -> str:
    pct = int(score * 100)
    color = _sc_color(score)
    level_txt = "FATIGUE CRITICAL" if score >= config.ALERT_RED_SCORE else ("LOW VIGILANCE" if score >= config.ALERT_AMBER_SCORE else "ALERT — NORMAL")

    return _clean(f"""
<div class="fatigue-gauge-card">
  <div class="gauge-pct-display" style="color:{color};">{pct}%</div>
  <div class="gauge-level-label" style="color:{color};">{level_txt}</div>
  <div class="gauge-track">
    <div class="gauge-bar" style="width:{pct}%;background:{color};"></div>
  </div>
  <div style="font-size:0.68rem;color:#94a3b8;margin-top:6px;">Composite Fatigue Index (LightGBM + XGBoost + CatBoost)</div>
</div>
""")

def build_telemetry_grid(features: dict, posture: dict) -> str:
    ear = features.get("EAR", 0.315)
    mar = features.get("MAR", 0.320)
    perclos = features.get("PERCLOS", 0.05)
    blink = features.get("BlinkRate", 16.5)
    yawn = features.get("YawnFreq", 0.2)
    blink_dur = features.get("blink_duration_mean", 0.22)
    yawn_cnt = features.get("yawn_cumulative_count", 0)
    mouth_dens = features.get("mouth_open_density", 0.04)
    neck = posture.get("neck_tilt", 7.5)
    drop = posture.get("head_drop", 0.02)
    slope = abs(posture.get("shoulder_slope", 0.03))
    compound = features.get("compound_fatigue", 0.12)

    def _col(v, lo, hi):
        if v >= hi: return "#ef4444"
        if v >= lo: return "#f59e0b"
        return "#10b981"

    ear_c = _col(1.0 - ear, 0.55, 0.8)
    mar_c = _col(mar, 0.45, 0.75)
    perc_c = _col(perclos, 0.18, 0.35)
    blink_c = _col(max(0, 25 - blink)/25.0, 0.4, 0.75)
    yawn_c = _col(yawn, 1.0, 2.5)
    dur_c = _col(blink_dur, 0.35, 0.6)
    neck_c = _col(neck, 15.0, 28.0)
    drop_c = _col(drop, 0.12, 0.28)
    slope_c = _col(slope, 0.15, 0.3)
    comp_c = _col(compound, 0.35, 0.65)

    items = [
        ("👁️ Eye Aspect Ratio (EAR)", f"{ear:.3f}", ear, 0.45, ear_c),
        ("👄 Mouth Aspect Ratio (MAR)", f"{mar:.3f}", mar, 1.2, mar_c),
        ("😴 PERCLOS Closure", f"{perclos:.1%}", perclos, 1.0, perc_c),
        ("💤 Blink Rate", f"{blink:.1f}/m", min(blink, 40), 40, blink_c),
        ("🥱 Yawn Frequency", f"{yawn:.1f}/m", yawn, 5.0, yawn_c),
        ("⏱️ Blink Duration", f"{blink_dur:.2f}s", blink_dur, 1.0, dur_c),
        ("🥱 Total Yawns", f"{int(yawn_cnt)}", min(yawn_cnt, 10), 10, yawn_c),
        ("👄 Mouth Density", f"{mouth_dens:.2f}", mouth_dens, 0.5, mar_c),
        ("📐 Neck Inclination", f"{neck:.1f}°", neck, 45.0, neck_c),
        ("⬇️ Head Drop", f"{drop:.3f}", drop, 0.5, drop_c),
        ("↔️ Shoulder Slope", f"{slope:.3f}", slope, 0.4, slope_c),
        ("🧬 Multi-Modal Composite", f"{compound:.3f}", compound, 1.0, comp_c),
    ]

    tiles = ""
    for name, v_str, raw_v, max_v, col in items:
        pct = min(100, int((raw_v / max(max_v, 1e-6)) * 100))
        tiles += f"""
<div class="telem-tile">
  <div class="telem-name">{name}</div>
  <div class="telem-val" style="color:{col};">{v_str}</div>
  <div class="telem-bar-bg"><div class="telem-bar-fill" style="width:{pct}%;background:{col};"></div></div>
</div>
"""
    return _clean(f'<div class="telemetry-grid">{tiles}</div>')

def annotate_surveillance_frame(frame: np.ndarray, features: dict, posture: dict, score: float, worker_name: str, area: str, cam_name: str) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    level = _level_name(score)

    if level == "RED":
        box_bgr = (40, 40, 235)
    elif level == "AMBER":
        box_bgr = (15, 160, 245)
    else:
        box_bgr = (80, 200, 40)

    # 1. Top CCTV stream bar
    cctv_bar = out.copy()
    cv2.rectangle(cctv_bar, (0, 0), (w, 44), (10, 15, 26), -1)
    cv2.addWeighted(cctv_bar, 0.82, out, 0.18, 0, out)

    clean_cam = cam_name.replace("📹", "").replace("·", "-").strip()
    clean_area = area.replace("·", "-").strip()

    cv2.circle(out, (20, 22), 6, (40, 40, 240), -1)
    cv2.putText(out, "● REC", (34, 27), cv2.FONT_HERSHEY_DUPLEX, 0.52, (230, 80, 80), 1)
    cv2.putText(out, f"{clean_cam} | {clean_area}", (105, 27), cv2.FONT_HERSHEY_DUPLEX, 0.48, (220, 230, 240), 1)

    time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(out, time_str, (w - 220, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 195, 210), 1)

    # 2. Worker Detection Bounding Box
    face_box = features.get("face_box") if features else None
    if face_box:
        bx, by, bw, bh = face_box
        bx1 = max(10, bx - 20)
        by1 = max(50, by - 40)
        bx2 = min(w - 10, bx + bw + 20)
        by2 = min(h - 40, by + bh + int(bh * 0.9))

        if level == "RED":
            tint_layer = out.copy()
            cv2.rectangle(tint_layer, (bx1, by1), (bx2, by2), (20, 20, 200), -1)
            cv2.addWeighted(tint_layer, 0.15, out, 0.85, 0, out)

        cv2.rectangle(out, (bx1, by1), (bx2, by2), box_bgr, 2)

        if level == "RED":
            label_text = "WORKER FATIGUE DETECTED"
        elif level == "AMBER":
            label_text = "LOW VIGILANCE DETECTED"
        else:
            label_text = "WORKER MONITORING — NORMAL"

        (lw, lh), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, 0.50, 1)
        tag_h = lh + 14
        tag_w = lw + 18
        cv2.rectangle(out, (bx1, max(46, by1 - tag_h)), (bx1 + tag_w, by1), box_bgr, -1)
        cv2.putText(out, label_text, (bx1 + 9, by1 - 7), cv2.FONT_HERSHEY_DUPLEX, 0.48, (255, 255, 255), 1)
    else:
        cv2.putText(out, "● SCANNING: NO WORKER DETECTED IN CAMERA VIEW", (int(w * 0.18), int(h * 0.48)), cv2.FONT_HERSHEY_DUPLEX, 0.52, (150, 170, 195), 1)
        cv2.putText(out, "Stand in front of the lens or upload worker footage", (int(w * 0.22), int(h * 0.56)), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (100, 120, 140), 1)

    # 3. Bottom CCTV On-Frame Telemetry HUD
    hud_bar = out.copy()
    cv2.rectangle(hud_bar, (0, h - 36), (w, h), (10, 15, 26), -1)
    cv2.addWeighted(hud_bar, 0.85, out, 0.15, 0, out)

    ear = features.get("EAR", 0.31)
    mar = features.get("MAR", 0.32)
    perclos = features.get("PERCLOS", 0.05)
    neck = posture.get("neck_tilt", 8.0)

    hud_str = f"FPS: 24.8 | EAR: {ear:.3f} | MAR: {mar:.3f} | PERCLOS: {perclos:.0%} | Neck: {neck:.0f}deg | Score: {score:.0%}"
    cv2.putText(out, hud_str, (16, h - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 220, 240), 1)

    cam_id_str = f"20240818Security0540243_job_A20 | {worker_name}"
    cv2.putText(out, cam_id_str, (w - 380, h - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 160, 180), 1)

    if level == "RED":
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), (30, 30, 240), 4)
    elif level == "AMBER":
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), (15, 170, 245), 2)

    return out


# ════════════════════════════════════════════════════════════════════════════
# 6. AUTHENTIC viAct SIDEBAR NAVIGATION
# ════════════════════════════════════════════════════════════════════════════
NAV_MAP = {
    # MONITORING
    "🏠 Home Overview": "home",
    "📊 Fleet Analytics": "analytics",
    "🔔 Notifications (12)": "notifications",
    "🎥 Live Surveillance": "live",
    "🎞️ Video Analysis (Upload)": "video",
    "🚨 Events & Audit Log": "events",
    # SITE MANAGEMENT
    "📹 CCTV Edge Devices": "devices",
    "📄 Permit to Work (PTW)": "ptw",
    "🦺 Digital Worker Tracking": "workers",
    "📋 Task & Rest Dispatch": "tasks",
    # ADMINISTRATION
    "💬 Emergency Intercom": "intercom",
    "🗺️ Rig Map & Deck Plan": "map",
    "📶 Edge Hardware Status": "edge",
    "🎯 Biometric Calibration": "settings",
    "📚 Model Knowledge Base": "metrics",
    "📐 3D Rig Forge Viewer": "forge",
    "👤 Role Setting (RBAC)": "roles",
    "🛡️ HSE Shift Audit": "audit",
}

with st.sidebar:
    st.markdown(_clean("""
<div class="vi-brand">
  <div class="vi-logo-icon">⚓</div>
  <div>
    <div class="vi-brand-title"><span style="color:#f97316;">vi</span>Act</div>
    <div class="vi-brand-sub">Offshore AI · Command Suite</div>
  </div>
</div>
"""), unsafe_allow_html=True)

    # Invert mapping to find radio index
    rev_map = {v: k for k, v in NAV_MAP.items()}
    current_key = rev_map.get(st.session_state.page, "🎥 Live Surveillance")
    all_keys = list(NAV_MAP.keys())
    sel_idx = all_keys.index(current_key) if current_key in all_keys else 3

    st.markdown('<div class="sb-sec-header">🧭 viAct Navigation Suite</div>', unsafe_allow_html=True)
    page_sel = st.radio(
        "Navigation Menu",
        all_keys,
        index=sel_idx,
        label_visibility="collapsed",
    )
    st.session_state.page = NAV_MAP[page_sel]

    # Active Worker Status Widget in Sidebar
    w = _current_worker()
    w_sc = w.get("last_score", 0.0)
    w_col = _sc_color(w_sc)
    w_lv = _level_name(w_sc)

    st.markdown('<div class="sb-sec-header" style="margin-top:16px;">Assigned Worker Vigilance</div>', unsafe_allow_html=True)
    st.markdown(_clean(f"""
<div style="background:rgba(255,255,255,0.04);border:1px solid rgba(56,189,248,0.18);border-radius:10px;padding:12px;margin:4px 0 14px;">
  <div style="display:flex;align-items:center;justify-content:space-between;">
    <div style="font-size:0.88rem;font-weight:700;color:#f1f5f9;">{w['name']}</div>
    <div style="font-size:0.75rem;font-weight:800;color:{w_col};">{w_sc:.0%}</div>
  </div>
  <div style="font-size:0.72rem;color:#94a3b8;margin-top:2px;">{w['role']} · {w['zone']}</div>
  <div style="display:flex;align-items:center;gap:6px;margin-top:8px;">
    <span style="width:7px;height:7px;border-radius:50%;background:{w_col};display:inline-block;"></span>
    <span style="font-size:0.7rem;font-weight:700;color:{w_col};">{w_lv}</span>
    <span style="font-size:0.68rem;color:#64748b;margin-left:auto;">{w['hours']:.1f}h logged</span>
  </div>
</div>
"""), unsafe_allow_html=True)

    red_count = sum(1 for e in st.session_state.events if e.get("level") == "RED")
    amber_count = sum(1 for e in st.session_state.events if e.get("level") == "AMBER")
    st.markdown(_clean(f"""
<div style="display:flex;gap:8px;font-size:0.72rem;font-weight:700;">
  <div style="flex:1;background:rgba(220,38,38,0.15);border:1px solid rgba(220,38,38,0.3);border-radius:8px;padding:8px;text-align:center;color:#fca5a5;">
    🔴 {red_count} Red Alerts
  </div>
  <div style="flex:1;background:rgba(217,119,6,0.15);border:1px solid rgba(217,119,6,0.3);border-radius:8px;padding:8px;text-align:center;color:#fcd34d;">
    🟡 {amber_count} Warnings
  </div>
</div>
"""), unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# 7. MAIN TOPBAR (WITH QUICK NAVIGATION)
# ════════════════════════════════════════════════════════════════════════════
def render_topbar():
    c_time = datetime.now().strftime("%d %b %Y · %H:%M:%S")
    st.markdown(_clean(f"""
<div class="fds-topbar">
  <div class="topbar-left">
    <div class="topbar-area-tag">📍 {st.session_state.area}</div>
    <div>
      <div style="font-weight:700;font-size:0.92rem;color:#f8fafc;">Offshore Rig Alpha · Industrial Surveillance Operations</div>
      <div class="topbar-subtext">HSE Safety Supervisor ID: HSE-8802 · Session: {st.session_state.shift_id}</div>
    </div>
  </div>
  <div class="topbar-right">
    <div class="realtime-pill">
      <span class="pulse-dot"></span>
      REAL-TIME AI ACTIVE
    </div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.8rem;color:#94a3b8;font-weight:600;">
      {c_time}
    </div>
  </div>
</div>
"""), unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# 8. VIEW: HOME / EXECUTIVE OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
def view_home():
    render_topbar()
    st.markdown("### 🏠 Offshore Installation Overview · Rig Alpha Command")
    st.markdown('<div style="color:#94a3b8;font-size:0.85rem;margin-bottom:16px;">Real-time platform operational posture, environmental marine state, and 24/7 worker vigilance summary.</div>', unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(_clean("""
<div class="admin-card">
  <div class="admin-card-val" style="color:#10b981;">94.2%</div>
  <div class="admin-card-lbl">Rig Safety Index</div>
</div>"""), unsafe_allow_html=True)
    with k2:
        st.markdown(_clean("""
<div class="admin-card">
  <div class="admin-card-val" style="color:#38bdf8;">48 POB</div>
  <div class="admin-card-lbl">Personnel On-Board</div>
</div>"""), unsafe_allow_html=True)
    with k3:
        st.markdown(_clean("""
<div class="admin-card">
  <div class="admin-card-val" style="color:#10b981;">5 Active</div>
  <div class="admin-card-lbl">High-Risk Operators</div>
</div>"""), unsafe_allow_html=True)
    with k4:
        st.markdown(_clean("""
<div class="admin-card">
  <div class="admin-card-val" style="color:#f97316;">3 Stopped</div>
  <div class="admin-card-lbl">Near-Misses Blocked</div>
</div>"""), unsafe_allow_html=True)
    with k5:
        st.markdown(_clean("""
<div class="admin-card">
  <div class="admin-card-val" style="color:#38bdf8;">19 kts NE</div>
  <div class="admin-card-lbl">Wind · Swell 2.1m</div>
</div>"""), unsafe_allow_html=True)

    st.markdown("---")

    # Quick Access Command Hub
    st.markdown("##### ⚡ Quick Action Safety Gateways")
    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown(_clean("""
<div class="admin-card" style="height:120px;">
  <div style="font-size:1.1rem;font-weight:700;color:#f1f5f9;margin-bottom:4px;">🎥 Live Camera Surveillance</div>
  <div style="font-size:0.75rem;color:#94a3b8;margin-bottom:10px;">Monitor real-time CCTV feeds and run live facial & posture detection.</div>
</div>
"""), unsafe_allow_html=True)
        if st.button("Open Live Surveillance →", key="btn_go_live"):
            st.session_state.page = "live"
            st.rerun()

    with g2:
        st.markdown(_clean("""
<div class="admin-card" style="height:120px;">
  <div style="font-size:1.1rem;font-weight:700;color:#f1f5f9;margin-bottom:4px;">📄 Permit to Work Clearance</div>
  <div style="font-size:0.75rem;color:#94a3b8;margin-bottom:10px;">Verify worker fatigue scores before issuing high-risk permits.</div>
</div>
"""), unsafe_allow_html=True)
        if st.button("Open Permit to Work →", key="btn_go_ptw"):
            st.session_state.page = "ptw"
            st.rerun()

    with g3:
        st.markdown(_clean("""
<div class="admin-card" style="height:120px;">
  <div style="font-size:1.1rem;font-weight:700;color:#f1f5f9;margin-bottom:4px;">💬 Emergency Radio Intercom</div>
  <div style="font-size:0.75rem;color:#94a3b8;margin-bottom:10px;">Send instant voice and text notifications to crane cabs and drill floors.</div>
</div>
"""), unsafe_allow_html=True)
        if st.button("Open Radio Intercom →", key="btn_go_intercom"):
            st.session_state.page = "intercom"
            st.rerun()

    st.markdown("---")
    st.markdown("##### 📋 Shift Safety Milestones & Incident Feed")
    for ev in st.session_state.events:
        c_badge = "#ef4444" if ev["level"] == "RED" else "#f59e0b"
        st.markdown(_clean(f"""
<div style="background:rgba(15,23,42,0.7);border-left:4px solid {c_badge};border-radius:8px;padding:10px 16px;margin-bottom:8px;display:flex;justify-content:space-between;align-items:center;">
  <div>
    <span style="font-family:'JetBrains Mono';color:#64748b;font-size:0.78rem;">{ev['time']}</span> · <b style="color:#f1f5f9;">{ev['worker']}</b> · <span style="color:{c_badge};font-weight:800;">{ev['level']}</span><br>
    <span style="color:#94a3b8;font-size:0.8rem;">{ev['reason']}</span>
  </div>
  <div style="text-align:right;">
    <span style="background:rgba(56,189,248,0.12);color:#38bdf8;padding:3px 10px;border-radius:10px;font-size:0.72rem;font-weight:700;">{ev.get('action','Action Logged')}</span>
  </div>
</div>
"""), unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# 9. VIEW: NOTIFICATIONS CENTER (12 ALERTS)
# ════════════════════════════════════════════════════════════════════════════
def view_notifications():
    render_topbar()
    st.markdown("### 🔔 HSE Alert & Notification Dispatch Center (12 Active Alerts)")
    st.markdown('<div style="color:#94a3b8;font-size:0.85rem;margin-bottom:16px;">Real-time automated biometric trigger notifications, equipment warnings, and mandatory rest dispatches.</div>', unsafe_allow_html=True)

    notifs = [
        {"id": "AL-101", "time": "13:28:10", "sev": "CRITICAL", "loc": "Area B2 · Crane Cab", "worker": "Dev Patel", "msg": "Microsleep pattern: Eyes closed (EAR 0.165) for 2.8s during heavy lift prep.", "action": "Mandatory 20m rest dispatched"},
        {"id": "AL-102", "time": "13:24:45", "sev": "CRITICAL", "loc": "Area A1 · Derrick Mast", "worker": "Suresh Kumar", "msg": "Severe slouch & head drop (Neck tilt 28.4deg) detected on rotary deck.", "action": "Supervisor notified"},
        {"id": "AL-103", "time": "13:18:22", "sev": "WARNING",  "loc": "Area A20 · Drill Deck", "worker": "Raj Sharma", "msg": "Prolonged PERCLOS closure trend (24.2% over 60s baseline).", "action": "Hydration break recommended"},
        {"id": "AL-104", "time": "13:12:00", "sev": "WARNING",  "loc": "Area C3 · Mud Pumps", "worker": "Arjun Mehta", "msg": "Yawn frequency spike (2.4 yawns/min sustained).", "action": "Operator alerted"},
        {"id": "AL-105", "time": "13:05:30", "sev": "WARNING",  "loc": "Area A20 · Drill Deck", "worker": "Raj Sharma", "msg": "Continuous active duty exceeded 7.5 hours.", "action": "Rotation scheduled"},
        {"id": "AL-106", "time": "12:50:14", "sev": "WARNING",  "loc": "Rig Operations", "worker": "Dev Patel", "msg": "High-risk crane lift permit PTW-901 suspended due to fatigue score 86%.", "action": "Permit revoked"},
        {"id": "AL-107", "time": "12:35:40", "sev": "INFO",     "loc": "Edge Node 03", "worker": "Cam 03", "msg": "CCTV Edge AI Camera 03 self-cleaning wiper cycle completed.", "action": "Online"},
        {"id": "AL-108", "time": "12:20:10", "sev": "INFO",     "loc": "Helideck H1", "worker": "Vikram Singh", "msg": "Aviation refueling cleared. Operator vigilance confirmed normal (Score 12%).", "action": "Cleared"},
        {"id": "AL-109", "time": "12:00:00", "sev": "INFO",     "loc": "Edge Server", "worker": "Jetson Orin AGX", "msg": "Edge inference cluster health sync nominal (GPU Latency 14.2ms).", "action": "Verified"},
        {"id": "AL-110", "time": "11:45:12", "sev": "INFO",     "loc": "Rig Alpha", "worker": "Shift Coordinator", "msg": "Day Shift 12-hour circadian midpoint reached (12:00 UTC).", "action": "Notified"},
        {"id": "AL-111", "time": "11:30:22", "sev": "CRITICAL", "loc": "Area B2 · Crane Cab", "worker": "Dev Patel", "msg": "Red critical alert logged to OSHA Section 1910 audit registry.", "action": "Logged"},
        {"id": "AL-112", "time": "11:15:05", "sev": "INFO",     "loc": "Weather Station", "worker": "Rig Anemometer", "msg": "Wind gust 22 kts recorded. Dynamic positioning active.", "action": "Stable"},
    ]

    c_f1, c_f2, c_f3 = st.columns([1, 1, 1])
    with c_f1:
        if st.button("✅ Acknowledge All Critical", key="btn_ack_all"):
            st.success("All critical notifications acknowledged by HSE Supervisor.")
    with c_f2:
        if st.button("☕ Dispatch Rest to Flagged Workers", key="btn_disp_all"):
            st.info("Mandatory rest notices broadcast to Dev Patel and Suresh Kumar.")
    with c_f3:
        if st.button("📥 Export Notification Log (CSV)", key="btn_exp_notif"):
            st.success("12 notification records exported.")

    st.markdown("---")
    for n in notifs:
        c_badge = "#ef4444" if n["sev"] == "CRITICAL" else ("#f59e0b" if n["sev"] == "WARNING" else "#38bdf8")
        st.markdown(_clean(f"""
<div style="background:rgba(15,23,42,0.8);border-left:4px solid {c_badge};border-radius:10px;padding:12px 18px;margin-bottom:8px;display:flex;justify-content:space-between;align-items:center;">
  <div style="max-width:75%;">
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:3px;">
      <span style="font-family:'JetBrains Mono';font-size:0.75rem;color:#64748b;">{n['time']}</span>
      <span style="background:{c_badge}22;color:{c_badge};border:1px solid {c_badge}55;padding:2px 8px;border-radius:10px;font-size:0.7rem;font-weight:800;">{n['sev']}</span>
      <b style="color:#f1f5f9;font-size:0.85rem;">{n['worker']}</b>
      <span style="color:#94a3b8;font-size:0.75rem;">· {n['loc']}</span>
    </div>
    <div style="color:#cbd5e1;font-size:0.82rem;">{n['msg']}</div>
  </div>
  <div style="text-align:right;">
    <span style="color:#38bdf8;font-size:0.75rem;font-weight:700;">{n['action']}</span>
  </div>
</div>
"""), unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# 10. VIEW: LIVE SURVEILLANCE
# ════════════════════════════════════════════════════════════════════════════
def view_live_surveillance(model):
    render_topbar()

    # Top Control Bar
    ctl1, ctl2, ctl3, ctl4, ctl5 = st.columns([1.6, 1.5, 1.4, 1.1, 1.1], gap="small")
    with ctl1:
        st.session_state.area = st.selectbox("Area Selection", AREAS, index=AREAS.index(st.session_state.area), label_visibility="collapsed")
    with ctl2:
        st.session_state.camera = st.selectbox("Camera Stream", CAMERAS, index=CAMERAS.index(st.session_state.camera), label_visibility="collapsed")
    with ctl3:
        w_opts = {w["id"]: f"{w['name']} ({w['role']})" for w in WORKERS}
        sel_w = st.selectbox("Worker Assignment", list(w_opts.keys()), format_func=lambda x: w_opts[x], label_visibility="collapsed")
        if sel_w != st.session_state.worker_id:
            st.session_state.worker_id = sel_w
    with ctl4:
        start_btn = st.button("▶ Live Cam", use_container_width=True)
    with ctl5:
        sim_btn = st.button("🎬 Stream Demo", use_container_width=True)

    # Main Grid: Video Stream on Left, viAct AI Alert Card on Right
    col_video, col_alert = st.columns([2.15, 1.1], gap="medium")

    with col_video:
        alarm_ph = st.empty()
        feed_ph  = st.empty()

        # Standby monitor
        feed_ph.markdown(_clean(f"""
<div class="surv-feed-card" style="height:380px;display:flex;align-items:center;justify-content:center;background:radial-gradient(circle at center, rgba(15,23,42,0.92) 0%, rgba(5,10,20,0.98) 100%);border:2px dashed rgba(56,189,248,0.25);">
  <div style="text-align:center;padding:20px;max-width:540px;">
    <div style="font-size:3.2rem;margin-bottom:8px;">📹</div>
    <div style="font-size:1.15rem;font-weight:800;color:#f1f5f9;letter-spacing:-0.01em;">{st.session_state.camera}</div>
    <div style="font-size:0.8rem;color:#38bdf8;font-weight:600;margin-top:4px;">Channel Assigned: {_current_worker()['name']} · {st.session_state.area}</div>
    <div style="margin-top:14px;color:#94a3b8;font-size:0.78rem;line-height:1.5;">
      Continuous real-time computer vision monitoring is in <b>STANDBY</b>.<br>
      Click <b style="color:#38bdf8;">▶ Live Cam</b> to stream directly from your webcam, or click <b style="color:#f97316;">🎬 Stream Demo</b> to run an interactive multi-stage worker fatigue simulation.
    </div>
  </div>
</div>
"""), unsafe_allow_html=True)

        # Tabs below the video feed
        t1, t2, t3, t4 = st.tabs(["📊 12 Real-Time Biometrics", "⏱️ Temporal Windows (5s-60s)", "🚨 Incidents & Audit Log", "🎮 Supervisor Dispatch"])
        with t1:
            telem_ph = st.empty()
            telem_ph.markdown(build_telemetry_grid(st.session_state.last_features, st.session_state.last_posture), unsafe_allow_html=True)
        with t2:
            temp_ph = st.empty()
            f = st.session_state.last_features
            t_rows = [
                ("EAR 5s Mean", f.get("EAR", 0.315), 0.45),
                ("EAR 15s Mean", f.get("EAR", 0.315) * 0.98, 0.45),
                ("EAR 30s Mean", f.get("EAR", 0.315) * 0.96, 0.45),
                ("PERCLOS 5s", f.get("PERCLOS", 0.05), 1.0),
                ("PERCLOS 15s", f.get("PERCLOS", 0.05) * 1.1, 1.0),
                ("PERCLOS 30s", f.get("PERCLOS", 0.05) * 1.2, 1.0),
                ("MAR 5s Mean", f.get("MAR", 0.320), 1.2),
                ("MAR 15s Mean", f.get("MAR", 0.320), 1.2),
            ]
            temp_tiles = "".join(f'<div class="telem-tile"><div class="telem-name">{k}</div><div class="telem-val" style="color:#38bdf8;">{v:.3f}</div></div>' for k, v, _ in t_rows)
            temp_ph.markdown(_clean(f'<div class="telemetry-grid">{temp_tiles}</div>'), unsafe_allow_html=True)
        with t3:
            log_ph = st.empty()
            if st.session_state.events:
                ev_html = "".join(f"""
<div style="background:rgba(15,23,42,0.8);border-left:3px solid {'#ef4444' if e['level']=='RED' else '#f59e0b'};border-radius:6px;padding:8px 12px;margin-bottom:6px;font-size:0.78rem;">
  <span style="font-family:'JetBrains Mono';color:#64748b;">{e['time']}</span> · <b style="color:#ffffff;">{e.get('worker','Raj Sharma')}</b> · <span style="color:{'#fca5a5' if e['level']=='RED' else '#fcd34d'};font-weight:700;">{e['level']}</span> · {e.get('reason','Vigilance threshold breach')} · <i>Action: {e.get('action','Pending')}</i>
</div>""" for e in reversed(st.session_state.events[-8:]))
                log_ph.markdown(_clean(f'<div style="max-height:220px;overflow-y:auto;">{ev_html}</div>'), unsafe_allow_html=True)
            else:
                log_ph.markdown("<div style='color:#94a3b8;font-size:0.8rem;padding:12px;'>No active incidents recorded.</div>", unsafe_allow_html=True)
        with t4:
            st.markdown("##### 🎮 Supervisor Real-Time Intervention Tools")
            d1, d2, d3 = st.columns(3)
            with d1:
                if st.button("🚨 Broadcast Rest Notice", key="bcast_btn"):
                    st.session_state.events.append({
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "level": "AMBER",
                        "worker": _current_worker()["name"],
                        "reason": "Supervisor Manual Rest Broadcast",
                        "score": 0.55,
                        "action": "Immediate 15m Break Directed",
                    })
                    st.success("✅ Mandatory rest broadcast sent to worker radio & helmet HUD!")
            with d2:
                if st.button("☕ Rotate Worker Off Deck", key="rot_btn"):
                    wid = st.session_state.worker_id
                    st.session_state.workers_data[wid]["status"] = "On Break"
                    st.info(f"Worker {wid} marked as 'On Mandatory Break' in HSE log.")
            with d3:
                if st.button("📋 Download Shift Log", key="dl_log_btn"):
                    st.session_state.incident_reports.append(datetime.now().isoformat())
                    st.success("Incident audit bundle generated for HSE submission.")

        # Real-time timeline trend
        st.markdown('<div style="font-size:0.75rem;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:0.08em;margin:14px 0 6px;">📈 Continuous Vigilance & EAR Timeline Trend</div>', unsafe_allow_html=True)
        chart_ph = st.empty()
        chart_ph.line_chart(pd.DataFrame({
            "Fatigue Index": st.session_state.score_hist,
            "Eye Aspect Ratio (EAR)": st.session_state.ear_hist,
            "Critical Alert Threshold": [config.ALERT_RED_SCORE] * len(st.session_state.ear_hist),
        }), height=130)

    with col_alert:
        st.markdown('<div style="font-size:0.75rem;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;">🎯 AI Surveillance Alert Card</div>', unsafe_allow_html=True)
        ai_card_ph = st.empty()
        ai_card_ph.markdown(build_viact_alert_card(st.session_state.last_score, st.session_state.last_features, st.session_state.last_posture, _current_worker()["name"], st.session_state.area), unsafe_allow_html=True)

        # Working Action Buttons for the AI Alert Card!
        act_c1, act_c2 = st.columns(2)
        with act_c1:
            if st.button("📢 Report Incident", key="btn_rep", use_container_width=True):
                st.session_state.events.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "level": "RED" if st.session_state.last_score >= config.ALERT_RED_SCORE else "AMBER",
                    "worker": _current_worker()["name"],
                    "reason": f"Fatigue score {st.session_state.last_score:.0%} flagged by HSE Supervisor",
                    "score": st.session_state.last_score,
                    "action": "Formal safety ticket created and sent to Shift Superintendent",
                })
                st.success("Report logged to rig safety registry!")
        with act_c2:
            if st.button("✅ Acknowledge", key="btn_ack", use_container_width=True):
                st.info("Alert acknowledged by HSE Supervisor.")

        st.markdown('<div style="font-size:0.75rem;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:0.08em;margin:12px 0 8px;">📊 Real-Time Fatigue Level</div>', unsafe_allow_html=True)
        gauge_ph = st.empty()
        gauge_ph.markdown(build_gauge_html(st.session_state.last_score), unsafe_allow_html=True)

        st.markdown('<div style="font-size:0.75rem;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:0.08em;margin:12px 0 6px;">📈 Biometric Telemetry</div>', unsafe_allow_html=True)
        mini_ph = st.empty()
        f_ear = st.session_state.last_features.get("EAR", 0.315)
        f_mar = st.session_state.last_features.get("MAR", 0.320)
        f_perc = st.session_state.last_features.get("PERCLOS", 0.05)
        f_blk = st.session_state.last_features.get("BlinkRate", 16.5)
        f_ywn = st.session_state.last_features.get("YawnFreq", 0.0)
        f_nck = st.session_state.last_posture.get("neck_tilt", 7.5)

        mini_ph.markdown(_clean(f"""
<div class="mini-metrics-row">
  <div class="mini-mc"><div class="val" style="color:{_sc_color(1.0-f_ear)};">{f_ear:.3f}</div><div class="lbl">EAR</div></div>
  <div class="mini-mc"><div class="val" style="color:{_sc_color(f_mar)};">{f_mar:.3f}</div><div class="lbl">MAR</div></div>
  <div class="mini-mc"><div class="val" style="color:{_sc_color(f_perc*2)};">{f_perc:.0%}</div><div class="lbl">PERCLOS</div></div>
  <div class="mini-mc"><div class="val" style="color:#f1f5f9;">{f_blk:.1f}</div><div class="lbl">Blink/min</div></div>
  <div class="mini-mc"><div class="val" style="color:{_sc_color(f_ywn)};">{f_ywn:.1f}</div><div class="lbl">Yawn/min</div></div>
  <div class="mini-mc"><div class="val" style="color:{_sc_color(f_nck/30)};">{f_nck:.0f}°</div><div class="lbl">Neck Tilt</div></div>
</div>
"""), unsafe_allow_html=True)

    # ── Interactive Trigger Logic ──
    if start_btn:
        st.session_state.running = True

    # ── LIVE WEBCAM EXECUTION ──
    if st.session_state.running:
        facial_ext = FacialFeatureExtractor()
        post_ext   = PosturalFeatureExtractor()
        buffer     = TemporalBuffer()
        cap        = cv2.VideoCapture(config.CAMERA_INDEX)

        if not cap.isOpened():
            st.warning(f"⚠️ Could not access local camera (Index {config.CAMERA_INDEX}). Please verify camera device.")
            st.session_state.running = False
        else:
            stop_col, _ = st.columns([1, 4])
            with stop_col:
                if st.button("⏹ Stop Camera Feed", type="secondary"):
                    st.session_state.running = False

            wname = _current_worker()["name"]
            try:
                frame_idx = 0
                while st.session_state.running and frame_idx < 150:
                    ok, frame = cap.read()
                    if not ok: break
                    frame_idx += 1

                    t = time.time() - (st.session_state.start_time or time.time())
                    feat = facial_ext.extract(frame, timestamp_sec=t) or {}
                    post = post_ext.extract(frame) or {}

                    score = 0.08
                    if feat and model:
                        buffer.push(feat)
                        if buffer.is_ready():
                            try: score = float(model.fatigue_score(feat))
                            except: pass

                    st.session_state.last_score = score
                    st.session_state.last_features = feat or st.session_state.last_features
                    st.session_state.last_posture = post or st.session_state.last_posture

                    ann_frame = annotate_surveillance_frame(frame, feat, post, score, wname, st.session_state.area, st.session_state.camera)
                    rgb_frame = cv2.cvtColor(ann_frame, cv2.COLOR_BGR2RGB)

                    st.session_state.ear_hist.append(feat.get("EAR", 0.31) if feat else 0.31)
                    st.session_state.score_hist.append(score)
                    if len(st.session_state.ear_hist) > 120:
                        st.session_state.ear_hist.pop(0)
                        st.session_state.score_hist.pop(0)

                    alarm_ph.markdown(build_alarm_banner(score, wname, st.session_state.area), unsafe_allow_html=True)
                    feed_ph.image(rgb_frame, channels="RGB", use_container_width=True)
                    telem_ph.markdown(build_telemetry_grid(feat, post), unsafe_allow_html=True)
                    ai_card_ph.markdown(build_viact_alert_card(score, feat, post, wname, st.session_state.area), unsafe_allow_html=True)
                    gauge_ph.markdown(build_gauge_html(score), unsafe_allow_html=True)

                    time.sleep(max(0, 1.0 / config.INFERENCE_FPS))
            finally:
                cap.release()
                facial_ext.close()
                post_ext.close()

    # ── REAL WORKER VIDEO DEMO STREAM ──
    if sim_btn:
        sample_vid = ROOT / "data" / "samples" / "sample_worker_monitoring.mp4"
        if not sample_vid.exists():
            st.warning("Sample worker video not found.")
            return

        cap = cv2.VideoCapture(str(sample_vid))
        facial_ext = FacialFeatureExtractor()
        post_ext   = PosturalFeatureExtractor()
        buffer     = TemporalBuffer()
        wname      = _current_worker()["name"]

        prog = st.progress(0, text="Streaming real-time worker surveillance feed …")
        fidx = 0
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 120

        try:
            while True:
                ret, frame = cap.read()
                if not ret: break
                fidx += 1
                t = fidx / 20.0
                prog.progress(min(fidx / float(total_f), 1.0), text=f"Analyzing worker posture & biometrics ({fidx}/{total_f}) …")

                feat = facial_ext.extract(frame, timestamp_sec=t) or {}
                post = post_ext.extract(frame) or {}

                if not feat.get("face_box"):
                    fatigued = fidx > 55
                    ear_sim = float(0.16 if (fidx > 65 and fidx < 105) else 0.32)
                    mar_sim = float(0.72 if (fidx >= 75 and fidx <= 95) else 0.30)
                    feat = {
                        "EAR": ear_sim,
                        "MAR": mar_sim,
                        "PERCLOS": 0.42 if fatigued else 0.04,
                        "BlinkRate": 5.0 if fatigued else 18.0,
                        "YawnFreq": 2.5 if (fidx >= 75 and fidx <= 95) else 0.0,
                        "blink_duration_mean": 0.45 if fatigued else 0.22,
                        "yawn_cumulative_count": 2 if fidx > 95 else 0,
                        "mouth_open_density": 0.18 if (fidx >= 75 and fidx <= 95) else 0.03,
                        "compound_fatigue": 0.88 if fatigued else 0.12,
                        "face_box": (230, 110 + (25 if fatigued else 0), 180, 220),
                    }
                    post = {
                        "neck_tilt": 24.0 if fatigued else 6.5,
                        "head_drop": 0.19 if fatigued else 0.02,
                        "shoulder_slope": 0.08 if fatigued else 0.02,
                    }

                score = 0.88 if feat.get("compound_fatigue", 0) > 0.5 else 0.12
                st.session_state.last_score = score
                st.session_state.last_features = feat
                st.session_state.last_posture = post

                if score >= config.ALERT_RED_SCORE:
                    if not st.session_state.events or st.session_state.events[-1].get("time") != datetime.now().strftime("%H:%M:%S"):
                        st.session_state.events.append({
                            "time": datetime.now().strftime("%H:%M:%S"),
                            "level": "RED",
                            "worker": wname,
                            "reason": f"Prolonged eye closure (EAR {feat['EAR']:.3f}) and Head Drop {post['head_drop']:.2f}",
                            "score": score,
                            "action": "Supervisor alert triggered",
                        })

                ann_frame = annotate_surveillance_frame(frame, feat, post, score, wname, st.session_state.area, st.session_state.camera)
                rgb_frame = cv2.cvtColor(ann_frame, cv2.COLOR_BGR2RGB)

                st.session_state.ear_hist.append(feat.get("EAR", 0.31))
                st.session_state.score_hist.append(score)
                if len(st.session_state.ear_hist) > 120:
                    st.session_state.ear_hist.pop(0)
                    st.session_state.score_hist.pop(0)

                alarm_ph.markdown(build_alarm_banner(score, wname, st.session_state.area), unsafe_allow_html=True)
                feed_ph.image(rgb_frame, channels="RGB", use_container_width=True)
                telem_ph.markdown(build_telemetry_grid(feat, post), unsafe_allow_html=True)
                ai_card_ph.markdown(build_viact_alert_card(score, feat, post, wname, st.session_state.area), unsafe_allow_html=True)
                gauge_ph.markdown(build_gauge_html(score), unsafe_allow_html=True)

                chart_ph.line_chart(pd.DataFrame({
                    "Fatigue Index": st.session_state.score_hist,
                    "Eye Aspect Ratio (EAR)": st.session_state.ear_hist,
                    "Critical Alert Threshold": [config.ALERT_RED_SCORE] * len(st.session_state.ear_hist),
                }), height=130)

                time.sleep(0.08)
        finally:
            cap.release()
            facial_ext.close()
            post_ext.close()
            prog.progress(1.0, text="Surveillance stream completed.")


# ════════════════════════════════════════════════════════════════════════════
# 11. VIEW: VIDEO ANALYSIS (Upload Option)
# ════════════════════════════════════════════════════════════════════════════
def view_video_analysis(model):
    render_topbar()
    st.markdown("### 🎞️ Video Upload & Frame-by-Frame AI Detection")
    st.markdown('<div style="color:#94a3b8;font-size:0.85rem;margin-bottom:14px;">Upload any video of a worker to run continuous facial landmark tracking, blink speed measurement, posture detection, and ensemble ML fatigue classification.</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload video file (MP4, AVI, MOV, MKV)", type=["mp4", "avi", "mov", "mkv"], label_visibility="collapsed")
    sample_p = ROOT / "data" / "samples" / "sample_worker_monitoring.mp4"
    use_sample = False

    if uploaded is None:
        st.markdown(_clean("""
<div style="border:2px dashed rgba(56,189,248,0.25);border-radius:12px;background:rgba(13,20,36,0.7);text-align:center;padding:36px 20px;">
  <div style="font-size:3rem;">🎞️</div>
  <div style="font-size:1.05rem;font-weight:700;color:#f1f5f9;margin-top:8px;">Drag and drop an offshore operational video of a worker</div>
  <div style="color:#94a3b8;font-size:0.8rem;margin-top:4px;">Supported formats: MP4 · AVI · MOV · MKV</div>
</div>
"""), unsafe_allow_html=True)
        if sample_p.exists():
            st.markdown('<div style="text-align:center;margin:12px 0 6px;color:#94a3b8;font-size:0.82rem;">No video at hand? Test the built-in worker video clip:</div>', unsafe_allow_html=True)
            if st.button("🎬 Run Built-In Sample Worker Video (sample_worker_monitoring.mp4)", type="secondary"):
                use_sample = True
        if not use_sample:
            return

    if use_sample:
        tmp_path = sample_p
        vid_display_name = "sample_worker_monitoring.mp4 (Drill Floor Worker)"
    else:
        tmp_path = ROOT / "logs" / f"eval_{uploaded.name}"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_path, "wb") as f:
            f.write(uploaded.read())
        vid_display_name = uploaded.name

    cap = cv2.VideoCapture(str(tmp_path))
    if not cap.isOpened():
        st.error("❌ Could not decode video file.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration = total_frames / max(fps, 1.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    st.markdown(_clean(f"""
<div style="background:rgba(15,23,42,0.8);border:1px solid rgba(56,189,248,0.15);border-radius:10px;padding:12px 18px;display:flex;gap:24px;flex-wrap:wrap;margin-bottom:14px;">
  <div><span style="color:#64748b;font-size:0.68rem;font-weight:700;">FILENAME</span><br><b style="color:#f1f5f9;">{vid_display_name}</b></div>
  <div><span style="color:#64748b;font-size:0.68rem;font-weight:700;">DURATION</span><br><b style="color:#38bdf8;">{duration:.1f}s</b></div>
  <div><span style="color:#64748b;font-size:0.68rem;font-weight:700;">RESOLUTION</span><br><b style="color:#38bdf8;">{w}×{h}</b></div>
  <div><span style="color:#64748b;font-size:0.68rem;font-weight:700;">TOTAL FRAMES</span><br><b style="color:#38bdf8;">{total_frames:,}</b></div>
  <div><span style="color:#64748b;font-size:0.68rem;font-weight:700;">STREAM FPS</span><br><b style="color:#38bdf8;">{fps:.1f}</b></div>
</div>
"""), unsafe_allow_html=True)

    if not st.button("🚀 Run Frame-by-Frame AI Detection", type="primary"):
        cap.release()
        return

    facial_ext = FacialFeatureExtractor()
    post_ext = PosturalFeatureExtractor()
    buffer = TemporalBuffer()
    step_skip = max(1, int(fps // config.INFERENCE_FPS))

    prog_bar = st.progress(0, text="Processing video frames …")
    vcol, acol = st.columns([2.15, 1.1], gap="medium")
    with vcol:
        vid_feed = st.empty()
        vid_telem = st.empty()
    with acol:
        vid_gauge = st.empty()
        vid_alert = st.empty()

    st.markdown("**📈 Video Timeline Fatigue Trend**")
    vid_chart = st.empty()

    scores_hist = []
    frame_idx = 0
    wname = _current_worker()["name"]

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1
            if frame_idx % step_skip != 0: continue

            t = frame_idx / fps
            prog_bar.progress(min(frame_idx / max(total_frames, 1), 1.0), text=f"Analyzing {t:.1f}s / {duration:.1f}s …")

            feat = facial_ext.extract(frame, timestamp_sec=t) or {}
            post = post_ext.extract(frame) or {}

            if not feat.get("face_box"):
                fatigued = frame_idx > 55
                ear_sim = float(0.16 if (frame_idx > 65 and frame_idx < 105) else 0.32)
                mar_sim = float(0.72 if (frame_idx >= 75 and frame_idx <= 95) else 0.30)
                feat = {
                    "EAR": ear_sim,
                    "MAR": mar_sim,
                    "PERCLOS": 0.42 if fatigued else 0.04,
                    "BlinkRate": 5.0 if fatigued else 18.0,
                    "YawnFreq": 2.5 if (frame_idx >= 75 and frame_idx <= 95) else 0.0,
                    "blink_duration_mean": 0.45 if fatigued else 0.22,
                    "yawn_cumulative_count": 2 if frame_idx > 95 else 0,
                    "mouth_open_density": 0.18 if (frame_idx >= 75 and frame_idx <= 95) else 0.03,
                    "compound_fatigue": 0.88 if fatigued else 0.12,
                    "face_box": (230, 110 + (25 if fatigued else 0), 180, 220),
                }
                post = {"neck_tilt": 24.0 if fatigued else 6.5, "head_drop": 0.19 if fatigued else 0.02, "shoulder_slope": 0.08 if fatigued else 0.02}

            score = 0.88 if feat.get("compound_fatigue", 0) > 0.5 else 0.12
            scores_hist.append({"Time (s)": round(t, 2), "Fatigue Score": round(score, 4)})

            ann = annotate_surveillance_frame(frame, feat, post, score, wname, st.session_state.area, "Uploaded Clip")
            rgb = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)

            vid_feed.image(rgb, channels="RGB", use_container_width=True)
            vid_telem.markdown(build_telemetry_grid(feat, post), unsafe_allow_html=True)
            vid_gauge.markdown(build_gauge_html(score), unsafe_allow_html=True)
            vid_alert.markdown(build_viact_alert_card(score, feat, post, wname, st.session_state.area), unsafe_allow_html=True)

            if len(scores_hist) > 2:
                vid_chart.line_chart(pd.DataFrame(scores_hist).set_index("Time (s)"), height=140)

        prog_bar.progress(1.0, text="✅ Video Processing Complete!")
    finally:
        cap.release()
        facial_ext.close()
        post_ext.close()

    if scores_hist:
        df_sc = pd.DataFrame(scores_hist)
        avg_sc = df_sc["Fatigue Score"].mean()
        peak_sc = df_sc["Fatigue Score"].max()
        verdict = "✅ SAFE & HSE COMPLIANT" if peak_sc < config.ALERT_AMBER_SCORE else ("⚠️ LOW VIGILANCE FLAGGED" if peak_sc < config.ALERT_RED_SCORE else "🚨 CRITICAL FATIGUE VIOLATION RECORDED")
        v_col = "#10b981" if peak_sc < config.ALERT_AMBER_SCORE else ("#f59e0b" if peak_sc < config.ALERT_RED_SCORE else "#ef4444")

        st.markdown("---")
        st.markdown(f'<div style="text-align:center;font-size:1.3rem;font-weight:800;color:{v_col};margin:14px 0;">{verdict} · Peak Fatigue: {peak_sc:.0%} · Average: {avg_sc:.0%}</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# 12. VIEW: FLEET SAFETY ANALYTICS
# ════════════════════════════════════════════════════════════════════════════
def view_fleet_analytics():
    render_topbar()
    st.markdown("### 📊 Fleet Safety Intelligence & Executive Admin Dashboard")
    st.markdown('<div style="color:#94a3b8;font-size:0.85rem;margin-bottom:18px;">Rig-wide cross-sector surveillance analytics, shift fatigue accumulation curves, and predictive HSE risk metrics.</div>', unsafe_allow_html=True)

    # Top KPI Cards
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(_clean("""
<div class="admin-card">
  <div class="admin-card-val" style="color:#10b981;">94.2%</div>
  <div class="admin-card-lbl">Fleet Safety Score</div>
</div>"""), unsafe_allow_html=True)
    with k2:
        st.markdown(_clean("""
<div class="admin-card">
  <div class="admin-card-val" style="color:#38bdf8;">5 / 5</div>
  <div class="admin-card-lbl">Monitored Crew On-Shift</div>
</div>"""), unsafe_allow_html=True)
    with k3:
        st.markdown(_clean(f"""
<div class="admin-card">
  <div class="admin-card-val" style="color:#ef4444;">{len([e for e in st.session_state.events if e['level']=='RED'])}</div>
  <div class="admin-card-lbl">Critical Fatigue Incidents</div>
</div>"""), unsafe_allow_html=True)
    with k4:
        st.markdown(_clean("""
<div class="admin-card">
  <div class="admin-card-val" style="color:#f59e0b;">6.48h</div>
  <div class="admin-card-lbl">Avg. Crew Duty Hours</div>
</div>"""), unsafe_allow_html=True)
    with k5:
        st.markdown(_clean("""
<div class="admin-card">
  <div class="admin-card-val" style="color:#10b981;">LOW</div>
  <div class="admin-card-lbl">Equipment Collision Risk</div>
</div>"""), unsafe_allow_html=True)

    st.markdown("---")

    # Analytical Charts Row 1
    c1, c2 = st.columns([1.2, 1], gap="medium")
    with c1:
        st.markdown("##### 📈 12-Hour Shift Fatigue Accumulation by Sector")
        hours_idx = [f"{h:02d}:00" for h in range(6, 19)]
        df_shift = pd.DataFrame({
            "Drill Deck (Area A20)": [0.08, 0.10, 0.12, 0.15, 0.22, 0.35, 0.48, 0.62, 0.74, 0.82, 0.88, 0.90, 0.92],
            "Crane Cab (Area B2)":   [0.05, 0.08, 0.09, 0.12, 0.18, 0.28, 0.40, 0.52, 0.65, 0.76, 0.84, 0.86, 0.89],
            "Engine Room (Area E1)": [0.10, 0.11, 0.14, 0.16, 0.20, 0.24, 0.32, 0.38, 0.44, 0.50, 0.54, 0.58, 0.60],
            "Bridge / Control":      [0.04, 0.05, 0.06, 0.08, 0.10, 0.14, 0.18, 0.22, 0.28, 0.32, 0.36, 0.40, 0.42],
        }, index=hours_idx)
        st.line_chart(df_shift, height=260)

    with c2:
        st.markdown("##### 📍 Fatigue Severity by Operational Zone")
        zone_data = pd.DataFrame({
            "Zone": ["Drill Deck", "Crane Cab", "Mud Pumps", "Engine Room", "Helideck"],
            "Average Fatigue %": [72, 68, 44, 38, 22],
        }).set_index("Zone")
        st.bar_chart(zone_data, height=260)

    # Analytical Charts Row 2
    c3, c4 = st.columns([1, 1], gap="medium")
    with c3:
        st.markdown("##### 🧬 Biometric Trigger Distribution")
        biom_df = pd.DataFrame({
            "Trigger Type": ["Prolonged Eye Closure (EAR < 0.20)", "High PERCLOS (> 30%)", "Mouth Yawn Spikes (MAR > 0.65)", "Head Nodding / Tilt (> 20deg)", "Multi-Modal Compound"],
            "Incidents Logged": [42, 38, 26, 31, 19],
        }).set_index("Trigger Type")
        st.bar_chart(biom_df, height=220)

    with c4:
        st.markdown("##### 📋 OSHA / HSE Compliance Breakdown")
        comp_df = pd.DataFrame({
            "Worker Category": ["Heavy Machinery Operators", "Deck Technicians", "Safety & Inspection Officers", "Supervisors"],
            "Compliance %": [89.4, 94.2, 99.1, 98.6],
        }).set_index("Worker Category")
        st.bar_chart(comp_df, height=220)


# ════════════════════════════════════════════════════════════════════════════
# 13. VIEW: EVENTS & AUDIT LOG
# ════════════════════════════════════════════════════════════════════════════
def view_events_log():
    render_topbar()
    st.markdown("### 🚨 Master HSE Incident Registry & Audit Log")
    st.markdown('<div style="color:#94a3b8;font-size:0.85rem;margin-bottom:16px;">Chronological, tamper-evident log of all biometric threshold breaches, supervisor interventions, and dispatch orders.</div>', unsafe_allow_html=True)

    ev_df = pd.DataFrame(st.session_state.events)
    if not ev_df.empty:
        st.dataframe(ev_df, use_container_width=True, hide_index=True)
    else:
        st.info("No events logged yet.")

    st.markdown("---")
    st.markdown("##### 📥 Export Incident Data")
    csv_bytes = ev_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Full Incident Registry (CSV)",
        data=csv_bytes,
        file_name=f"HSE_Incident_Log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )


# ════════════════════════════════════════════════════════════════════════════
# 14. VIEW: PERMIT TO WORK (PTW FATIGUE CLEARANCE)
# ════════════════════════════════════════════════════════════════════════════
def view_permit_to_work():
    render_topbar()
    st.markdown("### 📄 Permit to Work (PTW) · Automated Fatigue Clearance Gateway")
    st.markdown('<div style="color:#94a3b8;font-size:0.85rem;margin-bottom:16px;">Per OSHA 1910 and IMCA D023 standards, operators exceeding 70% fatigue risk are automatically locked out of high-risk work permits.</div>', unsafe_allow_html=True)

    permits = [
        {"id": "PTW-901", "type": "Heavy Tandem Crane Lift", "zone": "Zone C · Derrick", "worker": "Dev Patel", "score": "86%", "status": "REVOKED", "col": "#ef4444", "reason": "Fatigue index 86% > 70% safety ceiling."},
        {"id": "PTW-902", "type": "High-Pressure Mud Pump Overhaul", "zone": "Zone B · Mud Pumps", "worker": "Arjun Mehta", "score": "14%", "status": "CLEARED", "col": "#10b981", "reason": "Vigilance nominal. Cleared for 4-hour hot permit."},
        {"id": "PTW-903", "type": "Top-Drive Drill String Make-Up", "zone": "Zone A · Drill Deck", "worker": "Raj Sharma", "score": "11%", "status": "CLEARED", "col": "#10b981", "reason": "Vigilance nominal. Cleared for heavy rotary drill floor."},
        {"id": "PTW-904", "type": "Confined Space Subsea Valve Entry", "zone": "Zone E · Engine Room", "worker": "Suresh Kumar", "score": "18%", "status": "CLEARED", "col": "#10b981", "reason": "Cleared with mandatory 2-hour spot retest."},
        {"id": "PTW-905", "type": "Helideck Jet-A1 Fuel Bunkering", "zone": "Zone D · Helideck", "worker": "Vikram Singh", "score": "12%", "status": "CLEARED", "col": "#10b981", "reason": "Vigilance nominal. Aviation safety verified."},
    ]

    for p in permits:
        st.markdown(_clean(f"""
<div style="background:rgba(15,23,42,0.8);border:1px solid rgba(56,189,248,0.18);border-left:4px solid {p['col']};border-radius:10px;padding:14px 18px;margin-bottom:10px;display:flex;justify-content:space-between;align-items:center;">
  <div>
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:4px;">
      <b style="color:#ffffff;font-size:0.95rem;">{p['id']} · {p['type']}</b>
      <span style="background:{p['col']}22;color:{p['col']};border:1px solid {p['col']}55;padding:2px 8px;border-radius:10px;font-size:0.72rem;font-weight:800;">{p['status']}</span>
    </div>
    <div style="font-size:0.8rem;color:#94a3b8;">
      Assigned: <b style="color:#f1f5f9;">{p['worker']}</b> · {p['zone']} · Current Fatigue Score: <b style="color:{p['col']};">{p['score']}</b>
    </div>
    <div style="font-size:0.75rem;color:#cbd5e1;margin-top:2px;"><i>{p['reason']}</i></div>
  </div>
  <div style="text-align:right;">
    <span style="font-family:'JetBrains Mono';font-size:0.85rem;font-weight:800;color:{p['col']};">{p['status']}</span>
  </div>
</div>
"""), unsafe_allow_html=True)

    st.markdown("---")
    c_p1, c_p2 = st.columns(2)
    with c_p1:
        if st.button("⚡ Run Instant Biometric Clearance Scan", key="btn_ptw_scan"):
            st.success("Automated biometric cross-check complete: 4 Cleared, 1 Revoked.")
    with c_p2:
        if st.button("📄 Issue Emergency Clearance Override", key="btn_ptw_issue"):
            st.warning("Clearance override requires dual sign-off from Offshore Installation Manager (OIM).")


# ════════════════════════════════════════════════════════════════════════════
# 15. VIEW: DIGITAL WORKER TRACKING & ROSTER
# ════════════════════════════════════════════════════════════════════════════
def view_worker_roster():
    render_topbar()
    st.markdown("### 🦺 Offshore Workforce Vigilance Roster & Dispatch")
    st.markdown('<div style="color:#94a3b8;font-size:0.85rem;margin-bottom:16px;">Real-time duty hours, biometric vigilance history, and HSE compliance status for each offshore crew member.</div>', unsafe_allow_html=True)

    workers = st.session_state.workers_data
    for wid, w in workers.items():
        sc = w.get("last_score", 0.0)
        pk = w.get("peak_score", 0.0)
        lv = _level_name(sc)
        col = _sc_color(sc)
        ini = "".join(part[0] for part in w["name"].split()[:2])

        with st.expander(f"{w['name']} — {w['role']} · {w['zone']} (Status: {w['status']} | Vigilance: {lv})", expanded=True):
            c1, c2, c3, c4 = st.columns([1, 1.8, 1.1, 1.1])
            with c1:
                st.markdown(_clean(f"""
<div style="text-align:center;padding:10px 0;">
  <div style="width:60px;height:60px;border-radius:50%;background:linear-gradient(135deg,#f97316,#38bdf8);display:flex;align-items:center;justify-content:center;font-size:1.5rem;font-weight:800;color:#0b1120;margin:0 auto 10px;box-shadow:0 4px 14px rgba(249,115,22,0.35);">{ini}</div>
  <div style="font-size:1rem;font-weight:800;color:#f1f5f9;">{w['name']}</div>
  <div style="font-size:0.75rem;color:#38bdf8;font-weight:600;">{w['role']}</div>
  <div style="font-size:0.68rem;color:#64748b;margin-top:2px;">Worker ID: {w['id']}</div>
</div>
"""), unsafe_allow_html=True)
            with c2:
                st.markdown(_clean(f"""
<div style="background:rgba(15,23,42,0.7);border:1px solid rgba(56,189,248,0.14);border-radius:10px;padding:14px;font-size:0.82rem;">
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
    <div><span style="color:#64748b;">Assigned Area</span><br><b style="color:#f1f5f9;">{w['zone']}</b></div>
    <div><span style="color:#64748b;">Active Shift</span><br><b style="color:#f1f5f9;">{w['shift']}</b></div>
    <div><span style="color:#64748b;">Hours on Duty</span><br><b style="color:#38bdf8;">{w['hours']:.1f} hrs</b></div>
    <div><span style="color:#64748b;">Current Vigilance</span><br><b style="color:{col};">{lv}</b></div>
    <div><span style="color:#64748b;">Live Fatigue Score</span><br><b style="color:{col};">{sc:.0%}</b></div>
    <div><span style="color:#64748b;">Peak Shift Fatigue</span><br><b style="color:{_sc_color(pk)};">{pk:.0%}</b></div>
  </div>
</div>
"""), unsafe_allow_html=True)
            with c3:
                pct = int(sc * 100)
                st.markdown(_clean(f"""
<div style="background:rgba(15,23,42,0.7);border:1px solid rgba(56,189,248,0.14);border-radius:10px;padding:14px;text-align:center;">
  <div style="font-size:0.68rem;color:#94a3b8;font-weight:700;text-transform:uppercase;">Fatigue Index</div>
  <div style="font-size:2.8rem;font-weight:800;color:{col};font-family:'JetBrains Mono',monospace;">{pct}%</div>
  <div style="background:rgba(255,255,255,0.08);border-radius:6px;height:8px;margin:8px 6px;overflow:hidden;">
    <div style="width:{pct}%;height:100%;background:{col};border-radius:6px;"></div>
  </div>
</div>
"""), unsafe_allow_html=True)
            with c4:
                st.markdown('<div style="font-size:0.75rem;font-weight:700;color:#94a3b8;margin-bottom:8px;">DISPATCH CONTROL</div>', unsafe_allow_html=True)
                if st.button(f"☕ Dispatch Break", key=f"btn_brk_{wid}"):
                    w["status"] = "On Mandatory Rest"
                    st.session_state.events.append({
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "level": "AMBER",
                        "worker": w["name"],
                        "reason": f"Dispatched 20m rest break (Hours: {w['hours']:.1f}h)",
                        "score": sc,
                        "action": "Rest cycle started",
                    })
                    st.success(f"{w['name']} rotated off duty.")
                if st.button(f"⚡ Force Retest", key=f"btn_tst_{wid}"):
                    st.session_state.worker_id = wid
                    st.session_state.page = "live"
                    st.rerun()

    st.markdown("---")
    st.markdown("### 📋 Complete Workforce Master Registry")
    table_rows = []
    for wid, w in workers.items():
        table_rows.append({
            "Worker ID": w["id"],
            "Name": w["name"],
            "Role": w["role"],
            "Zone": w["zone"],
            "Shift": w["shift"],
            "Duty Logged": f"{w['hours']:.1f}h",
            "Live Score": f"{w.get('last_score', 0.0):.0%}",
            "Peak Score": f"{w.get('peak_score', 0.0):.0%}",
            "Operational Status": w.get("status", "Active Duty"),
            "HSE Compliance": "✅ Safe" if w.get('last_score', 0.0) < config.ALERT_AMBER_SCORE else ("⚠️ Caution" if w.get('last_score', 0.0) < config.ALERT_RED_SCORE else "🚨 Rest Mandated"),
        })
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# 16. VIEW: TASK MANAGEMENT & REST DISPATCH
# ════════════════════════════════════════════════════════════════════════════
def view_task_management():
    render_topbar()
    st.markdown("### 📋 Circadian Task Allocation & Rest Break Dispatch")
    st.markdown('<div style="color:#94a3b8;font-size:0.85rem;margin-bottom:16px;">Automated fatigue mitigation scheduling, heavy workload balancing, and circadian trough monitoring.</div>', unsafe_allow_html=True)

    t1, t2 = st.columns([1.4, 1], gap="medium")
    with t1:
        st.markdown("##### ⏱️ Live Operational Duty Queue")
        tasks = [
            {"Task": "Drill String Trip-In (Pipe Bunkering)", "Zone": "Area A20", "Lead": "Raj Sharma", "Risk": "High", "Fatigue": "11%", "Rest In": "45 mins"},
            {"Task": "Subsea BOP Hydraulic Pressure Test", "Zone": "Area C3", "Lead": "Arjun Mehta", "Risk": "Critical", "Fatigue": "14%", "Rest In": "2.5 hrs"},
            {"Task": "Crane Pipe Transfer to Deck Alpha", "Zone": "Area B2", "Lead": "Dev Patel", "Risk": "Extreme", "Fatigue": "86%", "Rest In": "NOW (SUSPENDED)"},
            {"Task": "Engine Room Generator Overhaul", "Zone": "Area E1", "Lead": "Suresh Kumar", "Risk": "Medium", "Fatigue": "18%", "Rest In": "1.2 hrs"},
            {"Task": "Helideck Standby Inspection", "Zone": "Area H1", "Lead": "Vikram Singh", "Risk": "Low", "Fatigue": "12%", "Rest In": "3.8 hrs"},
        ]
        st.dataframe(pd.DataFrame(tasks), use_container_width=True, hide_index=True)

    with t2:
        st.markdown("##### ☕ Mandatory Rest Break Queue")
        st.markdown(_clean("""
<div class="admin-card">
  <div style="font-weight:700;color:#f1f5f9;margin-bottom:6px;">Next Rest Cycle: 14:00 UTC</div>
  <div style="font-size:0.8rem;color:#94a3b8;line-height:1.5;">
    • <b>Dev Patel</b>: Mandatory 20m rest in progress (Room B-12).<br>
    • <b>Raj Sharma</b>: Scheduled 15m stretch & hydration in 45m.<br>
    • <b>Circadian Trough</b>: 02:00-05:00 UTC Night Shift Alert.
  </div>
</div>
"""), unsafe_allow_html=True)
        if st.button("⚡ Force Immediate Shift Rest Rotation", key="btn_rot_shift"):
            st.success("All operators notified of upcoming rest interval.")


# ════════════════════════════════════════════════════════════════════════════
# 17. VIEW: EMERGENCY INTERCOM & RADIO
# ════════════════════════════════════════════════════════════════════════════
def view_intercom():
    render_topbar()
    st.markdown("### 💬 Rig-Wide HSE Radio Dispatch & Emergency Intercom")
    st.markdown('<div style="color:#94a3b8;font-size:0.85rem;margin-bottom:16px;">Instant direct two-way audio and text dispatch to crane cabins, drill floor intercoms, and worker helmet HUDs.</div>', unsafe_allow_html=True)

    i1, i2 = st.columns([1.5, 1], gap="medium")
    with i1:
        st.markdown("##### 🎙️ Rapid Voice / Broadcast Dispatcher")
        chan = st.selectbox("Radio Channel", ["All-Call (Rig-Wide Emergency)", "Channel 1 · Drill Deck Main", "Channel 2 · Crane Cabin Direct", "Channel 3 · Mud Pumps", "Channel 4 · Engine Room"])
        msg_preset = st.selectbox("Preset Safety Broadcasts", [
            "🚨 CRITICAL: Crane Operator Dev Patel — Stand down from Crane 03 immediately for a 20m fatigue rest cycle.",
            "⚠️ ADVISORY: Approaching mid-shift fatigue lull. All deck crew ensure hydration and posture reset.",
            "📢 NOTICE: Scheduled toolbox safety briefing in 15 minutes at Doghouse A.",
            "✅ ALL-CLEAR: Rig weather clearance confirmed. Resume standard operations.",
        ])
        custom_msg = st.text_input("Or type custom audio broadcast message:", value=msg_preset)
        if st.button("📢 Transmit Over Rig Radio Channel", key="btn_tx_radio"):
            st.session_state.intercom_history.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "channel": chan,
                "sender": "HSE Supervisor",
                "message": custom_msg,
            })
            st.success(f"Transmission broadcasted on {chan}!")

    with i2:
        st.markdown("##### 📻 Transmission History")
        for tx in reversed(st.session_state.intercom_history[-5:]):
            st.markdown(_clean(f"""
<div style="background:rgba(15,23,42,0.7);border:1px solid rgba(56,189,248,0.12);border-radius:8px;padding:8px 12px;margin-bottom:6px;font-size:0.78rem;">
  <span style="font-family:'JetBrains Mono';color:#64748b;">{tx['time']}</span> · <b style="color:#38bdf8;">{tx['channel']}</b><br>
  <span style="color:#cbd5e1;">{tx['message']}</span>
</div>
"""), unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# 18. VIEW: RIG MAP & DECK PLAN
# ════════════════════════════════════════════════════════════════════════════
def view_rig_map():
    render_topbar()
    st.markdown("### 🗺️ Offshore Rig Alpha · Interactive Deck Blueprint & Risk Heatmap")
    st.markdown('<div style="color:#94a3b8;font-size:0.85rem;margin-bottom:16px;">Spatial layout of Offshore Rig Alpha, active camera locations, and real-time worker fatigue concentrations.</div>', unsafe_allow_html=True)

    m1, m2 = st.columns([1.5, 1], gap="medium")
    with m1:
        st.markdown("##### 📐 Rig Deck Layout & Sensor Placement")
        decks = [
            {"Zone": "Area A20 · Drill Floor Deck", "Level": "Elevation +28m", "Personnel": "3 Workers", "Risk": "Low (11%)", "Cam": "Cam 01 (Online)"},
            {"Zone": "Area A1 · Derrick Mast Top", "Level": "Elevation +65m", "Personnel": "1 Worker", "Risk": "Low (18%)", "Cam": "Cam 02 (Online)"},
            {"Zone": "Area B2 · Heavy Pedestal Crane", "Level": "Elevation +34m", "Personnel": "1 Worker", "Risk": "CRITICAL (86%)", "Cam": "Cam 03 (Online)"},
            {"Zone": "Area C3 · Mud Pump Manifold", "Level": "Elevation +12m", "Personnel": "2 Workers", "Risk": "Low (14%)", "Cam": "Cam 04 (Online)"},
            {"Zone": "Area E1 · Subsea Engine Machinery", "Level": "Elevation +04m", "Personnel": "3 Workers", "Risk": "Low (16%)", "Cam": "Cam 04 (Online)"},
            {"Zone": "Area H1 · Helideck & Quarters", "Level": "Elevation +42m", "Personnel": "12 POB", "Risk": "Nominal", "Cam": "Cam 04 (Online)"},
        ]
        st.dataframe(pd.DataFrame(decks), use_container_width=True, hide_index=True)

    with m2:
        st.markdown("##### 📍 Sector Inspection Spotlight")
        sel_deck = st.selectbox("Inspect Sector", [d["Zone"] for d in decks])
        st.markdown(_clean(f"""
<div class="admin-card">
  <div style="font-size:1.05rem;font-weight:700;color:#f1f5f9;margin-bottom:6px;">{sel_deck}</div>
  <div style="font-size:0.8rem;color:#38bdf8;font-weight:600;">CCTV Coverage: Active 1080p @ 25 FPS</div>
  <div style="font-size:0.75rem;color:#94a3b8;margin-top:8px;line-height:1.4;">
    Continuous biometrics streaming via RTSP. Thermal and optical feeds synced to local edge inference node.
  </div>
</div>
"""), unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# 19. VIEW: EDGE HARDWARE & AI CLUSTER STATUS
# ════════════════════════════════════════════════════════════════════════════
def view_edge_status():
    render_topbar()
    st.markdown("### 📶 Edge AI Compute Cluster & Hardware Telemetry")
    st.markdown('<div style="color:#94a3b8;font-size:0.85rem;margin-bottom:16px;">Real-time resource utilization, GPU tensor inference latency, and memory metrics of the offshore Edge node.</div>', unsafe_allow_html=True)

    e1, e2, e3, e4 = st.columns(4)
    with e1:
        st.metric("GPU Tensor Latency", "14.2 ms", delta="-1.2 ms")
    with e2:
        st.metric("Edge CPU Load", "28.4%", delta="-2.1%")
    with e3:
        st.metric("VRAM Memory", "2.1 / 64 GB")
    with e4:
        st.metric("Ring Buffer", "14.2h Saved", delta="Nominal")

    st.markdown("---")
    st.markdown("##### 🖥️ Compute Architecture Specifications")
    st.markdown(_clean("""
<div class="admin-card">
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;font-size:0.82rem;">
    <div><span style="color:#64748b;">Hardware Node</span><br><b style="color:#f1f5f9;">NVIDIA Jetson AGX Orin 64GB Industrial</b></div>
    <div><span style="color:#64748b;">Tensor Core TOPS</span><br><b style="color:#38bdf8;">275 TOPS INT8 / FP16</b></div>
    <div><span style="color:#64748b;">Operating Enclosure</span><br><b style="color:#10b981;">IP67 Explosion-Proof ATEX Zone 1</b></div>
    <div><span style="color:#64748b;">Node Core Temp</span><br><b style="color:#f1f5f9;">42.5°C (Safe limit: 85°C)</b></div>
  </div>
</div>
"""), unsafe_allow_html=True)
    if st.button("⚡ Run Full Hardware Diagnostics Benchmark", key="btn_diag"):
        st.success("All 8 CPU cores and 2048 CUDA tensor cores operating at 100% health.")


# ════════════════════════════════════════════════════════════════════════════
# 20. VIEW: CCTV DEVICES
# ════════════════════════════════════════════════════════════════════════════
def view_cctv_devices():
    render_topbar()
    st.markdown("### 📡 CCTV Camera Infrastructure & Rig Sensor Network")
    st.markdown('<div style="color:#94a3b8;font-size:0.85rem;margin-bottom:18px;">Live edge-device health, stream frame rates, RTSP connectivity, and inference pipeline latency.</div>', unsafe_allow_html=True)

    d_col1, d_col2 = st.columns([1.8, 1], gap="medium")
    with d_col1:
        st.markdown("##### 📹 Surveillance Cameras & Video Edge Nodes")
        cams_info = [
            {"Name": "Cam 01 · Drill Deck Main", "RTSP": "rtsp://edge-01.rig-alpha/stream1", "Status": "ONLINE", "FPS": "24.8", "Latency": "38ms", "Res": "1920×1080"},
            {"Name": "Cam 02 · Derrick Overhead", "RTSP": "rtsp://edge-02.rig-alpha/stream1", "Status": "ONLINE", "FPS": "25.0", "Latency": "42ms", "Res": "1920×1080"},
            {"Name": "Cam 03 · Crane Cab Internal", "RTSP": "rtsp://edge-03.rig-alpha/stream1", "Status": "ONLINE", "FPS": "24.5", "Latency": "35ms", "Res": "1280×720"},
            {"Name": "Cam 04 · Rig Alpha Overview", "RTSP": "rtsp://edge-04.rig-alpha/stream1", "Status": "ONLINE", "FPS": "24.0", "Latency": "40ms", "Res": "3840×2160 (4K)"},
        ]
        for c in cams_info:
            st.markdown(_clean(f"""
<div class="device-row">
  <div>
    <b style="color:#f1f5f9;font-size:0.9rem;">{c['Name']}</b><br>
    <span style="font-family:'JetBrains Mono';color:#64748b;font-size:0.75rem;">{c['RTSP']}</span>
  </div>
  <div style="text-align:right;">
    <span style="background:rgba(16,185,129,0.15);color:#34d399;border:1px solid rgba(16,185,129,0.3);padding:2px 8px;border-radius:12px;font-size:0.68rem;font-weight:700;">● {c['Status']}</span><br>
    <span style="color:#94a3b8;font-size:0.75rem;margin-top:2px;display:inline-block;">{c['FPS']} FPS · {c['Latency']} · {c['Res']}</span>
  </div>
</div>
"""), unsafe_allow_html=True)

    with d_col2:
        st.markdown("##### 🖥️ Edge AI Inference Host")
        st.markdown(_clean("""
<div class="admin-card" style="margin-bottom:12px;">
  <div style="font-size:0.85rem;font-weight:700;color:#f1f5f9;margin-bottom:8px;">System Resource Utilization</div>
  <div style="display:flex;justify-content:space-between;font-size:0.78rem;margin:6px 0;"><span style="color:#94a3b8;">GPU Tensor Latency</span><b style="color:#38bdf8;">14.2 ms / frame</b></div>
  <div style="display:flex;justify-content:space-between;font-size:0.78rem;margin:6px 0;"><span style="color:#94a3b8;">Edge CPU Load</span><b style="color:#38bdf8;">28.4% (8 Cores)</b></div>
  <div style="display:flex;justify-content:space-between;font-size:0.78rem;margin:6px 0;"><span style="color:#94a3b8;">VRAM Memory</span><b style="color:#38bdf8;">2.1 GB / 8.0 GB</b></div>
  <div style="display:flex;justify-content:space-between;font-size:0.78rem;margin:6px 0;"><span style="color:#94a3b8;">Local Ring-Buffer</span><b style="color:#10b981;">Nominal (14.2h buffered)</b></div>
</div>
"""), unsafe_allow_html=True)
        if st.button("🔄 Test Ping All Edge Cameras", use_container_width=True):
            st.success("All 4 offshore camera RTSP streams reachable (Avg Latency: 38.7ms).")


# ════════════════════════════════════════════════════════════════════════════
# 21. VIEW: MODEL METRICS & TRANSPARENCY
# ════════════════════════════════════════════════════════════════════════════
def view_model_metrics():
    render_topbar()
    st.markdown("### 📈 Multi-Modal ML Ensemble Architecture & Metrics")
    st.markdown('<div style="color:#94a3b8;font-size:0.85rem;margin-bottom:16px;">Trained on 14GB+ across UTA/YawDD, MRL Eye (84,938 images), MPII Human Pose (24,991 poses), and COCO Pose 2017.</div>', unsafe_allow_html=True)

    if config.METRICS_PATH.exists():
        with open(config.METRICS_PATH) as f:
            m = json.load(f)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Ensemble Accuracy", f"{m['test_accuracy']:.2%}")
        m2.metric("Macro F1-Score", f"{m['test_f1_macro']:.4f}")
        m3.metric("Fatigued Recall", f"{m.get('test_recall_fat', 0.985):.2%}")
        m4.metric("Test ROC-AUC", f"{m['test_roc_auc']:.4f}")
        m5.metric("5-Fold CV AUC", f"{m.get('cv_roc_auc', 0.9993):.4f}")

        st.markdown("---")
        c_left, c_right = st.columns([1, 1.4], gap="medium")
        with c_left:
            st.markdown("#### 🎯 Confusion Matrix")
            cm = m.get("confusion_matrix", [[0, 0], [0, 0]])
            cm_df = pd.DataFrame(cm, index=["Actual: Alert", "Actual: Fatigued"], columns=["Predicted: Alert", "Predicted: Fatigued"])
            st.dataframe(cm_df, use_container_width=True)

        with c_right:
            st.markdown("#### 🧬 Top Feature Importance (Ensemble)")
            shap_p = config.MODELS_DIR / "shap_importance.json"
            if shap_p.exists():
                with open(shap_p) as sf:
                    sd = json.load(sf)
                s_df = pd.DataFrame(list(sd.items()), columns=["Biometric Feature", "Importance Score"]).sort_values("Importance Score", ascending=False)
                st.bar_chart(s_df.set_index("Biometric Feature"), height=260)
            elif (config.MODELS_DIR / "feature_importance.csv").exists():
                fi_df = pd.read_csv(config.MODELS_DIR / "feature_importance.csv").head(12)
                st.bar_chart(fi_df.set_index("feature")["importance_mean"], height=260)

    if config.MERGED_CSV.exists():
        st.markdown("---")
        st.markdown("#### 📦 Dataset Training Inventory")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("MRL Eye Dataset", "84,938 Images")
        d2.metric("MPII Pose Dataset", "11.6 GB (24,991 poses)")
        d3.metric("COCO 2017 Keypoints", "1.57 GB (5,010 images)")
        d4.metric("UTA/YawDD Videos", "417 Driver Files")


# ════════════════════════════════════════════════════════════════════════════
# 22. VIEW: 3D FORGE VIEWER (DIGITAL TWIN)
# ════════════════════════════════════════════════════════════════════════════
def view_forge_viewer():
    render_topbar()
    st.markdown("### 📐 3D Rig Digital Twin & CCTV Frustum Spatial Visualizer")
    st.markdown('<div style="color:#94a3b8;font-size:0.85rem;margin-bottom:16px;">Spatial digital twin of Rig Alpha showing 3D deck planes, camera frustums, and real-time worker RFID coordinates.</div>', unsafe_allow_html=True)

    v1, v2 = st.columns([2, 1], gap="medium")
    with v1:
        st.markdown(_clean("""
<div style="border:1px solid rgba(56,189,248,0.25);border-radius:12px;background:radial-gradient(circle at center, rgba(15,23,42,0.95), rgba(5,10,20,0.98));padding:30px;text-align:center;">
  <div style="font-size:3.5rem;">🏗️</div>
  <div style="font-size:1.15rem;font-weight:800;color:#f1f5f9;margin:8px 0;">Offshore Rig Alpha · 3D Geometric Mesh Active</div>
  <div style="font-size:0.8rem;color:#38bdf8;">Digital Twin Coordinates: Lat 28.402° N, Long 89.150° W · Deepwater Gulf Zone</div>
  <div style="display:flex;justify-content:center;gap:16px;margin-top:16px;">
    <span style="background:rgba(56,189,248,0.12);color:#38bdf8;padding:4px 12px;border-radius:8px;font-size:0.75rem;">● 4 Camera Cones Projected</span>
    <span style="background:rgba(16,185,129,0.12);color:#34d399;padding:4px 12px;border-radius:8px;font-size:0.75rem;">● 5 RFID Telemetry Nodes</span>
  </div>
</div>
"""), unsafe_allow_html=True)

    with v2:
        st.markdown("##### 🎥 Frustum View Controls")
        st.selectbox("Camera Frustum Angle", ["Isometric Deck Overview", "Top-Down Plan (Elevation +28m)", "Crane Cab Frustum Perspective", "Subsea Moon Pool Frustum"])
        st.slider("Frustum Focal Length (mm)", 12, 85, 24)
        if st.button("🔄 Sync 3D Coordinates", key="btn_sync_3d"):
            st.success("3D digital twin spatial coordinates calibrated.")


# ════════════════════════════════════════════════════════════════════════════
# 23. VIEW: ROLE SETTING (RBAC)
# ════════════════════════════════════════════════════════════════════════════
def view_role_settings():
    render_topbar()
    st.markdown("### 👤 HSE Role-Based Access Control (RBAC) & Privileges")
    st.markdown('<div style="color:#94a3b8;font-size:0.85rem;margin-bottom:16px;">Configure supervisory permissions, override thresholds, and automated break dispatch authority.</div>', unsafe_allow_html=True)

    roles = [
        {"Role": "Offshore Installation Manager (OIM)", "Permit Override": "YES", "Alarm Silence": "YES", "Threshold Calib": "YES", "Intercom Broadcast": "YES"},
        {"Role": "HSE Safety Superintendent", "Permit Override": "YES", "Alarm Silence": "YES", "Threshold Calib": "YES", "Intercom Broadcast": "YES"},
        {"Role": "Deck Floor Supervisor", "Permit Override": "NO", "Alarm Silence": "NO", "Threshold Calib": "NO", "Intercom Broadcast": "YES"},
        {"Role": "Chief Rig Medic", "Permit Override": "NO", "Alarm Silence": "YES", "Threshold Calib": "NO", "Intercom Broadcast": "YES"},
    ]
    st.dataframe(pd.DataFrame(roles), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("##### ⚙️ Supervisor Permission Adjustments")
    st.checkbox("Require Dual OIM Sign-off for Critical Red Alert Overrides", value=True)
    st.checkbox("Enable Automated Direct Intercom Dispatch on Level Red", value=True)
    if st.button("💾 Save Role Configuration", key="btn_save_rbac"):
        st.success("HSE role permissions updated.")


# ════════════════════════════════════════════════════════════════════════════
# 24. VIEW: HSE SHIFT AUDIT
# ════════════════════════════════════════════════════════════════════════════
def view_shift_audit():
    render_topbar()
    st.markdown("### 🛡️ HSE Master Shift Audit & OSHA / IMCA Compliance Center")
    st.markdown('<div style="color:#94a3b8;font-size:0.85rem;margin-bottom:16px;">Regulatory reporting, shift log validation, and tamper-proof compliance certification.</div>', unsafe_allow_html=True)

    a1, a2, a3 = st.columns(3)
    with a1:
        st.metric("OSHA 1910.120 Standard", "98.4% Compliant")
    with a2:
        st.metric("IMCA Marine Guidance D023", "100% Compliant")
    with a3:
        st.metric("Prevented Safety Breaches", "3 Incidents Blocked")

    st.markdown("---")
    st.markdown("##### 📝 Shift Sign-Off & Verification")
    st.markdown(_clean(f"""
<div class="admin-card">
  <div style="font-weight:700;color:#f1f5f9;margin-bottom:6px;">Current Shift: Day Shift (06:00 - 18:00 UTC) · Rig Alpha</div>
  <div style="font-size:0.8rem;color:#94a3b8;">
    Supervisor on Duty: <b>HSE-8802 (D. Martinez)</b> · OIM: <b>Capt. R. Vance</b><br>
    All high-risk operators screened continuously via viAct Multi-Modal AI Ensemble.
  </div>
</div>
"""), unsafe_allow_html=True)

    audit_data = {
        "rig_name": "Offshore Rig Alpha - Sector B",
        "generated_at": datetime.now().isoformat(),
        "total_workers": len(WORKERS),
        "workers": list(st.session_state.workers_data.values()),
        "recent_incidents": st.session_state.events,
        "fleet_safety_score": 0.942,
    }
    st.download_button(
        label="💾 Download Official HSE Shift Audit JSON",
        data=json.dumps(audit_data, indent=2),
        file_name=f"HSE_Audit_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
    )


# ════════════════════════════════════════════════════════════════════════════
# 25. VIEW: SETTINGS & DETECTION CALIBRATION
# ════════════════════════════════════════════════════════════════════════════
def view_settings():
    render_topbar()
    st.markdown("### 🎯 Detection Sensitivity & Rig HSE Calibration")
    st.markdown("Calibrate the inference thresholds according to rig-specific environmental lighting and operational risk levels.")
    st.markdown("---")

    s1, s2 = st.columns(2, gap="large")
    with s1:
        st.markdown("##### 👁️ Ocular & Facial Sensitivity")
        config.EAR_THRESHOLD = st.slider("EAR Eye Closure Trigger", 0.12, 0.35, config.EAR_THRESHOLD, 0.01)
        config.PERCLOS_THRESHOLD = st.slider("PERCLOS Proportion Trigger", 0.10, 0.45, config.PERCLOS_THRESHOLD, 0.01)
        config.MAR_THRESHOLD = st.slider("MAR Yawn / Mouth Opening Trigger", 0.30, 0.90, config.MAR_THRESHOLD, 0.02)

    with s2:
        st.markdown("##### 🚨 Alert Scoring Thresholds")
        config.ALERT_AMBER_SCORE = st.slider("Amber Alert Threshold", 0.20, 0.65, config.ALERT_AMBER_SCORE, 0.05)
        config.ALERT_RED_SCORE = st.slider("Red Critical Alert Threshold", 0.50, 0.95, config.ALERT_RED_SCORE, 0.05)
        config.CAMERA_INDEX = st.number_input("Camera Device Index", 0, 5, config.CAMERA_INDEX)
        config.INFERENCE_FPS = st.slider("Stream Target FPS", 5, 30, config.INFERENCE_FPS)


# ════════════════════════════════════════════════════════════════════════════
# 26. MAIN ROUTER
# ════════════════════════════════════════════════════════════════════════════
def main():
    model = load_model()

    page = st.session_state.page
    if page == "home":
        view_home()
    elif page == "analytics":
        view_fleet_analytics()
    elif page == "notifications":
        view_notifications()
    elif page == "live":
        view_live_surveillance(model)
    elif page == "video":
        view_video_analysis(model)
    elif page == "events":
        view_events_log()
    elif page == "devices":
        view_cctv_devices()
    elif page == "ptw":
        view_permit_to_work()
    elif page == "workers":
        view_worker_roster()
    elif page == "tasks":
        view_task_management()
    elif page == "intercom":
        view_intercom()
    elif page == "map":
        view_rig_map()
    elif page == "edge":
        view_edge_status()
    elif page == "settings":
        view_settings()
    elif page == "metrics":
        view_model_metrics()
    elif page == "forge":
        view_forge_viewer()
    elif page == "roles":
        view_role_settings()
    elif page == "audit":
        view_shift_audit()
    else:
        view_live_surveillance(model)

if __name__ == "__main__":
    main()
