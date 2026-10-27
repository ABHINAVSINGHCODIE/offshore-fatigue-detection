"""
config.py — Central configuration for Offshore Fatigue Detection.
All paths are relative to this file's location. Override via environment variables.
No hardcoded absolute paths anywhere.
"""

import os
from pathlib import Path

# ── Base Directories ──────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = Path(os.environ.get("OFFSHORE_DATA_DIR",   str(BASE_DIR / "data")))
MODELS_DIR = Path(os.environ.get("OFFSHORE_MODELS_DIR", str(BASE_DIR / "models")))
LOGS_DIR   = Path(os.environ.get("OFFSHORE_LOGS_DIR",   str(BASE_DIR / "logs")))

# ── Raw Dataset Paths ─────────────────────────────────────────────────────────
RAW_DIR     = DATA_DIR / "raw"
UTA_DIR     = Path(os.environ.get("OFFSHORE_UTA_DIR",   str(RAW_DIR / "uta_yawdd")))
MRL_DIR     = Path(os.environ.get("OFFSHORE_MRL_DIR",   str(RAW_DIR / "mrl_eye")))
MPII_DIR    = Path(os.environ.get("OFFSHORE_MPII_DIR",  str(RAW_DIR / "mpii_pose")))
COCO_DIR    = Path(os.environ.get("OFFSHORE_COCO_DIR",  str(RAW_DIR / "coco_pose")))

# ── Source ZIP (user-provided dataset) ────────────────────────────────────────
# Default: look for the ZIP on the Desktop, but overridable via env var
_default_zip = Path.home() / "OneDrive" / "Desktop" / "Dataset Combined.zip"
DATASET_ZIP = Path(os.environ.get("OFFSHORE_DATASET_ZIP", str(_default_zip)))

# ── Processed / Merged Data ───────────────────────────────────────────────────
PROCESSED_DIR               = DATA_DIR / "processed"
MERGED_CSV                  = PROCESSED_DIR / "merged_features.csv"
UTA_EXTRACTED_CSV           = PROCESSED_DIR / "uta_yawdd_extracted.csv"
MASTER_DATASET_PARQUET      = PROCESSED_DIR / "master_training_dataset.parquet"
UTA_TEMPORAL_CSV            = PROCESSED_DIR / "uta_yawdd_temporal_features.csv"
MRL_FEATURES_CSV            = PROCESSED_DIR / "mrl_eye_features.csv"
POSTURAL_FEATURES_CSV       = PROCESSED_DIR / "postural_features.csv"

# ── Model Artifacts ───────────────────────────────────────────────────────────
LGBM_MODEL_PATH   = MODELS_DIR / "fatigue_lgbm.pkl"
XGB_MODEL_PATH    = MODELS_DIR / "fatigue_xgb.pkl"
CAT_MODEL_PATH    = MODELS_DIR / "fatigue_cat.pkl"
SCALER_PATH       = MODELS_DIR / "feature_scaler.pkl"
LABEL_ENC_PATH    = MODELS_DIR / "label_encoder.pkl"
METRICS_PATH      = MODELS_DIR / "training_metrics.json"
CALIBRATOR_PATH   = MODELS_DIR / "probability_calibrator.pkl"

# ── Logging ───────────────────────────────────────────────────────────────────
HSE_EVENTS_CSV      = LOGS_DIR / "hse_events.csv"
SHIFT_REPORTS_DIR   = LOGS_DIR / "shift_reports"

# ── Feature Engineering Parameters ───────────────────────────────────────────
EAR_THRESHOLD       = float(os.environ.get("OFFSHORE_EAR_THRESH",    "0.20"))
MAR_THRESHOLD       = float(os.environ.get("OFFSHORE_MAR_THRESH",    "0.60"))
PERCLOS_THRESHOLD   = float(os.environ.get("OFFSHORE_PERCLOS_THRESH", "0.20"))
TEMPORAL_WINDOW     = int(os.environ.get("OFFSHORE_TEMP_WIN",         "60"))    # frames
BLINK_WIN_SEC       = float(os.environ.get("OFFSHORE_BLINK_WIN",      "60.0"))  # seconds

# ── Temporal Feature Windows (seconds) ───────────────────────────────────────
# Used by feature_pipeline.py to compute multi-scale rolling statistics.
ROLLING_WINDOWS_SEC = [
    int(w) for w in
    os.environ.get("OFFSHORE_ROLL_WINS", "5,15,30,60").split(",")
]

# Video frame rate used for converting window sizes to frame counts
VIDEO_FPS = float(os.environ.get("OFFSHORE_VIDEO_FPS", "30.0"))

# ── Postural Feature Thresholds ───────────────────────────────────────────────
NECK_TILT_FATIGUE_DEG   = float(os.environ.get("OFFSHORE_NECK_TILT",    "20.0"))
HEAD_DROP_FATIGUE_RATIO = float(os.environ.get("OFFSHORE_HEAD_DROP",    "0.15"))
SHOULDER_SLOPE_MAX_RAD  = float(os.environ.get("OFFSHORE_SHLD_SLOPE",   "0.20"))

# ── Alert Thresholds ──────────────────────────────────────────────────────────
ALERT_AMBER_SCORE   = float(os.environ.get("OFFSHORE_AMBER_THRESH", "0.45"))
ALERT_RED_SCORE     = float(os.environ.get("OFFSHORE_RED_THRESH",   "0.70"))

# ── Webcam / Video ────────────────────────────────────────────────────────────
CAMERA_INDEX    = int(os.environ.get("OFFSHORE_CAM_IDX",   "0"))
FRAME_WIDTH     = int(os.environ.get("OFFSHORE_FRAME_W",   "640"))
FRAME_HEIGHT    = int(os.environ.get("OFFSHORE_FRAME_H",   "480"))
INFERENCE_FPS   = int(os.environ.get("OFFSHORE_INF_FPS",   "15"))

# ── Training Parameters ───────────────────────────────────────────────────────
TEST_SIZE       = float(os.environ.get("OFFSHORE_TEST_SIZE",   "0.15"))
VAL_SIZE        = float(os.environ.get("OFFSHORE_VAL_SIZE",    "0.15"))
RANDOM_STATE    = int(os.environ.get("OFFSHORE_RAND_STATE",    "42"))
CV_FOLDS        = int(os.environ.get("OFFSHORE_CV_FOLDS",      "5"))
N_OPTUNA_TRIALS = int(os.environ.get("OFFSHORE_OPTUNA_TRIALS", "50"))

# ── Ensemble Weights (tuned) ──────────────────────────────────────────────────
LGBM_WEIGHT = float(os.environ.get("OFFSHORE_LGBM_WEIGHT", "0.40"))
XGB_WEIGHT  = float(os.environ.get("OFFSHORE_XGB_WEIGHT",  "0.35"))
CAT_WEIGHT  = float(os.environ.get("OFFSHORE_CAT_WEIGHT",  "0.25"))

# ── Base Feature Columns (from CSV schema) ────────────────────────────────────
BASE_FEATURE_COLS = [
    "EAR", "MAR", "PERCLOS", "BlinkRate", "YawnFreq", "MouthOpen"
]

# ── Engineered Feature Names (computed during feature engineering) ─────────────
ENGINEERED_FEATURE_COLS = [
    # Interaction & polynomial
    "ear_mar_ratio", "ear_sq", "perclos_sq",
    # Binary flags
    "eye_closed_flag", "yawn_flag",
    # Compound fatigue signal
    "perclos_x_blink", "compound_fatigue",
    # Temporal dynamics (populated by feature_pipeline.py at various windows)
    # Format: {metric}_{window}s_{stat}  — e.g. EAR_5s_mean, EAR_30s_std
]

# ── Static feature set used by training (base + engineered) ──────────────────
# Full list is computed dynamically in train_models.py after feature engineering.
FEATURE_COLS = BASE_FEATURE_COLS.copy()

TARGET_COL  = "Label"
LABEL_MAP   = {"Alert": 0, "Fatigued": 1}

# ── Ensure critical dirs exist ────────────────────────────────────────────────
for _d in [
    DATA_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR, LOGS_DIR,
    SHIFT_REPORTS_DIR, UTA_DIR, MRL_DIR, MPII_DIR, COCO_DIR
]:
    _d.mkdir(parents=True, exist_ok=True)
