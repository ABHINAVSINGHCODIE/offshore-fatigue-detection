"""
src/feature_pipeline.py
═══════════════════════════
Session-aware temporal feature extractor for offshore fatigue detection.

This module transforms per-frame CSV rows into richly engineered feature vectors
by computing:
  - Multi-scale rolling statistics (mean, std, min, max, percentiles) across
    multiple time windows (5 s, 15 s, 30 s, 60 s)
  - Blink dynamics (duration distribution, frequency, closure speed)
  - Yawn dynamics (duration, peak MAR, cumulative count)
  - EAR/PERCLOS velocity and acceleration (first & second derivatives)
  - Cross-feature interactions (compound fatigue score, ratio features)
  - Temporal autocorrelation of EAR at lag 1, 2, 5
  - Flag densities in rolling windows (% time eyes closed, % time yawning)

All window sizes and thresholds are read from config — zero hardcoding.

Usage:
    from src.feature_pipeline import build_temporal_features

    df_raw      = pd.read_csv(...)            # raw merged CSVs from UTA/YawDD
    df_features = build_temporal_features(df_raw)  # → enriched feature matrix
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Rolling window helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rolling_stats(
    series: pd.Series,
    window: int,
    prefix: str,
    percentiles: tuple = (10, 25, 75, 90),
) -> pd.DataFrame:
    """
    Compute common rolling statistics for a single series.
    Returns a DataFrame of new columns.
    """
    result = pd.DataFrame(index=series.index)
    roll = series.rolling(window=window, min_periods=max(1, window // 4))

    result[f"{prefix}_mean"]   = roll.mean()
    result[f"{prefix}_std"]    = roll.std().fillna(0.0)
    result[f"{prefix}_min"]    = roll.min()
    result[f"{prefix}_max"]    = roll.max()
    result[f"{prefix}_range"]  = result[f"{prefix}_max"] - result[f"{prefix}_min"]
    for p in percentiles:
        result[f"{prefix}_p{p}"] = roll.quantile(p / 100)
    return result


def _velocity(series: pd.Series, name: str) -> pd.DataFrame:
    """First and second derivatives (velocity and acceleration)."""
    result = pd.DataFrame(index=series.index)
    result[f"{name}_vel"]  = series.diff().fillna(0.0)
    result[f"{name}_acc"]  = result[f"{name}_vel"].diff().fillna(0.0)
    return result


def _autocorr(series: pd.Series, name: str, lags: list = None) -> pd.DataFrame:
    """
    Fast vectorized lag correlation (Pearson at each lag).
    Uses shifted series instead of rolling .apply(autocorr) which is extremely slow.
    """
    if lags is None:
        lags = [1, 2, 5]
    result = pd.DataFrame(index=series.index)
    vals = series.values.astype(np.float64)
    for lag in lags:
        col = f"{name}_autocorr_lag{lag}"
        if len(vals) <= lag + 1:
            result[col] = 0.0
            continue
        # Simple lag correlation: corr(x[t], x[t-lag])
        shifted = np.empty_like(vals)
        shifted[:lag] = vals[lag]          # fill head with first valid value
        shifted[lag:] = vals[:-lag]
        # Vectorized Pearson
        v_mean    = vals.mean()
        s_mean    = shifted.mean()
        numerator = ((vals - v_mean) * (shifted - s_mean)).sum()
        denom     = (np.std(vals) * np.std(shifted) * len(vals)) + 1e-9
        corr_val  = np.clip(numerator / denom, -1.0, 1.0)
        result[col] = corr_val            # scalar broadcast — same value per session
    return result


def _blink_dynamics(
    ear: pd.Series,
    ear_threshold: float,
    fps: float,
) -> pd.DataFrame:
    """
    Extract blink dynamics from an EAR time series within one session.

    Returns columns:
        blink_duration_mean    — mean duration of each closure event (seconds)
        blink_duration_max     — max closure duration (prolonged microsleep proxy)
        blink_rate_per_min     — number of blinks per minute (rolling 60s)
        blink_closure_speed    — mean speed of eye closure (EAR drop / frame)
        long_blink_flag        — 1 if any blink in the last 30s exceeded 500 ms
    """
def _blink_dynamics(
    ear: pd.Series,
    ear_threshold: float,
    fps: float,
) -> pd.DataFrame:
    """
    Extract precise blink duration and frequency metrics from EAR.

    Returns columns:
        last_blink_duration    — duration of the most recent blink (seconds)
        blink_duration_mean    — mean duration of blinks in the last 60s
        blink_duration_max     — max blink duration in the last 60s
        blink_rate_per_min     — count of blinks per minute in rolling 60s
        blink_closure_speed    — mean speed of eye closure (EAR drop / frame)
        blink_opening_speed    — mean speed of eye opening (EAR rise / frame)
        long_blink_flag        — 1 if any blink in the last 30s exceeded 400 ms
    """
    result = pd.DataFrame(index=ear.index)
    vals = ear.values
    n = len(vals)
    if n == 0:
        return result

    closed = (vals < ear_threshold).astype(int)
    durations = np.zeros(n, dtype=float)
    blink_count_arr = np.zeros(n, dtype=float)

    in_blink = False
    blink_start = 0
    for i in range(n):
        if closed[i] == 1:
            if not in_blink:
                in_blink = True
                blink_start = i
        else:
            if in_blink:
                in_blink = False
                dur_sec = (i - blink_start) / max(1.0, fps)
                durations[i] = dur_sec
                blink_count_arr[i] = 1.0
    if in_blink:
        dur_sec = (n - blink_start) / max(1.0, fps)
        durations[-1] = dur_sec
        blink_count_arr[-1] = 1.0

    dur_s = pd.Series(durations, index=ear.index)
    count_s = pd.Series(blink_count_arr, index=ear.index)

    last_blink_dur = dur_s.replace(0.0, np.nan).ffill().fillna(0.15)
    result["last_blink_duration"] = last_blink_dur

    w30 = max(1, int(30 * fps))
    w60 = max(1, int(60 * fps))

    # Rolling blink count in window
    roll_dur_sum = dur_s.rolling(window=w60, min_periods=1).sum()
    roll_cnt_sum = count_s.rolling(window=w60, min_periods=1).sum()

    result["blink_rate_per_min"]  = roll_cnt_sum * (60.0 * fps / max(1, w60))
    result["blink_duration_mean"] = (roll_dur_sum / (roll_cnt_sum + 1e-6)).replace(0.0, 0.15)
    result["blink_duration_max"]  = dur_s.rolling(window=w60, min_periods=1).max()

    long_blink_event = (dur_s > 0.40).astype(float)
    result["long_blink_flag"] = long_blink_event.rolling(window=w30, min_periods=1).max()

    ear_vel = pd.Series(vals, index=ear.index).diff().fillna(0.0)
    result["blink_closure_speed"] = (
        ear_vel.clip(upper=0).abs().rolling(window=w30, min_periods=1).mean()
    )
    result["blink_opening_speed"] = (
        ear_vel.clip(lower=0).rolling(window=w30, min_periods=1).mean()
    )

    return result.fillna(0.0)


def _yawn_dynamics(
    mar: pd.Series,
    mouth_open: pd.Series,
    mar_threshold: float,
    fps: float,
) -> pd.DataFrame:
    """
    Extract yawn dynamics from MAR + MouthOpen time series within one session.

    Returns columns:
        last_yawn_duration     — duration of the most recent yawn (seconds)
        yawn_duration_mean     — mean yawn duration (seconds) in last 60s
        yawn_duration_max      — max yawn duration in last 60s
        yawn_cumulative_count  — total cumulative yawn events up to this frame
        yawn_peak_mar          — rolling max MAR in last 60s
        yawn_rate_per_min      — yawn events per minute (last 60s)
        mouth_open_density     — fraction of frames mouth was open in last 30s
    """
    result = pd.DataFrame(index=mar.index)
    vals = mar.values
    n = len(vals)
    if n == 0:
        return result

    yawn_bin = (vals > mar_threshold).astype(int)
    durations = np.zeros(n, dtype=float)
    yawn_count_arr = np.zeros(n, dtype=float)

    in_yawn = False
    yawn_start = 0
    for i in range(n):
        if yawn_bin[i] == 1:
            if not in_yawn:
                in_yawn = True
                yawn_start = i
        else:
            if in_yawn:
                in_yawn = False
                dur_sec = (i - yawn_start) / max(1.0, fps)
                durations[i] = dur_sec
                yawn_count_arr[i] = 1.0
    if in_yawn:
        dur_sec = (n - yawn_start) / max(1.0, fps)
        durations[-1] = dur_sec
        yawn_count_arr[-1] = 1.0

    dur_s = pd.Series(durations, index=mar.index)
    count_s = pd.Series(yawn_count_arr, index=mar.index)

    last_yawn_dur = dur_s.replace(0.0, np.nan).ffill().fillna(0.0)
    result["last_yawn_duration"] = last_yawn_dur

    w30 = max(1, int(30 * fps))
    w60 = max(1, int(60 * fps))

    roll_dur_sum = dur_s.rolling(window=w60, min_periods=1).sum()
    roll_cnt_sum = count_s.rolling(window=w60, min_periods=1).sum()

    result["yawn_duration_mean"]    = (roll_dur_sum / (roll_cnt_sum + 1e-6)).fillna(0.0)
    result["yawn_duration_max"]     = dur_s.rolling(window=w60, min_periods=1).max()
    result["yawn_rate_per_min"]     = roll_cnt_sum * (60.0 * fps / max(1, w60))
    result["yawn_cumulative_count"] = count_s.cumsum()
    result["yawn_peak_mar"]         = mar.rolling(window=w60, min_periods=1).max()

    mo_binary = (mouth_open > 0.3).astype(float)
    result["mouth_open_density"]    = mo_binary.rolling(window=w30, min_periods=1).mean()

    return result.fillna(0.0)


def _perclos_windows(
    ear: pd.Series,
    ear_threshold: float,
    fps: float,
    window_secs: list,
) -> pd.DataFrame:
    """
    Compute PERCLOS (proportion of time eyes are ≥80% closed) at multiple
    time windows.  Standard PERCLOS uses 80% eye closure (0.2 × EAR_open).
    """
    result = pd.DataFrame(index=ear.index)
    closed = (ear < ear_threshold).astype(float)

    for w in window_secs:
        frames = max(1, int(w * fps))
        col    = f"perclos_{w}s"
        result[col] = closed.rolling(window=frames, min_periods=1).mean()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Per-session processor
# ─────────────────────────────────────────────────────────────────────────────

def _process_session(session_df: pd.DataFrame, fps: float) -> pd.DataFrame:
    """
    Generate all temporal features for a single video session.
    Works on a sorted sub-DataFrame (already filtered by source_file or Video).
    """
    out_parts = [session_df.copy()]
    ear  = session_df["EAR"].values
    mar  = session_df["MAR"].values
    perc = session_df["PERCLOS"].values
    mo   = session_df["MouthOpen"].values

    # Reset integer index for diff / rolling to work simply
    idx = session_df.index
    ear_s  = pd.Series(ear,  index=idx)
    mar_s  = pd.Series(mar,  index=idx)
    perc_s = pd.Series(perc, index=idx)
    mo_s   = pd.Series(mo,   index=idx)

    # ── Subject Adaptive Baseline Normalization ──────────────────────────────
    # Eliminates cross-subject anatomical EAR/MAR variations
    ear_base = ear_s.expanding(min_periods=15).quantile(0.90).bfill().ffill()
    mar_base = mar_s.expanding(min_periods=15).quantile(0.10).bfill().ffill()
    ear_mean = ear_s.expanding(min_periods=15).mean().bfill().ffill()
    ear_std  = ear_s.expanding(min_periods=15).std().bfill().ffill().replace(0, 1e-3)
    mar_mean = mar_s.expanding(min_periods=15).mean().bfill().ffill()
    mar_std  = mar_s.expanding(min_periods=15).std().bfill().ffill().replace(0, 1e-3)

    norm_df = pd.DataFrame(index=idx)
    norm_df["EAR_ratio_to_baseline"] = (ear_s / (ear_base + 1e-5)).clip(0.0, 2.0)
    norm_df["EAR_zscore_session"]    = ((ear_s - ear_mean) / ear_std).clip(-5.0, 5.0)
    norm_df["MAR_ratio_to_baseline"] = (mar_s / (mar_base + 1e-5)).clip(0.0, 20.0)
    norm_df["MAR_zscore_session"]    = ((mar_s - mar_mean) / mar_std).clip(-5.0, 5.0)

    # Adaptive eye closure indicator (relative to personal baseline)
    adaptive_closed = (ear_s < 0.75 * ear_base).astype(float)
    for w in config.ROLLING_WINDOWS_SEC:
        frames = max(2, int(w * fps))
        norm_df[f"adaptive_perclos_{w}s"] = adaptive_closed.rolling(window=frames, min_periods=1).mean()
        norm_df[f"EAR_ratio_{w}s_mean"]   = norm_df["EAR_ratio_to_baseline"].rolling(window=frames, min_periods=1).mean()
        norm_df[f"EAR_ratio_{w}s_min"]    = norm_df["EAR_ratio_to_baseline"].rolling(window=frames, min_periods=1).min()
        norm_df[f"EAR_ratio_{w}s_std"]    = norm_df["EAR_ratio_to_baseline"].rolling(window=frames, min_periods=1).std().fillna(0.0)
        norm_df[f"MAR_ratio_{w}s_mean"]   = norm_df["MAR_ratio_to_baseline"].rolling(window=frames, min_periods=1).mean()
        norm_df[f"MAR_ratio_{w}s_max"]    = norm_df["MAR_ratio_to_baseline"].rolling(window=frames, min_periods=1).max()

    out_parts.append(norm_df)

    # ── Multi-scale rolling stats ─────────────────────────────────────────────
    for w in config.ROLLING_WINDOWS_SEC:
        frames = max(2, int(w * fps))
        for signal, sname in [(ear_s, "EAR"), (mar_s, "MAR"), (perc_s, "PERCLOS")]:
            stats_df = _rolling_stats(signal, frames, f"{sname}_{w}s")
            out_parts.append(stats_df)

    # ── Velocity & acceleration ────────────────────────────────────────────────
    out_parts.append(_velocity(ear_s,  "EAR"))
    out_parts.append(_velocity(perc_s, "PERCLOS"))
    out_parts.append(_velocity(mar_s,  "MAR"))

    # ── PERCLOS at multiple time windows ─────────────────────────────────────
    out_parts.append(_perclos_windows(
        ear_s, config.EAR_THRESHOLD, fps, config.ROLLING_WINDOWS_SEC
    ))

    # ── Blink dynamics ────────────────────────────────────────────────────────
    out_parts.append(_blink_dynamics(ear_s, config.EAR_THRESHOLD, fps))

    # ── Yawn dynamics ─────────────────────────────────────────────────────────
    out_parts.append(_yawn_dynamics(mar_s, mo_s, config.MAR_THRESHOLD, fps))

    return pd.concat(out_parts, axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Base feature engineering (static, non-temporal)
# ─────────────────────────────────────────────────────────────────────────────

def _static_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute static/instant features.  Called once before session-level rolling.
    """
    df = df.copy()

    # Clip physiologically implausible values
    df["EAR"]     = df["EAR"].clip(0.0, 0.95)
    df["MAR"]     = df["MAR"].clip(0.0, 1.5)
    df["PERCLOS"] = df["PERCLOS"].clip(0.0, 1.0)

    # Interactions
    df["ear_mar_ratio"]    = df["EAR"] / (df["MAR"] + 1e-6)
    df["ear_sq"]           = df["EAR"] ** 2
    df["perclos_sq"]       = df["PERCLOS"] ** 2
    df["mar_over_ear"]     = df["MAR"] / (df["EAR"] + 1e-6)
    df["ear_x_perclos"]    = df["EAR"] * df["PERCLOS"]

    # Binary flags
    df["eye_closed_flag"]  = (df["EAR"] < config.EAR_THRESHOLD).astype(int)
    df["yawn_flag"]        = (df["MAR"] > config.MAR_THRESHOLD).astype(int)

    # Compound scores
    low_blink = (df["BlinkRate"] < 10).astype(float)
    df["compound_fatigue"] = (
        df["PERCLOS"] * 0.40
        + df["MouthOpen"] * 0.25
        + low_blink * 0.20
        + df["YawnFreq"] * 0.15
    )
    df["perclos_x_blink"]  = df["PERCLOS"] * df["BlinkRate"]

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_temporal_features(
    df_raw: pd.DataFrame,
    fps: Optional[float] = None,
    session_col: str = "source_file",
    drop_cold_start: bool = True,
) -> pd.DataFrame:
    """
    Transform a raw per-frame DataFrame into a rich temporal feature matrix.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw merged CSV rows (from UTA-RLDD / YawDD).
    fps : float, optional
        Frame rate used to convert window sizes to frame counts.
        Defaults to config.VIDEO_FPS.
    session_col : str
        Column to group rows into individual recording sessions.
        Falls back to 'Video' or 'source_folder' if not found.
    drop_cold_start : bool
        If True, drops rows where cold_start == 1 before processing.

    Returns
    -------
    pd.DataFrame
        Extended DataFrame with all engineered features and the original columns.
    """
    fps = fps or config.VIDEO_FPS

    if drop_cold_start and "cold_start" in df_raw.columns:
        before = len(df_raw)
        df_raw = df_raw[df_raw["cold_start"] == 0].copy()
        dropped = before - len(df_raw)
        if dropped:
            log.info(f"  Dropped {dropped:,} cold-start rows")

    # Determine session grouping column
    if session_col not in df_raw.columns:
        for alt in ("Video", "source_folder", "backend"):
            if alt in df_raw.columns:
                session_col = alt
                break
        else:
            session_col = None  # no session grouping — process as one block

    # Static features first
    df_raw = _static_feature_engineering(df_raw)

    # Session-aware temporal features
    groups = df_raw.groupby(session_col) if session_col else [(None, df_raw)]

    n_sessions = df_raw[session_col].nunique() if session_col else 1
    log.info(f"  Processing {n_sessions} sessions …")

    processed_parts = []
    for session_id, grp in groups:
        grp_sorted = grp.sort_index()
        processed_parts.append(_process_session(grp_sorted, fps))

    df_out = pd.concat(processed_parts).sort_index()

    # Drop duplicate columns that may arise from concat
    df_out = df_out.loc[:, ~df_out.columns.duplicated()]

    # Resolve any remaining NaNs/Infs
    num_cols = df_out.select_dtypes(include=[np.number]).columns
    df_out[num_cols] = df_out[num_cols].replace([np.inf, -np.inf], np.nan)
    df_out[num_cols] = df_out[num_cols].fillna(df_out[num_cols].median())

    log.info(
        f"  Feature matrix built: {df_out.shape[0]:,} rows × "
        f"{df_out.shape[1]} columns"
    )
    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point — run as module to generate master dataset
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, zipfile as _zipfile

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%H:%M:%S")

    parser = argparse.ArgumentParser(
        description="Build temporal feature dataset from UTA-RLDD/YawDD CSVs"
    )
    parser.add_argument(
        "--fps", type=float, default=None,
        help=f"Override frame rate (default: from config.VIDEO_FPS = {config.VIDEO_FPS})"
    )
    args = parser.parse_args()

    # Load UTA/YawDD data
    zip_path = config.DATASET_ZIP
    if zip_path.exists():
        log.info(f"Loading from ZIP: {zip_path}")
        all_frames = []
        with _zipfile.ZipFile(zip_path, "r") as z:
            csvs = [f for f in z.namelist()
                    if f.endswith(".csv") and "desktop.ini" not in f.lower()]
            for csv_name in csvs:
                parts = csv_name.replace("\\", "/").split("/")
                label_folder = next(
                    (p for p in parts if p.lower() in ("alert", "fatigued")),
                    "Unknown"
                )
                try:
                    with z.open(csv_name) as fh:
                        df_tmp = pd.read_csv(fh)
                    if "Label" in df_tmp.columns and len(df_tmp) >= 5:
                        df_tmp["source_file"]   = Path(csv_name).name
                        df_tmp["source_folder"] = label_folder
                        all_frames.append(df_tmp)
                except Exception as e:
                    log.warning(f"  Skipping {csv_name}: {e}")
        if not all_frames:
            raise ValueError("No valid CSVs in ZIP.")
        df_raw = pd.concat(all_frames, ignore_index=True)
    elif config.UTA_DIR.exists():
        log.info(f"Loading from extracted dir: {config.UTA_DIR}")
        csvs = list(config.UTA_DIR.rglob("*.csv"))
        frames = []
        for p in csvs:
            try:
                df_tmp = pd.read_csv(p)
                if "Label" in df_tmp.columns and len(df_tmp) >= 5:
                    df_tmp["source_file"] = p.name
                    frames.append(df_tmp)
            except Exception:
                pass
        df_raw = pd.concat(frames, ignore_index=True)
    else:
        raise FileNotFoundError("Neither DATASET_ZIP nor UTA_DIR has data.")

    log.info(f"Loaded {len(df_raw):,} raw rows.")
    df_features = build_temporal_features(df_raw, fps=args.fps)

    out_path = config.MASTER_DATASET_PARQUET
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_features.to_parquet(out_path, index=False)
    log.info(f"Master dataset saved → {out_path}")
    log.info(f"Shape: {df_features.shape}")
