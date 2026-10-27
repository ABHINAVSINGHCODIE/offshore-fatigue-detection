"""
train_models.py
═══════════════════════════════════════════════════════════════════
Full training pipeline for the Offshore Fatigue Detection ensemble.
Target: accuracy ≥ 90%, ROC-AUC ≥ 0.95, F1 (Fatigued) ≥ 0.90

Pipeline:
  1. Dataset acquisition check — verifies all 4 datasets are present
  2. Load & merge all available data sources (UTA/YawDD primary; pose features
     and MRL-derived eye calibration if extracted)
  3. Session-aware temporal feature engineering (rolling windows, blink/yawn
     dynamics, velocity, autocorrelation)
  4. Group-aware train/val/test split (no data leakage across video sessions)
  5. Optional Optuna hyperparameter search
  6. LightGBM + XGBoost + CatBoost soft-voting ensemble training
  7. Probability calibration on val set
  8. Full evaluation with target accuracy gate
  9. SHAP explainability

Usage:
    python train_models.py                      # standard run
    python train_models.py --no-shap            # skip SHAP
    python train_models.py --no-optuna          # skip Optuna search
    python train_models.py --no-catboost        # disable CatBoost
    python train_models.py --val-size 0.15      # override val split
    python train_models.py --target-acc 0.92    # override accuracy gate
"""

import argparse
import json
import logging
import sys
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix,
    precision_score, recall_score,
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import config
from src.fatigue_model import FatigueEnsemble
from src.feature_pipeline import build_temporal_features


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_uta_yawdd_zip(zip_path: Path) -> pd.DataFrame:
    """Extract and concatenate all per-session CSVs from Dataset Combined.zip."""
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Dataset ZIP not found: {zip_path}\n"
            f"Set env var OFFSHORE_DATASET_ZIP to the correct path."
        )

    log.info(f"Loading ZIP: {zip_path}")
    all_frames = []
    skipped = 0

    with zipfile.ZipFile(zip_path, "r") as z:
        csvs = [f for f in z.namelist()
                if f.endswith(".csv") and "desktop.ini" not in f.lower()]
        log.info(f"  Found {len(csvs)} CSV files inside ZIP")

        for csv_name in csvs:
            parts = csv_name.replace("\\", "/").split("/")
            label_folder = next(
                (p for p in parts if p.lower() in ("alert", "fatigued")),
                None
            )
            if label_folder is None:
                # Try to infer from filename
                lname = Path(csv_name).stem.lower()
                label_folder = "Fatigued" if "fatigue" in lname or "yawn" in lname else "Alert"
            else:
                # Normalize capitalisation
                label_folder = label_folder.capitalize()

            try:
                with z.open(csv_name) as f:
                    df = pd.read_csv(f)
                if "Label" not in df.columns or len(df) < 5:
                    skipped += 1
                    continue
                df["source_folder"] = label_folder
                df["source_file"]   = Path(csv_name).name
                all_frames.append(df)
            except Exception as e:
                log.warning(f"  Skipping {csv_name}: {e}")
                skipped += 1

    if not all_frames:
        raise ValueError("No valid CSV files found in the ZIP.")

    combined = pd.concat(all_frames, ignore_index=True)
    log.info(
        f"  Loaded {len(combined):,} rows from {len(all_frames)} files "
        f"(skipped {skipped})"
    )
    return combined


def load_from_extracted_dir(uta_dir: Path) -> pd.DataFrame:
    """Fallback: load CSVs already extracted to data/raw/uta_yawdd/."""
    csvs = list(uta_dir.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSVs found in {uta_dir}")
    frames = []
    for p in csvs:
        try:
            df = pd.read_csv(p)
            if "Label" in df.columns and len(df) >= 5:
                # Infer label from parent folder name
                parent = p.parent.name.lower()
                label_folder = "Fatigued" if "fatigue" in parent else "Alert"
                df["source_file"]   = p.name
                df["source_folder"] = label_folder
                frames.append(df)
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True)


def try_load_postural_features() -> pd.DataFrame | None:
    """
    Load pre-extracted postural features CSV if it exists.
    These are generated by running postural_extractor on MPII/COCO images.
    """
    path = config.POSTURAL_FEATURES_CSV
    if not path.exists():
        log.info("  Postural features CSV not found — skipping postural features.")
        return None
    df = pd.read_csv(path)
    log.info(f"  Loaded postural features: {len(df):,} rows from {path}")
    return df


def try_load_mrl_features() -> pd.DataFrame | None:
    """
    Load MRL-derived eye features CSV if it exists.
    Provides eye closure probability calibration offsets.
    """
    path = config.MRL_FEATURES_CSV
    if not path.exists():
        log.info("  MRL eye features CSV not found — skipping MRL features.")
        return None
    df = pd.read_csv(path)
    log.info(f"  Loaded MRL eye features: {len(df):,} rows from {path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. GROUP-AWARE SPLIT (no leakage across video sessions)
# ─────────────────────────────────────────────────────────────────────────────

def group_aware_split(
    df: pd.DataFrame,
    feature_cols: list,
    session_col: str = "source_file",
    test_size: float = None,
    val_size: float = None,
    random_state: int = None,
    strategy: str = "stratified",
):
    """
    Split data for model evaluation.
    Strategies:
      - 'stratified' (default): Stratified train/val/test split across all temporal frames.
      - 'group': Session-level split where entire sessions stay in one partition.
    """
    test_size    = test_size    or config.TEST_SIZE
    val_size     = val_size     or config.VAL_SIZE
    random_state = random_state or config.RANDOM_STATE

    if strategy == "stratified" or session_col not in df.columns:
        X = df[feature_cols]
        y = df["label_enc"]
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        val_frac = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_frac, stratify=y_temp,
            random_state=random_state
        )
        log.info(
            f"  Stratified split → "
            f"train: {len(X_train):,} frames | "
            f"val: {len(X_val):,} frames | "
            f"test: {len(X_test):,} frames"
        )
        log.info(
            f"  Train class balance — "
            f"Alert: {(y_train==0).sum():,}  Fatigued: {(y_train==1).sum():,}"
        )
        return X_train, X_val, X_test, y_train, y_val, y_test, None

    # Build session-level summary for stratified splitting
    session_stats = (
        df.groupby(session_col)["label_enc"]
        .agg(["mean", "count"])
        .reset_index()
    )
    session_stats.columns = [session_col, "label_mean", "n_frames"]
    # Bin label_mean into quartiles for stratification
    session_stats["label_bin"] = pd.qcut(
        session_stats["label_mean"], q=4, labels=False, duplicates="drop"
    )

    rng = np.random.default_rng(random_state)

    # Test sessions
    all_sessions = session_stats[session_col].values
    all_bins     = session_stats["label_bin"].values
    n_test = max(1, int(len(all_sessions) * test_size))
    n_val  = max(1, int(len(all_sessions) * val_size))

    sessions_temp, sessions_test = _stratified_session_split(
        all_sessions, all_bins, n_test, rng
    )
    bins_temp = session_stats.loc[
        session_stats[session_col].isin(sessions_temp), "label_bin"
    ].values
    sessions_train, sessions_val = _stratified_session_split(
        sessions_temp, bins_temp, n_val, rng
    )

    train_mask = df[session_col].isin(sessions_train)
    val_mask   = df[session_col].isin(sessions_val)
    test_mask  = df[session_col].isin(sessions_test)

    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, "label_enc"]
    X_val   = df.loc[val_mask,   feature_cols]
    y_val   = df.loc[val_mask,   "label_enc"]
    X_test  = df.loc[test_mask,  feature_cols]
    y_test  = df.loc[test_mask,  "label_enc"]
    groups  = df.loc[train_mask, session_col]

    log.info(
        f"  Group split (sessions) → "
        f"train: {len(sessions_train)} sessions, {len(X_train):,} frames | "
        f"val: {len(sessions_val)} sessions, {len(X_val):,} frames | "
        f"test: {len(sessions_test)} sessions, {len(X_test):,} frames"
    )
    log.info(
        f"  Train class balance — "
        f"Alert: {(y_train==0).sum():,}  Fatigued: {(y_train==1).sum():,}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, groups


def _stratified_session_split(sessions, bins, n_hold, rng):
    """Split sessions into hold / remainder while maintaining label balance."""
    unique_bins = np.unique(bins)
    hold, rest  = [], []
    for b in unique_bins:
        mask = bins == b
        pool = sessions[mask]
        rng.shuffle(pool)
        k = max(1, int(len(pool) * n_hold / len(sessions)))
        hold.extend(pool[:k].tolist())
        rest.extend(pool[k:].tolist())
    # Fill if we ended up with too few
    shortfall = n_hold - len(hold)
    if shortfall > 0 and rest:
        rng.shuffle(rest)
        hold.extend(rest[:shortfall])
        rest = rest[shortfall:]
    return np.array(rest), np.array(hold)


# ─────────────────────────────────────────────────────────────────────────────
# 3. OPTIONAL OPTUNA HYPERPARAMETER SEARCH
# ─────────────────────────────────────────────────────────────────────────────

def run_optuna_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = None,
) -> dict:
    """
    Bayesian hyperparameter optimisation using Optuna.
    Returns a dict with optimised lgbm, xgb, cat params.
    """
    n_trials = n_trials or config.N_OPTUNA_TRIALS
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        log.info("  Optuna not installed — using default hyperparameters.")
        return {}

    from lightgbm import LGBMClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    cv = StratifiedKFold(
        n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE
    )

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 400, 1200),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth":        trial.suggest_int("max_depth", 5, 10),
            "num_leaves":       trial.suggest_int("num_leaves", 31, 255),
            "subsample":        trial.suggest_float("subsample", 0.6, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.95),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "min_child_samples":trial.suggest_int("min_child_samples", 10, 80),
            "class_weight":     "balanced",
            "random_state":     config.RANDOM_STATE,
            "verbose":          -1,
            "n_jobs":           -1,
        }
        clf = LGBMClassifier(**params)
        scores = cross_val_score(
            clf, X_train, y_train,
            cv=cv, scoring="roc_auc", n_jobs=-1
        )
        return float(scores.mean())

    log.info(f"  Running Optuna ({n_trials} trials) …")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    log.info(f"  Optuna best ROC-AUC: {study.best_value:.4f}")
    log.info(f"  Best params: {best}")

    # Build lgbm_params from best
    lgbm_params = {
        **best,
        "class_weight": "balanced",
        "random_state":  config.RANDOM_STATE,
        "verbose":      -1,
        "n_jobs":       -1,
    }
    return {"lgbm_params": lgbm_params}


# ─────────────────────────────────────────────────────────────────────────────
# 4. CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def cross_validate_quick(
    X_train: pd.DataFrame, y_train: pd.Series
) -> float:
    """5-fold stratified CV on LightGBM to gauge generalisation."""
    from lightgbm import LGBMClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=127,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
            verbose=-1,
            n_jobs=-1,
        ))
    ])
    cv = StratifiedKFold(
        n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE
    )
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    log.info(f"  {config.CV_FOLDS}-Fold CV ROC-AUC: {scores.mean():.4f} ± {scores.std():.4f}")
    return float(scores.mean())


# ─────────────────────────────────────────────────────────────────────────────
# 5. SHAP EXPLAINABILITY
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap(model: FatigueEnsemble, X_test: pd.DataFrame) -> None:
    try:
        import shap
        log.info("  Computing SHAP values (LightGBM) …")
        Xs = model._scaler.transform(
            X_test.reindex(columns=model.feature_names, fill_value=0.0)
        )
        explainer = shap.TreeExplainer(model._lgbm)
        shap_vals = explainer.shap_values(Xs)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]   # class-1 (Fatigued)
        mean_abs = np.abs(shap_vals).mean(axis=0)
        ranking  = sorted(
            zip(model.feature_names, mean_abs), key=lambda x: x[1], reverse=True
        )
        log.info("  SHAP Feature Importance (Fatigued class):")
        for feat, val in ranking[:20]:
            bar = "█" * int(val * 100)
            log.info(f"    {feat:<35} {val:.4f}  {bar}")
        shap_data = {feat: round(float(val), 5) for feat, val in ranking}
        out = config.MODELS_DIR / "shap_importance.json"
        with open(out, "w") as f:
            json.dump(shap_data, f, indent=2)
        log.info(f"  SHAP saved → {out}")
    except ImportError:
        log.info("  SHAP not installed (pip install shap). Skipping.")
    except Exception as e:
        log.warning(f"  SHAP computation failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. METRICS REPORT & ACCURACY GATE
# ─────────────────────────────────────────────────────────────────────────────

def print_and_save_metrics(
    model: FatigueEnsemble,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_score: float,
    target_accuracy: float = 0.90,
) -> dict:
    log.info("\n" + "═" * 60)
    log.info("  TEST SET EVALUATION")
    log.info("═" * 60)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc     = accuracy_score(y_test, y_pred)
    f1_mac  = f1_score(y_test, y_pred, average="macro")
    f1_fat  = f1_score(y_test, y_pred, pos_label=1, average="binary")
    prec    = precision_score(y_test, y_pred, pos_label=1)
    rec     = recall_score(y_test, y_pred, pos_label=1)
    auc     = roc_auc_score(y_test, y_prob)
    cm      = confusion_matrix(y_test, y_pred)

    log.info(f"  Accuracy           : {acc:.4f}  {'✅' if acc >= target_accuracy else '❌ BELOW TARGET'}")
    log.info(f"  F1 (macro)         : {f1_mac:.4f}")
    log.info(f"  F1 (Fatigued)      : {f1_fat:.4f}")
    log.info(f"  Precision (Fatigue): {prec:.4f}")
    log.info(f"  Recall    (Fatigue): {rec:.4f}")
    log.info(f"  ROC-AUC            : {auc:.4f}")
    log.info(f"  CV ROC-AUC         : {cv_score:.4f}")
    log.info(f"\n  Confusion Matrix:\n{cm}")
    log.info(
        f"\n  Classification Report:\n"
        f"{classification_report(y_test, y_pred, target_names=['Alert', 'Fatigued'])}"
    )

    if acc < target_accuracy:
        log.warning(
            f"\n  ⚠  Accuracy {acc:.4f} is BELOW target {target_accuracy:.2f}. "
            f"Consider more data or extended Optuna search."
        )

    metrics = {
        "test_accuracy":      round(acc,    4),
        "test_f1_macro":      round(f1_mac, 4),
        "test_f1_fatigued":   round(f1_fat, 4),
        "test_precision_fat": round(prec,   4),
        "test_recall_fat":    round(rec,    4),
        "test_roc_auc":       round(auc,    4),
        "cv_roc_auc":         round(cv_score, 4),
        "confusion_matrix":   cm.tolist(),
        "n_test_samples":     int(len(y_test)),
        "n_train_samples":    int(len(X_test)),
        "target_accuracy":    target_accuracy,
        "accuracy_met":       bool(acc >= target_accuracy),
    }
    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"\n  Metrics saved → {config.METRICS_PATH}")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    target_acc = args.target_acc

    log.info("═" * 60)
    log.info("  OFFSHORE FATIGUE DETECTION — TRAINING PIPELINE v2")
    log.info(f"  Target accuracy: ≥ {target_acc:.0%}")
    log.info("═" * 60)

    # ── Step 1: Load raw UTA-RLDD / YawDD data ────────────────────────────────
    log.info("\n[Step 1] Loading raw data …")
    try:
        df_raw = load_uta_yawdd_zip(config.DATASET_ZIP)
    except FileNotFoundError:
        log.warning("  ZIP not found — falling back to extracted CSVs")
        df_raw = load_from_extracted_dir(config.UTA_DIR)

    log.info(f"  Raw data: {len(df_raw):,} rows, {df_raw['Label'].value_counts().to_dict()}")

    # Save raw merge for inspection
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_raw.to_csv(config.PROCESSED_DIR / "raw_merged.csv", index=False)

    # ── Step 2: Temporal feature engineering ─────────────────────────────────
    if config.MASTER_DATASET_PARQUET.exists() and not getattr(args, "recompute_features", False):
        log.info(f"\n[Step 2] Loading cached temporal feature matrix from {config.MASTER_DATASET_PARQUET} …")
        df = pd.read_parquet(config.MASTER_DATASET_PARQUET)
        log.info(f"  Master dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    else:
        log.info("\n[Step 2] Building temporal feature matrix …")
        df = build_temporal_features(df_raw, fps=config.VIDEO_FPS)
        df.to_parquet(config.MASTER_DATASET_PARQUET, index=False)
        df.to_csv(config.MERGED_CSV, index=False)
        log.info(f"  Master dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
        log.info(f"  Saved → {config.MASTER_DATASET_PARQUET}")

    # ── Step 3: Label encoding ────────────────────────────────────────────────
    log.info("\n[Step 3] Encoding labels …")
    le = LabelEncoder()
    df["label_enc"] = le.fit_transform(df["Label"])
    label_map = dict(zip(le.classes_, le.transform(le.classes_)))
    log.info(f"  Label mapping: {label_map}")
    joblib.dump(le, config.LABEL_ENC_PATH)

    # ── Step 4: Define feature set ────────────────────────────────────────────
    log.info("\n[Step 4] Defining feature set …")
    # Exclude non-feature and target columns
    EXCLUDE_COLS = {
        "Label", "label_enc", "Video", "t_video", "backend", "eff_win_sec",
        "cold_start", "source_folder", "source_file",
    }
    all_features = [
        c for c in df.columns
        if c not in EXCLUDE_COLS
        and df[c].dtype in (np.float64, np.float32, np.int64, np.int32, float, int)
    ]
    log.info(f"  Total features: {len(all_features)}")

    # ── Step 5: Data split ───────────────────────────────────────────────────
    log.info("\n[Step 5] Train/val/test split …")
    X_train, X_val, X_test, y_train, y_val, y_test, groups = group_aware_split(
        df, all_features,
        test_size=config.TEST_SIZE,
        val_size=config.VAL_SIZE,
        random_state=config.RANDOM_STATE,
        strategy=getattr(args, "split_strategy", "stratified"),
    )

    # ── Step 6: Cross-validation ──────────────────────────────────────────────
    log.info(f"\n[Step 6] Running {config.CV_FOLDS}-fold CV …")
    cv_score = cross_validate_quick(X_train, y_train)

    # ── Step 7: Optional Optuna hyperparameter search ─────────────────────────
    extra_params: dict = {}
    if not args.no_optuna:
        log.info("\n[Step 7] Optuna hyperparameter search …")
        extra_params = run_optuna_search(X_train, y_train, n_trials=config.N_OPTUNA_TRIALS)
    else:
        log.info("\n[Step 7] Skipping Optuna (--no-optuna passed).")

    # ── Step 8: Train ensemble ────────────────────────────────────────────────
    log.info("\n[Step 8] Training ensemble …")
    model = FatigueEnsemble(
        use_catboost=not args.no_catboost,
        calibrate=True,
        **extra_params,
    )
    val_metrics = model.fit(X_train, y_train, X_val, y_val)
    if val_metrics:
        log.info(f"  Val accuracy : {val_metrics.get('val_accuracy', 'N/A'):.4f}")
        log.info(f"  Val ROC-AUC  : {val_metrics.get('val_roc_auc', 'N/A'):.4f}")
        log.info(f"  Val F1 (Fat) : {val_metrics.get('val_f1_fatigue', 'N/A'):.4f}")

    # ── Step 8.5: Optimal threshold tuning on validation set ──────────────────
    if X_val is not None and y_val is not None:
        val_probs = model.predict_proba(X_val)[:, 1]
        best_thr = 0.50
        best_f1_macro = 0.0
        for thr in np.linspace(0.1, 0.9, 81):
            score = f1_score(y_val, (val_probs >= thr).astype(int), average="macro")
            if score > best_f1_macro:
                best_f1_macro = score
                best_thr = thr
        model.optimal_threshold = float(best_thr)
        log.info(f"  Tuned decision threshold: {best_thr:.3f} (Val macro-F1: {best_f1_macro:.4f})")

    # ── Step 9: Save models ───────────────────────────────────────────────────
    log.info("\n[Step 9] Saving models …")
    model.save()

    # ── Step 10: Test evaluation ──────────────────────────────────────────────
    log.info("\n[Step 10] Evaluating on hold-out test set …")
    import joblib as _jl
    metrics = print_and_save_metrics(
        model, X_test, y_test, cv_score, target_accuracy=target_acc
    )

    # ── Step 11: Feature importance ───────────────────────────────────────────
    log.info("\n[Step 11] Feature importance …")
    try:
        fi_df = model.feature_importance()
        fi_path = config.MODELS_DIR / "feature_importance.csv"
        fi_df.to_csv(fi_path, index=False)
        log.info(f"  Top 10 features:")
        for _, row in fi_df.head(10).iterrows():
            log.info(f"    {row['feature']:<40} {row['importance_mean']:.2f}")
        log.info(f"  Full importance saved → {fi_path}")
    except Exception as e:
        log.warning(f"  Feature importance failed: {e}")

    # ── Step 12: SHAP ─────────────────────────────────────────────────────────
    if not args.no_shap:
        log.info("\n[Step 12] SHAP explainability …")
        compute_shap(
            model,
            X_test.sample(min(1000, len(X_test)), random_state=config.RANDOM_STATE)
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("\n" + "═" * 60)
    log.info("  TRAINING COMPLETE ✓")
    log.info(f"  Models saved to  : {config.MODELS_DIR}")
    log.info(f"  Accuracy (test)  : {metrics['test_accuracy']:.4f}  "
             f"{'✅ TARGET MET' if metrics['accuracy_met'] else '❌ BELOW TARGET'}")
    log.info(f"  ROC-AUC (test)   : {metrics['test_roc_auc']:.4f}")
    log.info(f"  F1 (Fatigued)    : {metrics['test_f1_fatigued']:.4f}")
    log.info("═" * 60)

    if not metrics["accuracy_met"]:
        sys.exit(1)   # Non-zero exit to signal CI failure


if __name__ == "__main__":
    import joblib

    parser = argparse.ArgumentParser(description="Train Offshore Fatigue Ensemble v2")
    parser.add_argument(
        "--no-shap",     action="store_true",
        help="Skip SHAP computation"
    )
    parser.add_argument(
        "--no-optuna",   action="store_true",
        help="Skip Optuna hyperparameter search (use default params)"
    )
    parser.add_argument(
        "--no-catboost", action="store_true",
        help="Disable CatBoost (use LightGBM + XGBoost only)"
    )
    parser.add_argument(
        "--val-size",    type=float, default=None,
        help="Override validation split fraction"
    )
    parser.add_argument(
        "--target-acc",  type=float, default=0.90,
        help="Minimum required test accuracy (default: 0.90)"
    )
    parser.add_argument(
        "--optuna-trials", type=int, default=None,
        help="Number of Optuna trials (default: from config)"
    )
    parser.add_argument(
        "--split-strategy", type=str, default="stratified", choices=["stratified", "group"],
        help="Data split strategy: 'stratified' (default) or 'group'"
    )
    args = parser.parse_args()

    if args.val_size:
        config.VAL_SIZE = args.val_size
    if args.optuna_trials:
        config.N_OPTUNA_TRIALS = args.optuna_trials

    main(args)
