"""
src/fatigue_model.py
════════════════════
LightGBM + XGBoost + CatBoost soft-voting ensemble with probability calibration.
Handles training, hyperparameter tuning, persistence, and inference.

Architecture:
  - Three gradient-boosted tree classifiers in soft-vote ensemble
  - Probability calibration via isotonic regression (Platt-scaling)
  - Dynamic feature importance tracking
  - Group-aware cross-validation support
"""

import sys
import json
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Optimal hyperparameter defaults (tuned from cross-validation experiments)
# All overridable via build args — nothing is hardcoded to final values.
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_LGBM_PARAMS = dict(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=7,
    num_leaves=127,
    subsample=0.75,
    colsample_bytree=0.75,
    reg_alpha=0.1,
    reg_lambda=1.0,
    min_child_samples=20,
    class_weight="balanced",
    random_state=config.RANDOM_STATE,
    verbose=-1,
    n_jobs=-1,
)

_DEFAULT_XGB_PARAMS = dict(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=7,
    subsample=0.75,
    colsample_bytree=0.75,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    min_child_weight=5,
    eval_metric="logloss",
    early_stopping_rounds=50,
    random_state=config.RANDOM_STATE,
    verbosity=0,
    n_jobs=-1,
)

_DEFAULT_CAT_PARAMS = dict(
    iterations=800,
    learning_rate=0.03,
    depth=7,
    l2_leaf_reg=3.0,
    subsample=0.75,
    random_seed=config.RANDOM_STATE,
    verbose=False,
    thread_count=-1,
    auto_class_weights="Balanced",
    eval_metric="AUC",
)

# ─────────────────────────────────────────────────────────────────────────────
# FatigueEnsemble
# ─────────────────────────────────────────────────────────────────────────────

class FatigueEnsemble:
    """
    Soft-voting ensemble of LightGBM, XGBoost, and CatBoost classifiers
    with optional probability calibration.

    Training contract:
        X  — pd.DataFrame with feature columns
        y  — pd.Series of numeric labels (0=Alert, 1=Fatigued)

    Inference:
        predict(X)       → np.ndarray of class labels
        predict_proba(X) → np.ndarray of shape (N, 2)
        fatigue_score(x) → float in [0, 1] (probability of Fatigued)
    """

    def __init__(
        self,
        lgbm_weight: float = None,
        xgb_weight: float  = None,
        cat_weight: float   = None,
        lgbm_params: dict   = None,
        xgb_params: dict    = None,
        cat_params: dict    = None,
        use_catboost: bool  = True,
        calibrate: bool     = True,
    ):
        # Weights default to config values (set by Optuna or user env vars)
        self.lgbm_weight = lgbm_weight if lgbm_weight is not None else config.LGBM_WEIGHT
        self.xgb_weight  = xgb_weight  if xgb_weight  is not None else config.XGB_WEIGHT
        self.cat_weight  = cat_weight  if cat_weight   is not None else config.CAT_WEIGHT
        self.use_catboost = use_catboost
        self.calibrate    = calibrate

        # Hyperparameters
        self._lgbm_params = lgbm_params or _DEFAULT_LGBM_PARAMS.copy()
        self._xgb_params  = xgb_params  or _DEFAULT_XGB_PARAMS.copy()
        self._cat_params  = cat_params  or _DEFAULT_CAT_PARAMS.copy()

        # Internal models
        self._lgbm        = None
        self._xgb         = None
        self._cat         = None
        self._scaler      = None
        self._calibrator  = None

        self.feature_names: list = []
        self.is_trained: bool    = False

    # ── Weight normalization ──────────────────────────────────────────────────
    @property
    def _weights(self) -> tuple:
        if self.use_catboost:
            total = self.lgbm_weight + self.xgb_weight + self.cat_weight
            return (
                self.lgbm_weight / total,
                self.xgb_weight  / total,
                self.cat_weight  / total,
            )
        else:
            total = self.lgbm_weight + self.xgb_weight
            return (
                self.lgbm_weight / total,
                self.xgb_weight  / total,
                0.0,
            )

    # ── Training ──────────────────────────────────────────────────────────────
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
    ) -> dict:
        """Train all sub-models. Returns validation metrics dict."""
        from lightgbm import LGBMClassifier, early_stopping, log_evaluation
        from xgboost import XGBClassifier
        from sklearn.preprocessing import StandardScaler

        self.feature_names = list(X_train.columns)

        # Scale features
        self._scaler   = StandardScaler()
        Xs_train       = self._scaler.fit_transform(X_train)
        Xs_val         = self._scaler.transform(X_val) if X_val is not None else None

        # ── LightGBM ──────────────────────────────────────────────────────────
        log.info("[FatigueEnsemble] Training LightGBM …")
        self._lgbm = LGBMClassifier(**self._lgbm_params)
        if Xs_val is not None:
            self._lgbm.fit(
                Xs_train, y_train,
                eval_set=[(Xs_val, y_val)],
                callbacks=[
                    early_stopping(50, verbose=False),
                    log_evaluation(period=-1),
                ],
            )
        else:
            self._lgbm.fit(Xs_train, y_train)

        # ── XGBoost ──────────────────────────────────────────────────────────
        log.info("[FatigueEnsemble] Training XGBoost …")
        scale_pos = float((y_train == 0).sum()) / max(1.0, float((y_train == 1).sum()))
        xgb_p = {**self._xgb_params, "scale_pos_weight": scale_pos}
        self._xgb = XGBClassifier(**xgb_p)
        if Xs_val is not None:
            self._xgb.fit(
                Xs_train, y_train,
                eval_set=[(Xs_val, y_val)],
                verbose=False,
            )
        else:
            self._xgb.fit(Xs_train, y_train)

        # ── CatBoost (optional) ───────────────────────────────────────────────
        if self.use_catboost:
            log.info("[FatigueEnsemble] Training CatBoost …")
            try:
                from catboost import CatBoostClassifier, Pool
                self._cat = CatBoostClassifier(**self._cat_params)
                if Xs_val is not None:
                    train_pool = Pool(Xs_train, label=y_train)
                    val_pool   = Pool(Xs_val,   label=y_val)
                    self._cat.fit(
                        train_pool,
                        eval_set=val_pool,
                        early_stopping_rounds=50,
                        use_best_model=True,
                    )
                else:
                    self._cat.fit(Xs_train, y_train)
            except ImportError:
                log.warning("[FatigueEnsemble] CatBoost not installed — skipping.")
                self.use_catboost = False
                self._cat = None

        self.is_trained = True
        log.info("[FatigueEnsemble] All models trained.")

        # ── Probability calibration on validation set ─────────────────────────
        if self.calibrate and Xs_val is not None:
            self._fit_calibrator(X_val, y_val)

        # ── Validation metrics ────────────────────────────────────────────────
        metrics = {}
        if Xs_val is not None and y_val is not None:
            from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
            y_pred = self.predict(X_val)
            y_prob = self.predict_proba(X_val)[:, 1]
            metrics = {
                "val_accuracy":  float(accuracy_score(y_val, y_pred)),
                "val_f1_macro":  float(f1_score(y_val, y_pred, average="macro")),
                "val_roc_auc":   float(roc_auc_score(y_val, y_prob)),
                "val_f1_fatigue": float(f1_score(y_val, y_pred, pos_label=1, average="binary")),
            }
        return metrics

    def _fit_calibrator(self, X_val: pd.DataFrame, y_val: pd.Series) -> None:
        """Fit isotonic calibrator on validation set raw scores."""
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.base import BaseEstimator, ClassifierMixin

        log.info("[FatigueEnsemble] Fitting probability calibrator …")
        # Get raw uncalibrated probabilities
        raw_proba = self._raw_predict_proba(X_val)

        # Wrap as a simple calibration input
        # We calibrate only the fatigued-class probability (class 1)
        from sklearn.isotonic import IsotonicRegression
        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(raw_proba[:, 1], y_val.values)
        self._calibrator = cal
        log.info("[FatigueEnsemble] Calibration done.")

    # ── Inference ─────────────────────────────────────────────────────────────
    def _scale(self, X: pd.DataFrame) -> np.ndarray:
        if self._scaler is None:
            raise RuntimeError("Model not trained or loaded. Call fit() or load().")
        # Select only known feature columns; fill any missing with 0
        X_aligned = X.reindex(columns=self.feature_names, fill_value=0.0)
        return self._scaler.transform(X_aligned)

    def _raw_predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Soft-voting ensemble probability BEFORE calibration."""
        Xs = self._scale(X)
        w_lgbm, w_xgb, w_cat = self._weights

        p = w_lgbm * self._lgbm.predict_proba(Xs)
        p = p + w_xgb * self._xgb.predict_proba(Xs)
        if self.use_catboost and self._cat is not None:
            p = p + w_cat * self._cat.predict_proba(Xs)
        return p

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Ensemble probability, optionally post-calibrated."""
        raw = self._raw_predict_proba(X)
        if self._calibrator is not None:
            p1_cal = self._calibrator.predict(raw[:, 1])
            p1_cal = np.clip(p1_cal, 0.0, 1.0)
            return np.column_stack([1 - p1_cal, p1_cal])
        return raw

    def predict(self, X: pd.DataFrame, threshold: float = None) -> np.ndarray:
        thr = threshold if threshold is not None else getattr(self, "optimal_threshold", 0.50)
        proba = self.predict_proba(X)
        return (proba[:, 1] >= thr).astype(int)

    def fatigue_score(self, feature_dict: dict) -> float:
        """
        Single-sample inference.

        Args:
            feature_dict : {feature_name: value, …}

        Returns:
            float in [0, 1] — probability of being Fatigued
        """
        row = {k: [feature_dict.get(k, 0.0)] for k in self.feature_names}
        X = pd.DataFrame(row)
        return float(self.predict_proba(X)[0, 1])

    # ── Feature importance ────────────────────────────────────────────────────
    def feature_importance(self) -> pd.DataFrame:
        """Return a sorted DataFrame of mean feature importances across models."""
        records = {}
        if self._lgbm is not None:
            for feat, imp in zip(self.feature_names, self._lgbm.feature_importances_):
                records.setdefault(feat, []).append(float(imp))
        if self._xgb is not None:
            imp_dict = self._xgb.get_booster().get_fscore()
            for feat in self.feature_names:
                records.setdefault(feat, []).append(float(imp_dict.get(f"f{self.feature_names.index(feat)}", 0)))
        if self.use_catboost and self._cat is not None:
            for feat, imp in zip(self.feature_names, self._cat.get_feature_importance()):
                records.setdefault(feat, []).append(float(imp))

        rows = [{"feature": f, "importance_mean": np.mean(v), "importance_std": np.std(v)}
                for f, v in records.items()]
        df = pd.DataFrame(rows).sort_values("importance_mean", ascending=False).reset_index(drop=True)
        return df

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self) -> None:
        """Save ensemble to config-defined paths."""
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._lgbm,   config.LGBM_MODEL_PATH)
        joblib.dump(self._xgb,    config.XGB_MODEL_PATH)
        joblib.dump(self._scaler, config.SCALER_PATH)
        if self.use_catboost and self._cat is not None:
            joblib.dump(self._cat, config.CAT_MODEL_PATH)
        if self._calibrator is not None:
            joblib.dump(self._calibrator, config.CALIBRATOR_PATH)

        meta = {
            "feature_names":  self.feature_names,
            "lgbm_weight":    self.lgbm_weight,
            "xgb_weight":     self.xgb_weight,
            "cat_weight":     self.cat_weight,
            "use_catboost":   self.use_catboost,
            "calibrated":     self._calibrator is not None,
            "saved_at":       time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(config.MODELS_DIR / "ensemble_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        log.info(f"[FatigueEnsemble] Saved to {config.MODELS_DIR}")

    def load(self) -> None:
        """Load ensemble from config-defined paths."""
        self._lgbm   = joblib.load(config.LGBM_MODEL_PATH)
        self._xgb    = joblib.load(config.XGB_MODEL_PATH)
        self._scaler = joblib.load(config.SCALER_PATH)

        if config.CAT_MODEL_PATH.exists():
            try:
                self._cat = joblib.load(config.CAT_MODEL_PATH)
                self.use_catboost = True
            except Exception:
                self._cat = None
                self.use_catboost = False
        else:
            self._cat = None
            self.use_catboost = False

        if config.CALIBRATOR_PATH.exists():
            self._calibrator = joblib.load(config.CALIBRATOR_PATH)

        meta_path = config.MODELS_DIR / "ensemble_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self.feature_names = meta.get("feature_names", config.FEATURE_COLS)
            self.lgbm_weight   = meta.get("lgbm_weight", config.LGBM_WEIGHT)
            self.xgb_weight    = meta.get("xgb_weight",  config.XGB_WEIGHT)
            self.cat_weight    = meta.get("cat_weight",  config.CAT_WEIGHT)
        else:
            self.feature_names = config.FEATURE_COLS

        self.is_trained = True
        log.info(f"[FatigueEnsemble] Loaded from {config.MODELS_DIR}")

    @classmethod
    def from_saved(cls) -> "FatigueEnsemble":
        """Convenience factory: load and return a ready ensemble."""
        obj = cls()
        obj.load()
        return obj
