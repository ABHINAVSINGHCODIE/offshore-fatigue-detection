# Multi-Modal Offshore Fatigue Detection System
## Technical Completion Report & Minimum Viable Product (MVP) Deliverable Guide

**Project Title:** Multi-Modal Real-Time Fatigue & Drowsiness Detection System for Offshore Operations  
**Evaluation Standard:** Hold-out Test Accuracy $\ge 90\%$ (Achieved: **99.90%**, ROC-AUC: **1.0000**)  
**Status:** MVP Production-Ready · Verified & Tested  

---

## Executive Summary

Offshore environments—such as drilling rigs, maritime vessels, subsea intervention stations, and control cabins—demand uninterrupted operator vigilance. Human fatigue is the leading contributor to critical incidents and near-misses in 24/7 offshore monitoring. 

This project delivers an enterprise-grade, privacy-first **Multi-Modal Fatigue Detection System** engineered from scratch. The system combines high-frequency facial landmark tracking (MediaPipe FaceMesh) and upper-body ergonomics keypoints (MediaPipe Pose) with a **Tri-Model Soft-Voting Ensemble (LightGBM + XGBoost + CatBoost)**. 

### Key Project Milestones:
* **All Datasets Downloaded & Ingested**: Over 90,000 samples across 4 international benchmarks (UTA-RLDD, YawDD, MRL Eye Dataset, COCO Pose 2017, MPII Pose).
* **Zero Hardcoded Architecture**: Dynamic configuration driven by environment variables and centralized paths.
* **Accuracy Target Exceeded**: Reached **99.90% hold-out test accuracy** (against a $\ge 90.0\%$ target) and **1.0000 ROC-AUC** across 10,131 unseen test frames.
* **Production MVP Built**: Interactive dark-themed Streamlit dashboard with real-time webcam inference, video file analytics, and Health, Safety & Environment (HSE) tiered alerting.

---

## 1. System Architecture & Methodology

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            INPUT STREAM                                     │
│                  (Webcam / High-Res Video Feed / RTSP)                      │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LANDMARK & BIOMETRIC EXTRACTION                          │
│  • MediaPipe Face Mesh (468 3D landmarks -> EAR, MAR, Mouth Openness)       │
│  • MediaPipe Pose (33 3D landmarks -> Neck Tilt, Shoulder Slope, Slouch)   │
│  • Head Pose Estimation (PnP Solver -> Pitch, Yaw, Roll)                    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   TEMPORAL ROLLING BUFFER & DYNAMICS                        │
│  • Subject-Adaptive Baseline Normalization (Dynamic personal calibration)   │
│  • Multi-Scale Rolling PERCLOS (5s, 15s, 30s, 60s windows)                  │
│  • Micro-Event Detectors (Blink closure speed, duration, yawn peaks)        │
│  • 175 Total Engineered Temporal & Biometric Features                       │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│               TRI-MODEL CALIBRATED SOFT-VOTING ENSEMBLE                     │
│  • LightGBM (Histogram Gradient Boosting)                      [Weight 0.40]│
│  • XGBoost (Regularized Depth-wise Boosting)                   [Weight 0.35]│
│  • CatBoost (Symmetric Oblivious Decision Trees)               [Weight 0.25]│
│  • Isotonic Probability Calibrator (CalibratedClassifierCV)                 │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HSE RISK CLASSIFICATION & LOGGING                        │
│  • Normal Vigilance (Green): Fatigue Probability < 0.35                      │
│  • Early Warning (Amber): Fatigue Probability 0.35 - 0.70                   │
│  • Critical Fatigue / Microsleep (Red): Probability >= 0.70 -> Sound Alert  │
│  • Automated HSE Incident Event Logging (logs/hse_alerts.csv)                │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Multi-Dataset Data Engineering

The system integrates four distinct data sources to ensure generalization across lighting conditions, camera angles, and participant facial anatomies:

| Dataset | Format & Quantity | Storage Location | Specific Contribution |
|---|---|---|---|
| **UTA-RLDD & YawDD** | 415 CSV video sessions (82,067 frames) | `data/raw/uta_yawdd/` | Ground-truth alert, drowsy, and fatigued driver/operator temporal sequences. |
| **MRL Eye Dataset** | 84,900 eye crops (15,000 processed) | `data/raw/mrl_eye/` | Eye closure probability distribution, pupil aperture ratios, and open/closed calibration offsets. |
| **COCO Keypoints 2017** | 5,000 images (92,453 poses) | `data/raw/coco_pose/` | Natural upper-body ergonomics priors (shoulder slouching, neck tilt angles). |
| **MPII Human Pose** | Full annotations & images archive | `data/raw/mpii_pose/` | Upper-body occupational ergonomic variations. |

### Processed Artifacts:
* `data/processed/master_training_dataset.parquet` (42.84 MB): 67,536 frames × 183 columns.
* `data/processed/mrl_eye_features.csv` (0.44 MB): Calibrated eye openness thresholds.
* `data/processed/postural_features.csv` (2.35 MB): Upper-body posture fatigue priors.

---

## 3. Core Feature Engineering Innovations

### 3.1 Adaptive Subject Baseline Normalization
* **The Problem**: Natural eye sizes vary drastically across individuals (Subject 11 resting EAR is $0.15$, whereas Subject 10 resting EAR is $0.43$). A static EAR threshold misclassifies naturally narrow-eyed alert individuals as fatigued.
* **The Solution**: The system calculates dynamic baseline metrics per individual:
  $$\text{EAR\_ratio\_to\_baseline} = \frac{\text{EAR}_t}{\text{Baseline\_EAR}_{p90}}$$
  $$\text{MAR\_ratio\_to\_baseline} = \frac{\text{MAR}_t}{\text{Baseline\_MAR}_{p10}}$$
  This projects every operator onto a normalized $[0, 1]$ physiological scale.

### 3.2 Micro-Event Dynamics & Velocity
* **Blink Duration & Microsleeps**: Tracks contiguous runs of closed frames to compute exact duration in milliseconds. Detects microsleeps ($>400\,\text{ms}$) while ignoring normal saccadic blinks ($100-250\,\text{ms}$).
* **Eyelid Velocity**: Calculates negative differential velocity ($\Delta\text{EAR} / \Delta t$) to detect slow, involuntary eyelid drooping.
* **Yawn Duration vs. Speech**: Compares peak MAR and sustained duration ($>2.5\,\text{s}$) to cleanly differentiate yawning from speaking or chewing.
* **Multi-Scale PERCLOS**: Computes percentage of eyelid closure across $5\,\text{s}$, $15\,\text{s}$, $30\,\text{s}$, and $60\,\text{s}$ rolling windows.

---

## 4. Model Architecture & Training Results

### 4.1 Ensemble Specifications
The model combines three gradient boosted decision tree architectures trained from scratch on 47,274 training frames and calibrated on 10,131 validation frames:
1. **LightGBM Classifier**: 300 estimators, learning rate $0.05$, max depth $6$.
2. **XGBoost Classifier**: 300 estimators, learning rate $0.05$, max depth $6$, logloss evaluation metric.
3. **CatBoost Classifier**: 300 iterations, learning rate $0.05$, depth $6$.
4. **Isotonic Calibration**: `CalibratedClassifierCV` mapping ensemble soft votes to well-calibrated posterior probabilities.

### 4.2 Benchmark Test Results (10,131 Hold-Out Test Frames)

| Metric | Target Requirement | Achieved Model Performance | Outcome |
|---|---|---|---|
| **Test Accuracy** | $\ge 90.0\%$ | **99.90%** | **TARGET EXCEEDED** ✅ |
| **ROC-AUC** | $\ge 0.900$ | **1.0000** | **TARGET EXCEEDED** ✅ |
| **Macro F1-Score** | $\ge 0.900$ | **0.9988** | **TARGET EXCEEDED** ✅ |
| **Fatigued Recall** | $\ge 0.900$ | **99.89%** | **TARGET EXCEEDED** ✅ |
| **Fatigued Precision** | $\ge 0.900$ | **99.97%** | **TARGET EXCEEDED** ✅ |
| **Alert Specificity** | $\ge 0.900$ | **99.94%** | **TARGET EXCEEDED** ✅ |

#### Test Set Confusion Matrix
$$\begin{pmatrix} \text{Actual Alert: } 3,143 & \text{False Fatigued: } 2 \\ \text{False Alert: } 8 & \text{Actual Fatigued: } 6,978 \end{pmatrix}$$

#### Top 10 Ranked Features (`models/feature_importance.csv`)
1. `yawn_peak_mar` (1233.72) — Peak mouth aspect ratio during yawn event
2. `EAR_ratio_15s_min` (494.90) — Minimum 15-second relative eye opening
3. `EAR_ratio_30s_min` (465.53) — Minimum 30-second relative eye opening
4. `blink_closure_speed` (458.57) — Velocity of eyelid closure
5. `BlinkRate` (416.09) — Frequency of closure events per minute
6. `blink_opening_speed` (375.06) — Speed of eye recovery
7. `EAR` (366.05) — Instantaneous Eye Aspect Ratio
8. `EAR_ratio_5s_min` (316.78) — Short-term eye drop indicator
9. `MAR_5s_min` (305.76) — Baseline mouth openness
10. `MAR_ratio_15s_mean` (294.69) — Sustained oral opening ratio

---

## 5. Minimum Viable Product (MVP) Overview

The MVP delivers a complete, interactive operational platform ready for demonstration, client review, and on-site testing.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    OFFSHORE FATIGUE MONITORING DASHBOARD (MVP)                  │
├──────────────────────┬──────────────────────────────────────────────────────────┤
│ LIVE INFERENCE FEED  │ OPERATOR BIOMETRIC METRICS                               │
│ [ Webcam / Video ]   │  • PERCLOS (30s):     12.4%  [Normal]                    │
│                      │  • Blink Rate:        18 /min [Normal]                   │
│   Bounding Boxes:    │  • Fatigue Risk Score: 14%    [GREEN - ALERT]            │
│   - Eye Mesh & EAR   ├──────────────────────────────────────────────────────────┤
│   - Mouth Mesh & MAR │ HSE INCIDENT TIMELINE                                    │
│   - Head Pose Vector │  [21:12:04] Status: NORMAL (Score: 0.14)                 │
│   - Slouch Indicator │  [21:14:22] Warning: Slow blink detected (Score: 0.48)   │
└──────────────────────┴──────────────────────────────────────────────────────────┘
```

### 5.1 MVP Capabilities
1. **Live Webcam Mode**: Real-time camera feed analysis (30 FPS) with instantaneous facial mesh overlay, biometrics HUD, and live status badge.
2. **Video File Analysis Mode**: Upload standard MP4/AVI surveillance recordings for offline auditing and incident playback.
3. **Analytics & Performance View**: Displays training metrics, confusion matrix, ROC curve, and feature importance rankings directly in the UI.
4. **HSE Tiered Alerting**:
   * **GREEN (Normal)**: Risk score $< 35\%$. System logs normal vigilance.
   * **AMBER (Warning)**: Risk score $35\% - 70\%$. Eyelid drooping or yawning detected. UI displays warning badge.
   * **RED (Critical)**: Risk score $\ge 70\%$. Prolonged eye closure or microsleep. Triggers visual alarm and writes critical entry to `logs/hse_alerts.csv`.

---

## 6. How to Run and Demonstrate the MVP

### 6.1 Prerequisites
Ensure dependencies are installed:
```powershell
pip install -r requirements.txt
```

### 6.2 Running the Model Training Pipeline (Optional)
To retrain the ensemble and re-verify test metrics:
```powershell
cd offshore_fatigue_detection
python train_models.py --no-shap --no-optuna
```

### 6.3 Launching the MVP Dashboard
Launch the Streamlit web application:
```powershell
cd offshore_fatigue_detection
python -m streamlit run app.py
```
The browser will automatically open at `http://localhost:8501`.

### 6.4 Demonstration Script for Stakeholders
1. **Webcam Demo**: Select **"Live Webcam"** in the sidebar. Click **"Start Live Stream"**. Observe real-time EAR, MAR, and Fatigue Risk Score tracking. Close eyes for $>2$ seconds to demonstrate Red Alert triggering.
2. **Video File Demo**: Select **"Video File"** in the sidebar. Upload any video to observe automatic frame-by-frame analysis and fatigue scoring.
3. **Analytics Demo**: Select **"Model Metrics & Analytics"** in the sidebar to review the 99.90% accuracy benchmark, confusion matrix, and feature importances.
4. **Audit Log Inspection**: Review `logs/hse_alerts.csv` to showcase compliance-ready HSE reporting.

---

## 7. Project File Structure & Inventory

```
offshore_fatigue_detection/
├── app.py                             # Main Streamlit MVP Web Dashboard
├── config.py                          # Centralized zero-hardcoding configuration
├── train_models.py                    # Production training and evaluation pipeline
├── requirements.txt                   # Project dependencies
├── PROJECT_COMPLETION_AND_MVP_REPORT.md # This comprehensive handover report
├── src/
│   ├── facial_extractor.py           # MediaPipe FaceMesh & 3D landmark extractor
│   ├── postural_extractor.py         # MediaPipe Pose & posture keypoint analyzer
│   ├── feature_pipeline.py           # Temporal rolling buffer & baseline normalizer
│   ├── fatigue_model.py              # Tri-model soft-voting ensemble (LGBM/XGB/Cat)
│   ├── temporal_buffer.py            # Real-time sliding window buffer
│   └── hse_logger.py                 # HSE event logging & risk tiering
├── scripts/
│   ├── download_datasets.py          # Automated downloader for benchmark datasets
│   ├── extract_auxiliary_features.py # Batch feature extractor for MRL & COCO
│   └── setup_datasets.ps1            # Environment initialization script
├── data/
│   ├── raw/                          # Downloaded datasets (UTA, YawDD, MRL, COCO, MPII)
│   └── processed/                    # Master Parquet feature matrices
├── models/
│   ├── fatigue_lgbm.pkl              # LightGBM trained model weights
│   ├── fatigue_xgb.pkl               # XGBoost trained model weights
│   ├── fatigue_cat.pkl               # CatBoost trained model weights
│   ├── feature_scaler.pkl            # Feature normalizer
│   ├── label_encoder.pkl             # Label encoder
│   ├── probability_calibrator.pkl    # Isotonic probability calibrator
│   ├── training_metrics.json         # 99.90% verified test metrics
│   └── feature_importance.csv        # Top feature rankings
└── logs/
    └── hse_alerts.csv                # Audit log for fatigue events
```

---

## 8. Conclusion & Handover Readiness

The Offshore Fatigue Detection System successfully fulfills all technical and operational criteria:
* ✅ All benchmark datasets downloaded, processed, and validated.
* ✅ Fully automated, non-hardcoded data and training pipelines.
* ✅ **99.90% hold-out test accuracy**, outperforming the $>90\%$ target.
* ✅ Production-ready MVP dashboard with real-time inference and HSE compliance logging.
