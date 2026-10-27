# Multi-Modal Real-Time Offshore Fatigue Detection System
### Enterprise Industrial AI Surveillance Suite (viAct Design Spec)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Accuracy](https://img.shields.io/badge/Holdout_Accuracy-99.90%25-brightgreen.svg)]()
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-1.0000-success.svg)]()
[![Inference Latency](https://img.shields.io/badge/Edge_Latency-14.2ms_(70.4_FPS)-orange.svg)]()
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Executive Overview

Offshore oil and gas drilling platforms, marine crane vessels, and subsea intervention stations operate in continuous 24/7 high-hazard rotational shifts (12-hour duty cycles). Physiological operator fatigue is the leading contributor to critical maritime and drilling incidents, including blowouts, dropped-object crane impacts, and subsea manifold collisions.

This repository delivers an enterprise-grade, non-intrusive **Multi-Modal Fatigue Detection System (FDS)** engineered for offshore installations. The framework combines high-frequency facial landmark tracking (MediaPipe FaceMesh 468 points with automatic OpenCV Haar Cascade fallback) and upper-body ergonomics (MediaPipe Pose 33 points) with an **Adaptive Subject-Baseline Normalization Engine** and a **Calibrated Tri-Model Soft-Voting Ensemble (LightGBM + XGBoost + CatBoost)**.

> 📖 **Full Academic Paper Available:** See [`RESEARCH_PAPER_OFFSHORE_FATIGUE_DETECTION.md`](RESEARCH_PAPER_OFFSHORE_FATIGUE_DETECTION.md) for complete mathematical formulations, literature review, and sustainability analysis.

---

## 🚀 Key Highlights & Empirical Performance

* **Hold-out Test Accuracy**: **99.90%** across 10,131 unseen frames (exceeding the $\ge 90.0\%$ target by $+9.90\%$).
* **Discriminative Capacity**: **1.0000 ROC-AUC** and **0.9988 Macro F1-Score**.
* **Near-Zero False Alarms**: Only 2 false alarms out of 3,145 vigilant test frames ($0.06\%$ error rate).
* **High Fatigued Sensitivity**: **99.89% Recall** on fatigued frames, eliminating undetected microsleeps.
* **Edge-AI Optimized**: **14.2 ms / frame** total latency (70.4 FPS throughput) on embedded NVIDIA Jetson AGX Orin nodes.
* **Zero Hardcoded Detection**: Biometrics are dynamically extracted from live video/webcam; empty scenes cleanly display scanning status without phantom bounding boxes.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       OFFSHORE SENSOR INPUT STREAM                          │
│               (Industrial CCTV / RTSP / USB 2.0 WebCam / MP4)               │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 DUAL-STREAM BIOMETRIC LANDMARK EXTRACTION                   │
│   • Stream A: MediaPipe FaceMesh (468 3D Points) -> OpenCV Haar Fallback   │
│   • Stream B: MediaPipe Pose (33 3D Points)      -> Head Geometry Fallback  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 ADAPTIVE BASELINE & KINEMATIC NORMALIZATION                 │
│   • Running 90th percentile baseline EAR: EAR_baseline                     │
│   • Dynamic ratio mapping: EAR_ratio = EAR_t / EAR_baseline                │
│   • Running 10th percentile baseline MAR: MAR_baseline                     │
│   • Dynamic ratio mapping: MAR_ratio = MAR_t / MAR_baseline                │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 MULTI-SCALE TEMPORAL ROLLING BUFFER ENGINE                  │
│   • Multi-Window Rolling Statistics: 5s, 15s, 30s, and 60s windows         │
│   • Kinematics: Eyelid Velocity (dEAR/dt), Acceleration, & Micro-Events     │
│   • 175 Total Engineered Features per Timestep                              │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│               TRI-MODEL CALIBRATED SOFT-VOTING ML ENSEMBLE                  │
│       ┌─────────────────┐ ┌─────────────────┐ ┌──────────────────┐          │
│       │ LightGBM (0.40) │ │ XGBoost (0.35)  │ │ CatBoost (0.25)  │          │
│       └────────┬────────┘ └────────┬────────┘ └────────┬─────────┘          │
│                └───────────────────┼───────────────────┘                    │
│                                    ▼                                        │
│                    Isotonic Probability Calibration                         │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│               HSE TIERED CLASSIFICATION & TELEMETRY DISPATCH                │
│   • Level Green (Normal): Fatigue Score < 0.35                              │
│   • Level Amber (Low Vigilance Warning): Score 0.35 - 0.70                 │
│   • Level Red (Critical Fatigue / Microsleep): Score >= 0.70                │
│   • Dispatches: Intercom Broadcast, PTW Auto-Lock, OSHA Audit Log           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🖥️ viAct Enterprise Surveillance Suite (17 Modules)

The application provides a comprehensive dark-themed industrial command suite matching the viAct surveillance design:

```
viAct Offshore AI · Command Suite
│
├── ── MONITORING ──
│   ├── 🏠 Home Overview            -> Rig operational status, 48 POB, near-misses blocked, marine weather
│   ├── 📊 Fleet Analytics          -> 12-hour shift fatigue accumulation, zone heatmaps, OSHA compliance
│   ├── 🔔 Notifications (12)       -> 12 active offshore alerts with 1-click Acknowledge & Break Dispatch
│   ├── 🎥 Live Surveillance        -> Real-time webcam (▶ Live Cam) + 1-Click Worker Demo (🎬 Stream Demo)
│   ├── 🎞️ Video Analysis (Upload)  -> Upload custom MP4/AVI/MOV/MKV video for frame-by-frame inference
│   └── 🚨 Events & Audit Log       -> Master chronological incident log with CSV export
│
├── ── SITE MANAGEMENT ──
│   ├── 📹 CCTV Edge Devices        -> Cam 01-04 RTSP streams, latency, FPS, and live ping test
│   ├── 📄 Permit to Work (PTW)     -> Automated fatigue clearance gateway for high-risk permits
│   ├── 🦺 Digital Worker Tracking  -> 5 worker cards with duty hours, live gauges, and Break Dispatch
│   └── 📋 Task & Rest Dispatch     -> Circadian task scheduler and mandatory rest queue
│
└── ── ADMINISTRATION ──
    ├── 💬 Emergency Intercom       -> Direct two-way radio and voice broadcast console
    ├── 🗺️ Rig Map & Deck Plan      -> Interactive 2D schematic blueprint of Rig Alpha deck planes
    ├── 📶 Edge Hardware Status     -> Jetson Orin AGX cluster telemetry (GPU latency 14.2ms, VRAM)
    ├── 🎯 Biometric Calibration    -> Sliders for EAR, PERCLOS, MAR, and alert thresholds
    ├── 📚 Model Knowledge Base     -> 98.63% test accuracy, ROC-AUC 0.9995, Confusion Matrix, SHAP
    ├── 📐 3D Rig Forge Viewer      -> 3D digital twin spatial visualizer with CCTV camera frustums
    ├── 👤 Role Setting (RBAC)      -> Role-based access control matrix (OIM, HSE Lead, Medic)
    └── 🛡️ HSE Shift Audit          -> Signed shift compliance certificate and JSON report download
```

---

## 📊 Training Datasets Ingested (14GB+)

The model ensemble was trained and cross-validated across four international benchmarks:

| Dataset | Modality & Quantity | Storage Location | Specific Contribution |
|---|---|---|---|
| **UTA-RLDD & YawDD** | 415 Video Sessions (82,067 frames) | `data/raw/uta_yawdd/` | Ground-truth alert, drowsy, and fatigued driver/operator temporal sequences. |
| **MRL Eye Dataset** | 84,938 Localized Eye Crops | `data/raw/mrl_eye/` | Eye closure probability distribution, pupil aperture ratios under varied lighting. |
| **COCO Keypoints 2017** | 5,000 Images (92,453 poses) | `data/raw/coco_pose/` | Natural upper-body ergonomics priors (shoulder slouching, neck tilt angles). |
| **MPII Human Pose** | 24,991 Pose Annotations | `data/raw/mpii_pose/` | Occupational ergonomic variations across physical work tasks. |

---

## 🔬 Benchmark Results & Confusion Matrix

### Holdout Test Metrics (10,131 Unseen Frames):
* **Accuracy:** `99.90%`
* **Macro F1-Score:** `0.9988`
* **Fatigued Precision:** `99.97%`
* **Fatigued Recall:** `99.89%`
* **Test ROC-AUC:** `1.0000`
* **5-Fold Cross-Validation AUC:** `1.0000`

### Confusion Matrix:
```
                      PREDICTED ALERT     PREDICTED FATIGUED
ACTUAL ALERT                3,143                  2
ACTUAL FATIGUED                 8              6,978
```

---

## ⚡ Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/ABHINAVSINGHCODIE/offshore-fatigue-detection.git
cd offshore-fatigue-detection
```

### 2. Create Virtual Environment & Install Dependencies
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Launch the viAct Command Suite Dashboard
```bash
streamlit run app.py
```
Open your web browser and navigate to **`http://localhost:8501`**.

---

## 📂 Repository Structure

```
offshore-fatigue-detection/
├── app.py                                       # Enterprise viAct 17-Module Streamlit Application
├── config.py                                    # Dynamic environment paths and threshold calibration
├── train_models.py                              # Tri-model ensemble training & isotonic probability calibration
├── requirements.txt                             # Production dependencies (Streamlit, OpenCV, LightGBM, etc.)
├── RESEARCH_PAPER_OFFSHORE_FATIGUE_DETECTION.md # Full 7-section academic research paper manuscript
├── PROJECT_COMPLETION_AND_MVP_REPORT.md         # Comprehensive engineering report & technical audit
├── src/
│   ├── facial_extractor.py                      # MediaPipe FaceMesh (468 pts) with OpenCV Haar fallback
│   ├── postural_extractor.py                    # MediaPipe Pose (33 pts) neck tilt and head drop tracking
│   ├── temporal_buffer.py                       # Multi-scale rolling buffer (5s, 15s, 30s, 60s windows)
│   ├── fatigue_model.py                         # Tri-model soft-voting ensemble inference wrapper
│   ├── feature_pipeline.py                      # 175-dimensional dynamical feature synthesis
│   ├── hse_logger.py                            # OSHA Section 1910 tiered alert logging engine
│   └── video_ingestion.py                       # RTSP and video file frame generator
├── models/
│   ├── ensemble_meta.json                       # Ensemble weights and 175-feature registry
│   ├── feature_importance.csv                   # Global feature ranking
│   ├── shap_importance.json                     # Top SHAP values for UI bar chart
│   └── training_metrics.json                    # Exact test metrics, confusion matrix, and AUC
├── data/
│   ├── DATASET_MANIFEST.md                      # Audit of all ingested benchmark datasets
│   └── samples/
│       └── sample_worker_monitoring.mp4         # 1-Click built-in worker surveillance test video
└── static/
    └── offshore_bg.jpg                          # Offshore oil rig ocean platform background wallpaper
```

---

## 📜 Regulatory Standards & Compliance

The fatigue scoring thresholds and escalation actions are mapped to international occupational frameworks:
* **OSHA Standard 1910.120**: Hazardous waste operations and emergency response fatigue limits.
* **IMCA Marine Guidance D023**: Guidance on diving and dynamic positioning (DP) operational fatigue.
* **IOGP Report 626**: Fatigue management in the oil and gas industry.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
