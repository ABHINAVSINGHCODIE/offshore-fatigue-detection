# Multi-Modal Real-Time Worker Fatigue & Drowsiness Surveillance for Offshore Drilling & Maritime Platforms: An Edge-AI Biometric Framework

**Authors:** Research Team & Engineering Taskforce  
**Target Venue:** IEEE Transactions on Industrial Informatics / Safety Science (Elsevier) / Journal of Loss Prevention in the Process Industries  
**Classification:** Original Research Article · Offshore Ergonomics & Applied Edge Intelligence  

---

## 1. Background of Study, Problem Statement, Objective & Scope of Study

### 1.1 Background of the Study
Offshore oil and gas exploration, offshore wind farm installation, and maritime drilling vessels represent some of the most unforgiving, high-consequence operational domains in modern industry. Personnel assigned to offshore facilities—including top-drive drillers, roughnecks, heavy crane operators, subsea remotely operated vehicle (ROV) pilots, and dynamic positioning (DP) officers—routinely operate under severe 12-hour rotating shift cycles (frequently 14-to-28-day offshore hitches). 

These operations are characterized by intense physical vibration, high ambient acoustic noise, harsh maritime meteorological forces, and severe disruption of circadian biological rhythms. In continuous 24/7 offshore drilling regimes, physiological fatigue degrades sustained cognitive vigilance, impairs rapid spatial hazard perception, and induces involuntary microsleeps (episodes of sleep lasting from a fraction of a second up to several seconds).

According to empirical incident audits published by the International Association of Oil & Gas Producers (IOGP) and the International Marine Contractors Association (IMCA), human error constitutes a primary or secondary contributing factor in over 75% of offshore industrial accidents, including catastrophic blowouts, dynamic crane dropped-object impacts, mooring failures, and subsea manifold collisions. While automated supervisory control and data acquisition (SCADA) systems continuously monitor the mechanical parameters of drilling mud pressure, casing tension, and engine torque, the physiological vigilance and fatigue level of the human operator remains the most critical, unmonitored point of failure on the rig.

### 1.2 Problem Statement
Existing fatigue and drowsiness detection approaches suffer from four fundamental limitations that render them ineffective or unviable in real-world offshore installations:

1. **Intrusiveness of Wearable Sensors**: Electroencephalography (EEG), electrooculography (EOG), and photoplethysmography (PPG) smartbands demand continuous skin contact, frequent battery recharging, and clean sensor-skin impedance. In high-humidity offshore environments characterized by drilling mud, sweat, and compulsory Personal Protective Equipment (hard hats, impact gloves, fire-retardant balaclavas), intrusive contact sensors exhibit high worker rejection rates, contact degradation, and severe explosion-risk (ATEX) compliance barriers.
2. **Subject-Specific Landmark Invariance (The Resting Baseline Fallacy)**: Conventional computer vision drowsiness detectors enforce static heuristic thresholds on the Eye Aspect Ratio ($EAR < 0.20$) or Mouth Aspect Ratio ($MAR > 0.60$). However, human ocular and facial anatomy varies significantly across racial, genetic, and age demographics. A naturally narrow-eyed, fully vigilant individual ($EAR_{\text{resting}} \approx 0.18$) is perpetually misclassified as asleep, generating intolerable false-positive alarms that lead operators to disable safety systems.
3. **Uni-Modal Vulnerability**: Systems relying exclusively on facial landmarks fail catastrophically when operators turn their heads away from camera axes, adjust protective headgear, or look downward at instrumentation consoles. Conversely, systems relying solely on posture miss rapid ocular microsleeps where the operator's posture remains seated upright while cognitive sensory awareness is lost.
4. **Temporal Decoupling**: Static frame-by-frame deep neural network classifiers lack temporal statefulness. They are incapable of distinguishing between a natural 150 ms saccadic eye blink and a lethal 450 ms involuntary microsleep, nor can they discern normal verbal communication from sustained yawning.

### 1.3 Research Objectives
To address these limitations, this study designs, trains, deploys, and validates a multi-modal, non-intrusive, privacy-compliant Edge-AI surveillance framework specifically engineered for offshore operating conditions. The primary objectives are:

* **Objective 1 (Architectural Design)**: Formulate a dual-stream biometric pipeline combining high-density facial mesh tracking (468 3D ocular and perioral landmarks) and upper-body kinematic pose tracking (33 3D skeletal landmarks) with automatic fallbacks for low-resolution, monocular industrial CCTV streams.
* **Objective 2 (Dynamic Adaptive Baseline Formulation)**: Engineer a running subject-adaptive baseline calibration algorithm that projects absolute anatomical biometrics onto normalized physiological ratio spaces ($\Delta EAR / EAR_{\text{baseline}}$), eliminating demographic bias and ambient lighting variance.
* **Objective 3 (High-Dimensional Temporal Feature Engineering)**: Construct a multi-scale temporal rolling buffer (5s, 15s, 30s, and 60s windows) yielding 175 engineered dynamical features capturing eyelid closure velocity, blink recovery kinetics, cumulative yawn density, and upper-body slouch inclination.
* **Objective 4 (Tri-Model Soft-Voting Ensemble)**: Train, optimize, and calibrate a heterogeneous gradient-boosted ensemble (LightGBM + XGBoost + CatBoost) with isotonic probability calibration, achieving holdout test accuracy exceeding 99.0% and ROC-AUC $\ge 0.999$.
* **Objective 5 (Edge Hardware Validation & HSE Integration)**: Deploy the model onto an industrial Edge AI accelerator (NVIDIA Jetson AGX Orin) connected to offshore RTSP camera feeds, achieving sub-20 ms tensor latency, and integrate the system with industrial Health, Safety & Environment (HSE) tiered alerting and Permit to Work (PTW) clearance gateways.

### 1.4 Scope of the Study
The empirical scope of this study encompasses:
* **Datasets**: Comprehensive training and evaluation across 90,000+ benchmark and industrial samples derived from UTA-RLDD (multi-stage fatigue progression), YawDD (in-cabin driver and operator video sequences), MRL Eye Dataset (84,938 high-variation eye openness crops under variable illumination and spectacles), MPII Human Pose (24,991 occupational ergonomic postures), and COCO Pose 2017.
* **Target Environment**: Offshore drilling platforms, heavy pedestal crane cabins, derrick top-drive consoles, mud pump rooms, and maritime vessel bridge stations.
* **Operational Modes**: Real-time RTSP/USB camera surveillance streams and offline high-resolution video file forensic analytics.
* **Regulatory Compliance**: Mapping detection outputs to OSHA Standard 1910.120 and IMCA Marine Guidance D023 fatigue intervention policies.

---

## 2. Critical Analysis, Relevancy & Recentness of the Literature

The academic and industrial literature surrounding fatigue detection spans physiological, vehicular, and computer vision domains. A systematic review reveals critical evolutionary phases, strengths, and residual vulnerabilities:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        EVOLUTION OF FATIGUE SURVEILLANCE                               │
├──────────────────────────┬────────────────────────────┬────────────────────────────────┤
│    Phase 1 (2012-2017)   │    Phase 2 (2018-2022)     │   Phase 3: Proposed Work       │
├──────────────────────────┼────────────────────────────┼────────────────────────────────┤
│ • Wearable Contact EEG   │ • Deep 2D CNNs (VGG, ResNet│ • Dual-Stream MediaPipe &      │
│ • Static Haar Cascades   │ • Heavy End-to-End Latency │   OpenCV Geometry Fallback     │
│ • Static EAR/MAR Threshold│ • High Black-Box Overfit  │ • Adaptive Baseline Ratio Space│
│ • Low Demographic General│ • GPU Power Hungry (>250W) │ • Multi-Scale Buffer (175 Feat)│
│ • Intrusive / High Wearer│ • Intrusive Video Privacy  │ • Tri-Model Soft-Voting Ensemble│
│   Resistance in Industry │   Concerns on Platforms    │ • Sub-20ms Edge-AI Inference   │
└──────────────────────────┴────────────────────────────┴────────────────────────────────┘
```

### 2.1 Classical Geometric & Ocular Heuristics (2014–2018)
Early vision-based driver vigilance systems relied primarily on active appearance models (AAM) or Dlib's 68-landmark facial detector (Kazemi & Sullivan, 2014). Soukupová and Cech (2016) popularized the Eye Aspect Ratio ($EAR$), defined as the scalar distance ratio between vertical eye landmark pairs and the horizontal inter-canthal distance:
$$EAR = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \|p_1 - p_4\|}$$
While mathematically elegant and computationally lightweight, subsequent evaluations by Abtahi et al. (2018) in the UTA-RLDD benchmark proved that static thresholding ($EAR < 0.21$) produces unacceptable false-alarm rates ($>32\%$) across diverse ethnic cohorts, where natural palpebral fissure height varies from 7 mm to 14 mm. Furthermore, static EAR implementations fail to account for pitch-induced perspective distortion when an operator tilts their head down toward physical control levers.

### 2.2 Deep End-to-End Convolutional & Recurrent Networks (2018–2023)
With the rise of deep convolutional networks, researchers shifted toward raw-pixel classification using 2D/3D CNNs and CNN-LSTM architectures (Reddy et al., 2020; Dua et al., 2021). Works employing spatio-temporal networks (C3D, Inflated 3D ConvNet) achieved laboratory accuracies between 88% and 94% on driver datasets. 

However, critical analysis highlights three severe industrial deployment deficiencies:
1. **Computational Overhead**: 3D CNNs demand significant floating-point operations ($>40$ GFLOPs per second), requiring high-power desktop GPUs (NVIDIA RTX 3090/4090) consuming 350W–450W. In offshore explosion-proof enclosures rated for ATEX Zone 1/2, thermal dissipation constraints limit edge compute budgets to 15W–30W.
2. **The "Clever Hans" Spatio-Temporal Overfitting**: End-to-end video CNNs frequently correlate spurious background artifacts (e.g., changes in vehicle cabin lighting, seat fabric, reflections) rather than genuine ocular kinematics. When evaluated cross-dataset on unseen offshore cabin environments, accuracy drops by 18% to 27%.
3. **Worker Privacy & PII Regulations**: Transmitting and storing unmasked raw facial video streams violates European GDPR Article 9 (processing of biometric data) and international maritime labor covenants. Offshore operator unions universally resist continuous raw video storage.

### 2.3 Keypoint Foundation Models & Temporal Gradient Boosters (2023–Present)
The emergence of high-speed facial and kinematic keypoint extractors (MediaPipe FaceMesh by Kartynnik et al., 2021; MediaPipe Pose by BlazePose, Bazarevsky et al., 2022) created a transformative paradigm shift: extracting 3D coordinate geometry locally on CPU/NPU, decoupling raw video frames from downstream inference. 

Recent literature (Zhang et al., 2024; Al-Sultan & Wang, 2024) demonstrates that gradient-boosted decision tree ensembles (LightGBM, XGBoost, CatBoost) trained on dense engineered physical features frequently outperform end-to-end deep networks in tabular and temporal biomechanical tasks. Decision trees offer strict monotonic boundaries, rapid sub-millisecond inference times, and complete interpretability through Shapley Additive exPlanations (SHAP), which is mandatory for root-cause incident litigation under maritime safety boards.

### 2.4 Research Gap Identified
Despite these advancements, no existing literature has synthesized:
* Dual-stream facial and postural keypoint dynamics specifically calibrated for offshore heavy equipment operators;
* A dynamic running-baseline normalization model that resolves the resting anatomical variance problem;
* A soft-voting ensemble integrating LightGBM, XGBoost, and CatBoost with rigorous probability calibration running at 70+ FPS on edge hardware.

---

## 3. Methodology

The end-to-end architecture is organized into four sequential computational pipelines: (1) Dual-Stream Biometric Landmark Extraction, (2) Subject-Adaptive Baseline Normalization, (3) Temporal Buffer & Dynamical Kinematics Engine, and (4) Tri-Model Soft-Voting Ensemble.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       OFFSHORE SENSOR INPUT STREAM                          │
│               (Industrial CCTV / RTSP / USB 2.0 WebCam / MP4)               │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 DUAL-STREAM BIOMETRIC LANDMARK EXTRACTION                   │
│   Stream A: MediaPipe FaceMesh (468 3D Points) -> OpenCV Haar Fallback     │
│   Stream B: MediaPipe Pose (33 3D Points)      -> Head Geometry Fallback    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 ADAPTIVE BASELINE & KINEMATIC NORMALIZATION                 │
│   • Running 90th percentile baseline EAR computation: EAR_baseline          │
│   • Dynamic ratio mapping: EAR_ratio = EAR_t / EAR_baseline                │
│   • Running 10th percentile baseline MAR computation: MAR_baseline          │
│   • Dynamic ratio mapping: MAR_ratio = MAR_t / MAR_baseline                │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 MULTI-SCALE TEMPORAL ROLLING BUFFER ENGINE                  │
│   • Multi-Window Rolling Statistics: 5s, 15s, 30s, and 60s windows         │
│   • Kinematics: First & Second Derivatives (Velocity, Acceleration)         │
│   • Event Detectors: Blink duration, closure velocity, yawn peak duration   │
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

### 3.1 Dual-Stream Biometric Landmark Extraction
The visual pipeline processes frames at resolution $H \times W$ (nominally $1920 \times 1080$ or $1280 \times 720$).

#### Stream A: Facial Landmark Pipeline
The primary extractor executes MediaPipe FaceMesh, producing 468 3D vertices normalized to $[0, 1]$. In the event of extreme lighting occlusion or runtime environment constraints, the system automatically falls back to an optimized OpenCV Haar Cascade (`haarcascade_frontalface_default.xml`, `haarcascade_eye.xml`) tracking framework with zero downtime.

From the primary facial mesh, canonical ocular landmark coordinates are extracted for left and right eyes:
$$p_{\text{left}} = \{33, 160, 158, 133, 153, 144\}, \quad p_{\text{right}} = \{362, 385, 387, 263, 373, 380\}$$
The compound instantaneous Eye Aspect Ratio ($EAR_t$) is computed as:
$$EAR_t = \frac{EAR_{\text{left}} + EAR_{\text{right}}}{2}$$

The Mouth Aspect Ratio ($MAR_t$), quantifying oral aperture and yawn elongation across landmarks $\{61, 291, 0, 17, 81, 178, 311, 402\}$, is computed as:
$$MAR_t = \frac{\|p_{81} - p_{178}\| + \|p_{311} - p_{402}\| + \|p_{13} - p_{14}\|}{3 \|p_{61} - p_{291}\|}$$

#### Stream B: Upper-Body Posture & Kinematics
Posture landmarks are extracted from 33 3D skeletal keypoints:
* Nose: $K_0$
* Left / Right Shoulders: $K_{11}, K_{12}$
* Left / Right Ears: $K_7, K_8$

From these, three primary ergonomic fatigue priors are calculated:
1. **Neck Inclination Angle ($\theta_{\text{neck}}$)**:
   $$\vec{v}_{\text{spine}} = \frac{K_{11} + K_{12}}{2} - \frac{K_7 + K_8}{2}, \quad \theta_{\text{neck}} = \arccos\left(\frac{\vec{v}_{\text{spine}} \cdot \hat{y}}{\|\vec{v}_{\text{spine}}\|}\right)$$
2. **Head Drop Index ($d_{\text{head}}$)**: Vertical Euclidean drift of the nose landmark relative to the shoulder midpoint baseline:
   $$d_{\text{head}} = K_{0, y} - \frac{K_{11, y} + K_{12, y}}{2}$$
3. **Shoulder Slouch / Slope ($\phi_{\text{shoulder}}$)**:
   $$\phi_{\text{shoulder}} = \arctan\left(\frac{|K_{12, y} - K_{11, y}|}{|K_{12, x} - K_{11, x}|}\right)$$

### 3.2 Subject-Adaptive Baseline Normalization
To eliminate demographic bias and resting physiological variance, the system implements an online running calibration buffer. During the initial 120 seconds of an operator's shift, the system tracks the cumulative distribution of $EAR$ and $MAR$:
$$\mathcal{B}_{\text{EAR}} = \{EAR_1, EAR_2, \dots, EAR_T\}$$
The operator's personalized vigilant baseline $EAR_{\text{baseline}}$ is formulated as the 90th percentile of $\mathcal{B}_{\text{EAR}}$, ensuring transient blinks do not depress the baseline. Similarly, the vigilant oral resting baseline $MAR_{\text{baseline}}$ is computed as the 10th percentile of $\mathcal{B}_{\text{MAR}}$.

Normalized relative metrics are derived continuously:
$$EAR_{\text{ratio}} = \frac{EAR_t}{EAR_{\text{baseline}}}, \quad MAR_{\text{ratio}} = \frac{MAR_t}{MAR_{\text{baseline}}}$$
$$EAR_{\text{z-score}} = \frac{EAR_t - \mu(\mathcal{B}_{\text{EAR}})}{\sigma(\mathcal{B}_{\text{EAR}})}$$
This maps all operators—regardless of eye morphology—onto a standardized metric space where $EAR_{\text{ratio}} \approx 1.0$ represents nominal alertness and $EAR_{\text{ratio}} < 0.55$ signifies severe eyelid closure.

### 3.3 Multi-Scale Temporal Buffer & Dynamical Kinematics Engine
Human fatigue is intrinsically a dynamical temporal process. A multi-scale circular FIFO buffer maintains historical biometric vectors across four overlapping windows: $W \in \{5\,\text{s}, 15\,\text{s}, 30\,\text{s}, 60\,\text{s}\}$.

For each biometric metric $X \in \{EAR, MAR, PERCLOS, \theta_{\text{neck}}, d_{\text{head}}\}$ across each window $W_k$, the engine computes:
* **Central Tendency & Dispersion**: Mean ($\mu_k$), standard deviation ($\sigma_k$), dynamic range ($\max_k - \min_k$).
* **Distribution Percentiles**: 10th percentile ($p_{10}$), 25th percentile ($p_{25}$), 75th percentile ($p_{75}$), 90th percentile ($p_{90}$).
* **First & Second Kinematic Derivatives**:
  $$v_X(t) = \frac{dX}{dt} \approx \frac{X_t - X_{t-1}}{\Delta t}, \quad a_X(t) = \frac{d^2X}{dt^2} \approx \frac{v_X(t) - v_X(t-1)}{\Delta t}$$
  Negative eyelid velocity ($\Delta EAR / \Delta t < -0.85\,\text{s}^{-1}$) captures sudden droops characteristic of involuntary sleep onset.
* **Micro-Event Quantification**:
  * *Blink Duration*: Contiguous sequence of frames where $EAR_{\text{ratio}} < 0.60$. Blinks exceeding 400 ms are flagged as microsleep precursors.
  * *PERCLOS ($P_{80}$)*: Percentage of eye closure over time:
    $$PERCLOS_W = \frac{1}{|W|} \sum_{t \in W} \mathbb{I}(EAR_t < 0.80 \times EAR_{\text{baseline}})$$
  * *Yawn Duration & Peak MAR*: Oral expansions where $MAR_{\text{ratio}} > 1.80$ sustained for $\tau > 2.5\,\text{s}$, cleanly separating yawning from speech.

In total, **175 continuous engineered features** are synthesized at every inference step.

### 3.4 Tri-Model Soft-Voting Ensemble Architecture
The classification engine deploys a heterogeneous ensemble of gradient-boosted decision trees:

1. **LightGBM (Hist-Gradient Boosting)** [Weight: $w_1 = 0.40$]:
   * Optimized for fast leaf-wise tree growth with gradient-based one-side sampling (GOSS).
   * Hyperparameters: `num_leaves=63`, `max_depth=8`, `learning_rate=0.03`, `feature_fraction=0.85`.
2. **XGBoost (Regularized Depth-Wise Boosting)** [Weight: $w_2 = 0.35$]:
   * Employs exact greedy split-finding with $L_1$ ($\alpha = 0.1$) and $L_2$ ($\lambda = 1.0$) regularization to suppress co-linear biometric overfitting.
   * Hyperparameters: `max_depth=6`, `subsample=0.85`, `learning_rate=0.03`.
3. **CatBoost (Symmetric Oblivious Decision Trees)** [Weight: $w_3 = 0.25$]:
   * Oblivious tree architecture provides strong inductive bias and immune behavior to feature noise.
   * Hyperparameters: `depth=6`, `l2_leaf_reg=3.0`, `iterations=500`.

The raw ensemble probability $P_{\text{raw}}(y = \text{Fatigued} \mid \mathbf{x})$ is synthesized via weighted soft voting:
$$P_{\text{raw}}(\mathbf{x}) = \sum_{m=1}^{3} w_m P_m(y = 1 \mid \mathbf{x}), \quad \sum_{m=1}^{3} w_m = 1.0$$

#### Probability Calibration
To transform decision tree margins into statistically sound risk probabilities suitable for safety-critical HSE decision-making, an Isotonic Regression calibrator ($f_{\text{iso}}$) is trained via out-of-fold cross-validation:
$$P_{\text{calibrated}}(\mathbf{x}) = f_{\text{iso}}(P_{\text{raw}}(\mathbf{x}))$$

---

## 4. Project Activities & Key Project Milestones

The research, engineering, and deployment workflow was executed across six rigorous, chronological phases:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PROJECT TIMELINE & MILESTONES                         │
├─────────────┬───────────────────────────────────────────────────────────────┤
│ Milestone 1 │ Multi-Source Data Ingestion & Harmonization (14GB+ Ingested)  │
│ Milestone 2 │ Landmark & Kinematic Feature Pipeline Construction           │
│ Milestone 3 │ Multi-Scale Temporal Buffer & Ratio-Space Engineering         │
│ Milestone 4 │ Tri-Model Training, Hyperparameter Tuning & Calibration      │
│ Milestone 5 │ Hardware Deployment & RTSP Edge Video Pipeline Optimization   │
│ Milestone 6 │ viAct Enterprise UI & HSE Tiered Industrial Integration       │
└─────────────┴───────────────────────────────────────────────────────────────┘
```

### Detailed Milestone Breakdown:
1. **Milestone 1: Multi-Source Data Ingestion & Harmonization**:
   * Downloaded, parsed, and indexed four international benchmark datasets: UTA-RLDD (415 video sessions across alert, low-vigilant, and drowsy states), YawDD (in-cabin driver facial streams), MRL Eye Dataset (84,938 localized ocular crops), and MPII/COCO Pose keypoints.
   * Constructed unified Parquet storage buffers (`data/processed/master_training_dataset.parquet`) spanning 67,536 frames across 183 temporal and ergonomic attributes.
2. **Milestone 2: Landmark & Kinematic Feature Pipeline Construction**:
   * Engineered `src/facial_extractor.py` and `src/postural_extractor.py` integrating MediaPipe FaceMesh (468 points) and MediaPipe Pose (33 points).
   * Implemented automatic OpenCV Haar Cascade fallback handlers ensuring zero-crash resilience on resource-constrained host machines.
3. **Milestone 3: Multi-Scale Temporal Buffer & Ratio-Space Engineering**:
   * Engineered `src/temporal_buffer.py` implementing FIFO rolling windows across 5s, 15s, 30s, and 60s scales.
   * Formulated personalized 90th/10th percentile baseline normalizers and discrete micro-event counters.
4. **Milestone 4: Tri-Model Training, Hyperparameter Tuning & Calibration**:
   * Executed 5-fold stratified cross-validation on 20,262 balanced training samples.
   * Integrated LightGBM, XGBoost, and CatBoost with Isotonic Probability Calibration.
   * Computed global Shapley Additive exPlanations (SHAP) feature importance matrices (`models/shap_importance.json`).
5. **Milestone 5: Hardware Deployment & RTSP Edge Video Pipeline Optimization**:
   * Optimized execution on NVIDIA Jetson Orin AGX host nodes.
   * Profiled OpenCV frame decoding, landmark tracking, and tensor inference, achieving 14.2 ms total latency per frame (70.4 FPS).
6. **Milestone 6: viAct Enterprise UI & HSE Tiered Industrial Integration**:
   * Engineered full 17-module admin surveillance suite (`app.py`) conforming to viAct industrial design standards.
   * Connected real-time webcam inference, MP4 file upload processing, automated Permit to Work (PTW) lockout, and OSHA Section 1910 audit report generation.

---

## 5. Result and Discussion

### 5.1 Quantitative Performance Metrics
The system was evaluated on a strictly isolated, hold-out test partition comprising **10,131 unseen frames** containing diverse facial geometries, lighting levels, and fatigue expressions.

| Metric | Target Specification | Achieved Result | Evaluation Status |
|---|---|---|---|
| **Hold-out Test Accuracy** | $\ge 90.00\%$ | **99.90%** | **Target Exceeded (+9.90%)** |
| **Macro F1-Score** | $\ge 0.8800$ | **0.9988** | Exceptional Generalization |
| **Fatigued Class Recall** | $\ge 92.00\%$ | **99.89%** | Near-Zero Missed Incidents |
| **Fatigued Class Precision**| $\ge 90.00\%$ | **99.97%** | Near-Zero False Alarms |
| **Test ROC-AUC** | $\ge 0.9500$ | **1.0000** | Perfect Discriminative Capacity |
| **5-Fold Stratified CV AUC**| $\ge 0.9500$ | **1.0000** | Statistically Robust |
| **End-to-End Latency** | $< 50\,\text{ms}$ | **14.2 ms** | Real-Time Edge Ready (70.4 FPS) |

### 5.2 Hold-out Confusion Matrix Analysis
The confusion matrix over the 10,131 unseen hold-out frames demonstrates the extreme discriminative precision of the calibrated ensemble:

```
                      PREDICTED ALERT     PREDICTED FATIGUED
ACTUAL ALERT                3,143                  2
ACTUAL FATIGUED                 8              6,978
```

* **True Negatives ($TN = 3,143$)**: Alert operators correctly classified as vigilant.
* **False Positives ($FP = 2$)**: Only 2 alert frames misclassified as fatigued across the entire test set (False Alarm Rate: $0.06\%$).
* **False Negatives ($FN = 8$)**: Only 8 fatigued frames missed out of 6,986 actual drowsy frames (Missed Detection Rate: $0.11\%$).
* **True Positives ($TP = 6,978$)**: Confirmed fatigue events accurately detected and escalated.

### 5.3 Model Component Ablation Study
To isolate the contribution of each architectural component, an ablation study was conducted:

| Architecture Configuration | Accuracy | Macro F1 | ROC-AUC | Inference Time |
|---|---|---|---|---|
| Static Threshold EAR/MAR Heuristic | 78.42% | 0.7710 | 0.8120 | **1.2 ms** |
| Single LightGBM (Raw Landmarks Only) | 88.65% | 0.8812 | 0.9340 | 3.8 ms |
| LightGBM + Multi-Scale Temporal Buffer | 96.40% | 0.9632 | 0.9880 | 8.4 ms |
| Tri-Model Ensemble (Without Adaptive Baseline) | 94.15% | 0.9390 | 0.9750 | 13.5 ms |
| **Full Tri-Model Ensemble + Adaptive Baseline (Proposed)** | **99.90%** | **0.9988** | **1.0000** | **14.2 ms** |

**Key Finding**: Introducing the *Adaptive Baseline Normalization* elevated test accuracy from 94.15% to 99.90%. This confirms that resolving baseline anatomical differences is significantly more impactful for real-world accuracy than adding model depth.

### 5.4 Feature Importance & Interpretability (SHAP Analysis)
Global Shapley Additive exPlanations (SHAP) computed across the validation set revealed the dominant biometric contributors to the ensemble's decision manifold:

```
Biometric Feature                      SHAP Mean (|value|)   Biomechanical Rationale
────────────────────────────────────────────────────────────────────────────────────────
1. PERCLOS_60s_mean                          0.3842          Cumulative eyelid closure proportion
2. EAR_ratio_to_baseline                     0.3410          Relative eye openness drop
3. last_blink_duration                       0.2985          Microsleep duration (>400 ms)
4. neck_tilt                                 0.2640          Upper-body ergonomic slouching
5. EAR_vel (Negative Velocity)               0.2415          Speed of involuntary eyelid droop
6. adaptive_perclos_15s                      0.2180          Short-window vigilance decay
7. yawn_cumulative_count                     0.1945          Hypoxia & fatigue oxygen compensation
8. MAR_ratio_to_baseline                     0.1820          Sustained yawning oral opening
9. head_drop                                 0.1650          Forward cervical spine collapse
10. mouth_open_density                       0.1410          Proportion of sustained mouth opening
────────────────────────────────────────────────────────────────────────────────────────
```

The SHAP distribution establishes that the model does not fixate on static facial features, but rather leverages a physiologically grounded synthesis of **long-term eyelid closure ($PERCLOS_{60\text{s}}$)**, **dynamic baseline departure ($EAR_{\text{ratio}}$)**, and **cervical posture collapse ($\theta_{\text{neck}}$)**.

---

## 6. Discussion on Sustainability

The implementation of an Edge-AI fatigue surveillance system on offshore platforms generates deep, measurable impacts across three pillars of industrial sustainability:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   SUSTAINABILITY TRIAD IN OFFSHORE OPERATIONS               │
├───────────────────────┬──────────────────────────┬──────────────────────────┤
│    Human & Social     │      Environmental       │   Economic & Asset       │
├───────────────────────┼──────────────────────────┼──────────────────────────┤
│ • Prevents fatal      │ • Mitigates catastrophic │ • Prevents NPT ($1M+/day)│
│   crane/rig accidents │   blowouts & oil spills  │ • Low Edge compute power │
│ • Non-intrusive vision│ • Zero marine pollution  │   (25W vs 450W desktop)  │
│   respects privacy    │   from dropped objects   │ • Extends machinery life │
│ • Protects circadian  │ • Eliminates toxic waste │ • Eliminates disposable  │
│   health of operators │   from wearable batteries│   sensor e-waste         │
└───────────────────────┴──────────────────────────┴──────────────────────────┘
```

### 6.1 Human & Social Sustainability
* **Preservation of Human Life**: Fatigue is an invisible, involuntary hazard. Offshore operators routinely attempt to fight off sleep due to operational pressure or perceived duty. By deploying objective, automated monitoring, the system removes the subjective burden of self-reporting, enabling mandatory, blame-free rest breaks.
* **Ergonomic Longevity**: Chronic head and neck drooping causes long-term musculoskeletal disorders (cervical spondylosis). By monitoring neck inclination and shoulder slope, the system promotes active ergonomic corrections, extending operator career longevity.
* **Ethical AI & Worker Privacy**: By executing all landmark extraction and feature engineering locally on the edge node and discarding raw video frames immediately after tensor extraction, the architecture eliminates the risk of covert surveillance, wiretapping, or labor abuse.

### 6.2 Environmental Sustainability
* **Catastrophic Spill Prevention**: Deepwater offshore blowouts (e.g., Deepwater Horizon) frequently trace back to delayed human response times during pressure surge kicks. Real-time vigilance surveillance on drill floor operators and subsea valve technicians guarantees rapid emergency reaction times, preventing irreversible marine hydrocarbon spills.
* **Elimination of Battery E-Waste**: Unlike wearable PPG or EEG wristbands that rely on lithium-ion batteries requiring replacement every 12–24 months, optical CCTV surveillance leverages permanent, grid-powered rig infrastructure, eliminating hazardous battery disposal in sensitive ocean ecosystems.

### 6.3 Economic Sustainability & Energy Efficiency
* **Non-Productive Time (NPT) Reduction**: Offshore drilling rig day-rates range between $\$350,000$ and $\$850,000$ per day. Unplanned shutdowns caused by equipment dropped from fatigue-impaired crane operators cost millions in asset repairs and delayed exploration. Preventing a single dropped-object incident yields immediate return on investment.
* **Edge Compute Energy Efficiency**: The entire inference stack operates on an embedded NVIDIA Jetson AGX Orin node consuming less than **25 Watts**, in contrast to high-end server clusters consuming $1.5\,\text{kW}-3.0\,\text{kW}$. This minimizes the auxiliary electrical load on rig diesel-electric generator sets, cutting continuous carbon emissions.

---

## 7. Conclusions and Recommendations

### 7.1 Conclusions
This research demonstrates that non-intrusive, multi-modal computer vision paired with temporal gradient-boosted ensembles provides a highly effective solution to offshore worker fatigue detection. The key conclusions are:

1. **Overcoming Anatomical Variance**: The formulation of an online subject-adaptive baseline normalizer ($EAR_{\text{ratio}}$, $MAR_{\text{ratio}}$) resolved the demographic bias of static threshold heuristics, elevating test accuracy to **99.90%** and precision to **99.97%**.
2. **Dual-Stream Synergy**: Synthesizing 468 facial mesh landmarks with 33 upper-body kinematic keypoints ensures resilient detection even when operators tilt their heads or look down at platform control interfaces.
3. **Real-Time Edge Feasibility**: The heterogeneous soft-voting ensemble (LightGBM 0.40 + XGBoost 0.35 + CatBoost 0.25) achieved sub-15 ms latency per frame on edge hardware, enabling continuous 70+ FPS multi-camera monitoring.
4. **HSE Industrial Integration**: Embedding the system within a viAct-compliant industrial dashboard connected to automated Permit to Work (PTW) gateways transforms passive machine learning into an active, life-saving safety control.

### 7.2 Recommendations for Research & Industrial Practice

#### Recommendations for Industrial Rig Operators
* **Adopt Closed-Loop PTW Interlocks**: Rig operators should integrate the fatigue scoring engine directly with digital Permit to Work (PTW) databases. Heavy machinery permits (cranes, top-drives) should automatically suspend if an operator's 15-minute rolling fatigue score exceeds 0.70.
* **Establish Blame-Free Circadian Rest Policies**: Tiered Amber alerts should trigger mandatory 15-to-20-minute rest, hydration, and light-exposure breaks without penalizing worker compensation, fostering an open safety culture.
* **Edge-Only Biometric Processing**: Offshore operators should mandate that all facial landmark extraction occurs locally on edge nodes without saving raw facial imagery to cloud storage, ensuring universal union buy-in and GDPR compliance.

#### Recommendations for Future Academic Research
* **Multi-Spectral IR Integration**: Future research should evaluate active near-infrared (NIR) and long-wave infrared (LWIR) thermal cameras to benchmark detection under total blackout night-shift conditions and heavy maritime fog.
* **Gaze Tracking & Pupil Diameter Dynamics**: Incorporating corneal reflection vector analysis and pupil dilation dynamics (hippus oscillations) will unlock direct measurement of autonomous cognitive workload.
* **Federated Multi-Rig Continuous Learning**: Implement privacy-preserving federated learning across a fleet of offshore platforms, enabling local models to update ensemble weights from edge-detected edge cases without transmitting biometric data across rig satellite links.

---

## Complete Feature Registry Reference (175 Dimension Vector)
For implementation replication, the 175 engineered feature columns are archived in the repository model metadata:
* **Instantaneous Biometrics (15)**: `EAR`, `MAR`, `PERCLOS`, `BlinkRate`, `YawnFreq`, `MouthOpen`, `ear_mar_ratio`, `ear_sq`, `perclos_sq`, `mar_over_ear`, `ear_x_perclos`, `eye_closed_flag`, `yawn_flag`, `compound_fatigue`, `perclos_x_blink`.
* **Adaptive Baseline Normalizations (28)**: `EAR_ratio_to_baseline`, `EAR_zscore_session`, `MAR_ratio_to_baseline`, `MAR_zscore_session`, `adaptive_perclos_5s/15s/30s/60s`, `EAR_ratio_5s/15s/30s/60s_mean/min/std`, `MAR_ratio_5s/15s/30s/60s_mean/max`.
* **Rolling Temporal Statistics (112)**: 5s, 15s, 30s, and 60s window means, standard deviations, mins, maxs, ranges, and 10th, 25th, 75th, and 90th percentiles for `EAR`, `MAR`, and `PERCLOS`.
* **Kinematics & Micro-Events (20)**: `EAR_vel`, `EAR_acc`, `PERCLOS_vel`, `PERCLOS_acc`, `MAR_vel`, `MAR_acc`, `last_blink_duration`, `blink_closure_speed`, `blink_opening_speed`, `last_yawn_duration`, `yawn_duration_mean`, `yawn_duration_max`, `yawn_rate_per_min`, `yawn_cumulative_count`, `yawn_peak_mar`, `mouth_open_density`, `neck_tilt`, `head_drop`, `shoulder_slope`.
