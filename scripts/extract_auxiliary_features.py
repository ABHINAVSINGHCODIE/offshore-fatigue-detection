"""
scripts/extract_auxiliary_features.py
═════════════════════════════════════
Extracts auxiliary calibration and reference distributions from:
1. MRL Eye Dataset (84,900 images) -> mrl_eye_features.csv
2. COCO Pose Keypoints (person_keypoints_val2017.json) -> postural_features.csv

This enables multi-modal calibration:
- Precision eye-closure thresholds and aperture distributions from MRL
- Natural upper-body posture variation priors (neck tilt, shoulder slope, head drop) from COCO
"""

import json
import logging
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("extract_aux")


# ─────────────────────────────────────────────────────────────────────────────
# 1. MRL Eye Dataset Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_mrl_features(max_samples: int = 15000) -> pd.DataFrame:
    """
    Extract geometric and photometric features from MRL Eye dataset.
    Filename format:
      <subject_id>_<image_id>_<gender>_<glasses>_<eye_state>_<reflection>_<lighting>_<frame_id>.png
      where eye_state: 0 = closed, 1 = open
    """
    log.info("═" * 60)
    log.info("  Extracting MRL Eye Dataset Features …")
    log.info("═" * 60)

    mrl_dir = config.MRL_DIR
    pngs = list(mrl_dir.rglob("*.png")) + list(mrl_dir.rglob("*.jpg"))
    if not pngs:
        log.warning(f"No images found in {mrl_dir}")
        return pd.DataFrame()

    log.info(f"  Found {len(pngs):,} images in {mrl_dir}")
    if len(pngs) > max_samples:
        rng = np.random.default_rng(config.RANDOM_STATE)
        pngs = rng.choice(pngs, size=max_samples, replace=False).tolist()
        log.info(f"  Sampled {len(pngs):,} images for feature extraction")

    records = []
    for p in pngs:
        stem = Path(p).stem
        parts = stem.split("_")
        if len(parts) >= 8:
            try:
                subject_id = parts[0]
                gender = int(parts[2])
                glasses = int(parts[3])
                eye_state = int(parts[4])  # 0=closed, 1=open
                refl = int(parts[5])
                lighting = int(parts[6])
            except ValueError:
                continue
        else:
            # Fallback if in open/closed subdirectories
            parent = Path(p).parent.name.lower()
            eye_state = 1 if "open" in parent else (0 if "close" in parent else None)
            if eye_state is None:
                continue
            subject_id, gender, glasses, refl, lighting = "unknown", 0, 0, 0, 0

        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        h, w = img.shape[:2]
        # Binarize / threshold to measure eye aperture ratio
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        aperture_ratio = 0.0
        if contours:
            largest = max(contours, key=cv2.contourArea)
            _, _, cw, ch = cv2.boundingRect(largest)
            aperture_ratio = round(ch / (cw + 1e-6), 4)

        mean_intensity = float(np.mean(img))
        std_intensity = float(np.std(img))

        records.append({
            "subject_id": subject_id,
            "eye_state": eye_state,  # 0=closed, 1=open
            "aperture_ratio": aperture_ratio,
            "mean_intensity": round(mean_intensity, 2),
            "std_intensity": round(std_intensity, 2),
            "glasses": glasses,
            "lighting": lighting,
        })

    df = pd.DataFrame(records)
    log.info(f"  Extracted {len(df):,} MRL records (Open: {(df['eye_state']==1).sum():,}, Closed: {(df['eye_state']==0).sum():,})")
    
    out_path = config.MRL_FEATURES_CSV
    df.to_csv(out_path, index=False)
    log.info(f"  MRL features saved → {out_path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. COCO Pose 2017 Keypoints Postural Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_coco_postural_features() -> pd.DataFrame:
    """
    Extract upper body posture angles (shoulder slope, neck tilt, head drop)
    from COCO val2017 keypoint annotations.
    """
    log.info("═" * 60)
    log.info("  Extracting COCO Pose 2017 Postural Features …")
    log.info("═" * 60)

    ann_file = config.COCO_DIR / "annotations" / "person_keypoints_val2017.json"
    if not ann_file.exists():
        # Look for annotations in subdirectories
        candidates = list(config.COCO_DIR.rglob("*person_keypoints*.json"))
        if candidates:
            ann_file = candidates[0]
        else:
            log.warning(f"COCO annotations not found at {ann_file}")
            return pd.DataFrame()

    log.info(f"  Reading COCO annotations: {ann_file} …")
    with open(ann_file, "r") as f:
        coco_data = json.load(f)

    annotations = coco_data.get("annotations", [])
    log.info(f"  Found {len(annotations):,} total annotations")

    # COCO Keypoint format (17 keypoints: [x, y, v], 0=unlabeled, 1=labeled invisible, 2=visible)
    # 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
    # 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
    # 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip ...
    records = []
    for ann in annotations:
        kpts = ann.get("keypoints", [])
        if len(kpts) < 17 * 3:
            continue

        # Extract visible points
        def get_pt(idx):
            x = kpts[idx * 3]
            y = kpts[idx * 3 + 1]
            v = kpts[idx * 3 + 2]
            return (x, y, v)

        nose = get_pt(0)
        le = get_pt(3)
        re = get_pt(4)
        ls = get_pt(5)
        rs = get_pt(6)
        lh = get_pt(11)
        rh = get_pt(12)

        # Require shoulders and at least one head reference point to be visible (v > 0)
        if ls[2] == 0 or rs[2] == 0 or nose[2] == 0:
            continue

        # Shoulder slope
        dx = rs[0] - ls[0] + 1e-9
        dy = rs[1] - ls[1]
        shoulder_slope = math.atan2(dy, dx)

        # Mid shoulder
        mid_s_x = (ls[0] + rs[0]) / 2.0
        mid_s_y = (ls[1] + rs[1]) / 2.0

        # Head centre (mid ear if visible, else nose)
        if le[2] > 0 and re[2] > 0:
            mid_head_x = (le[0] + re[0]) / 2.0
            mid_head_y = (le[1] + re[1]) / 2.0
        else:
            mid_head_x = nose[0]
            mid_head_y = nose[1]

        # Neck vector and tilt from vertical
        neck_dx = mid_head_x - mid_s_x
        neck_dy = mid_head_y - mid_s_y
        mag = math.sqrt(neck_dx**2 + neck_dy**2) + 1e-9
        # Vertical is (0, -1)
        cos_tilt = max(-1.0, min(1.0, (-neck_dy) / mag))
        neck_tilt_deg = math.degrees(math.acos(cos_tilt))

        # Torso height
        if lh[2] > 0 and rh[2] > 0:
            mid_hip_y = (lh[1] + rh[1]) / 2.0
            torso_h = abs(mid_hip_y - mid_s_y) + 1e-9
        else:
            # Fallback estimation based on shoulder width
            torso_h = math.sqrt(dx**2 + dy**2) * 1.5 + 1e-9

        head_drop = (nose[1] - mid_s_y) / torso_h

        # Label fatigue postural risk: severe slouching or neck tilt > 20 deg or head drop
        postural_fatigue_flag = int(
            neck_tilt_deg > config.NECK_TILT_FATIGUE_DEG
            or head_drop > config.HEAD_DROP_FATIGUE_RATIO
            or abs(shoulder_slope) > config.SHOULDER_SLOPE_MAX_RAD
        )

        records.append({
            "shoulder_slope": round(shoulder_slope, 5),
            "neck_tilt_deg": round(neck_tilt_deg, 3),
            "head_drop_ratio": round(head_drop, 5),
            "postural_fatigue_flag": postural_fatigue_flag,
        })

    df = pd.DataFrame(records)
    log.info(f"  Extracted {len(df):,} valid postural records from COCO")
    
    out_path = config.POSTURAL_FEATURES_CSV
    df.to_csv(out_path, index=False)
    log.info(f"  Postural features saved → {out_path}")
    return df


if __name__ == "__main__":
    extract_mrl_features()
    extract_coco_postural_features()
