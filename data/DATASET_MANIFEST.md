# Dataset Manifest — Offshore Fatigue Detection

All datasets used in training. Paths are relative to `data/raw/`.

---

## 1. UTA-RLDD + YawDD (Combined CSV)

| Property | Value |
|----------|-------|
| **Source** | User-provided (`Dataset Combined.zip`) |
| **License** | UTA-RLDD: Free for non-commercial research; YawDD: Academic use |
| **Folder** | `data/raw/uta_yawdd/` (loaded directly from ZIP via config) |
| **Size** | ~20 MB compressed, 415 CSV files, 82,067 rows |
| **Format** | Per-session CSV files (one per video) |
| **Columns** | `EAR, MAR, PERCLOS, BlinkRate, YawnFreq, MouthOpen, Label, Video, t_video, backend, eff_win_sec, cold_start` |
| **Labels** | `Alert` (29,617 rows) · `Fatigued` (52,450 rows) |
| **Backend** | MediaPipe FaceMesh (pre-extracted) |
| **Notes** | Cold-start rows (`cold_start==1`) dropped during training |

### Folder Structure
```
data/raw/uta_yawdd/          ← optional extracted location
  Alert/
    *.csv                    ← one CSV per alert-state video session
  Fatigued/
    *.csv                    ← one CSV per fatigued-state video session
```

---

## 2. MRL Eye Dataset

| Property | Value |
|----------|-------|
| **Source** | [Kaggle: akashshingha850/mrl-eye-dataset](https://www.kaggle.com/datasets/akashshingha850/mrl-eye-dataset) |
| **License** | CC BY 4.0 |
| **Folder** | `data/raw/mrl_eye/` |
| **Size** | ~2.8 GB compressed |
| **Format** | Per-image labels (open/closed, IR/visible light) |
| **Annotation** | Folder-based: `open/` and `closed/` subdirectories |
| **Usage** | Eye-state classifier calibration; supplements EAR threshold |
| **Download** | `kaggle datasets download -d akashshingha850/mrl-eye-dataset` |

### Folder Structure
```
data/raw/mrl_eye/
  open/
    *.png / *.jpg            ← open-eye images
  closed/
    *.png / *.jpg            ← closed-eye images
```

---

## 3. MPII Human Pose

| Property | Value |
|----------|-------|
| **Source** | [MPI Informatics](https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/) |
| **License** | Free for non-commercial research |
| **Folder** | `data/raw/mpii_pose/` |
| **Images** | `mpii_human_pose_v1.tar.gz` (~12.9 GB) |
| **Annotations** | `mpii_human_pose_v1_u12_2.zip` (`.mat` format) |
| **Annotation Format** | MATLAB `.mat` struct; 16-joint skeleton (MPII format) |
| **Joints** | 0=RAnkle, 1=RKnee, 2=RHip, 3=LHip, 4=LKnee, 5=LAnkle, 6=Pelvis, 7=Thorax, 8=UpperNeck, 9=HeadTop, 10=RWrist, 11=RElbow, 12=RShoulder, 13=LShoulder, 14=LElbow, 15=LWrist |
| **Usage** | Postural feature extractor training (shoulder angle, neck tilt) |

### Folder Structure
```
data/raw/mpii_pose/
  images/
    *.jpg                    ← ~25,000 activity images
  annotations/
    mpii_human_pose_v1_u12_2.mat
```

---

## 4. COCO Pose 2017 (val set)

| Property | Value |
|----------|-------|
| **Source** | [cocodataset.org](http://cocodataset.org) |
| **License** | CC BY 4.0 |
| **Images** | `http://images.cocodataset.org/zips/val2017.zip` (~780 MB) |
| **Annotations** | `http://images.cocodataset.org/annotations/annotations_trainval2017.zip` (~240 MB) |
| **Folder** | `data/raw/coco_pose/` |
| **Annotation Format** | JSON; 17-point keypoint format |
| **Keypoints** | 0=Nose, 1=LEye, 2=REye, 3=LEar, 4=REar, 5=LShoulder, 6=RShoulder, 7=LElbow, 8=RElbow, 9=LWrist, 10=RWrist, 11=LHip, 12=RHip, 13=LKnee, 14=RKnee, 15=LAnkle, 16=RAnkle |
| **Usage** | Supplements MPII for postural model (broader pose diversity) |

### Folder Structure
```
data/raw/coco_pose/
  images/
    val2017/
      *.jpg                  ← 5,000 validation images
  annotations/
    person_keypoints_val2017.json
    instances_val2017.json
```

---

## Notes

- `YawDD` and `UTA-RLDD` were pre-processed with MediaPipe FaceMesh to produce per-frame feature CSVs. The raw videos are NOT included.
- MPII and COCO datasets require feature extraction via `postural_extractor.py` before they can be used in training.
- Do NOT commit any files in `data/raw/` — see `data/.gitignore`.
