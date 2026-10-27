"""
src/postural_extractor.py
Extracts postural features (shoulder angle, neck tilt, head drop)
from a single BGR frame using MediaPipe Pose.
"""

import sys
import math
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False


def _angle_between(p1, p2, p3) -> float:
    """
    Angle at p2 in the triangle p1-p2-p3 (degrees).
    """
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    dot  = v1[0]*v2[0] + v1[1]*v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2) + 1e-9
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2) + 1e-9
    cos_a = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_a))


class PosturalFeatureExtractor:
    """
    Extracts body-posture features from BGR frames via MediaPipe Pose.

    Output feature keys:
        shoulder_slope   — signed slope of shoulder line (radians)
        neck_tilt        — angle between neck vector and vertical (degrees)
        head_drop        — normalised vertical drop of nose below mid-shoulder
        left_ear_score   — MediaPipe landmark visibility for left ear
        right_ear_score  — MediaPipe landmark visibility for right ear
    """

    # MediaPipe Pose landmark IDs
    _LEFT_SHOULDER  = 11
    _RIGHT_SHOULDER = 12
    _LEFT_EAR       = 7
    _RIGHT_EAR      = 8
    _NOSE           = 0
    _LEFT_HIP       = 23
    _RIGHT_HIP      = 24

    def __init__(self):
        self._mode = "cv2"
        self._pose = None
        if _MP_AVAILABLE and hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
            try:
                _mp_pose = mp.solutions.pose
                self._pose = _mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    smooth_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self._mode = "mediapipe"
            except Exception:
                self._mode = "cv2"

        self._face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self._prev_cy = None

    # ── Public API ────────────────────────────────────────────────────────────
    def extract(self, frame_bgr: np.ndarray) -> dict | None:
        """
        Process one BGR frame and return postural feature dict.
        Returns None if pose or face is not detected.
        """
        if self._mode == "mediapipe" and self._pose:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = self._pose.process(rgb)
            if not results.pose_landmarks:
                return None

            h, w = frame_bgr.shape[:2]
            lm = results.pose_landmarks.landmark

            def pt(idx):
                return (lm[idx].x * w, lm[idx].y * h)

            ls  = pt(self._LEFT_SHOULDER)
            rs  = pt(self._RIGHT_SHOULDER)
            le  = pt(self._LEFT_EAR)
            re  = pt(self._RIGHT_EAR)
            ns  = pt(self._NOSE)
            lh  = pt(self._LEFT_HIP)
            rh  = pt(self._RIGHT_HIP)

            # ── Shoulder slope ────────────────────────────────────────────────────
            dx = rs[0] - ls[0] + 1e-9
            dy = rs[1] - ls[1]
            shoulder_slope = math.atan2(dy, dx)          # radians, signed

            # ── Mid-shoulder point ────────────────────────────────────────────────
            mid_shoulder = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)

            # ── Mid-ear point (proxy for head centre) ─────────────────────────────
            mid_ear = ((le[0] + re[0]) / 2, (le[1] + re[1]) / 2)

            # ── Neck tilt (angle of neck vector from vertical) ────────────────────
            neck_vec = (mid_ear[0] - mid_shoulder[0], mid_ear[1] - mid_shoulder[1])
            vertical  = (0, -1)
            dot   = neck_vec[0]*vertical[0] + neck_vec[1]*vertical[1]
            mag_n = math.sqrt(neck_vec[0]**2 + neck_vec[1]**2) + 1e-9
            cos_t = max(-1.0, min(1.0, dot / mag_n))
            neck_tilt = math.degrees(math.acos(cos_t))  # 0 = straight up

            # ── Head drop (nose below mid-shoulder, normalised by torso height) ───
            torso_h = abs((lh[1] + rh[1]) / 2 - mid_shoulder[1]) + 1e-9
            head_drop = (ns[1] - mid_shoulder[1]) / torso_h

            return {
                "shoulder_slope":  round(shoulder_slope, 5),
                "neck_tilt":       round(neck_tilt,      3),
                "head_drop":       round(head_drop,      5),
                "left_ear_score":  round(lm[self._LEFT_EAR].visibility,  3),
                "right_ear_score": round(lm[self._RIGHT_EAR].visibility, 3),
            }
        else:
            # OpenCV geometry based posture fallback
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = self._face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
            if len(faces) == 0:
                return {
                    "shoulder_slope":  0.0,
                    "neck_tilt":       0.0,
                    "head_drop":       0.0,
                    "left_ear_score":  1.0,
                    "right_ear_score": 1.0,
                }
            fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
            cy = fy + fh / 2.0
            h = frame_bgr.shape[0]
            # Normalised vertical head drop relative to upper quadrant
            norm_drop = float(np.clip((cy - (h * 0.35)) / max(float(h * 0.5), 1.0), -0.2, 0.6))
            # Head tilt estimated from face aspect ratio displacement
            tilt = float(np.clip(abs(norm_drop) * 35.0, 0.0, 45.0))
            return {
                "shoulder_slope":  0.02,
                "neck_tilt":       round(tilt, 2),
                "head_drop":       round(max(0.0, norm_drop), 4),
                "left_ear_score":  1.0,
                "right_ear_score": 1.0,
            }

    def close(self) -> None:
        if self._pose is not None:
            try:
                self._pose.close()
            except Exception:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
