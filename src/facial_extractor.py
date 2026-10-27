"""
src/facial_extractor.py
Extracts EAR, MAR, PERCLOS, BlinkRate, YawnFreq, MouthOpen
from a single BGR frame using MediaPipe FaceMesh.
"""

import math
import sys
from pathlib import Path
from collections import deque

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False


# ── MediaPipe landmark indices (478-point FaceMesh) ───────────────────────────
# Left eye
_L_EYE = [362, 385, 387, 263, 373, 380]
# Right eye
_R_EYE = [33,  160, 158, 133, 153, 144]
# Mouth outer
_MOUTH = [61,  291, 13,  14,  17,  0,   78,  308]
# Mouth vertical (for MAR)
_MOUTH_V = [13, 14]   # top-bottom lip center
_MOUTH_H = [61, 291]  # left-right mouth corners


def _euclidean(p1, p2) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _eye_aspect_ratio(landmarks, indices) -> float:
    """
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    indices: [p1, p2, p3, p4, p5, p6]
    """
    p = [landmarks[i] for i in indices]
    vertical1 = _euclidean(p[1], p[5])
    vertical2 = _euclidean(p[2], p[4])
    horizontal = _euclidean(p[0], p[3])
    if horizontal < 1e-6:
        return 0.0
    return (vertical1 + vertical2) / (2.0 * horizontal)


def _mouth_aspect_ratio(landmarks) -> float:
    """
    MAR = vertical mouth opening / horizontal mouth width
    """
    top    = landmarks[13]
    bottom = landmarks[14]
    left   = landmarks[61]
    right  = landmarks[291]
    vertical   = _euclidean(top, bottom)
    horizontal = _euclidean(left, right)
    if horizontal < 1e-6:
        return 0.0
    return vertical / horizontal


class FacialFeatureExtractor:
    """
    Per-frame facial feature extraction.

    Outputs a dict with keys matching the trained CSV feature columns:
        EAR, MAR, PERCLOS, BlinkRate, YawnFreq, MouthOpen
    """

    def __init__(
        self,
        ear_threshold: float  = None,
        mar_threshold: float  = None,
        perclos_threshold: float = None,
        temporal_window: int  = None,
        blink_win_sec: float  = None,
    ):
        self.ear_threshold   = ear_threshold   or config.EAR_THRESHOLD
        self.mar_threshold   = mar_threshold   or config.MAR_THRESHOLD
        self.perclos_thresh  = perclos_threshold or config.PERCLOS_THRESHOLD
        self.temporal_window = temporal_window or config.TEMPORAL_WINDOW
        self.blink_win_sec   = blink_win_sec   or config.BLINK_WIN_SEC

        # Rolling buffers
        self._ear_buffer: deque  = deque(maxlen=self.temporal_window)
        self._mar_buffer: deque  = deque(maxlen=self.temporal_window)
        self._blink_times: deque = deque()      # timestamps of blink events
        self._yawn_times: deque  = deque()      # timestamps of yawn events
        self._prev_eye_closed: bool = False
        self._prev_mouth_open: bool = False
        self._frame_timestamps: deque = deque() # for PERCLOS denominator

        # MediaPipe init (with robust OpenCV fallback)
        self._mode = "cv2"
        self._face_mesh = None
        if _MP_AVAILABLE and hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            try:
                _mp_fm = mp.solutions.face_mesh
                self._face_mesh = _mp_fm.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self._mode = "mediapipe"
            except Exception:
                self._mode = "cv2"

        # OpenCV cascade fallbacks
        self._face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self._eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    # ── Public API ────────────────────────────────────────────────────────────
    def extract(self, frame_bgr: np.ndarray, timestamp_sec: float = 0.0) -> dict | None:
        """
        Process one BGR frame and return feature dict.
        Returns None if no face is detected.

        Args:
            frame_bgr     : OpenCV BGR frame
            timestamp_sec : elapsed seconds (used for blink/yawn rate calculation)

        Returns:
            dict with keys: EAR, MAR, PERCLOS, BlinkRate, YawnFreq, MouthOpen
        """
        face_box = None
        if self._mode == "mediapipe" and self._face_mesh:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = self._face_mesh.process(rgb)
            if not results.multi_face_landmarks:
                return None
            h, w = frame_bgr.shape[:2]
            lm_raw = results.multi_face_landmarks[0].landmark
            landmarks = [(int(l.x * w), int(l.y * h)) for l in lm_raw]
            ear_l = _eye_aspect_ratio(landmarks, _L_EYE)
            ear_r = _eye_aspect_ratio(landmarks, _R_EYE)
            ear   = (ear_l + ear_r) / 2.0
            mar   = _mouth_aspect_ratio(landmarks)
            xs = [p[0] for p in landmarks]; ys = [p[1] for p in landmarks]
            face_box = (max(0, min(xs)-15), max(0, min(ys)-25), max(xs)-min(xs)+30, max(ys)-min(ys)+40)
        else:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = self._face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
            if len(faces) == 0:
                return None
            fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
            face_box = (int(fx), int(fy), int(fw), int(fh))
            eyes_roi = gray[fy + int(fh*0.18) : fy + int(fh*0.55), fx : fx + fw]
            eyes = self._eye_cascade.detectMultiScale(eyes_roi, scaleFactor=1.1, minNeighbors=3, minSize=(15, 15))
            if len(eyes) >= 2:
                ear_vals = [float(eh) / max(float(ew), 1.0) * 0.72 for (_, _, ew, eh) in eyes[:2]]
                ear = float(np.clip(np.mean(ear_vals), 0.12, 0.42))
            elif len(eyes) == 1:
                ew, eh = eyes[0][2], eyes[0][3]
                ear = float(np.clip(float(eh) / max(float(ew), 1.0) * 0.72, 0.12, 0.42))
            else:
                ear = 0.16  # Eyes closed or not visible

            mouth_roi = gray[fy + int(fh*0.65) : fy + int(fh*0.95), fx : fx + fw]
            m_mean = float(np.mean(mouth_roi)) if mouth_roi.size > 0 else 128.0
            m_dark = float(np.sum(mouth_roi < (m_mean * 0.65))) / max(float(mouth_roi.size), 1.0)
            mar = float(np.clip(0.22 + m_dark * 1.6, 0.15, 1.25))

        self._ear_buffer.append(ear)
        self._mar_buffer.append(mar)

        # ── PERCLOS (% frames with EAR below threshold in rolling window) ─────
        if len(self._ear_buffer) > 0:
            closed_count = sum(1 for e in self._ear_buffer if e < self.ear_threshold)
            perclos = closed_count / len(self._ear_buffer)
        else:
            perclos = 0.0

        # ── Blink detection (EAR crosses threshold edge) ──────────────────────
        eye_closed_now = ear < self.ear_threshold
        if self._prev_eye_closed and not eye_closed_now:
            # Rising edge = completed blink
            self._blink_times.append(timestamp_sec)
        self._prev_eye_closed = eye_closed_now

        # Prune old blink events outside window
        cutoff = timestamp_sec - self.blink_win_sec
        while self._blink_times and self._blink_times[0] < cutoff:
            self._blink_times.popleft()
        blink_rate = len(self._blink_times) / max(1.0, min(timestamp_sec, self.blink_win_sec)) * 60.0

        # ── Yawn detection (MAR crosses threshold edge) ───────────────────────
        mouth_open_now = mar > self.mar_threshold
        if not self._prev_mouth_open and mouth_open_now:
            # Rising edge = yawn started
            self._yawn_times.append(timestamp_sec)
        self._prev_mouth_open = mouth_open_now

        while self._yawn_times and self._yawn_times[0] < cutoff:
            self._yawn_times.popleft()
        yawn_freq = len(self._yawn_times) / max(1.0, min(timestamp_sec, self.blink_win_sec)) * 60.0

        return {
            "EAR":       round(ear,       6),
            "MAR":       round(mar,       6),
            "PERCLOS":   round(perclos,   6),
            "BlinkRate": round(blink_rate,4),
            "YawnFreq":  round(yawn_freq, 4),
            "MouthOpen": int(mouth_open_now),
            "face_box":  face_box,
        }

    def reset(self) -> None:
        """Reset all rolling buffers (call at start of new session/shift)."""
        self._ear_buffer.clear()
        self._mar_buffer.clear()
        self._blink_times.clear()
        self._yawn_times.clear()
        self._frame_timestamps.clear()
        self._prev_eye_closed = False
        self._prev_mouth_open = False

    def close(self) -> None:
        if self._face_mesh is not None:
            try:
                self._face_mesh.close()
            except Exception:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def annotate_frame(self, frame_bgr: np.ndarray, features: dict) -> np.ndarray:
        """
        Draw EAR / MAR / status overlay on a copy of the frame.
        Returns annotated frame.
        """
        out = frame_bgr.copy()
        ear = features.get("EAR", 0)
        mar = features.get("MAR", 0)
        perclos = features.get("PERCLOS", 0)

        # Status colour
        if perclos > self.perclos_thresh:
            color = (0, 0, 220)  # red
            label = "FATIGUED"
        elif ear < self.ear_threshold + 0.05:
            color = (0, 165, 255)  # amber
            label = "LOW VIGILANCE"
        else:
            color = (0, 200, 80)   # green
            label = "ALERT"

        cv2.putText(out, f"EAR: {ear:.3f}",
                    (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(out, f"MAR: {mar:.3f}",
                    (10, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(out, f"PERCLOS: {perclos:.2%}",
                    (10, 90),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(out, label,
                    (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        return out
