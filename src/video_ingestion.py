"""
src/video_ingestion.py
Handles webcam and video-file input with configurable resolution and FPS.
"""

import cv2
import sys
from pathlib import Path

# Allow running from any directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


class VideoIngestion:
    """
    Unified video source abstraction.
    Supports webcam index, local video files, and RTSP/HTTP streams.
    """

    def __init__(self, source=None, width: int = None, height: int = None):
        """
        Args:
            source : int (camera index) | str (file path or stream URL) | None
                     None defaults to config.CAMERA_INDEX
            width  : frame width  (defaults to config.FRAME_WIDTH)
            height : frame height (defaults to config.FRAME_HEIGHT)
        """
        self.source = source if source is not None else config.CAMERA_INDEX
        self.width  = width  or config.FRAME_WIDTH
        self.height = height or config.FRAME_HEIGHT
        self._cap: cv2.VideoCapture | None = None

    # ── Context Manager ───────────────────────────────────────────────────────
    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.release()

    # ── Public API ────────────────────────────────────────────────────────────
    def open(self) -> None:
        """Open the video source and configure resolution."""
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"[VideoIngestion] Cannot open source: {self.source!r}. "
                "Check camera index or file path."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read(self) -> tuple[bool, object]:
        """
        Read the next frame.

        Returns:
            (success: bool, frame: np.ndarray | None)
        """
        if self._cap is None:
            raise RuntimeError("[VideoIngestion] Source not opened. Call open() first.")
        return self._cap.read()

    def release(self) -> None:
        """Release the video capture resource."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def fps(self) -> float:
        if self._cap is None:
            return 0.0
        return self._cap.get(cv2.CAP_PROP_FPS) or config.INFERENCE_FPS

    @property
    def frame_count(self) -> int:
        """Total frames (only meaningful for file sources)."""
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self):
        """Iterate frames until source is exhausted or stream fails."""
        while self.is_opened():
            ok, frame = self.read()
            if not ok:
                break
            yield frame
