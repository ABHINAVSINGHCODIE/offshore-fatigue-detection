"""
src/hse_logger.py
HSE-compliant alert engine and shift-report writer.

Alert levels:
    GREEN  (score < AMBER_THRESH)  — Alert / Normal
    AMBER  (AMBER_THRESH ≤ score < RED_THRESH) — Low Vigilance
    RED    (score ≥ RED_THRESH)    — Fatigued / Critical
"""

import csv
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

log = logging.getLogger(__name__)

# ── Alert level constants ─────────────────────────────────────────────────────
LEVEL_GREEN = "GREEN"
LEVEL_AMBER = "AMBER"
LEVEL_RED   = "RED"


def score_to_level(score: float) -> str:
    if score >= config.ALERT_RED_SCORE:
        return LEVEL_RED
    elif score >= config.ALERT_AMBER_SCORE:
        return LEVEL_AMBER
    return LEVEL_GREEN


class HSELogger:
    """
    Writes per-event rows to hse_events.csv and generates shift
    summary JSON reports.

    Thread-safe via file-level append (no shared state between processes).
    """

    _CSV_HEADERS = [
        "timestamp", "shift_id", "worker_id", "fatigue_score",
        "alert_level", "ear", "mar", "perclos", "blink_rate",
        "yawn_freq", "mouth_open", "duration_sec",
    ]

    def __init__(self, worker_id: str = "unknown", shift_id: str = None):
        self.worker_id = worker_id
        self.shift_id  = shift_id or datetime.now().strftime("shift_%Y%m%d_%H%M%S")
        self._shift_start: float = time.time()
        self._events: list[dict]  = []
        self._last_level: str     = LEVEL_GREEN
        self._level_start: float  = time.time()

        # Ensure log dir and CSV header exist
        config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        config.SHIFT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        self._ensure_csv_header()

    # ── Public API ────────────────────────────────────────────────────────────
    def log_event(self, score: float, features: dict) -> str:
        """
        Evaluate score, detect level transitions, write event to CSV.

        Returns the current alert level string.
        """
        now   = time.time()
        level = score_to_level(score)

        # Detect level transition
        if level != self._last_level:
            duration = now - self._level_start
            event = {
                "timestamp":    datetime.now().isoformat(timespec="seconds"),
                "shift_id":     self.shift_id,
                "worker_id":    self.worker_id,
                "fatigue_score": round(score, 4),
                "alert_level":  level,
                "ear":          round(features.get("EAR",        0), 4),
                "mar":          round(features.get("MAR",        0), 4),
                "perclos":      round(features.get("PERCLOS",    0), 4),
                "blink_rate":   round(features.get("BlinkRate",  0), 4),
                "yawn_freq":    round(features.get("YawnFreq",   0), 4),
                "mouth_open":   int(features.get("MouthOpen",    0)),
                "duration_sec": round(duration, 2),
            }
            self._events.append(event)
            self._write_csv_row(event)
            log.info(f"[HSE] {self._last_level} → {level}  (score={score:.3f})")
            self._last_level  = level
            self._level_start = now

        return level

    def generate_shift_report(self) -> Path:
        """
        Write a JSON shift report and return its path.
        """
        now     = datetime.now()
        elapsed = time.time() - self._shift_start

        amber_events = [e for e in self._events if e["alert_level"] == LEVEL_AMBER]
        red_events   = [e for e in self._events if e["alert_level"] == LEVEL_RED]

        report = {
            "shift_id":          self.shift_id,
            "worker_id":         self.worker_id,
            "generated_at":      now.isoformat(timespec="seconds"),
            "shift_duration_sec": round(elapsed, 1),
            "total_events":      len(self._events),
            "amber_alerts":      len(amber_events),
            "red_alerts":        len(red_events),
            "avg_fatigue_score": (
                round(sum(e["fatigue_score"] for e in self._events) / len(self._events), 4)
                if self._events else 0.0
            ),
            "peak_fatigue_score": (
                round(max(e["fatigue_score"] for e in self._events), 4)
                if self._events else 0.0
            ),
            "hse_compliant": len(red_events) == 0,
            "events": self._events,
        }

        filename = f"{self.shift_id}_{now.strftime('%H%M%S')}.json"
        out_path = config.SHIFT_REPORTS_DIR / filename
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)

        log.info(f"[HSE] Shift report saved → {out_path}")
        return out_path

    def get_summary(self) -> dict:
        """Return a lightweight in-memory summary (for Streamlit sidebar)."""
        amber_c = sum(1 for e in self._events if e["alert_level"] == LEVEL_AMBER)
        red_c   = sum(1 for e in self._events if e["alert_level"] == LEVEL_RED)
        return {
            "shift_id":     self.shift_id,
            "worker_id":    self.worker_id,
            "elapsed_sec":  round(time.time() - self._shift_start, 1),
            "amber_alerts": amber_c,
            "red_alerts":   red_c,
            "total_events": len(self._events),
        }

    def recent_events(self, n: int = 10) -> list[dict]:
        """Return the last n logged events."""
        return list(reversed(self._events[-n:]))

    # ── Private helpers ───────────────────────────────────────────────────────
    def _ensure_csv_header(self) -> None:
        path = config.HSE_EVENTS_CSV
        if not path.exists():
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._CSV_HEADERS)
                writer.writeheader()

    def _write_csv_row(self, event: dict) -> None:
        with open(config.HSE_EVENTS_CSV, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._CSV_HEADERS,
                                    extrasaction="ignore")
            writer.writerow(event)
