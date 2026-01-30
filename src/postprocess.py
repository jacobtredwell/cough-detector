# src/postprocess.py
from __future__ import annotations

from collections import deque


class EventPostProcessor:
    """
    Converts frame-level scores into cough events.

    The live loop produces a score p in [0, 1] per hop (10 ms by default).
    This class turns those frame scores into discrete events using:
      - smoothing (moving average)
      - hysteresis (start_thresh, end_thresh)
      - minimum event duration
      - merge gap (treat very-close bursts as one event)
      - refractory period (suppress repeated triggers)

    By default, `update(...)` returns True on EVENT START.
    """

    def __init__(
        self,
        start_thresh: float = 0.7,
        end_thresh: float = 0.4,
        min_event_sec: float = 0.10,
        merge_gap_sec: float = 0.30,
        refractory_sec: float = 0.75,
        smooth_len: int = 5,
        fire_on: str = "start",  # "start" or "end"
    ):
        if not (0.0 <= end_thresh <= 1.0 and 0.0 <= start_thresh <= 1.0):
            raise ValueError("Thresholds must be in [0, 1].")
        if end_thresh > start_thresh:
            raise ValueError("end_thresh should be <= start_thresh for hysteresis.")
        if fire_on not in ("start", "end"):
            raise ValueError("fire_on must be 'start' or 'end'.")

        self.start_thresh = float(start_thresh)
        self.end_thresh = float(end_thresh)
        self.min_event_sec = float(min_event_sec)
        self.merge_gap_sec = float(merge_gap_sec)
        self.refractory_sec = float(refractory_sec)
        self.fire_on = fire_on

        self.smooth = deque(maxlen=int(smooth_len))

        self.in_event = False
        self.event_start_t: float | None = None
        self.last_end_t: float | None = None
        self.last_fire_t: float | None = None

    def _smooth_p(self, p: float) -> float:
        self.smooth.append(float(p))
        return sum(self.smooth) / len(self.smooth)

    def update(self, p: float, t: float) -> bool:
        """
        Update with current frame score p at time t (seconds, monotonic).

        Returns True when an event is detected (start or end depending on fire_on).
        """
        p_s = self._smooth_p(p)

        # Refractory: suppress any new fires for a bit after we fired.
        if self.last_fire_t is not None and (t - self.last_fire_t) < self.refractory_sec:
            # Still allow state to end naturally so we do not get stuck in_event forever.
            if self.in_event and p_s < self.end_thresh:
                self.in_event = False
                self.last_end_t = t
                self.event_start_t = None
            return False

        fired = False

        if not self.in_event:
            gap_ok = (self.last_end_t is None) or ((t - self.last_end_t) >= self.merge_gap_sec)
            if gap_ok and p_s >= self.start_thresh:
                self.in_event = True
                self.event_start_t = t
                if self.fire_on == "start":
                    fired = True
                    self.last_fire_t = t
        else:
            # End condition
            if p_s < self.end_thresh:
                start_t = self.event_start_t if self.event_start_t is not None else t
                dur = t - start_t

                self.in_event = False
                self.last_end_t = t
                self.event_start_t = None

                if self.fire_on == "end" and dur >= self.min_event_sec:
                    fired = True
                    self.last_fire_t = t

        return fired
