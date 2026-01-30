# src/postprocess.py
from collections import deque

class EventPostProcessor:
    """
    Converts frame-level probabilities/scores into couch event detections.
    Implements:
      - smoothing
      - hysteresis / thresholding
      - min duration 
      - refractory period 
    """

    def __init__(
        self,
        start_thresh: float = 0.7,
        end_thresh: float = 0.4,
        min_event_sec: float = 0.10,
        merge_gap_sec: float = 0.30,
        refactory_sec: float = 0.75,
        smooth_len: int = 5,
    ):
        self.start_thresh = start_thresh
        self.end_thresh = end_thresh
        self.min_event_sec = min_event_sec
        self.merge_gap_sec = merge_gap_sec
        self.refractory_sec = refactory_sec
        self.smooth_len = smooth_len
        self.smooth = deque(maxlen=smooth_len)

        self.in_event = False
        self.event_start_t = None
        self.last_end_t = None
        self.last_fire_t = None

    def update(self, p: float, t: float) -> bool:
        # Smooth
        self.smooth.append(float(p))
        p_s = sum(self.smooth) / len(self.smooth)

        # Refractory: supress new events for a bit after a fire
        if self.last_fire_t is not None and (t- self.last_fire_t) < self.refractory_sec:
            return False
        
        fired = False

        if not self.in_event:
            # Start condition (with merge-gap logic consideration)
            gap_ok = ((self.last_end_t is None) or (t - self.last_end_t) >= self.merge_gap_sec)
            if gap_ok and p_s >= self.start_thresh:
                self.in_event = True
                self.event_start_t = t
        else:
            # End condition
            if p_s < self.end_thresh:
                self.in_event = False
                self.last_end_t = t

                # fire only if long enough
                if self.event_start_t is not None and (t - self.event_start_t) >= self.min_event_sec:
                    fired = True
                    self.last_fire_t = t

            self.event_start_t = None

        return fired