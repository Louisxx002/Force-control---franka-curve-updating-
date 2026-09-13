"""Old whiteboard complete-frame driver, adapted to the demo sample contract."""
import time
from .serial_frames import KunweiFT
from .sensor import ForceSample, StaleSampleError, DEFAULT_PORT


class KunweiSensor(KunweiFT):
    def __init__(self):
        super().__init__(port=DEFAULT_PORT)

    def latest(self, max_age_s=.05):
        with self._lock:
            if not self._stamp or not 0 <= time.monotonic()-self._stamp <= max_age_s:
                raise StaleSampleError('No fresh complete serial frame')
            return ForceSample(tuple(self._force)+tuple(self._torque), self._frames,
                               self._frames*28, self._stamp)

    def wait_next(self, after_frame=0, timeout_s=.1, max_age_s=.05):
        deadline = time.monotonic()+timeout_s
        while time.monotonic() < deadline:
            s = self.latest(max_age_s)
            if s.frame > after_frame:
                return s
            time.sleep(.001)
        raise StaleSampleError('No new complete serial frame')
