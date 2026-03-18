import threading
import time

from config import logger


class HeartBeat:
    """Context manager that logs a periodic message while a long task runs.

    Usage::

        with _heartbeat("TFLite float export", interval=30):
            model.export(format="tflite")   # silent for minutes
    """

    def __init__(self, label: str, interval: float = 30):
        self._label = label
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            elapsed = time.perf_counter() - self._t0
            logger.info("  ⏳ %s still running … %.0fs elapsed", self._label, elapsed)

    def __enter__(self):
        self._t0 = time.perf_counter()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        elapsed = time.perf_counter() - self._t0
        logger.info("  ✔ %s finished in %.1fs", self._label, elapsed)
        return False
