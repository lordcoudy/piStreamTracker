"""Start/stop lifecycle for the web tracker loop."""

from __future__ import annotations

from threading import Event, Lock, Thread
from typing import Any, Callable, Optional


class TrackerLifecycle:
    """Serialize start/stop so only one tracker loop is alive."""

    def __init__(
        self,
        factory: Callable[[], Any],
        loop_fn: Callable[[Any], None],
        join_timeout: float = 3.0,
    ):
        self._factory = factory
        self._loop_fn = loop_fn
        self._join_timeout = join_timeout
        self._lock = Lock()
        self._starting = False
        self.tracker = None
        self.thread: Optional[Thread] = None

    def start(self, connect_timeout: float = 15.0) -> dict:
        with self._lock:
            if self._starting or (self.tracker is not None and getattr(self.tracker, 'running', False)):
                return {'status': 'already_running'}
            self._join_unlocked()
            tracker = self._factory()
            ready = Event()
            cancelled = Event()
            error: list[str] = []
            self._starting = True
            self.tracker = tracker

            def loop():
                try:
                    if not tracker.connect():
                        error.append('Failed to connect to stream')
                        ready.set()
                        return
                    if cancelled.is_set():
                        return
                    tracker.running = True
                    ready.set()
                    self._loop_fn(tracker)
                finally:
                    tracker.running = False
                    try:
                        tracker.cleanup()
                    except Exception:
                        pass
                    ready.set()

            self.thread = Thread(target=loop, daemon=True, name='tracker-loop')
            self.thread.start()

        if not ready.wait(timeout=connect_timeout):
            cancelled.set()
            tracker.running = False
            with self._lock:
                self._starting = False
                self._join_unlocked()
                self.tracker = None
                self.thread = None
            return {'status': 'error', 'message': 'Connect timed out'}

        with self._lock:
            self._starting = False
            if error:
                self._join_unlocked()
                self.tracker = None
                self.thread = None
                return {'status': 'error', 'message': error[0]}
        return {'status': 'ok'}

    def stop(self) -> dict:
        with self._lock:
            if self.tracker is not None:
                self.tracker.running = False
            self._join_unlocked()
            self.tracker = None
            self.thread = None
        return {'status': 'ok'}

    def _join_unlocked(self):
        thread = self.thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=self._join_timeout)
