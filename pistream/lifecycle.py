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
        self._cancel: Optional[Event] = None
        self.tracker = None
        self.thread: Optional[Thread] = None

    def start(self, connect_timeout: float = 15.0) -> dict:
        with self._lock:
            if self._starting or (self.thread is not None and self.thread.is_alive()):
                return {'status': 'already_running'}
            self._clear_finished_unlocked()
            ready = Event()
            cancelled = Event()
            error: list[str] = []
            holder: list[Any] = []
            self._starting = True
            self._cancel = cancelled
            self.tracker = None

            def loop():
                tracker = None
                try:
                    tracker = self._factory()
                    holder.append(tracker)
                    self.tracker = tracker
                    if cancelled.is_set():
                        return
                    if not tracker.connect():
                        error.append('Failed to connect to stream')
                        ready.set()
                        return
                    if cancelled.is_set():
                        return
                    tracker.running = True
                    ready.set()
                    self._loop_fn(tracker)
                except Exception as exc:
                    error.append(str(exc) or exc.__class__.__name__)
                finally:
                    if tracker is not None:
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
            if holder:
                holder[0].running = False
            with self._lock:
                self._starting = False
                if self._join_unlocked():
                    self._clear_finished_unlocked()
            return {'status': 'error', 'message': 'Connect timed out'}

        with self._lock:
            self._starting = False
            tracker = holder[0] if holder else None
            if error:
                self._join_unlocked()
                self._clear_finished_unlocked()
                return {'status': 'error', 'message': error[0]}
            if tracker is None or self.tracker is not tracker or cancelled.is_set():
                self._join_unlocked()
                self._clear_finished_unlocked()
                return {'status': 'error', 'message': 'Start cancelled'}
        return {'status': 'ok'}

    def stop(self) -> dict:
        with self._lock:
            if self._cancel is not None:
                self._cancel.set()
            if self.tracker is not None:
                self.tracker.running = False
            stopped = self._join_unlocked()
            if stopped:
                self._clear_finished_unlocked()
                return {'status': 'ok'}
            return {'status': 'stopping'}

    def _join_unlocked(self) -> bool:
        thread = self.thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=self._join_timeout)
        return thread is None or not thread.is_alive()

    def _clear_finished_unlocked(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        self.tracker = None
        self.thread = None
        self._cancel = None
