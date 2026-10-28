"""Start/stop lifecycle for the web tracker loop."""

from __future__ import annotations

import logging
from threading import Event, Lock, Thread, Timer
from typing import Any, Callable, Optional


logger = logging.getLogger(__name__)


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
        self._cleaning = False
        self._cancel: Optional[Event] = None
        self.tracker = None
        self.last_motor_error = None
        self.thread: Optional[Thread] = None
        self._start_timer: Optional[Timer] = None

    def schedule_start(self, delay: float) -> None:
        """Schedule one start; manual Start/Stop consumes the pending request."""
        with self._lock:
            self._cancel_start_timer_unlocked()

            def autostart():
                result = self._start(scheduled=timer)
                if result['status'] == 'ok':
                    logger.info('Automatic tracking start completed')
                elif result['status'] != 'cancelled':
                    logger.warning('Automatic tracking start: %s', result)

            timer = Timer(delay, autostart)
            timer.daemon = True
            self._start_timer = timer
            timer.start()
        logger.info('Tracking and configured motors will start in %s seconds', delay)

    def start(self, connect_timeout: float = 15.0) -> dict:
        return self._start(connect_timeout)

    def _start(self, connect_timeout: float = 15.0, scheduled: Optional[Timer] = None) -> dict:
        with self._lock:
            # Cancel and admit starts under the same lock as Stop: an already
            # entered timer callback must never reopen hardware after Stop.
            if scheduled is not None and scheduled is not self._start_timer:
                return {'status': 'cancelled'}
            self._cancel_start_timer_unlocked()
            if self._stopping_unlocked():
                return {'status': 'stopping'}
            if self._starting or (self.thread is not None and self.thread.is_alive()):
                return {'status': 'already_running'}
            self._clear_finished_unlocked()
            if self.tracker is not None and self.tracker.remote_stop_pending:
                return self._pending_result()
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
                    if cancelled.is_set():
                        tracker.running = False
                        return
                    ready.set()
                    self._loop_fn(tracker)
                except Exception as exc:
                    error.append(str(exc) or exc.__class__.__name__)
                finally:
                    self._cleaning = True
                    if tracker is not None:
                        tracker.running = False
                        try:
                            tracker.cleanup()
                        except Exception:
                            pass
                    self._cleaning = False
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
            self._cancel_start_timer_unlocked()
            retry_pending = (
                not self._worker_alive_unlocked()
                and self.tracker is not None
                and (self.tracker.cleanup_pending or self.tracker.remote_stop_pending)
            )
            if self._cancel is not None:
                self._cancel.set()
            if self.tracker is not None:
                self.tracker.running = False
                stop_motion = getattr(self.tracker, 'request_stop', None)
                if stop_motion is not None:
                    stop_motion()
            stopped = self._join_unlocked()
            if stopped:
                if retry_pending:
                    self.tracker.stop_recording()
                self._clear_finished_unlocked()
                if self.tracker is not None:
                    return self._pending_result()
                return {'status': 'ok'}
            return {'status': 'stopping'}

    def _cancel_start_timer_unlocked(self) -> None:
        if self._start_timer is not None:
            self._start_timer.cancel()
            self._start_timer = None

    @property
    def stopping(self) -> bool:
        with self._lock:
            return self._stopping_unlocked()

    def _worker_alive_unlocked(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def _stopping_unlocked(self) -> bool:
        if self._worker_alive_unlocked():
            return self._cleaning or (self._cancel is not None and self._cancel.is_set())
        return bool(
            self.tracker is not None
            and self.tracker.cleanup_pending
            and (
                not self.tracker.remote_stop_pending
                or self.tracker.local_recording_path is not None
            )
        )

    def _pending_result(self) -> dict:
        if self.tracker.remote_stop_pending:
            return {
                'status': 'recording_stop_pending',
                'message': 'Camera recording stop is unconfirmed. Press Stop to retry.',
            }
        return {'status': 'stopping'}

    def _join_unlocked(self) -> bool:
        thread = self.thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=self._join_timeout)
        return thread is None or not thread.is_alive()

    def _clear_finished_unlocked(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        self.thread = None
        self._cancel = None
        if self.tracker is not None and (
            self.tracker.cleanup_pending or self.tracker.remote_stop_pending
        ):
            return
        if self.tracker is not None:
            error = getattr(getattr(self.tracker, 'motors', None), 'error', None)
            self.last_motor_error = error if isinstance(error, str) else None
        self.tracker = None
