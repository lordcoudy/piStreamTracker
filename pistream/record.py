"""ffmpeg / OpenCV recording thread."""

import logging
import shutil
import subprocess
import time
from threading import Event, Lock, Thread
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _probe_encoder() -> str:
    """Return the best available ffmpeg H.264 encoder.

    Preference order:
      1. h264_v4l2m2m  — Pi hardware encoder (near-zero CPU)
      2. libx264       — CPU-based but vastly more efficient than MJPG

    Each candidate is validated by encoding a single tiny test frame so we
    detect missing hardware devices (e.g. no V4L2 encode node) instead of
    only checking whether ffmpeg was compiled with the encoder.

    Returns the encoder name or '' if ffmpeg is not available.
    """
    if not shutil.which('ffmpeg'):
        return ''

    for enc in ('h264_v4l2m2m', 'libx264'):
        try:
            # Encode one 64×64 black frame to /dev/null — proves the encoder
            # can actually initialise on this hardware.
            cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                '-s', '64x64', '-r', '1',
                '-i', 'pipe:0',
                '-frames:v', '1',
                '-c:v', enc,
                '-pix_fmt', 'yuv420p',
                '-f', 'null', '-',
            ]
            r = subprocess.run(
                cmd,
                input=b'\x00' * (64 * 64 * 3),
                capture_output=True, timeout=10,
            )
            if r.returncode == 0:
                logger.info(f"Verified ffmpeg encoder: {enc}")
                return enc
            logger.debug(
                f"Encoder {enc} failed probe: "
                + r.stderr.decode(errors='replace').strip()
            )
        except Exception as e:
            logger.debug(f"Encoder {enc} probe error: {e}")
            continue
    return ''


class _RecordingThread:
    """Writes frames at a fixed framerate via an ffmpeg subprocess pipe.

    Uses hardware-accelerated H.264 when available (h264_v4l2m2m on Pi 5),
    falling back to software libx264 ultrafast, then to the legacy OpenCV
    MJPG writer as a last resort.

    The thread ticks at exactly ``1/fps`` intervals.  If the main loop is
    slower than the target FPS the last available frame is repeated so the
    output file always has a constant frame rate.
    """

    def __init__(self, path: str, width: int, height: int, fps: float,
                 encoder: str = 'auto'):
        self._interval = 1.0 / max(fps, 1.0)
        self._frame: Optional[np.ndarray] = None
        self._lock = Lock()
        self._stop = Event()
        self._thread: Optional[Thread] = None
        self._proc: Optional[subprocess.Popen] = None
        self._cv_writer: Optional[cv2.VideoWriter] = None
        self._width = width
        self._height = height
        self._stderr_lines: list[str] = []
        self._stderr_thread: Optional[Thread] = None
        self.output_path = path

        self._init_writer(path, width, height, fps, encoder)

    # ---- writer init with fallback chain -----------------------------

    def _init_writer(self, path: str, w: int, h: int, fps: float,
                     encoder: str):
        """Try ffmpeg pipe first, fall back to OpenCV MJPG."""
        if encoder == 'auto':
            encoder = _probe_encoder()
        elif encoder == 'mjpg':
            encoder = ''

        if encoder and shutil.which('ffmpeg'):
            try:
                self._start_ffmpeg(path, w, h, fps, encoder)
                return
            except Exception as e:
                logger.warning(f"ffmpeg ({encoder}) failed: {e} — falling back")

        # Fallback: OpenCV MJPG
        logger.info("Using fallback OpenCV MJPG recorder")
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        fallback_path = path.rsplit('.', 1)[0] + '.avi'
        self._cv_writer = cv2.VideoWriter(fallback_path, fourcc, fps, (w, h))
        if not self._cv_writer.isOpened():
            self._cv_writer.release()
            self._cv_writer = None
            raise RuntimeError(f"Could not open recording output: {fallback_path}")
        self.output_path = fallback_path

    def _start_ffmpeg(self, path: str, w: int, h: int, fps: float,
                      encoder: str):
        """Launch ffmpeg as a subprocess accepting raw BGR frames on stdin."""
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning', '-nostats',
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{w}x{h}',
            '-r', str(fps),
            '-i', 'pipe:0',
            '-c:v', encoder,
        ]
        if encoder == 'libx264':
            cmd += ['-preset', 'ultrafast', '-crf', '23']
        elif encoder == 'h264_v4l2m2m':
            cmd += ['-b:v', '4M']

        cmd += [
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            path,
        ]

        logger.info(f"Recording via ffmpeg [{encoder}]: {path}")
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        # Drain stderr in a background thread to prevent pipe-buffer
        # deadlock — ffmpeg writes progress/stats continuously and if
        # the OS pipe buffer (~64 KB on Linux) fills up, ffmpeg blocks
        # and the stdin write from Python raises BrokenPipeError.
        self._stderr_thread = Thread(
            target=self._drain_stderr, daemon=True,
            name="ffmpeg-stderr",
        )
        self._stderr_thread.start()

    # ---- public API --------------------------------------------------

    def update_frame(self, frame: np.ndarray):
        """Feed the latest frame (non-blocking)."""
        with self._lock:
            self._frame = frame

    def start(self):
        """Start the recording thread."""
        if not self.ready:
            raise RuntimeError("Recording writer is not available")
        self._stop.clear()
        self._thread = Thread(target=self._run, daemon=True, name="RecordingThread")
        self._thread.start()

    def stop(self):
        """Stop the recording thread and flush/close the writer."""
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=max(self._interval * 4, 2.0))

        if self._proc and self._proc.stdin:
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=10)
            except Exception as e:
                logger.debug(f"ffmpeg close: {e}")
                self._proc.kill()
                try:
                    self._proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    logger.warning("ffmpeg did not exit after being killed")

        if self._stderr_thread and self._stderr_thread.is_alive():
            self._stderr_thread.join(timeout=2.0)

        # Log ffmpeg exit status
        if self._proc and self._proc.returncode and self._proc.returncode != 0:
            tail = self._stderr_lines[-20:]
            logger.warning(
                f"ffmpeg exited with code {self._proc.returncode}:\n"
                + "\n".join(tail)
            )

        if self._cv_writer:
            self._cv_writer.release()

    # ---- internal ----------------------------------------------------

    def _drain_stderr(self):
        """Read ffmpeg stderr continuously to prevent pipe-buffer deadlock."""
        try:
            for raw_line in self._proc.stderr:
                line = raw_line.decode(errors='replace').rstrip()
                if line:
                    self._stderr_lines.append(line)
                    del self._stderr_lines[:-100]
        except (ValueError, OSError):
            pass  # stderr closed

    def _write(self, frame: np.ndarray):
        """Write one frame to whichever backend is active."""
        if self._proc and self._proc.stdin:
            # Detect ffmpeg death early before attempting write
            if self._proc.poll() is not None:
                tail = self._stderr_lines[-10:]
                logger.warning(
                    f"ffmpeg died (exit {self._proc.returncode}) — stopping recording\n"
                    + "\n".join(tail)
                )
                self._stop.set()
                return
            try:
                self._proc.stdin.write(frame.tobytes())
            except (BrokenPipeError, OSError, ValueError):
                tail = self._stderr_lines[-10:]
                logger.warning(
                    "ffmpeg pipe broken — stopping recording\n"
                    + "\n".join(tail)
                )
                self._stop.set()
        elif self._cv_writer:
            self._cv_writer.write(frame)

    def _run(self):
        next_tick = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            sleep_for = next_tick - now
            if sleep_for > 0 and self._stop.wait(sleep_for):
                break
            if self._stop.is_set():
                break

            # Skip missed ticks instead of burst-writing duplicate frames
            # after a stall in ffmpeg or the scheduler.
            now = time.monotonic()
            next_tick = max(next_tick + self._interval, now + self._interval)

            with self._lock:
                frame = self._frame
            if frame is not None:
                self._write(frame)

    @property
    def ready(self) -> bool:
        if self._proc is not None:
            return self._proc.poll() is None and self._proc.stdin is not None
        return self._cv_writer is not None and self._cv_writer.isOpened()
