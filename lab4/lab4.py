
import argparse
import logging
import os
import queue
import sys
import threading
import time

import cv2
import numpy as np

# ─── Logging setup ────────────────────────────────────────────────────────────
os.makedirs("log", exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("log", "app.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("main")


# ─── Base Sensor ──────────────────────────────────────────────────────────────
class Sensor:
    def get(self):
        raise NotImplementedError("Subclasses must implement method get()")


# ─── SensorX ──────────────────────────────────────────────────────────────────
class SensorX(Sensor):
    """Numeric sensor simulating data acquisition at a fixed frequency."""

    def __init__(self, delay: float):
        self._delay = delay
        self._data = 0

    def get(self) -> int:
        time.sleep(self._delay)
        self._data += 1
        return self._data


# ─── SensorCam (RAII) ─────────────────────────────────────────────────────────
class SensorCam(Sensor):
    """USB camera sensor via OpenCV. RAII: opened in __init__, released in __del__."""

    _log = logging.getLogger("SensorCam")

    def __init__(self, cam_name: str, resolution: str):
        try:
            w, h = resolution.lower().split("x")
            self._width, self._height = int(w), int(h)
        except Exception:
            self._log.warning(
                "Invalid resolution '%s', falling back to 640x480", resolution
            )
            self._width, self._height = 640, 480

        try:
            src = int(cam_name)
        except ValueError:
            src = cam_name

        self._cap = cv2.VideoCapture(src)
        if not self._cap.isOpened():
            self._log.error("Cannot open camera: %s", cam_name)
            raise RuntimeError(f"Cannot open camera: {cam_name}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._log.info("Camera '%s' opened at %dx%d", cam_name, self._width, self._height)

    def get(self):
        """Return the latest frame, or None if the camera failed/was disconnected."""
        ret, frame = self._cap.read()
        if not ret:
            self._log.error(
                "cap.read() failed — camera disconnected or read error"
            )
            return None
        return frame

    def __del__(self):
        if hasattr(self, "_cap") and self._cap.isOpened():
            self._cap.release()
            self._log.info("Camera released")


# ─── WindowImage (RAII) ───────────────────────────────────────────────────────
class WindowImage:
    """OpenCV display window. RAII: created in __init__, destroyed in __del__."""

    _log = logging.getLogger("WindowImage")
    _WIN = "Sensor Feed"
    _POLL_MS = 10  # waitKey chunk size (ms) — keeps stop_event responsive

    def __init__(self, fps: float):
        if fps <= 0:
            self._log.error("fps must be positive, got %s", fps)
            raise ValueError(f"fps must be positive, got {fps}")
        self._fps = fps
        self._frame_ms = max(1, int(1000 / fps))
        try:
            cv2.namedWindow(self._WIN, cv2.WINDOW_NORMAL)
            self._log.info("Window created — target fps=%.1f", fps)
        except Exception as exc:
            self._log.error("Cannot create window: %s", exc)
            raise

    def show(self, img, stop_event: threading.Event) -> bool:
        """Render *img* and wait up to one frame period, polling *stop_event*.
        Returns False when the user presses 'q' OR *stop_event* is set."""
        try:
            cv2.imshow(self._WIN, img)
        except Exception as exc:
            self._log.error("imshow failed: %s", exc)
            return False

        elapsed = 0
        while elapsed < self._frame_ms:
            key = cv2.waitKey(self._POLL_MS) & 0xFF
            elapsed += self._POLL_MS
            if key == ord("q"):
                return False
            if stop_event.is_set():
                return False
        return True

    def __del__(self):
        try:
            cv2.destroyWindow(self._WIN)
            self._log.info("Window destroyed")
        except Exception:
            pass


# ─── Worker threads ───────────────────────────────────────────────────────────

def _drain_and_put(q: queue.Queue, value) -> None:
    """Discard stale value (if any) and publish the fresh one."""
    try:
        q.get_nowait()
    except queue.Empty:
        pass
    q.put(value)


def sensor_worker(sensor: Sensor, q: queue.Queue, stop_event: threading.Event) -> None:
    """Continuously read *sensor* and publish latest value to *q*.
    Uses stop_event.wait() instead of time.sleep() so shutdown is instant."""
    log = logging.getLogger("sensor_worker")
    while not stop_event.is_set():
        try:
            data = sensor.get()
        except Exception as exc:
            log.error("Sensor read error: %s", exc)
            stop_event.set()
            return
        _drain_and_put(q, data)


def cam_worker(sensor: SensorCam, q: queue.Queue, stop_event: threading.Event) -> None:
    """Continuously grab frames; signals stop_event on camera failure."""
    log = logging.getLogger("cam_worker")
    while not stop_event.is_set():
        frame = sensor.get()
        if frame is None:
            log.error("Camera read returned None — triggering shutdown")
            stop_event.set()   # <-- notifies main thread AND all other workers
            return
        _drain_and_put(q, frame)


# ─── Overlay helper ───────────────────────────────────────────────────────────

def overlay_sensor_values(frame: np.ndarray, values: dict) -> np.ndarray:
    """Draw a semi-transparent panel with sensor readings in the bottom-right corner."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.55, 1
    line_h = 22

    lines = [f"{name}: {val}" for name, val in values.items()]
    box_w = max(cv2.getTextSize(ln, font, scale, thick)[0][0] for ln in lines) + 16
    box_h = len(lines) * line_h + 10
    x0 = w - box_w - 10
    y0 = h - box_h - 10

    roi = frame[y0 : y0 + box_h, x0 : x0 + box_w]
    white = np.ones_like(roi, dtype=np.uint8) * 255
    frame[y0 : y0 + box_h, x0 : x0 + box_w] = cv2.addWeighted(roi, 0.3, white, 0.7, 0)

    for i, line in enumerate(lines):
        cv2.putText(
            frame, line,
            (x0 + 8, y0 + line_h * (i + 1)),
            font, scale, (0, 0, 0), thick, cv2.LINE_AA,
        )
    return frame


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-sensor real-time viewer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--camera", default="0",
                   help="Camera index or device path (e.g. 0 or /dev/video0)")
    p.add_argument("--resolution", default="640x480", metavar="WxH",
                   help="Desired camera resolution, e.g. 1280x720")
    p.add_argument("--fps", type=float, default=30.0,
                   help="Display refresh rate in Hz")
    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    stop_event = threading.Event()

    try:
        cam = SensorCam(args.camera, args.resolution)
    except RuntimeError:
        logger.critical("Failed to open camera '%s'. Exiting.", args.camera)
        sys.exit(1)

    sensors = [
        SensorX(0.01),   # 100 Hz
        SensorX(0.1),    # 10 Hz
        SensorX(1.0),    # 1 Hz
    ]

    cam_q: queue.Queue = queue.Queue(maxsize=1)
    sensor_qs = [queue.Queue(maxsize=1) for _ in sensors]

    threads: list[threading.Thread] = []

    t_cam = threading.Thread(
        target=cam_worker, args=(cam, cam_q, stop_event), daemon=True
    )
    t_cam.start()
    threads.append(t_cam)

    for s, q_ in zip(sensors, sensor_qs):
        t = threading.Thread(
            target=sensor_worker, args=(s, q_, stop_event), daemon=True
        )
        t.start()
        threads.append(t)

    try:
        win = WindowImage(args.fps)
    except Exception:
        logger.critical("Failed to create display window. Exiting.")
        stop_event.set()
        sys.exit(1)

    last_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    last_vals: dict = {f"Sensor{i}": 0 for i in range(len(sensors))}

    logger.info("Starting main loop — press 'q' to quit")
    try:
        while not stop_event.is_set():
            try:
                last_frame = cam_q.get_nowait()
            except queue.Empty:
                pass

            for i, q_ in enumerate(sensor_qs):
                try:
                    last_vals[f"Sensor{i}"] = q_.get_nowait()
                except queue.Empty:
                    pass

            display = last_frame.copy()
            display = overlay_sensor_values(display, last_vals)

            # show() now polls stop_event every 10 ms — reacts quickly to USB yank
            if not win.show(display, stop_event):
                logger.info("Shutdown signal received ('q' pressed or camera lost)")
                break
    finally:
        stop_event.set()
        logger.info("Waiting for threads to finish...")
        for t in threads:
            t.join(timeout=2.0)
            if t.is_alive():
                logger.warning("Thread %s did not exit cleanly within timeout", t.name)
        del win   # cv2.destroyWindow
        del cam   # cap.release()
        logger.info("All resources released. Bye.")


if __name__ == "__main__":
    main()
