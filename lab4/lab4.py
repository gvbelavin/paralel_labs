import argparse
import logging
import os
import queue
import signal
import sys
import threading
import time

import cv2
import numpy as np

os.makedirs("log", exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("log", "app.log"), encoding="utf-8"),
        logging.StreamHandler(
            open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False)
        ),
    ],
)
logger = logging.getLogger("main")


class Sensor:
    def get(self):
        raise NotImplementedError("должен быть реализован в подклассе")


class SensorX(Sensor):
    """Простой датчик-счётчик."""

    def __init__(self, delay: float):
        self._delay = delay
        self._data = 0

    def get(self) -> int:
        time.sleep(self._delay)
        self._data += 1
        return self._data


class SensorCam(Sensor):
    """Камера через OpenCV."""

    _log = logging.getLogger("SensorCam")

    def __init__(self, cam_name: str, resolution: str):
        try:
            w, h = resolution.lower().split("x")
            self._width, self._height = int(w), int(h)
        except Exception:
            self._log.warning(
                "Некорректное разрешение '%s', используем 640x480", resolution
            )
            self._width, self._height = 640, 480

        src = self._parse_camera_name(cam_name)

        self._cap = cv2.VideoCapture(src)
        if not self._cap.isOpened():
            self._log.error("Не удалось открыть камеру: %s (проанализировано как %s)", cam_name, src)
            raise RuntimeError(f"Не удалось открыть камеру: {cam_name}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._log.info(
            "Камера '%s' открыта (индекс %s), разрешение %dx%d",
            cam_name,
            src,
            self._width,
            self._height,
        )

    def _parse_camera_name(self, name: str):
        if name.startswith("/dev/video"):
            try:
                return int(name.replace("/dev/video", ""))
            except ValueError:
                pass
        try:
            return int(name)
        except ValueError:
            return name

    def get(self):
        ret, frame = self._cap.read()
        if not ret:
            self._log.error("Ошибка чтения кадра")
            return None
        return frame

    def __del__(self):
        if hasattr(self, "_cap") and self._cap.isOpened():
            self._cap.release()
            self._log.info("Камера освобождена")


class WindowImage:
    """Окно вывода OpenCV."""

    _log = logging.getLogger("WindowImage")
    _WIN = "Sensor Feed"

    def __init__(self, fps: float, stop_event: threading.Event):
        if fps <= 0:
            self._log.error("fps должен быть положительным, получено: %s", fps)
            raise ValueError(f"fps должен быть положительным, получено: {fps}")

        self._frame_ms = max(1, int(1000 / fps))
        self._stop = stop_event

        try:
            cv2.namedWindow(self._WIN, cv2.WINDOW_NORMAL)
            self._log.info("Окно создано, целевой fps=%.1f", fps)
        except Exception as exc:
            self._log.error("Не удалось создать окно: %s", exc)
            raise

    def show(self, img) -> bool:
        try:
            cv2.imshow(self._WIN, img)
        except Exception as exc:
            self._log.error("Ошибка отображения: %s", exc)
            return False

        key = cv2.waitKey(self._frame_ms) & 0xFF

        if key in (ord("q"), ord("Q")):
            return False

        if self._stop.is_set():
            return False

        if cv2.getWindowProperty(self._WIN, cv2.WND_PROP_VISIBLE) < 1:
            return False

        return True

    def __del__(self):
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            self._log.info("Окно закрыто")
        except Exception:
            pass


def _drain_and_put(q: queue.Queue, value) -> None:
    try:
        q.get_nowait()
    except queue.Empty:
        pass
    q.put(value)


def sensor_worker(sensor: Sensor, q: queue.Queue, stop_event: threading.Event) -> None:
    log = logging.getLogger("sensor_worker")

    while not stop_event.is_set():
        try:
            data = sensor.get()
        except Exception as exc:
            log.error("Ошибка датчика: %s", exc)
            stop_event.set()
            return

        if data is None:
            log.error("Датчик вернул None — завершаем работу")
            stop_event.set()
            return

        _drain_and_put(q, data)


def overlay_sensor_values(frame: np.ndarray, values: dict) -> np.ndarray:
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thick = 1
    line_h = 22

    lines = [f"{name}: {val}" for name, val in values.items()]
    box_w = max(cv2.getTextSize(line, font, scale, thick)[0][0] for line in lines) + 16
    box_h = len(lines) * line_h + 10
    x0 = w - box_w - 10
    y0 = h - box_h - 10

    roi = frame[y0:y0 + box_h, x0:x0 + box_w]
    white = np.ones_like(roi, dtype=np.uint8) * 255
    frame[y0:y0 + box_h, x0:x0 + box_w] = cv2.addWeighted(roi, 0.3, white, 0.7, 0)

    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x0 + 8, y0 + line_h * (i + 1)),
            font,
            scale,
            (0, 0, 0),
            thick,
            cv2.LINE_AA,
        )

    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Просмотр данных с датчиков в реальном времени",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--camera", default="0", help="Индекс или путь к камере (например /dev/video0)")
    parser.add_argument("--resolution", default="640x480", metavar="WxH", help="Разрешение камеры")
    parser.add_argument("--fps", type=float, default=30.0, help="Частота обновления окна")
        
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stop_event = threading.Event()

    def _sigint_handler(sig, frame):
        logger.info("Получен сигнал SIGINT. Завершение работы")
        stop_event.set()
    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        cam = SensorCam(args.camera, args.resolution)
    except RuntimeError:
        logger.critical("Не удалось открыть камеру '%s'. Выход.", args.camera)
        sys.exit(1)

    sensors = [SensorX(0.01), SensorX(0.1), SensorX(1.0)]

    cam_q: queue.Queue = queue.Queue(maxsize=1)
    sensor_qs = [queue.Queue(maxsize=1) for _ in sensors]

    threads: list[threading.Thread] = []
    for sensor, q in zip([cam] + sensors, [cam_q] + sensor_qs):
        t = threading.Thread(
            target=sensor_worker,
            args=(sensor, q, stop_event),
            daemon=True,
        )
        t.start()
        threads.append(t)

    try:
        win = WindowImage(args.fps, stop_event)
    except Exception:
        logger.critical("Не удалось создать окно. Выход.")
        stop_event.set()
        sys.exit(1)

    last_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    last_vals = {f"Sensor{i}": 0 for i in range(len(sensors))}

    logger.info("Запуск. Для выхода нажмите q")

    try:
        while not stop_event.is_set():
            try:
                last_frame = cam_q.get_nowait()
            except queue.Empty:
                pass

            for i, q in enumerate(sensor_qs):
                try:
                    last_vals[f"Sensor{i}"] = q.get_nowait()
                except queue.Empty:
                    pass

            display = overlay_sensor_values(last_frame.copy(), last_vals)

            if not win.show(display):
                logger.info("Получен сигнал завершения окна")
                break
    finally:
        stop_event.set()
        logger.info("Ожидаем завершения потоков...")

        for t in threads:
            t.join(timeout=2.0)
            if t.is_alive():
                logger.warning(
                    "Поток %s не завершился за отведённое время",
                    t.name,
                )

        threads.clear()
        sensors.clear()

        del win
        del cam

        logger.info("Все ресурсы освобождены")


if __name__ == "__main__":
    main()