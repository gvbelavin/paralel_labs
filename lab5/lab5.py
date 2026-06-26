import argparse
import queue
import threading
import time
import cv2
import torch
from ultralytics import YOLO


class VideoSource:
    def __init__(self, src):
        if isinstance(src, str) and src.startswith('/dev/video'):
            src = int(src.replace('/dev/video', ''))
        elif str(src).isdigit():
            src = int(src)

        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError(f"Не удалось открыть {src}")

        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not self.fps or self.fps <= 1:
            self.fps = 30.0

    def get(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()


class VideoSink:
    def __init__(self, path, fps, w, h, rt):
        self.rt = rt
        self.window_name = "YOLO"
        self.out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        if self.rt:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def process(self, frame):
        self.out.write(frame)

        if self.rt:
            cv2.imshow(self.window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return False

            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                return False

        return True

    def release(self):
        if hasattr(self, 'out') and self.out is not None:
            self.out.release()


def producer(cam, q_in, stop, workers):
    idx = 0
    while not stop.is_set():
        frame = cam.get()
        if frame is None:
            break

        while not stop.is_set():
            try:
                q_in.put((idx, frame), timeout=0.05)
                idx += 1
                break
            except queue.Full:
                pass

    for _ in range(workers):
        try:
            q_in.put((-1, None), timeout=0.1)
        except queue.Full:
            pass


def worker(q_in, q_out, stop):
    model = YOLO("yolov8s-pose.pt")

    while not stop.is_set():
        try:
            idx, frame = q_in.get(timeout=0.05)

            if frame is None:
                q_out.put((-1, None))
                break

            res = model(frame, verbose=False)[0].plot()
            q_out.put((idx, res))

        except queue.Empty:
            continue


def main():
    torch.set_num_threads(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="0", help="Путь к видео или 0 для камеры")
    parser.add_argument("--mode", choices=["single", "multi"], default="multi")
    parser.add_argument("--output", default="output.mp4")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--realtime", action="store_true")
    args = parser.parse_args()

    workers_cnt = 1 if args.mode == "single" else args.workers
    stop_event = threading.Event()

    cam = VideoSource(args.input)
    sink = VideoSink(args.output, cam.fps, cam.w, cam.h, args.realtime)

    q_in = queue.Queue(maxsize=workers_cnt * 2)
    q_out = queue.Queue(maxsize=workers_cnt * 2)
    threads = []

    threads.append(
        threading.Thread(
            target=producer,
            args=(cam, q_in, stop_event, workers_cnt),
            daemon=True
        )
    )

    for _ in range(workers_cnt):
        threads.append(
            threading.Thread(
                target=worker,
                args=(q_in, q_out, stop_event),
                daemon=True
            )
        )

    for t in threads:
        t.start()

    start_t = time.time()
    buf = {}
    next_id = 0
    finished = 0
    running = True

    print(f"Запуск: {args.input}, режим {args.mode} ({workers_cnt} потоков)")

    try:
        while running and finished < workers_cnt:
            try:
                idx, frame = q_out.get(timeout=0.05)

                if frame is None:
                    finished += 1
                    continue

                buf[idx] = frame

                while next_id in buf:
                    ok = sink.process(buf.pop(next_id))
                    if not ok:
                        stop_event.set()
                        running = False
                        break
                    next_id += 1

            except queue.Empty:
                if stop_event.is_set():
                    break

    except KeyboardInterrupt:
        print("\nОстановка по Ctrl+C...")
        stop_event.set()

    finally:
        stop_event.set()

        for t in threads:
            t.join(timeout=1.0)

        sink.release()
        cam.release()

        if args.realtime:
            try:
                cv2.destroyWindow("YOLO")
            except cv2.error:
                pass

            for _ in range(8):
                cv2.waitKey(1)

        elapsed = time.time() - start_t
        print(f"Обработано кадров: {next_id}")
        print(f"Время: {elapsed:.2f}s, FPS: {next_id / elapsed if elapsed > 0 else 0:.1f}")


if __name__ == "__main__":
    main()