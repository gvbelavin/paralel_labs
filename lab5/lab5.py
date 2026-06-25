import argparse
import queue
import threading
import time
import cv2
import torch
from ultralytics import YOLO

torch.set_num_threads(1)

class VideoSource:
    def __init__(self, src):
        if isinstance(src, str) and src.startswith('/dev/video'):
            src = int(src.replace('/dev/video', ''))
        elif str(src).isdigit():
            src = int(src)
            
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Не удалось открыть {src}")
            
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

    def get(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def __del__(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

class VideoSink:
    def __init__(self, path, fps, w, h, rt):
        self.rt = rt
        self.out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        if self.rt:
            cv2.namedWindow("YOLO", cv2.WINDOW_NORMAL)

    def process(self, frame):
        self.out.write(frame)
        if self.rt:
            cv2.imshow("YOLO", frame)
            if cv2.waitKey(1) & 0xFF in (ord('q')):
                return False
        return True

    def __del__(self):
        if hasattr(self, 'out'):
            self.out.release()
        if getattr(self, 'rt', False):
            cv2.destroyAllWindows()
            cv2.waitKey(1)


def producer(cam, q_in, stop, workers):
    idx = 0
    while not stop.is_set():
        frame = cam.get()
        if frame is None:
            break
            
        while not stop.is_set():
            try:
                q_in.put((idx, frame), timeout=0.1)
                idx += 1
                break
            except queue.Full:
                pass
                
    for _ in range(workers):
        q_in.put((-1, None))


def worker(q_in, q_out, stop):
    model = YOLO("yolov8s-pose.pt")
    while not stop.is_set():
        try:
            idx, frame = q_in.get(timeout=0.1)
            if frame is None:
                q_out.put((-1, None))
                break
                
            res = model(frame, verbose=False)[0].plot()
            q_out.put((idx, res))
        except queue.Empty:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="0", help="Путь к видео или 0 для камеры")
    parser.add_argument("--mode", choices=["single", "multi"], default="multi")
    parser.add_argument("--output", default="output.mp4")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--realtime", action="store_true")
    args = parser.parse_args()

    workers_cnt = 1 if args.mode == "single" else args.workers
    stop_event = threading.Event()
    
    cam = VideoSource(args.input)
    sink = VideoSink(args.output, cam.fps, cam.w, cam.h, args.realtime)

    q_in = queue.Queue(maxsize=workers_cnt * 2)
    q_out = queue.Queue(maxsize=workers_cnt * 2)
    threads = []

    threads.append(threading.Thread(target=producer, args=(cam, q_in, stop_event, workers_cnt)))
    for _ in range(workers_cnt):
        threads.append(threading.Thread(target=worker, args=(q_in, q_out, stop_event)))

    for t in threads:
        t.start()

    start_t = time.time()
    buf = {}
    next_id = 0
    finished = 0

    print(f"Запуск: {args.input}, режим {args.mode} ({workers_cnt} потоков)")

    try:
        while finished < workers_cnt:
            try:
                idx, frame = q_out.get(timeout=0.1)
                if frame is None:
                    finished += 1
                    continue
                    
                buf[idx] = frame
                
                # Выдача кадров по порядку
                while next_id in buf:
                    if not sink.process(buf.pop(next_id)):
                        stop_event.set()
                    next_id += 1
                    
            except queue.Empty:
                if stop_event.is_set() and q_in.empty():
                    break
                    
    except KeyboardInterrupt:
        print("\nОстановка по Ctrl+C...")
    finally:
        stop_event.set()
        for t in threads:
            t.join(timeout=1.0)
            
        elapsed = time.time() - start_t
        print(f"Обработано кадров: {next_id}")
        print(f"Время: {elapsed:.2f}s, FPS: {next_id/elapsed if elapsed > 0 else 0:.1f}")

        threads.clear()
        buf.clear()
        del sink
        del cam

if __name__ == "__main__":
    main()