import argparse
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path


THREADS = [1, 2, 4, 7, 8, 16, 20, 40]
CHUNKS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
TIME_RE = re.compile(r"Time\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*sec")


def run_once(exe: Path, schedule: str, chunk: int, threads: int) -> float:
    cmd = [str(exe), schedule, str(chunk), str(threads)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )

    match = TIME_RE.search(proc.stdout)
    if not match:
        raise RuntimeError(
            f"Could not parse execution time from output of {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return float(match.group(1))


def run_average(exe: Path, schedule: str, chunk: int, threads: int, runs: int) -> float:
    times = [run_once(exe, schedule, chunk, threads) for _ in range(runs)]
    return statistics.fmean(times)


def benchmark_threads(exe: Path, schedule: str, chunk: int, runs: int) -> None:
    print(f"\n=== {exe.name} | schedule={schedule}, chunk={chunk} ===")
    print(f"Average over {runs} runs for each threads count")
    for th in THREADS:
        avg = run_average(exe, schedule, chunk, th, runs)
        print(f"Threads = {th:>2} | Avg time = {avg:.6f} sec")


def benchmark_chunks(exe: Path, schedule: str, threads: int, runs: int) -> None:
    print(f"\n=== {exe.name} | schedule={schedule}, threads={threads} ===")
    print(f"Chunk search, average over {runs} runs")
    best_chunk = None
    best_time = float("inf")
    for chunk in CHUNKS:
        avg = run_average(exe, schedule, chunk, threads, runs)
        print(f"Chunk = {chunk:>4} | Avg time = {avg:.6f} sec")
        if avg < best_time:
            best_time = avg
            best_chunk = chunk

    print(f"Best chunk = {best_chunk} | Best avg time = {best_time:.6f} sec")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sequential benchmark runner for lab2_3 variants"
    )
    parser.add_argument(
        "--build-dir",
        default="build",
        help="Directory containing lab2_3(.exe) and lab2_3_2(.exe)",
    )
    parser.add_argument(
        "--schedule",
        default="static",
        choices=["static", "dynamic", "guided"],
        help="OpenMP schedule type",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Runs per point for averaging",
    )
    parser.add_argument(
        "--chunk-for-threads",
        type=int,
        default=64,
        help="Chunk used in threads benchmark",
    )
    parser.add_argument(
        "--threads-for-chunk",
        type=int,
        default=8,
        help="Thread count used in chunk benchmark",
    )
    parser.add_argument(
        "--skip-threads",
        action="store_true",
        help="Skip benchmark over predefined thread counts",
    )
    parser.add_argument(
        "--skip-chunk",
        action="store_true",
        help="Skip benchmark over chunk candidates",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    build_dir = (root / args.build_dir).resolve()
    exe_suffix = ".exe" if os.name == "nt" else ""
    exe1 = build_dir / f"lab2_3{exe_suffix}"
    exe2 = build_dir / f"lab2_3_2{exe_suffix}"

    for exe in (exe1, exe2):
        if not exe.exists():
            print(f"Missing executable: {exe}", file=sys.stderr)
            return 1

    for exe in (exe1, exe2):
        if not args.skip_threads:
            benchmark_threads(exe, args.schedule, args.chunk_for_threads, args.runs)
        if not args.skip_chunk:
            benchmark_chunks(exe, args.schedule, args.threads_for_chunk, args.runs)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
