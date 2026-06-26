import subprocess
import os
import re
import sys

SIZES = [128, 256, 512, 1024]
MODE_SIZES = {
    "CPU-onecore": [128, 256, 512],
    "CPU-multicore": [128, 256, 512, 1024],
    "GPU": [128, 256, 512, 1024],
}
EPS = "1e-6"
MAX_ITER = 1000000
CHECK_INTERVALS = [1, 100]
BUILD_DIR = "./build"

EXECUTABLES = {
    "CPU-onecore": os.path.join(BUILD_DIR, "heat_host"),
    "CPU-multicore": os.path.join(BUILD_DIR, "heat_multicore"),
    "GPU": os.path.join(BUILD_DIR, "heat_gpu"),
}

TIME_PATTERNS = [
    r"Time:\s*([0-9eE+\-.]+)",
    r"Время(?: выполнения)?:\s*([0-9eE+\-.]+)",
    r"Your calculations took\s*([0-9eE+\-.]+)",
]
ITER_PATTERN = r"Iterations:\s*(\d+)"
ERROR_PATTERN = r"Error:\s*([0-9eE+\-.]+)"


def parse_value(patterns, text, cast_func, field_name):
    if isinstance(patterns, str):
        patterns = [patterns]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return cast_func(match.group(1))
    raise ValueError(f"Не удалось найти поле '{field_name}' в выводе программы")


def build_command(executable, size, check_interval):
    return [
        executable,
        "--size", str(size),
        "--eps", EPS,
        "--iter", str(MAX_ITER),
        "--check", str(check_interval),
    ]


def run_test(label, executable, size, check_interval):
    if not os.path.exists(executable):
        print(f"Не найден исполняемый файл для режима {label}: {executable}")
        return None

    cmd = build_command(executable, size, check_interval)
    print(f"Запуск ({label}, check={check_interval}): {' '.join(cmd)}")

    env = os.environ.copy()
    if label == "CPU-multicore":
        env.setdefault("OMP_NUM_THREADS", "20")
        env.setdefault("ACC_NUM_CORES", "20")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
        stdout = result.stdout

        elapsed = parse_value(TIME_PATTERNS, stdout, float, "время")
        iterations = parse_value(ITER_PATTERN, stdout, int, "итерации")
        error = parse_value(ERROR_PATTERN, stdout, float, "ошибка")

        print(f" Время: {elapsed:.6f} сек | Итерации: {iterations} | Ошибка: {error:.6e}")
        return {
            "time": elapsed,
            "iterations": iterations,
            "error": error,
            "stdout": stdout,
        }
    except subprocess.CalledProcessError as e:
        print(f" Ошибка выполнения {label}:\n{e.stderr}")
        return None
    except ValueError as e:
        print(f"Ошибка парсинга вывода {label}: {e}")
        print("----- STDOUT BEGIN -----")
        print(result.stdout)
        print("----- STDOUT END -----")
        return None


def print_section_header(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def print_mode_table(mode_name, rows, check_interval):
    print_section_header(f"{mode_name} | check={check_interval}")
    print(f"{'Размер сетки':<12} | {'Время (с)':<14} | {'Ошибка':<14} | {'Итерации':<12}")
    print("-" * 90)
    allowed_sizes = set(MODE_SIZES[mode_name])
    for size in SIZES:
        if size not in allowed_sizes:
            print(f"{size}x{size:<7} | {'SKIPPED':<14} | {'SKIPPED':<14} | {'SKIPPED':<12}")
            continue
        row = rows.get(size)
        if row is None:
            print(f"{size}x{size:<7} | {'ERROR':<14} | {'ERROR':<14} | {'ERROR':<12}")
        else:
            print(
                f"{size}x{size:<7} | "
                f"{row['time']:<14.6f} | "
                f"{row['error']:<14.6e} | "
                f"{row['iterations']:<12}"
            )


def print_raw_arrays(results):
    print(f"SIZES = {SIZES}")
    print(f"MODE_SIZES = {MODE_SIZES}")
    for check_interval in CHECK_INTERVALS:
        print(f"CHECK_INTERVAL = {check_interval}")
        for mode_name in EXECUTABLES:
            mode_rows = results[check_interval][mode_name]
            times = [round(mode_rows[s]["time"], 6) if s in mode_rows else None for s in SIZES]
            errors = [mode_rows[s]["error"] if s in mode_rows else None for s in SIZES]
            iterations = [mode_rows[s]["iterations"] if s in mode_rows else None for s in SIZES]
            key = mode_name.replace('-', '_').replace(' ', '_').upper()
            print(f"{key}_TIMES = {times}")
            print(f"{key}_ERRORS = {errors}")
            print(f"{key}_ITERATIONS = {iterations}")


def main():
    results = {
        check_interval: {mode_name: {} for mode_name in EXECUTABLES}
        for check_interval in CHECK_INTERVALS
    }

    print_section_header("Запуск бенчмарка heat.cpp")
    print(f"Размеры сеток: {SIZES}")
    print(f"Режимы и размеры: {MODE_SIZES}")
    print(f"Точность: {EPS}")
    print(f"Максимум итераций: {MAX_ITER}")
    print(f"check_interval: {CHECK_INTERVALS}")
    print(f"Исполняемые файлы: {EXECUTABLES}")

    for check_interval in CHECK_INTERVALS:
        for mode_name, executable in EXECUTABLES.items():
            for size in MODE_SIZES[mode_name]:
                print_section_header(f"Сетка {size}x{size} | {mode_name} | check={check_interval}")
                test_result = run_test(mode_name, executable, size, check_interval)
                if test_result is not None:
                    results[check_interval][mode_name][size] = test_result

    for check_interval in CHECK_INTERVALS:
        for mode_name in EXECUTABLES:
            print_mode_table(mode_name, results[check_interval][mode_name], check_interval)

    print_raw_arrays(results)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nПрервано пользователем")