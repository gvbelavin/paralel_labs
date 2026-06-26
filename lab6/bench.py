import subprocess
import os
import re
import sys

SIZES = [128, 256, 512, 1024]
TOL = "1e-6"
MAX_ITER = 1000000

EXECUTABLES = {
    "CPU-onecore": "./build-host/task6",
    "CPU-multicore": "./build-multicore/task6",
    "GPU": "./build-gpu/task6",
}

TIME_PATTERNS = [
    r"Время выполнения:\s*([0-9eE+\-.]+)",
    r"Your calculations took\s*([0-9eE+\-.]+)",
    r"Время:\s*([0-9eE+\-.]+)",
]
ITER_PATTERN = r"Количество итераций:\s*(\d+)"
ERROR_PATTERN = r"Достигнутая ошибка:\s*([0-9eE+\-.]+)"


def parse_value(patterns, text, cast_func, field_name):
    if isinstance(patterns, str):
        patterns = [patterns]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return cast_func(match.group(1))
    raise ValueError(f"Не удалось найти поле '{field_name}' в выводе программы")


def run_test(label, executable, size):
    if not os.path.exists(executable):
        print(f"Не найден исполняемый файл для режима {label}: {executable}")
        return None

    cmd = [executable, "--size", str(size), "--tol", TOL, "--iter", str(MAX_ITER)]
    print(f"Запуск ({label}): {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        stdout = result.stdout

        elapsed = parse_value(TIME_PATTERNS, stdout, float, "время")
        iterations = parse_value(ITER_PATTERN, stdout, int, "итерации")
        error = parse_value(ERROR_PATTERN, stdout, float, "ошибка")

        print(f"    Время: {elapsed:.6f} сек | Итерации: {iterations} | Ошибка: {error:.6e}")
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
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_mode_table(mode_name, rows):
    print_section_header(mode_name)
    print(f"{'Размер сетки':<12} | {'Время (с)':<14} | {'Точность':<14} | {'Итерации':<12}")
    print("-" * 80)
    for size in SIZES:
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
    for mode_name in EXECUTABLES:
        mode_rows = results[mode_name]
        times = [round(mode_rows[s]["time"], 6) if s in mode_rows else None for s in SIZES]
        errors = [mode_rows[s]["error"] if s in mode_rows else None for s in SIZES]
        iterations = [mode_rows[s]["iterations"] if s in mode_rows else None for s in SIZES]
        key = mode_name.replace('-', '_').replace(' ', '_').upper()
        print(f"{key}_TIMES = {times}")
        print(f"{key}_ERRORS = {errors}")
        print(f"{key}_ITERATIONS = {iterations}")


def main():
    results = {mode_name: {} for mode_name in EXECUTABLES}

    print_section_header("Запуск бенчмарка Task 6")
    print(f"Размеры сеток: {SIZES}")
    print(f"Точность: {TOL}")
    print(f"Максимум итераций: {MAX_ITER}")

    for size in SIZES:
        print_section_header(f"Сетка {size}x{size}")
        for mode_name, executable in EXECUTABLES.items():
            test_result = run_test(mode_name, executable, size)
            if test_result is not None:
                results[mode_name][size] = test_result

    for mode_name in EXECUTABLES:
        print_mode_table(mode_name, results[mode_name])

    print_raw_arrays(results)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\n Прервано пользователем")