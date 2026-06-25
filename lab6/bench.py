import subprocess
import time
import os

# Настройки согласно заданию
SIZES = [128, 256, 512, 1024]
MAX_ITER = 1000  # Фиксируем число итераций для честного сравнения
EXEC_CPU = "./heat_cpu"
EXEC_GPU = "./heat_gpu"

def run_test(executable, size):
    """Запускает C++ программу и замеряет время выполнения."""
    if not os.path.exists(executable):
        print(f"[!] Ошибка: Файл {executable} не найден.")
        return None

    cmd = [executable, "--size", str(size), "--iter", str(MAX_ITER)]
    print(f"[*] Запуск: {' '.join(cmd)}")
    
    start_time = time.perf_counter()
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        end_time = time.perf_counter()
        
        elapsed = end_time - start_time
        print(f"    Время: {elapsed:.4f} сек")
        return elapsed
    except subprocess.CalledProcessError as e:
        print(f"[!] Ошибка выполнения: {e.stderr}")
        return None

def main():
    cpu_times = []
    gpu_times = []
    valid_sizes = []

    print("=== Начало тестирования CPU vs GPU ===")
    for size in SIZES:
        print(f"\n--- Сетка {size}x{size} ---")
        t_cpu = run_test(EXEC_CPU, size)
        t_gpu = run_test(EXEC_GPU, size)
        
        if t_cpu is not None and t_gpu is not None:
            cpu_times.append(t_cpu)
            gpu_times.append(t_gpu)
            valid_sizes.append(size)

    if not valid_sizes:
        print("\n[!] Тестирование не удалось.")
        return

    # Расчет ускорения (Speedup)
    speedups = [cpu / gpu for cpu, gpu in zip(cpu_times, gpu_times)]

    # Вывод результатов для копирования
    print("\n" + "="*70)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ ДЛЯ ПОСТРОЕНИЯ ГРАФИКОВ")
    print("="*70)
    print(f"{'Размер сетки':<15} | {'Время CPU (с)':<15} | {'Время GPU (с)':<15} | {'Ускорение':<15}")
    print("-" * 70)
    for sz, t_c, t_g, sp in zip(valid_sizes, cpu_times, gpu_times, speedups):
        print(f"{sz:<15} | {t_c:<15.6f} | {t_g:<15.6f} | {sp:.2f}x")
    print("="*70)
    
    # Формат сырых данных для удобной вставки в чат
    print("\nСырые данные (скопируйте их):")
    print(f"SIZES = {valid_sizes}")
    print(f"CPU_TIMES = {[round(t, 6) for t in cpu_times]}")
    print(f"GPU_TIMES = {[round(t, 6) for t in gpu_times]}")

if __name__ == "__main__":
    main()