import subprocess
import time
import os
import matplotlib.pyplot as plt

# Настройки согласно заданию
SIZES = [128, 256, 512, 1024]
MAX_ITER = 1000  # Фиксируем число итераций для честного сравнения скорости (чтобы не ждать сходимости часами)
EXEC_CPU = "./heat_cpu"
EXEC_GPU = "./heat_gpu"

def run_test(executable, size):
    """Запускает C++ программу и замеряет время выполнения."""
    if not os.path.exists(executable):
        print(f"[!] Ошибка: Исполняемый файл {executable} не найден.")
        return None

    cmd = [executable, "--size", str(size), "--iter", str(MAX_ITER)]
    print(f"[*] Запуск: {' '.join(cmd)}")
    
    start_time = time.perf_counter()
    try:
        # Запуск с перехватом вывода, чтобы он не засорял консоль
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
        print("\n[!] Тестирование не удалось. Проверьте наличие исполняемых файлов.")
        return

    # Расчет ускорения (Speedup)
    speedups = [cpu / gpu for cpu, gpu in zip(cpu_times, gpu_times)]

    print("\n=== Результаты ===")
    print(f"{'Размер':<10} | {'Время CPU (с)':<15} | {'Время GPU (с)':<15} | {'Ускорение GPU':<15}")
    print("-" * 62)
    for sz, t_c, t_g, sp in zip(valid_sizes, cpu_times, gpu_times, speedups):
        print(f"{sz:<10} | {t_c:<15.4f} | {t_g:<15.4f} | {sp:.2f}x")

    # Построение графиков
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Анализ производительности OpenACC: Multicore CPU vs GPU', fontsize=16)

    # График времени выполнения (в логарифмическом масштабе для наглядности)
    ax1.plot(valid_sizes, cpu_times, marker='o', linewidth=2, label='CPU (multicore)', color='blue')
    ax1.plot(valid_sizes, gpu_times, marker='s', linewidth=2, label='GPU', color='red')
    ax1.set_title('Время выполнения (чем меньше, тем лучше)')
    ax1.set_xlabel('Размер сетки (N)')
    ax1.set_ylabel('Время (секунды)')
    ax1.set_yscale('log') # Логарифмическая шкала, так как N^2 растет очень быстро
    ax1.set_xticks(valid_sizes)
    ax1.legend()
    ax1.grid(True, which="both", ls="--", alpha=0.5)

    # График ускорения
    ax2.plot(valid_sizes, speedups, marker='^', linewidth=2, color='green')
    ax2.axhline(y=1.0, color='gray', linestyle='--', label='Нет ускорения (1x)')
    ax2.set_title('Ускорение на GPU (Speedup = Time_CPU / Time_GPU)')
    ax2.set_xlabel('Размер сетки (N)')
    ax2.set_ylabel('Ускорение (разы)')
    ax2.set_xticks(valid_sizes)
    ax2.legend()
    ax2.grid(True, ls="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig("task6_benchmark.png", dpi=300)
    print("\n[*] Графики успешно сохранены в файл 'task6_benchmark.png'")
    plt.show()

if __name__ == "__main__":
    main()