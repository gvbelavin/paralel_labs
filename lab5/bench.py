import subprocess
import re
import matplotlib.pyplot as plt

VIDEO_PATH = r"C:\Users\ironm\paralel_labs\lab5\output.mp4"
SCRIPT_PATH = "lab5.py"
MAX_THREADS = 16


def run_benchmark(workers):
    print(f"[*] Запуск теста для {workers} потоков...")
    cmd = [
        "python", SCRIPT_PATH,
        "--input", VIDEO_PATH,
        "--mode", "multi",
        "--workers", str(workers),
        "--output", "temp_benchmark.mp4"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    
    stdout = result.stdout if result.stdout else ""
    stderr = result.stderr if result.stderr else ""
    output = stdout + stderr

    match = re.search(r"Время:\s*([\d\.]+)s", output)
    if match:
        time_sec = float(match.group(1))
        print(f"    Время выполнения: {time_sec:.2f} сек")
        return time_sec
    else:
        print(f"[!] Ошибка парсинга времени для {workers} потоков. Вывод:")
        print(output)
        return None

def main():
    threads = list(range(1, MAX_THREADS + 1))
    times = []

    for t in threads:
        exec_time = run_benchmark(t)
        if exec_time is None:
            print("Бенчмарк прерван из-за ошибки.")
            return
        times.append(exec_time)

    t1 = times[0]
    speedups = [t1 / tn for tn in times]
    efficiencies = [s / n for s, n in zip(speedups, threads)]

    optimal_threads = 1
    max_speedup = 1
    for i, (s, e) in enumerate(zip(speedups, efficiencies)):
        if s > max_speedup and e >= 0.5:
            max_speedup = s
            optimal_threads = threads[i]

    print("\n" + "="*40)
    print(f"Оптимальное количество потоков: {optimal_threads}")
    print(f"Ускорение при этом: {max_speedup:.2f}x")
    print(f"Эффективность потоков: {efficiencies[optimal_threads-1]*100:.1f}%")
    print("="*40 + "\n")

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Анализ многопоточной обработки YOLOv8-pose на CPU', fontsize=16)

    axs[0].plot(threads, times, marker='o', color='blue', linewidth=2)
    axs[0].set_title('Время выполнения (Time)')
    axs[0].set_xlabel('Количество потоков')
    axs[0].set_ylabel('Время (секунды)')
    axs[0].grid(True, linestyle='--', alpha=0.7)

    axs[1].plot(threads, speedups, marker='s', color='green', linewidth=2)
    axs[1].plot(threads, threads, linestyle='--', color='gray', label='Идеальное ускорение')
    axs[1].axvline(x=optimal_threads, color='red', linestyle=':', label='Оптимально')
    axs[1].set_title('Ускорение (Speedup)')
    axs[1].set_xlabel('Количество потоков')
    axs[1].set_ylabel('S = T(1) / T(n)')
    axs[1].legend()
    axs[1].grid(True, linestyle='--', alpha=0.7)

    axs[2].plot(threads, efficiencies, marker='^', color='orange', linewidth=2)
    axs[2].axhline(y=0.5, color='red', linestyle='--', label='Порог 50%')
    axs[2].axvline(x=optimal_threads, color='red', linestyle=':')
    axs[2].set_title('Эффективность (Efficiency)')
    axs[2].set_xlabel('Количество потоков')
    axs[2].set_ylabel('E = S(n) / n')
    axs[2].set_ylim(0, 1.1)
    axs[2].legend()
    axs[2].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("benchmark_results.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()