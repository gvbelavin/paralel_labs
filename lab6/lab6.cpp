#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

double lerp(double v0, double v1, double t) {
    return (1.0 - t) * v0 + t * v1;
}

void init_grid(std::vector<double>& grid, int N) {
    const double top_left = 10.0;
    const double top_right = 20.0;
    const double bottom_right = 30.0;
    const double bottom_left = 20.0;

    std::fill(grid.begin(), grid.end(), 0.0);

    for (int i = 0; i < N; ++i) {
        double t = static_cast<double>(i) / (N - 1);

        grid[i] = lerp(top_left, top_right, t);
        grid[(N - 1) * N + i] = lerp(bottom_left, bottom_right, t);
        grid[i * N] = lerp(top_left, bottom_left, t);
        grid[i * N + (N - 1)] = lerp(top_right, bottom_right, t);
    }
}

void print_grid(const std::vector<double>& grid, int N) {
    if (N > 13) return;

    std::cout << "\nGrid state (" << N << "x" << N << "):\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << std::fixed << std::setprecision(2) << std::setw(6) << grid[i * N + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main(int argc, char* argv[]) {
    int N = 128;
    double tol = 1e-6;
    int max_iter = 1000000;

    try {
        po::options_description desc("Разрешенные опции");
        desc.add_options()
            ("help,h", "Вывести справочное сообщение")
            ("size,n", po::value<int>(&N)->default_value(128), "Размер сетки (N x N)")
            ("tol,t", po::value<double>(&tol)->default_value(1e-6), "Требуемая точность")
            ("iter,i", po::value<int>(&max_iter)->default_value(1000000), "Максимальное число итераций");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);

        if (vm.count("help")) {
            std::cout << desc << "\n";
            return 0;
        }
    } catch (std::exception& e) {
        std::cerr << "Ошибка парсинга аргументов: " << e.what() << "\n";
        return 1;
    }

    std::vector<double> grid(N * N);
    std::vector<double> new_grid(N * N);

    init_grid(grid, N);
    init_grid(new_grid, N);

    double* d_grid = grid.data();
    double* d_new_grid = new_grid.data();

    int iter = 0;
    double error = 1.0;

    // Начинаем замер времени
    auto start = std::chrono::steady_clock::now();

    #pragma acc data copy(d_grid[0:N*N], d_new_grid[0:N*N])
    {
        while (error > tol && iter < max_iter) {
            error = 0.0;

            if (iter % 2 == 0) {
                // Итерация 0, 2, 4... Читаем из d_grid, пишем в d_new_grid
                #pragma acc parallel loop collapse(2) present(d_grid[0:N*N], d_new_grid[0:N*N])
                for (int i = 1; i < N - 1; ++i) {
                    for (int j = 1; j < N - 1; ++j) {
                        d_new_grid[i * N + j] = 0.25 * (
                            d_grid[(i - 1) * N + j] +
                            d_grid[(i + 1) * N + j] +
                            d_grid[i * N + (j - 1)] +
                            d_grid[i * N + (j + 1)]
                        );
                    }
                }
                
                // Отдельный цикл для ошибки (PGI компилятор так лучше векторизует)
                #pragma acc parallel loop collapse(2) reduction(max:error) present(d_grid[0:N*N], d_new_grid[0:N*N])
                for (int i = 1; i < N - 1; ++i) {
                    for (int j = 1; j < N - 1; ++j) {
                        double diff = d_new_grid[i * N + j] - d_grid[i * N + j];
                        diff = (diff > 0.0) ? diff : -diff; // вместо std::abs
                        if (diff > error) error = diff;
                    }
                }
            } else {
                // Итерация 1, 3, 5... Читаем из d_new_grid, пишем в d_grid
                #pragma acc parallel loop collapse(2) present(d_grid[0:N*N], d_new_grid[0:N*N])
                for (int i = 1; i < N - 1; ++i) {
                    for (int j = 1; j < N - 1; ++j) {
                        d_grid[i * N + j] = 0.25 * (
                            d_new_grid[(i - 1) * N + j] +
                            d_new_grid[(i + 1) * N + j] +
                            d_new_grid[i * N + (j - 1)] +
                            d_new_grid[i * N + (j + 1)]
                        );
                    }
                }
                
                // Отдельный цикл для ошибки
                #pragma acc parallel loop collapse(2) reduction(max:error) present(d_grid[0:N*N], d_new_grid[0:N*N])
                for (int i = 1; i < N - 1; ++i) {
                    for (int j = 1; j < N - 1; ++j) {
                        double diff = d_grid[i * N + j] - d_new_grid[i * N + j];
                        diff = (diff > 0.0) ? diff : -diff; // вместо std::abs
                        if (diff > error) error = diff;
                    }
                }
            }

            iter++;
        }
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;

    // В конце определяем, где лежат актуальные данные
    const std::vector<double>& result_grid = (iter % 2 == 0) ? grid : new_grid;

    std::cout << "Количество итераций: " << iter << "\n";
    std::cout << "Достигнутая ошибка: " << std::scientific << error << "\n";
    std::cout << "Время выполнения: " << std::fixed << std::setprecision(6) << elapsed_seconds.count() << " сек\n";

    if (N == 10 || N == 13) {
        print_grid(result_grid, N);
    }

    return 0;
}