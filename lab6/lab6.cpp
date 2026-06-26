#include <boost/program_options.hpp>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

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
        double t = static_cast<double>(i) / static_cast<double>(N - 1);

        grid[i] = lerp(top_left, top_right, t);
        grid[(N - 1) * N + i] = lerp(bottom_left, bottom_right, t);
        grid[i * N] = lerp(top_left, bottom_left, t);
        grid[i * N + (N - 1)] = lerp(top_right, bottom_right, t);
    }
}

void print_grid(const std::vector<double>& grid, int N) {
    std::cout << "\nGrid state (" << N << "x" << N << "):\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << std::fixed << std::setprecision(6)
                      << std::setw(12) << grid[i * N + j] << " ";
        }
        std::cout << '\n';
    }
    std::cout << '\n';
}

void save_grid_to_file(const std::vector<double>& grid, int N, const std::string& filename) {
    std::ofstream out(filename);
    if (!out) {
        throw std::runtime_error("Не удалось открыть файл: " + filename);
    }

    out << std::fixed << std::setprecision(10);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            out << grid[i * N + j];
            if (j + 1 != N) out << ' ';
        }
        out << '\n';
    }
}

int main(int argc, char* argv[]) {
    int N = 128;
    double tol = 1e-6;
    int max_iter = 1000000;
    std::string output_file = "result_matrix.txt";

    try {
        po::options_description desc("Разрешенные опции");
        desc.add_options()
            ("help,h", "Вывести справку")
            ("size,n", po::value<int>(&N)->default_value(128), "Размер сетки N")
            ("tol,t", po::value<double>(&tol)->default_value(1e-6), "Требуемая точность")
            ("iter,i", po::value<int>(&max_iter)->default_value(1000000), "Максимум итераций")
            ("output,o", po::value<std::string>(&output_file)->default_value("result_matrix.txt"),
             "Файл для сохранения результирующей матрицы");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);

        if (vm.count("help")) {
            std::cout << desc << '\n';
            return 0;
        }
    } catch (const std::exception& e) {
        std::cerr << "Ошибка парсинга аргументов: " << e.what() << '\n';
        return 1;
    }

    if (N < 2) {
        std::cerr << "Размер сетки должен быть >= 2\n";
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

    const auto start = std::chrono::steady_clock::now();

    #pragma acc data copy(d_grid[0:N*N], d_new_grid[0:N*N])
    {
        while (error > tol && iter < max_iter) {
            error = 0.0;

            #pragma acc parallel loop collapse(2) reduction(max:error) present(d_grid[0:N*N], d_new_grid[0:N*N])
            for (int i = 1; i < N - 1; ++i) {
                for (int j = 1; j < N - 1; ++j) {
                    const double new_value = 0.25 * (
                        d_grid[(i - 1) * N + j] +
                        d_grid[(i + 1) * N + j] +
                        d_grid[i * N + (j - 1)] +
                        d_grid[i * N + (j + 1)]
                    );

                    d_new_grid[i * N + j] = new_value;

                    const double diff = std::abs(new_value - d_grid[i * N + j]);
                    if (diff > error) {
                        error = diff;
                    }
                }
            }

            std::swap(d_grid, d_new_grid);
            ++iter;
        }
    }

    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed_seconds = end - start;

    const std::vector<double>& result_grid = (d_grid == grid.data()) ? grid : new_grid;

    std::cout << "Количество итераций: " << iter << '\n';
    std::cout << "Достигнутая ошибка: " << std::scientific << error << '\n';
    std::cout << "Время выполнения: " << std::fixed
              << elapsed_seconds.count() << " сек.\n";
    std::cout << "Матрица сохранена в файл: " << output_file << '\n';

    if (N == 10 || N == 13) {
        print_grid(result_grid, N);
    }

    try {
        save_grid_to_file(result_grid, N, output_file);
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }

    return 0;
}