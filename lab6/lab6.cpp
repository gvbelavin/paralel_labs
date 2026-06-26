#include <boost/program_options.hpp>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

namespace po = boost::program_options;

double lerp(double v0, double v1, double t) {
    return (1.0 - t) * v0 + t * v1;
}

void init_grid(std::vector<double>& grid, int N) {
    std::fill(grid.begin(), grid.end(), 0.0);
    for (int i = 0; i < N; ++i) {
        double t = static_cast<double>(i) / (N - 1);
        grid[i]           = lerp(10.0, 20.0, t);
        grid[(N-1)*N + i] = lerp(20.0, 30.0, t);
        grid[i*N]         = lerp(10.0, 20.0, t);
        grid[i*N + (N-1)] = lerp(20.0, 30.0, t);
    }
}

void print_grid(const std::vector<double>& grid, int N) {
    if (N > 13) return;
    std::cout << "\nGrid (" << N << "x" << N << "):\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j)
            std::cout << std::fixed << std::setprecision(2)
                      << std::setw(6) << grid[i*N+j] << " ";
        std::cout << "\n";
    }
}

int main(int argc, char* argv[]) {
    int N        = 128;
    double tol   = 1e-6;
    int max_iter = 1000000;

    try {
        po::options_description desc("Опции");
        desc.add_options()
            ("help,h",  "Справка")
            ("size,n",  po::value<int>(&N)->default_value(128),           "Размер сетки N")
            ("tol,t",   po::value<double>(&tol)->default_value(1e-6),     "Точность")
            ("iter,i",  po::value<int>(&max_iter)->default_value(1000000),"Макс. итераций");
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);
        if (vm.count("help")) { std::cout << desc; return 0; }
    } catch (std::exception& e) {
        std::cerr << "Ошибка: " << e.what() << "\n";
        return 1;
    }

    std::vector<double> grid(N * N), new_grid(N * N);
    init_grid(grid, N);
    init_grid(new_grid, N);

    double* __restrict__ g  = grid.data();
    double* __restrict__ ng = new_grid.data();
    int sz = N * N;

    int iter     = 0;
    double error = 1.0;

    auto start = std::chrono::steady_clock::now();

    #pragma acc data copy(g[0:sz]) create(ng[0:sz])
    {
        while (error > tol && iter < max_iter) {
            error = 0.0;

            #pragma acc parallel loop collapse(2) reduction(max:error) \
                present(g[0:sz], ng[0:sz])
            for (int i = 1; i < N - 1; ++i) {
                for (int j = 1; j < N - 1; ++j) {
                    double new_val = 0.25 * (g[(i-1)*N+j] + g[(i+1)*N+j] +
                                             g[i*N+j-1]   + g[i*N+j+1]);
                    double d = new_val - g[i*N+j]; // ← diff от g до записи в ng
                    if (d < 0.0) d = -d;
                    if (d > error) error = d;
                    ng[i*N+j] = new_val;            // ← запись в конце
                }
            }

            #pragma acc parallel loop present(g[0:sz], ng[0:sz])
            for (int k = 0; k < sz; ++k)
                g[k] = ng[k];

            ++iter;
        }
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Итераций: " << iter << "\n"
              << "Ошибка:   " << std::scientific << error << "\n"
              << "Время:    " << std::fixed << std::setprecision(6)
              << elapsed.count() << " сек\n";

    if (N == 10 || N == 13)
        print_grid(grid, N);

    std::ofstream out("result_matrix.txt");
    out << std::fixed << std::setprecision(10);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            out << grid[i*N+j] << (j + 1 == N ? "\n" : " ");

    return 0;
}