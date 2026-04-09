#include <iostream>
#include <vector>
#include <cstdint>
#include <iomanip>
#include <ctime>
#include <omp.h>

#ifndef M
#define M 40000
#endif

#ifndef N
#define N 40000
#endif

constexpr int m = M;
constexpr int n = N;

double cpuSecond()
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1.e-9;
}

void matrix_vector_product_omp(const std::vector<double>& a,
                               const std::vector<double>& b,
                               std::vector<double>& c,
                               int m, int n)
{
#pragma omp parallel for
    for (int i = 0; i < m; ++i) {
        c[i] = 0.0;
        for (int j = 0; j < n; ++j) {
            c[i] += a[i * n + j] * b[j];
        }
    }
}


void run_parallel()
{
    std::vector<double> a(static_cast<size_t>(m) * n);
    std::vector<double> b(n);
    std::vector<double> c(m);

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            a[i * n + j] = static_cast<double>(i + j);
        }
    }

    for (int j = 0; j < n; ++j) {
        b[j] = static_cast<double>(j);
    }

    double t = cpuSecond();
    matrix_vector_product_omp(a, b, c, m, n);
    t = cpuSecond() - t;

    std::cout << "Elapsed time (parallel): "
              << std::fixed << std::setprecision(6)
              << t << " sec.\n";
    std::cout << "c[0] = " << c[0] << ", c[m-1] = " << c[m - 1] << "\n";
}

int main()
{
    std::cout << "Matrix-vector product (c[m] = a[m, n] * b[n]; "
              << "m = " << m << ", n = " << n << ")\n";

    std::uint64_t memory_bytes =
        (static_cast<std::uint64_t>(m) * n +
         static_cast<std::uint64_t>(m) +
         static_cast<std::uint64_t>(n)) * sizeof(double);

    std::cout << "Memory used: " << (memory_bytes >> 20) << " MiB\n";
    std::cout << "OpenMP threads: " << omp_get_max_threads() << "\n";

    run_parallel();

    return 0;
}