#include <iostream>
#include <iomanip>
#include <cmath>
#include <omp.h>

double func(double x)
{
    return std::exp(-x * x);
}

double integrate_omp(double (*func)(double), double a, double b, int n)
{
    double h = (b - a) / n;
    double sum = 0.0;

    #pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = n / nthreads;

        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (n - 1) : (lb + items_per_thread - 1);

        double sumloc = 0.0;

        for (int i = lb; i <= ub; i++) {
            sumloc += func(a + h * (i + 0.5));
        }

        #pragma omp atomic
        sum += sumloc;
    }

    return sum * h;
}

int main()
{
    double res = integrate_omp(func, -4.0, 4.0, 40000000);
    std::cout << "Result = " << std::fixed << std::setprecision(12) << res << '\n';
    return 0;
}