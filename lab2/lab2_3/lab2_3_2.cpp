#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <omp.h>
#include <ctime>

using namespace std;

double cpuSecond()
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1.e-9;
}

double norm2(const vector<double>& v)
{
    double s = 0.0;
    for (double val : v)
        s += val * val;
    return sqrt(s);
}

int main()
{
    const int N = 3800;
    const double tau = 0.00001;
    const double eps = 1e-5;
    const int max_iter = 250000;

    vector<vector<double>> A(N, vector<double>(N));
    vector<double> b(N), x(N, 0.0), x_new(N, 0.0), r(N, 0.0);

    for (int i = 0; i < N; i++) {
        b[i] = N + 1.0;
        for (int j = 0; j < N; j++) {
            A[i][j] = (i == j) ? 2.0 : 1.0;
        }
    }

    double norm_b = norm2(b);
    double rel = 0.0;
    double sumsq = 0.0;
    int iter = 0;
    bool done = false;

    double t = cpuSecond();

    #pragma omp parallel shared(x, x_new, r, rel, sumsq, iter, done)
    {
        while (true) {
            #pragma omp for
            for (int i = 0; i < N; i++) {
                double ax = 0.0;
                for (int j = 0; j < N; j++) {
                    ax += A[i][j] * x[j];
                }
                r[i] = ax - b[i];
            }

            #pragma omp single
            sumsq = 0.0;

            #pragma omp for reduction(+:sumsq)
            for (int i = 0; i < N; i++) {
                sumsq += r[i] * r[i];
            }

            #pragma omp single
            {
                rel = sqrt(sumsq) / norm_b;
                if (rel < eps || iter >= max_iter)
                    done = true;
            }

            #pragma omp barrier
            if (done)
                break;

            #pragma omp for
            for (int i = 0; i < N; i++) {
                x_new[i] = x[i] - tau * r[i];
            }

            #pragma omp for
            for (int i = 0; i < N; i++) {
                x[i] = x_new[i];
            }

            #pragma omp single
            iter++;
        }
    }

    t = cpuSecond() - t;

    cout << fixed << setprecision(8);
    cout << "Result = " << rel << '\n';
    cout << "Time = " << t << " sec\n";
    cout << "Threads = " << omp_get_max_threads() << '\n';

    return 0;
}