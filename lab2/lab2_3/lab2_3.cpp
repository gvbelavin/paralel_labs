#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <omp.h>
#include <ctime>
#include <string>

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
    for (double val : v) {
        s += val * val;
    }
    return sqrt(s);
}

omp_sched_t parseSchedule(const string& name)
{
    if (name == "static") {
        return omp_sched_static;
    }
    if (name == "dynamic") {
        return omp_sched_dynamic;
    }
    if (name == "guided") {
        return omp_sched_guided;
    }
    return omp_sched_static;
}

int main(int argc, char* argv[])
{
    const int N = 3800;
    const double tau = 0.00001;
    const double eps = 1e-5;
    const int max_iter = 250000;
    string schedule_name = "static";
    int chunk_size = 0;
    int num_threads = omp_get_max_threads();
    if (argc >= 2) {
        schedule_name = argv[1];
    }
    if (argc >= 3) {
        chunk_size = stoi(argv[2]);
    }
    if (argc >= 4) {
        num_threads = stoi(argv[3]);
    }

    omp_sched_t schedule_kind = parseSchedule(schedule_name);
    omp_set_schedule(schedule_kind, chunk_size);
    omp_set_num_threads(num_threads);

    vector<vector<double>> A(N, vector<double>(N));
    vector<double> b(N);
    vector<double> x(N, 0.0), x_new(N, 0.0), r(N, 0.0);
    for (int i = 0; i < N; i++) {
        b[i] = N + 1.0;
        for (int j = 0; j < N; j++) {
            A[i][j] = (i == j) ? 2.0 : 1.0;
        }
    }
    const double norm_b = norm2(b);

    double rel = 0.0;
    int iter = 0;
    double t = cpuSecond();

    do {
#pragma omp parallel for schedule(runtime)
        for (int i = 0; i < N; i++) {
            double ax = 0.0;
            for (int j = 0; j < N; j++) {
                ax += A[i][j] * x[j];
            }
            r[i] = ax - b[i];
        }

        double sumsq = 0.0;
#pragma omp parallel for schedule(runtime) reduction(+:sumsq)
        for (int i = 0; i < N; i++) {
            sumsq += r[i] * r[i];
        }

        rel = sqrt(sumsq) / norm_b;
        if (rel < eps) {
            break;
        }

#pragma omp parallel for schedule(runtime)
        for (int i = 0; i < N; i++) {
            x_new[i] = x[i] - tau * r[i];
        }

#pragma omp parallel for schedule(runtime)
        for (int i = 0; i < N; i++) {
            x[i] = x_new[i];
        }

        iter++;
    } while (iter < max_iter);

    t = cpuSecond() - t;

    cout << fixed << setprecision(8);
    cout << "Result = " << rel << '\n';
    cout << "Iterations = " << iter << '\n';
    cout << "Time = " << t << " sec\n";
    cout << "Threads = " << num_threads << '\n';
    cout << "Schedule = " << schedule_name << ", chunk = " << chunk_size << '\n';
    return 0;
}