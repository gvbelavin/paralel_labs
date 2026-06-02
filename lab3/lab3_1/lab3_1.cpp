#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <algorithm>

using namespace std;

double cpuSecond() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

void init_chunk(vector<double>& a, vector<double>& b, vector<double>& c,
                long long m, long long n, long long row_begin, long long row_end,
                long long b_begin, long long b_end)
{
    for (long long i = row_begin; i < row_end; ++i) {
        for (long long j = 0; j < n; ++j)
            a[i * n + j] = static_cast<double>(i + j);
        c[i] = 0.0;
    }
    for (long long j = b_begin; j < b_end; ++j)
        b[j] = static_cast<double>(j);
}

void matvec_chunk(const vector<double>& a, const vector<double>& b, vector<double>& c,
                  long long n, long long row_begin, long long row_end)
{
    for (long long i = row_begin; i < row_end; ++i) {
        double sum = 0.0;
        for (long long j = 0; j < n; ++j)
            sum += a[i * n + j] * b[j];
        c[i] = sum;
    }
}

int main(int argc, char* argv[])
{
    long long m = 20000;
    long long n = 20000;
    unsigned int num_threads = 4;

    if (argc > 1) num_threads = static_cast<unsigned int>(atoi(argv[1]));
    if (argc > 2) { long long sz = atoll(argv[2]); m = sz; n = sz; }
    if (num_threads == 0) num_threads = 1;

    vector<double> a(static_cast<size_t>(m) * n);
    vector<double> b(n);
    vector<double> c(m, 0.0);

    vector<thread> threads;
    threads.reserve(num_threads);

    double t_init = cpuSecond();
    for (unsigned int t = 0; t < num_threads; ++t) {
        long long row_begin = static_cast<long long>(t) * m / num_threads;
        long long row_end   = static_cast<long long>(t + 1) * m / num_threads;

        long long b_begin = row_begin * n / m;
        long long b_end   = row_end   * n / m;
        threads.emplace_back(init_chunk,
            ref(a), ref(b), ref(c), m, n,
            row_begin, row_end, b_begin, b_end);
    }
    for (auto& th : threads) th.join();
    threads.clear();
    t_init = cpuSecond() - t_init;

    double t_calc = cpuSecond();
    for (unsigned int t = 0; t < num_threads; ++t) {
        long long row_begin = static_cast<long long>(t) * m / num_threads;
        long long row_end   = static_cast<long long>(t + 1) * m / num_threads;
        threads.emplace_back(matvec_chunk,
            cref(a), cref(b), ref(c), n, row_begin, row_end);
    }
    for (auto& th : threads) th.join();
    t_calc = cpuSecond() - t_calc;

    cout << fixed << setprecision(6);
    cout << "Matrix: " << m << "x" << n << "\n";
    cout << "Threads: " << num_threads << "\n";
    cout << "Init time:  " << t_init << " sec\n";
    cout << "Calc time:  " << t_calc << " sec\n";
    cout << "Total time: " << (t_init + t_calc) << " sec\n";

    cout << "\nFirst 5 results: ";
    for (int i = 0; i < min(5LL, m); ++i)
        cout << "c[" << i << "]=" << c[i] << " ";
    cout << "\n";

    return 0;
}
