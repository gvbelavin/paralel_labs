#include <iostream>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <string>
#include <memory>
#include <utility>
#include <boost/program_options.hpp>

#ifdef USE_NVTX
#include <nvtx3/nvToolsExt.h>
#define NVTX_PUSH(s) nvtxRangePushA(s)
#define NVTX_POP() nvtxRangePop()
#else
#define NVTX_PUSH(s)
#define NVTX_POP()
#endif

namespace po = boost::program_options;

int main(int argc, char* argv[]) {
    int N;
    double eps;
    int max_iter;
    int check_interval;

    po::options_description desc("Heat equation solver (2D Jacobi, OpenACC)");
    desc.add_options()
        ("help,h", "Show help")
        ("size,n", po::value<int>(&N)->default_value(128), "Grid size N (NxN)")
        ("eps,e", po::value<double>(&eps)->default_value(1e-6), "Convergence tolerance")
        ("iter,i", po::value<int>(&max_iter)->default_value(1000000), "Max iterations")
        ("check,c", po::value<int>(&check_interval)->default_value(1), "Check error every N iterations");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 0;
    }

    const long long size = (long long)N * N;
    const long long total = 2LL * size;

    // Double buffer: buf[0..size-1] = A, buf[size..2*size-1] = Anew
    // cur/nxt are offsets; std::swap(cur,nxt) replaces the copy loop
    auto buf_owner = std::make_unique<double[]>(total);
    double* buf = buf_owner.get();
    long long cur = 0, nxt = size;

    const double TL = 10.0, TR = 20.0, BR = 30.0, BL = 20.0;

    NVTX_PUSH("init");

    for (int j = 0; j < N; j++) {
        buf[cur + 0*N + j]     = TL + (TR - TL) * j / (N - 1);
        buf[cur + (N-1)*N + j] = BL + (BR - BL) * j / (N - 1);
        buf[nxt + 0*N + j]     = buf[cur + 0*N + j];
        buf[nxt + (N-1)*N + j] = buf[cur + (N-1)*N + j];
    }
    for (int i = 1; i < N - 1; i++) {
        buf[cur + i*N + 0]     = TL + (BL - TL) * i / (N - 1);
        buf[cur + i*N + (N-1)] = TR + (BR - TR) * i / (N - 1);
        buf[nxt + i*N + 0]     = buf[cur + i*N + 0];
        buf[nxt + i*N + (N-1)] = buf[cur + i*N + (N-1)];
    }

    NVTX_POP();

    std::cout << "Grid: " << N << "x" << N << "  eps=" << std::scientific << eps
              << "  max_iter=" << max_iter << "  check=" << check_interval << "\n";

    double error = 1.0;
    int iter = 0;

    auto t_start = std::chrono::high_resolution_clock::now();

    #pragma acc data copy(buf[0:total])
    {
        NVTX_PUSH("jacobi_loop");

        while (iter < max_iter && error > eps) {
            NVTX_PUSH("calc");
            if (iter % check_interval == 0) {
                #pragma acc wait(1)  // дождаться завершения всех async-ядер перед редукцией
                error = 0.0;
                #pragma acc parallel loop collapse(2) reduction(max:error)
                for (int i = 1; i < N - 1; i++) {
                    for (int j = 1; j < N - 1; j++) {
                        double val = 0.25 * (buf[cur + (i-1)*N+j] + buf[cur + (i+1)*N+j] +
                                             buf[cur + i*N+j-1]   + buf[cur + i*N+j+1]);
                        buf[nxt + i*N+j] = val;
                        error = fmax(error, fabs(val - buf[cur + i*N+j]));
                    }
                }
            } else {
                #pragma acc parallel loop collapse(2) async(1)
                for (int i = 1; i < N - 1; i++) {
                    for (int j = 1; j < N - 1; j++) {
                        buf[nxt + i*N+j] = 0.25 * (buf[cur + (i-1)*N+j] + buf[cur + (i+1)*N+j] +
                                                    buf[cur + i*N+j-1]   + buf[cur + i*N+j+1]);
                    }
                }
            }
            NVTX_POP();

            // Swap src/dst offsets — zero GPU overhead (just two integers on host)
            std::swap(cur, nxt);
            iter++;
        }

        NVTX_POP();
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "Iterations: " << iter << "\n"
              << "Error:      " << std::scientific << std::setprecision(6) << error << "\n"
              << "Time:       " << std::fixed << std::setprecision(3) << elapsed << " s\n";

    double* result = buf + cur;
    std::string fname = "result_" + std::to_string(N) + ".txt";
    std::ofstream out(fname);
    out << std::fixed << std::setprecision(6);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            out << result[i*N + j];
            if (j < N - 1) out << " ";
        }
        out << "\n";
    }
    out.close();
    std::cout << "Result saved to " << fname << "\n";

    if (N <= 13) {
        std::cout << "\nGrid " << N << "x" << N << ":\n";
        std::cout << std::fixed << std::setprecision(4);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                std::cout << std::setw(8) << result[i*N + j];
            }
            std::cout << "\n";
        }
    }

    return 0;
}
