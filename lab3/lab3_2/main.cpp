#include <iostream>
#include <fstream>
#include <thread>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>
#include "server.h"

template <typename Scalar>
Scalar fun_sin(Scalar arg) { return std::sin(arg); }

template <typename Scalar>
Scalar fun_sqrt(Scalar arg) { return std::sqrt(std::abs(arg)); }

template <typename Scalar>
Scalar fun_pow(Scalar base, Scalar exp) { return std::pow(base, exp); }

void client_sin(Server<double>& server, int N) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-3.14159265, 3.14159265);

    std::vector<std::pair<size_t, double>> tasks;
    tasks.reserve(N);

    for (int i = 0; i < N; i++) {
        double arg = dist(rng);
        size_t id = server.add_task([arg]() -> double {
            return fun_sin<double>(arg);
        });
        tasks.emplace_back(id, arg);
    }

    std::ofstream file("client1_sin.txt");
    file << std::fixed << std::setprecision(10);
    file << "ID\targ\tresult\n";

    for (auto& [id, arg] : tasks) {
        double result = server.request_result(id);
        file << id << "\t" << arg << "\t" << result << "\n";
    }

    std::cout << "[Client 1] sin:  " << N << " tasks in client1_sin.txt\n";
}


void client_sqrt(Server<double>& server, int N) {
    std::mt19937 rng(123);
    std::uniform_real_distribution<double> dist(0.0, 1000.0);

    std::vector<std::pair<size_t, double>> tasks;
    tasks.reserve(N);

    for (int i = 0; i < N; i++) {
        double arg = dist(rng);
        size_t id = server.add_task([arg]() -> double {
            return fun_sqrt<double>(arg);
        });
        tasks.emplace_back(id, arg);
    }

    std::ofstream file("client2_sqrt.txt");
    file << std::fixed << std::setprecision(10);
    file << "ID\targ\tresult\n";

    for (auto& [id, arg] : tasks) {
        double result = server.request_result(id);
        file << id << "\t" << arg << "\t" << result << "\n";
    }

    std::cout << "[Client 2] sqrt: " << N << " tasks in client2_sqrt.txt\n";
}


void client_pow(Server<double>& server, int N) {
    std::mt19937 rng(777);
    std::uniform_real_distribution<double> baseDist(0.1, 10.0);
    std::uniform_real_distribution<double> expDist(0.5, 5.0);

    std::vector<std::tuple<size_t, double, double>> tasks;
    tasks.reserve(N);

    for (int i = 0; i < N; i++) {
        double base = baseDist(rng);
        double exp  = expDist(rng);
        size_t id = server.add_task([base, exp]() -> double {
            return fun_pow<double>(base, exp);
        });
        tasks.emplace_back(id, base, exp);
    }

    std::ofstream file("client3_pow.txt");
    file << std::fixed << std::setprecision(10);
    file << "ID\tbase\texp\tresult\n";

    for (auto& [id, base, exp] : tasks) {
        double result = server.request_result(id);
        file << id << "\t" << base << "\t" << exp << "\t" << result << "\n";
    }

    std::cout << "[Client 3] pow:  " << N << " tasks in client3_pow.txt\n";
}
int main() {
    const int N = 100;

    Server<double> server;
    server.start();

    std::thread t1(client_sin,  std::ref(server), N);
    std::thread t2(client_sqrt, std::ref(server), N);
    std::thread t3(client_pow,  std::ref(server), N);

    t1.join();
    t2.join();
    t3.join();

    server.stop();

    std::cout << "\nDone\n";
    return 0;
}
