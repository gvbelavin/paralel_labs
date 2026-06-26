#pragma once

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>
#include <future>
#include <unordered_map>
#include <memory>
#include <stdexcept>
#include <iostream>

class ThreadPool {
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> jobQueue;
    std::mutex jobMutex;
    std::condition_variable cv;
    std::atomic<bool> running{false};

    void workerLoop() {
        while (true) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lock(jobMutex);
                cv.wait(lock, [this]() {
                    return !running || !jobQueue.empty();
                });
                if (!running && jobQueue.empty()) return;
                job = std::move(jobQueue.front());
                jobQueue.pop();
            }
            job();
        }
    }

public:
    explicit ThreadPool(size_t threadCount) {
        running = true;
        workers.reserve(threadCount);
        for (size_t i = 0; i < threadCount; i++)
            workers.emplace_back(&ThreadPool::workerLoop, this);
    }

    ~ThreadPool() { shutdown(); }

    void submit(std::function<void()> job) {
        {
            std::lock_guard<std::mutex> lock(jobMutex);
            jobQueue.push(std::move(job));
        }
        cv.notify_one();
    }

    void shutdown() {
        if (!running) return;
        running = false;
        cv.notify_all();
        for (auto& t : workers)
            if (t.joinable()) t.join();
        workers.clear();
    }
};

template <typename T>
class Server {
private:

    struct Task {
        size_t id;
        std::function<T()> func;
        std::promise<T> promise;
    };

    std::queue<std::shared_ptr<Task>> taskQueue;
    std::unordered_map<size_t, std::future<T>> resultFutures;

    std::mutex queueMutex;
    std::mutex resultsMutex;
    std::condition_variable cv;

    std::atomic<bool> running{false};
    std::atomic<size_t> nextId{1};

    std::unique_ptr<ThreadPool> pool;
    std::thread dispatchThread;

    void dispatchLoop() {
        while (true) {
            std::shared_ptr<Task> task;
            {
                std::unique_lock<std::mutex> lock(queueMutex);
                cv.wait(lock, [this]() {
                    return !running || !taskQueue.empty();
                });
                if (!running && taskQueue.empty()) return;
                task = std::move(taskQueue.front());
                taskQueue.pop();
            }
            pool->submit([task]() {
                try {
                    T result = task->func();
                    task->promise.set_value(result);
                } catch (...) {
                    task->promise.set_exception(std::current_exception());
                }
            });
        }
    }

public:
    Server() = default;
    ~Server() { stop(); }

    void start(size_t threadCount = std::thread::hardware_concurrency()) {
        if (running) return;
        running = true;
        pool = std::make_unique<ThreadPool>(threadCount);
        dispatchThread = std::thread(&Server::dispatchLoop, this);
        std::cout << "[Server] started with ThreadPool(" << threadCount << " threads)\n";
    }

    void stop() {
        if (!running) return;
        running = false;
        cv.notify_all();
        if (dispatchThread.joinable()) dispatchThread.join();
        if (pool) pool->shutdown();
    }

    size_t add_task(std::function<T()> func) {
        size_t id = nextId++;

        auto task = std::make_shared<Task>();
        task->id = id;
        task->func = std::move(func);

        std::future<T> fut = task->promise.get_future();
        {
            std::lock_guard<std::mutex> lock(resultsMutex);
            resultFutures.emplace(id, std::move(fut));
        }
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            taskQueue.push(std::move(task));
        }
        cv.notify_one();
        return id;
    }

    T request_result(size_t id) {
        std::future<T> fut;
        {
            std::lock_guard<std::mutex> lock(resultsMutex);
            auto it = resultFutures.find(id);
            if (it == resultFutures.end())
                throw std::runtime_error("Unknown task id: " + std::to_string(id));
            fut = std::move(it->second);
            resultFutures.erase(it);
        }
        return fut.get();
    }
};