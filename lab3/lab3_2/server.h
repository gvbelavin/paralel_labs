#pragma once

#include <queue>
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <functional>
#include <future>
#include <memory>
#include <atomic>
#include <stdexcept>

// ═══════════════════════════════════════════════════════════════════
//  ThreadPool — пул воркер-потоков, которые берут задачи из очереди
//  Используется внутри Server<T> как механизм параллельного выполнения
// ═══════════════════════════════════════════════════════════════════
class ThreadPool {
private:
    std::vector<std::thread>          workers;
    std::queue<std::function<void()>> jobQueue;
    std::mutex                        jobMutex;
    std::condition_variable           cv;
    std::atomic<bool>                 running{false};

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
            job();  // выполняем задачу вне лока — параллельно между потоками
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

    // Добавить void-задачу в очередь пула
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


// ═══════════════════════════════════════════════════════════════════
//  Server<T> — шаблонный сервер с интерфейсом: start / stop /
//              add_task / request_result
//  Внутри использует ThreadPool для параллельного выполнения задач.
//  Пара promise/future — механизм передачи результата между потоками:
//    • promise.set_value()  — воркер кладёт результат (один раз)
//    • future.get()         — клиент забирает результат (один раз)
// ═══════════════════════════════════════════════════════════════════
template <typename T>
class Server {
private:

    struct Task {
        size_t              id;
        std::function<T()>  func;
        std::promise<T>     promise;   // воркер кладёт сюда результат
    };

    // Очередь задач, ожидающих выполнения
    std::queue<std::shared_ptr<Task>>             taskQueue;
    // Словарь: id → future  (клиент забирает результат отсюда)
    // unordered_map: O(1) insert / erase / find
    std::unordered_map<size_t, std::future<T>>    resultFutures;

    std::mutex              queueMutex;
    std::mutex              resultsMutex;
    std::condition_variable cv;

    std::atomic<bool>   running{false};
    std::atomic<size_t> nextId{1};

    std::unique_ptr<ThreadPool> pool;   // ← ThreadPool внутри сервера

    // Диспетчер: один поток читает taskQueue и отдаёт задачи в пул
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

            // Передаём задачу в ThreadPool — воркер выполнит её в своём потоке
            // Захватываем task по значению (shared_ptr) — безопасно
            pool->submit([task]() {
                try {
                    T result = task->func();
                    task->promise.set_value(result);           // кладём один раз
                } catch (...) {
                    task->promise.set_exception(std::current_exception());
                }
            });
        }
    }

public:
    Server() = default;
    ~Server() { stop(); }

    // threadCount — сколько потоков в пуле (по умолчанию = числу ядер)
    void start(size_t threadCount = std::thread::hardware_concurrency()) {
        if (running) return;
        running = true;
        pool           = std::make_unique<ThreadPool>(threadCount);
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

    // Добавить задачу → вернуть id
    // promise создаётся здесь, future сохраняется в map,
    // promise уходит вместе с task в очередь диспетчеру → потом воркеру пула
    size_t add_task(std::function<T()> func) {
        size_t id = nextId++;

        auto task    = std::make_shared<Task>();
        task->id     = id;
        task->func   = std::move(func);

        // Получаем future ДО передачи promise в другой поток
        std::future<T> fut = task->promise.get_future();   // связанная пара

        {
            std::lock_guard<std::mutex> lock(resultsMutex);
            resultFutures.emplace(id, std::move(fut));     // future → в map
        }
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            taskQueue.push(std::move(task));               // task (с promise) → в очередь
        }

        cv.notify_one();
        return id;
    }

    // Блокирующий: ждёт пока воркер не вызовет promise.set_value()
    // future.get() вызывается строго ОДИН РАЗ — запись сразу удаляется из map
    T request_result(size_t id) {
        std::future<T> fut;
        {
            std::lock_guard<std::mutex> lock(resultsMutex);
            auto it = resultFutures.find(id);
            if (it == resultFutures.end())
                throw std::runtime_error("Unknown task id: " + std::to_string(id));
            fut = std::move(it->second);    // забираем future из map
            resultFutures.erase(it);        // удаляем — get() только один раз
        }
        return fut.get();                   // ← единственный вызов get()
    }
};
