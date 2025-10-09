#pragma once

#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

#include "types.h"

class ThreadPool {
 public:
  explicit ThreadPool(size_t numThreads = std::thread::hardware_concurrency())
      : stop_(false) {
    workers_.reserve(numThreads);
    for (size_t i = 0; i < numThreads; ++i) {
      workers_.emplace_back([this] {
        while (true) {
          std::function<void()> task;

          {
            std::unique_lock<std::mutex> lock(queueMutex_);
            condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });

            if (stop_ && tasks_.empty()) {
              return;
            }

            task = std::move(tasks_.front());
            tasks_.pop();
          }
          task();
        }
      });
    }
  }

  ~ThreadPool() {
    {
      std::unique_lock<std::mutex> lock(queueMutex_);
      stop_ = true;
    }

    condition_.notify_all();

    for (std::thread& worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }

  template <typename F, typename... Args>
  auto enqueue(F&& f, Args&&... args) -> std::future<typename std::invoke_result<F, Args...>::type> {
    using return_type = typename std::invoke_result<F, Args...>::type;

    auto task =
        std::make_shared<std::packaged_task<return_type()>>(std::bind(std::forward<F>(f), std::forward<Args>(args)...));

    std::future<return_type> res = task->get_future();

    {
      std::unique_lock<std::mutex> lock(queueMutex_);

      if (stop_) {
        throw std::runtime_error("enqueue on stopped ThreadPool");
      }

      tasks_.emplace([task]() { (*task)(); });
    }

    condition_.notify_one();
    return res;
  }

  size_t getThreadCount() const {
    return workers_.size();
  }

  size_t getPendingTaskCount() const {
    std::unique_lock<std::mutex> lock(queueMutex_);
    return tasks_.size();
  }

 private:
  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;

  mutable std::mutex queueMutex_;
  std::condition_variable condition_;
  bool stop_;
};

class GlobalThreadPool {
 public:
  static ThreadPool& getInstance() {
    static ThreadPool instance;
    return instance;
  }

  GlobalThreadPool(const GlobalThreadPool&) = delete;
  GlobalThreadPool& operator=(const GlobalThreadPool&) = delete;

 private:
  GlobalThreadPool() = default;
};

class ParallelUtil {
 public:
  template <typename BinaryOp>
  static val_t p_transform_reduce(const val_t* first1, const val_t* first2, size_t size, BinaryOp op,
      val_t init = val_t(0), size_t threshold = 10000) {
    if (size < threshold) {
      return std::transform_reduce(first1, first1 + size, first2, init, std::plus<>(), op);
    }

    auto& pool = GlobalThreadPool::getInstance();
    const size_t numThreads = pool.getThreadCount();
    const size_t chunkSize = (size + numThreads - 1) / numThreads;

    std::vector<std::future<val_t>> futures;
    futures.reserve(numThreads);

    for (size_t t = 0; t < numThreads; ++t) {
      const size_t start = t * chunkSize;
      if (start >= size)
        break;
      const size_t end = std::min(start + chunkSize, size);

      futures.push_back(pool.enqueue([first1, first2, start, end, op, init]() {
        return std::transform_reduce(first1 + start, first1 + end, first2 + start, init, std::plus<>(), op);
      }));
    }

    val_t result = init;
    for (auto& future : futures) {
      result += future.get();
    }

    return result;
  }

  template <typename UnaryOp>
  static void p_transform(const val_t* first, val_t* result, size_t size, UnaryOp op, size_t threshold = 10000) {
    if (size < threshold) {
      std::transform(first, first + size, result, op);
      return;
    }

    auto& pool = GlobalThreadPool::getInstance();
    const size_t numThreads = pool.getThreadCount();
    const size_t chunkSize = (size + numThreads - 1) / numThreads;

    std::vector<std::future<void>> futures;
    futures.reserve(numThreads);

    for (size_t t = 0; t < numThreads; ++t) {
      const size_t start = t * chunkSize;
      if (start >= size)
        break;
      const size_t end = std::min(start + chunkSize, size);

      futures.push_back(pool.enqueue(
          [first, result, start, end, op]() { std::transform(first + start, first + end, result + start, op); }));
    }

    for (auto& future : futures) {
      future.get();
    }
  }

  template <typename UnaryOp>
  static void p_transform(const val_t* first1, const val_t* first2, val_t* result, size_t size, UnaryOp op,
      size_t threshold = 10000) {
    if (size < threshold) {
      std::transform(first1, first1 + size, first2, result, op);
      return;
    }

    auto& pool = GlobalThreadPool::getInstance();
    const size_t numThreads = pool.getThreadCount();
    const size_t chunkSize = (size + numThreads - 1) / numThreads;

    std::vector<std::future<void>> futures;
    futures.reserve(numThreads);

    for (size_t t = 0; t < numThreads; ++t) {
      const size_t start = t * chunkSize;
      if (start >= size)
        break;
      const size_t end = std::min(start + chunkSize, size);

      futures.push_back(pool.enqueue([first1, first2, result, start, end, op]() {
        std::transform(first1 + start, first1 + end, first2 + start, result + start, op);
      }));
    }

    for (auto& future : futures) {
      future.get();
    }
  }
};
