// CPU BA 共用进程级工作池，避免 bottom-up 每个原子工作线程再创建线程池而过度占用 CPU。
// 并行区域互斥使用完整池宽度，小问题直接在调用线程求解。
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "core/Env.h"

namespace bacpu {

class Pool {
public:
    // 池宽度按整机设置，可由 SS_SFM_BA_THREADS 覆盖；单次求解通过任务数限流，避免首个单线程原子任务将全池永久限制为一线程。
    static Pool& get() {
        static Pool p;
        return p;
    }

    int size() const { return (int)workers_.size() + 1; }

    // 动态分配 fn(task, tid)，最多使用 maxWorkers 个线程；归约必须按 task 索引，不能按 tid，否则结果依赖调度。
    template <class F>
    void run(int ntasks, int maxWorkers, F&& fn) {
        if (ntasks <= 0) return;
        if (workers_.empty() || ntasks == 1 || maxWorkers <= 1) {
            for (int t = 0; t < ntasks; t++) fn(t, 0);
            return;
        }
        Job<F> job{std::forward<F>(fn)};
        dispatch(ntasks, maxWorkers, job);
    }
    template <class F>
    void run(int ntasks, F&& fn) {
        run(ntasks, size(), std::forward<F>(fn));
    }

private:
    struct AnyJob {
        virtual void call(int task, int tid) const = 0;
        virtual ~AnyJob() = default;
    };
    template <class F>
    struct Job : AnyJob {
        F f;
        explicit Job(F&& fn) : f(std::forward<F>(fn)) {}
        void call(int task, int tid) const override { f(task, tid); }
    };

    Pool() {
        int threads = 0;
        if (const char* e = spirula::env("SFM_BA_THREADS")) threads = atoi(e);
        unsigned hc = std::thread::hardware_concurrency();
        int n = threads > 0 ? threads : (int)(hc ? hc : 1u);
        workers_.reserve((size_t)n - 1);
        for (int i = 1; i < n; i++) workers_.emplace_back([this, i] { loop(i); });
    }

    ~Pool() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        for (std::thread& t : workers_) t.join();
    }

    void dispatch(int ntasks, int maxWorkers, const AnyJob& job) {
        std::lock_guard<std::mutex> region(region_);  // 同一时刻仅运行一个并行区域
        {
            std::lock_guard<std::mutex> lk(mu_);
            job_ = &job;
            ntasks_ = ntasks;
            limit_ = maxWorkers;
            next_.store(0, std::memory_order_relaxed);
            done_ = 0;
            gen_++;
        }
        cv_.notify_all();
        drain(0);
        std::unique_lock<std::mutex> lk(mu_);
        cv_done_.wait(lk, [&] { return done_ == (int)workers_.size(); });
        job_ = nullptr;
    }

    void drain(int tid) {
        if (tid >= limit_) return;
        for (int t = next_.fetch_add(1, std::memory_order_relaxed); t < ntasks_;
             t = next_.fetch_add(1, std::memory_order_relaxed))
            job_->call(t, tid);
    }

    void loop(int tid) {
        uint64_t seen = 0;
        for (;;) {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&] { return stop_ || gen_ != seen; });
            if (stop_) return;
            seen = gen_;
            lk.unlock();
            drain(tid);
            lk.lock();
            if (++done_ == (int)workers_.size()) cv_done_.notify_one();
        }
    }

    std::vector<std::thread> workers_;
    std::mutex region_, mu_;
    std::condition_variable cv_, cv_done_;
    const AnyJob* job_ = nullptr;
    std::atomic<int> next_{0};
    int ntasks_ = 0, done_ = 0, limit_ = 0;
    uint64_t gen_ = 0;
    bool stop_ = false;
};

// 将 [0, n) 分成 ntasks 个连续区间，第 t 段为 [lo, hi)。
inline void taskRange(int64_t n, int ntasks, int t, int64_t& lo, int64_t& hi) {
    int64_t q = n / ntasks, r = n % ntasks;
    lo = q * t + (t < r ? t : r);
    hi = lo + q + (t < r ? 1 : 0);
}

// 按 grain 粒度划分 n 项并限制在池容量内；仅一个任务时内联执行，避免小型求解进入线程池。
inline int taskCount(int64_t n, int64_t grain, int nthreads) {
    if (n <= grain || nthreads <= 1) return 1;
    int64_t k = (n + grain - 1) / grain;
    return (int)(k < nthreads ? k : nthreads);
}

}  // 命名空间 bacpu
