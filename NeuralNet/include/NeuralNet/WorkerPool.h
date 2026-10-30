#pragma once


#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <exception>
#include <stdexcept>
#include <utility>

using namespace std;
namespace NN
{
    class WorkerPool
    {
    public:
        // A job receives the index of the worker executing it.
        using Job = function<void(size_t workerIndex)>;

        WorkerPool() = default;
        ~WorkerPool();

        // Prevent copying a pool that owns threads and synchronization objects.
        WorkerPool(const WorkerPool&) = delete;
        WorkerPool& operator=(const WorkerPool&) = delete;

        // Create the worker threads.
        void Start(size_t workerCount);

        // Wait for pending jobs and stop the worker threads.
        void Stop();

        // Add jobs to the shared queue.
        void SubmitJobs(vector<Job> jobs);

        // Wait until all submitted jobs have finished.
        void WaitForJobs();

        size_t GetWorkerCount() const;

    private:
        // Function executed by each worker thread.
        void WorkerLoop(size_t workerIndex);

        vector<Job> activeJobs_;
        atomic<size_t> nextJobIndex_{0};
        atomic<size_t> pendingJobs_{0};

        vector<thread> threads_;
        queue<Job> jobQueue_;

        mutex queueMutex_;
        condition_variable queueCondition_;
        condition_variable completionCondition_;

        //size_t pendingJobs_ = 0;
        bool stopping_ = false;

        exception_ptr jobException_ = nullptr;
    };
}