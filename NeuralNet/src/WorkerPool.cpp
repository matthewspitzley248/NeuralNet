#include <NeuralNet/WorkerPool.h>


namespace NN
{
    WorkerPool::~WorkerPool()
    {
        Stop();
    }

    void WorkerPool::Start(size_t workerCount)
    {
        unique_lock<mutex> lock(queueMutex_);

        if (!threads_.empty())
        {
            throw logic_error("WorkerPool has already been started.");
        }

        if (workerCount == 0)
        {
            throw invalid_argument("WorkerPool requires at least one worker.");
        }

        stopping_ = false;
        jobException_ = nullptr;
        nextJobIndex_ = 0;
        pendingJobs_ = 0;

        try
        {
            for (size_t i = 0; i < workerCount; i++)
            {
                threads_.emplace_back(&WorkerPool::WorkerLoop, this, i);
            }
        }
        catch (...)
        {
            stopping_ = true;
            lock.unlock();

            queueCondition_.notify_all();

            for (thread& thread : threads_)
            {
                if (thread.joinable())
                {
                    thread.join();
                }
            }

            threads_.clear();

            lock.lock();
            stopping_ = false;

            throw;
        }
    }

    void WorkerPool::Stop()
    {
        {
            unique_lock<mutex> lock(queueMutex_);

            // Allow submitted jobs to finish before shutting down.
            completionCondition_.wait(lock, [this]()
                {
                    return pendingJobs_.load() == 0;
                });

            stopping_ = true;
        }

        queueCondition_.notify_all();

        for (thread& thread : threads_)
        {
            if (thread.joinable())
            {
                thread.join();
            }
        }

        threads_.clear();
        activeJobs_.clear();
    }

    void WorkerPool::SubmitJobs(vector<Job> jobs)
    {
        if (jobs.empty())
        {
            return;
        }

        {
            lock_guard<mutex> lock(queueMutex_);

            if (stopping_)
            {
                throw logic_error("Cannot submit jobs while WorkerPool is stopping.");
            }

            if (threads_.empty())
            {
                throw logic_error("WorkerPool must be started before submitting jobs.");
            }

            // Zero Allocation Hand-Off Optimization:
            // Store the jobs into a contiguous vector and reset the atomic counter.
            // This replaces pushing elements one-by-one into a slow queue.
            activeJobs_ = std::move(jobs);
            nextJobIndex_ = 0;
            pendingJobs_ = activeJobs_.size();
        }

        queueCondition_.notify_all();
    }

    void WorkerPool::WaitForJobs()
    {
        exception_ptr exception;

        {
            unique_lock<mutex> lock(queueMutex_);

            completionCondition_.wait(lock, [this]()
                {
                    return pendingJobs_.load() == 0;
                });

            exception = exchange(jobException_, nullptr);
        }

        // Re-throw on the calling thread, not on a worker thread.
        if (exception)
        {
            rethrow_exception(exception);
        }
    }

    size_t WorkerPool::GetWorkerCount() const
    {
        return threads_.size();
    }

    void WorkerPool::WorkerLoop(size_t workerIndex)
    {
        while (true)
        {
            size_t myJobIdx = 0;
            bool foundJob = false;

            // 1. Lock-Free Stealing Optimization Phase
            // Threads grab work by incrementing the atomic pointer index.
            // This completely bypasses the queueMutex_ lock!
            size_t currentIdx = nextJobIndex_.load(std::memory_order_relaxed);
            while (currentIdx < activeJobs_.size())
            {
                if (nextJobIndex_.compare_exchange_weak(currentIdx, currentIdx + 1,
                    std::memory_order_seq_cst,
                    std::memory_order_relaxed))
                {
                    myJobIdx = currentIdx;
                    foundJob = true;
                    break;
                }
            }

            // 2. Mutex Fallback Phase (Only triggered when the active batch is empty)
            if (!foundJob)
            {
                std::unique_lock<std::mutex> lock(queueMutex_);

                queueCondition_.wait(lock, [this]()
                    {
                        return stopping_ || (nextJobIndex_.load() < activeJobs_.size());
                    });

                if (stopping_ && (nextJobIndex_.load() >= activeJobs_.size()))
                {
                    return;
                }

                // Try to steal again immediately after waking up
                continue;
            }

            // 3. Thread Safe Execution Phase
            try
            {
                // Run the job completely independent of lock parameters
                activeJobs_[myJobIdx](workerIndex);
            }
            catch (const std::exception& e)
            {
                printf("[Worker %zu] Exception in job: %s\n", workerIndex, e.what());
                std::lock_guard<std::mutex> lock(queueMutex_);
                if (!jobException_)
                {
                    jobException_ = std::current_exception();
                }
            }
            catch (...)
            {
                printf("[Worker %zu] Unknown exception in job.\n", workerIndex);
                std::lock_guard<std::mutex> lock(queueMutex_);
                if (!jobException_)
                {
                    jobException_ = std::current_exception();
                }
            }

            // 4. Lock-Free Atomic Progress Reduction
            // If this thread finishes the very last job in the active pool, 
            // it notifies the condition variable cleanly.
            if (pendingJobs_.fetch_sub(1, std::memory_order_acq_rel) == 1)
            {
                std::lock_guard<std::mutex> lock(queueMutex_);
                completionCondition_.notify_all();
            }
        }
    }
}