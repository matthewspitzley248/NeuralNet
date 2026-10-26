#include <NeuralNet/WorkerPool.h>


namespace NN
{
    WorkerPool::~WorkerPool()
    {
        Stop();
    }

    void WorkerPool::Start(size_t workerCount)
    {
        std::unique_lock<std::mutex> lock(queueMutex_);

        if (!threads_.empty())
        {
            throw std::logic_error("WorkerPool has already been started.");
        }

        if (workerCount == 0)
        {
            throw std::invalid_argument("WorkerPool requires at least one worker.");
        }

        stopping_ = false;
        jobException_ = nullptr;

        try
        {
            for (size_t i = 0; i < workerCount; ++i)
            {
                threads_.emplace_back(&WorkerPool::WorkerLoop, this, i);
            }
        }
        catch (...)
        {
            stopping_ = true;
            lock.unlock();

            queueCondition_.notify_all();

            for (std::thread& thread : threads_)
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
            std::unique_lock<std::mutex> lock(queueMutex_);

            // Allow submitted jobs to finish before shutting down.
            completionCondition_.wait(lock, [this]()
                {
                    return pendingJobs_ == 0;
                });

            stopping_ = true;
        }

        queueCondition_.notify_all();

        for (std::thread& thread : threads_)
        {
            if (thread.joinable())
            {
                thread.join();
            }
        }

        threads_.clear();

        std::lock_guard<std::mutex> lock(queueMutex_);
        stopping_ = false;
    }

    void WorkerPool::SubmitJobs(std::vector<Job> jobs)
    {
        if (jobs.empty())
        {
            return;
        }

        {
            std::lock_guard<std::mutex> lock(queueMutex_);

            if (stopping_)
            {
                throw std::logic_error(
                    "Cannot submit jobs while WorkerPool is stopping.");
            }

            if (threads_.empty())
            {
                throw std::logic_error(
                    "WorkerPool must be started before submitting jobs.");
            }

            pendingJobs_ += jobs.size();

            for (Job& job : jobs)
            {
                jobQueue_.push(std::move(job));
            }
        }

        queueCondition_.notify_all();
    }

    void WorkerPool::WaitForJobs()
    {
        std::exception_ptr exception;

        {
            std::unique_lock<std::mutex> lock(queueMutex_);

            completionCondition_.wait(lock, [this]()
                {
                    return pendingJobs_ == 0;
                });

            exception = std::exchange(jobException_, nullptr);
        }

        // Re-throw on the calling thread, not on a worker thread.
        if (exception)
        {
            std::rethrow_exception(exception);
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
            Job job;

            {
                std::unique_lock<std::mutex> lock(queueMutex_);

                queueCondition_.wait(lock, [this]()
                    {
                        return stopping_ || !jobQueue_.empty();
                    });

                // Exit only after shutdown is requested and the queue
                // has been drained.
                if (stopping_ && jobQueue_.empty())
                {
                    return;
                }

                job = std::move(jobQueue_.front());
                jobQueue_.pop();
            }

            try
            {
                // Run the job without holding the queue mutex.
                job(workerIndex);
            }
            catch (const std::exception& e)
            {
                printf("[Worker %zu] Exception in job: %s\n",
                    workerIndex, e.what());

                {
                    std::lock_guard<std::mutex> lock(queueMutex_);

                    if (!jobException_)
                    {
                        jobException_ = std::current_exception();
                    }
                }
            }
            catch (...)
            {
                printf("[Worker %zu] Unknown exception in job.\n", workerIndex);

                {
                    std::lock_guard<std::mutex> lock(queueMutex_);

                    if (!jobException_)
                    {
                        jobException_ = std::current_exception();
                    }
                }
            }

            {
                std::lock_guard<std::mutex> lock(queueMutex_);

                --pendingJobs_;

                if (pendingJobs_ == 0)
                {
                    completionCondition_.notify_all();
                }
            }
        }
    }
}