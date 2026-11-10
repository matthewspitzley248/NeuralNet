#include <NeuralNet/Layers/Flatten.h>

namespace NN::Layers
{
    Flatten::Flatten()
    {
    }

    void Flatten::Initialize(std::mt19937& generator, const std::vector<size_t>& inputShape)
    {
        inputShape_ = inputShape;
    }

    std::vector<size_t> Flatten::GetOutputShape(const std::vector<size_t>& inputShape) const
    {
        size_t totalSize = 1;

        for (size_t dimension : inputShape)
        {
            totalSize *= dimension;
        }

        return { totalSize };
    }

    void Flatten::RunBatch(ImageBatch& batch, const size_t batchSize, std::vector<std::unique_ptr<WorkerContext>>& contexts, WorkerPool& workerPool)
    {
        vector<WorkerPool::Job> jobs;
        jobs.reserve(batchSize);

        imageContexts_.clear();
        imageContexts_.resize(batchSize);

        //For each image in the batch add it to the jobs list for the threads or workers to handle
        for (size_t i = 0; i < batchSize; i++)
        {
            jobs.emplace_back([this, &batch, &contexts, i](size_t workerIndex)
                {
                    ImageLabel& image = batch.imageLabels[i];

                    //Forward pass using this worker's contexts.
                    Run(image.image);
                });
        }

        workerPool.SubmitJobs(std::move(jobs));
        workerPool.WaitForJobs();
    }

    void Flatten::Run(Tensor& source)
    {
        //Force the destination buffer size to match without hitting the OS heap
        source.SetShape({ source.GetSize() });
    }

    void Flatten::Backpropagate(Tensor& gradient, WorkerContext& workerContext, ImageContext& imageContext)
    {
        gradient.ReshapeAndResize(inputShape_);
    }
}