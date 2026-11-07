#include <NeuralNet/Layers/Flatten.h>

namespace NN::Layers
{
    Flatten::Flatten()
    {
    }

    unique_ptr<LayerContext> Flatten::CreateContext() const
    {
        auto context = make_unique<FlattenContext>();
        return context;
    }

    void Flatten::Initialize(std::mt19937& generator, const std::vector<size_t>& inputShape)
    {
        inputShape_ = inputShape;
    }

    std::vector<size_t> Flatten::GetOutputShape(const std::vector<size_t>& inputShape)
    {
        size_t totalSize = 1;

        for (size_t dimension : inputShape)
        {
            totalSize *= dimension;
        }

        return { totalSize };
    }

    void Flatten::RunBatch(ImageBatch& batch, const size_t batchSize, std::vector<std::unique_ptr<LayerContext>>& contexts, WorkerPool& workerPool)
    {
        vector<WorkerPool::Job> jobs;
        jobs.reserve(batchSize);

        //For each image in the batch add it to the jobs list for the threads or workers to handle
        for (size_t i = 0; i < batchSize; i++)
        {
            jobs.emplace_back([this, &batch, &contexts, i](size_t workerIndex)
                {
                    ImageLabel& image = batch.imageLabels[i];

                    //Forward pass using this worker's contexts.
                    Run(image.image, *contexts[workerIndex]);
                });
        }

        workerPool.SubmitJobs(std::move(jobs));
        workerPool.WaitForJobs();
    }

    void Flatten::Run(Tensor& source, LayerContext& context)
    {
        //Force the destination buffer size to match without hitting the OS heap
        source.SetShape({ source.GetSize() });
    }

    void Flatten::BackpropagateBatch(vector<Tensor>& batchGradients, const size_t batchSize, std::vector<std::unique_ptr<LayerContext>>& contexts, WorkerPool& workerPool)
    {
        vector<WorkerPool::Job> jobs;
        jobs.reserve(batchSize);

        //For each image in the batch add it to the jobs list for the threads or workers to handle
        for (size_t i = 0; i < batchSize; i++)
        {
            jobs.emplace_back([this, &batchGradients, &contexts, i](size_t workerIndex)
                {
                    //Forward pass using this worker's contexts.
                    Backpropagate(batchGradients[i], *contexts[workerIndex]);
                });
        }

        workerPool.SubmitJobs(std::move(jobs));
        workerPool.WaitForJobs();
    }

    void Flatten::Backpropagate(Tensor& gradient, LayerContext& context)
    {
        gradient.ReshapeAndResize(inputShape_);
    }

    void Flatten::ApplyGradients(Optimizer& optimizer, int batchSize)
    {
    }

    void Flatten::ResetGradients(LayerContext& context)
    {
    }

    void Flatten::AccumulateGradients(LayerContext& context)
    {
    }
}