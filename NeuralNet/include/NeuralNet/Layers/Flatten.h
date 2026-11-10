#pragma once

#include <NeuralNet/Layers/Layer.h>

namespace NN::Layers
{
    class Flatten : public Layer
    {
    public:
        Flatten();

        void Initialize(std::mt19937& generator, const std::vector<size_t>& inputShape) override;

        std::vector<size_t> GetOutputShape(const std::vector<size_t>& inputShape) const override;

        void RunBatch(ImageBatch& batch, const size_t batchSize, std::vector<std::unique_ptr<WorkerContext>>& workerContexts, WorkerPool& workerPool) override;

        void Run(Tensor& data) override;

    protected:
        //for batch methods to run
        void Backpropagate(Tensor& gradient, WorkerContext& workerContext, ImageContext& imageContext) override;

        std::vector<size_t> inputShape_;
    };
}