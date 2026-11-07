#pragma once

#include <NeuralNet/Layers/Layer.h>

namespace NN::Layers
{
    struct FlattenContext : LayerContext
    {
    };

    class Flatten : public Layer
    {
    public:
        Flatten();

        unique_ptr<LayerContext> CreateContext() const override;

        void Initialize(std::mt19937& generator, const std::vector<size_t>& inputShape) override;

        std::vector<size_t> GetOutputShape(const std::vector<size_t>& inputShape) override;

        void RunBatch(ImageBatch& batch, const size_t batchSize, std::vector<std::unique_ptr<LayerContext>>& contexts, WorkerPool& workerPool) override;

        void Run(Tensor& source, LayerContext& context) override;

        void BackpropagateBatch(vector<Tensor>& batchGradients, const size_t batchSize, std::vector<std::unique_ptr<LayerContext>>& contexts, WorkerPool& workerPool) override;

        void Backpropagate(Tensor& gradient, LayerContext& context) override;

        void ApplyGradients(Optimizer& optimizer, int batchSize) override;

        void ResetGradients(LayerContext& context) override;

        void AccumulateGradients(LayerContext& context) override;

    protected:
        std::vector<size_t> inputShape_;
    };
}