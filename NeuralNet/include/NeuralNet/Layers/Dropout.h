#pragma once

#include <NeuralNet/Layers/Layer.h>

namespace NN::Layers
{
    struct DropoutContext : LayerContext
    {
        std::mt19937 randomGenerator;
    };

    struct DropoutImageContext
    {
        std::vector<bool> dropoutMask;
    };

    class Dropout : public Layer
    {
    public:
        Dropout(float dropoutRate);

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
        void Run(Tensor& source, LayerContext& gradientContext, DropoutImageContext& imageContext);
        void Backpropagate(Tensor& gradient, LayerContext& gradientContext, DropoutImageContext& imageContext);

        float dropoutRate_;
        float scale_;

        //used to setup the gens for the worker contexts set in init
        std::mt19937* randomGenerator_ = nullptr;

        std::vector<DropoutImageContext> imageContexts_;
    };
}