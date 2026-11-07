#pragma once

#include <algorithm>
#include <stdexcept>

#include <NeuralNet/Layers/Layer.h>
#include <NeuralNet/Padding.h>

namespace NN::Layers
{
    struct MaxPool2DContext : LayerContext
    {
        // Maps each output position to the absolute flat source index of the maximum value
        //std::vector<size_t> maxIndices;
    };

    struct MaxPool2DImageContext
    {
        //Maps each output position to the absolute flat source index of the maximum value
        std::vector<size_t> maxIndices;
    };


    class MaxPool2D : public Layer
    {
    public:
        MaxPool2D(std::pair<size_t, size_t> poolSize = { 2, 2 }, std::pair<size_t, size_t> strides = { 0, 0 }, Padding padding = Padding::Valid);

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
        void Run(Tensor& source, LayerContext& gradientContext, MaxPool2DImageContext& imageContext);
        void Backpropagate(Tensor& gradient, LayerContext& gradientContext, MaxPool2DImageContext& imageContext);

        std::vector<MaxPool2DImageContext> imageContexts_;

        std::pair<size_t, size_t> poolSize_; 
        std::pair<size_t, size_t> strides_; 
        Padding padding_;

        std::vector<size_t> inputShape_;
    };
}