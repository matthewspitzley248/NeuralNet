#pragma once

#include <NeuralNet/Layers/Layer.h>
#include <NeuralNet/Padding.h>
#include <utility>
#include <NeuralNet/Math.h>

namespace NN::Layers
{
    struct Conv2DContext : LayerContext
    {
        //Accumulated gradients
        Tensor weightGradients;
        Tensor biasGradients;
        Tensor activationGradient;

        // Caches the massive spatial column footprints safely
        Tensor columnsCache;
        Tensor transposedGradCache;
        Tensor columnGradCache;
    };

    struct Conv2DImageContext
    {
        //Forward pass info
        Tensor input;
        Tensor output;
    };

    class Conv2D : public Layer
    {
    public:
        Conv2D(
            size_t filters,
            std::pair<size_t, size_t> kernelSize,
            std::pair<size_t, size_t> strides = {1, 1},
            ActivationType activationType = ActivationType::None,
            Padding padding = Padding::Valid
        );

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
        void UpdateTransposedWeightsCache();

        void Run(Tensor& source, LayerContext& gradientContext, Conv2DImageContext& imageContext);
        void Backpropagate(Tensor& gradient, LayerContext& gradientContext, Conv2DImageContext& imageContext);

        std::vector<Conv2DImageContext> imageContexts_;

        size_t filters_;
        std::pair<size_t, size_t> kernelSize_;
        std::pair<size_t, size_t> strides_;
        Padding padding_;
        ActivationType activationType_;

        size_t inputChannels_ = 0;

        //Padding amounts calculated from the input and output dimensions
        size_t paddingTop_ = 0;
        size_t paddingLeft_ = 0;

        Tensor transposedWeights_;
    };
}