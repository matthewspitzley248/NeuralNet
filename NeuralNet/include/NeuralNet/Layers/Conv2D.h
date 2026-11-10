#pragma once

#include <NeuralNet/Layers/Layer.h>
#include <NeuralNet/Padding.h>
#include <utility>
#include <NeuralNet/Math.h>

namespace NN::Layers
{
    struct Conv2DContext : WorkerContext
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

    struct Conv2DImageContext : ImageContext
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

        unique_ptr<WorkerContext> CreateWorkerContext() const override;

        void Initialize(std::mt19937& generator, const std::vector<size_t>& inputShape) override;

        std::vector<size_t> GetOutputShape(const std::vector<size_t>& inputShape) const override;

        void Run(Tensor& data) override;

        void ApplyGradients(Optimizer& optimizer, int batchSize) override;

        void ResetGradients(WorkerContext& workerContext) override;

        void AccumulateGradients(WorkerContext& workerContext) override;
         
    protected:
        void UpdateTransposedWeightsCache();

        void Run(Tensor& data, WorkerContext& workerContext, ImageContext& imageContext) override;
        void Backpropagate(Tensor& gradient, WorkerContext& workerContext, ImageContext& imageContext) override;
        unique_ptr<ImageContext> CreateImageContext() const override;


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