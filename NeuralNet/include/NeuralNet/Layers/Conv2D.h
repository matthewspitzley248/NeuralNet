#pragma once

#include <NeuralNet/Layers/Layer.h>
#include <NeuralNet/Padding.h>
#include <utility>
#include <NeuralNet/Math.h>

namespace NN::Layers
{
    struct Conv2DContext : LayerContext
    {
        Tensor input;
        Tensor preActivation;
        Tensor activationGradient;

        //Accumulated gradients
        Tensor weightGradients;
        Tensor biasGradients;

        // Caches the massive spatial column footprints safely
        Tensor columnsCache;
        Tensor transposedGradCache;
        Tensor columnGradCache;
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

        void Run(const Tensor& input, Tensor& output, LayerContext& context) override;

        void Backpropagate(const Tensor& input, Tensor& output, LayerContext& context) override;

        void ApplyGradients(Optimizer& optimizer, int batchSize) override;

        void ResetGradients(LayerContext& context) override;

        void AccumulateGradients(LayerContext& context) override;
         
    protected:
        void UpdateTransposedWeightsCache();

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