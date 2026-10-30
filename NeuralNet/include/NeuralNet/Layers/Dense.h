#pragma once

#include <NeuralNet/Layers/Layer.h>

namespace NN::Layers
{
    struct DenseContext : LayerContext
    {
        //Forward pass info
        Tensor input;
        Tensor preActivation;
        Tensor activationGradient;

        //Accumulated gradients
        Tensor weightGradients;
        Tensor biasGradients;
    };

    class Dense : public Layer
    {
    public:
        /// <summary>
        /// Dense layer based on Keras
        /// </summary>
        /// <param name="units">Dimensionality of the output space.</param>
        /// <param name="activation">Activation function to use.</param>
        Dense(int units, ActivationType activationType = ActivationType::None);

        unique_ptr<LayerContext> CreateContext() const override;

        void Initialize(std::mt19937& generator, const std::vector<size_t>& inputShape) override;

        std::vector<size_t> GetOutputShape(const std::vector<size_t>& inputShape) override;

        void Run(const Tensor& input, Tensor& output, LayerContext& context) override;

        void Backpropagate(const Tensor& input, Tensor& output, LayerContext& context) override;

        void ApplyGradients(Optimizer& optimizer, int batchSize) override;

        void ResetGradients(LayerContext& context) override;

        void AccumulateGradients(LayerContext& context) override;

    protected:
        size_t units_;
        ActivationType activationType_;

        //Tensor lastInput_;
        //Tensor lastOutput_;
    };
}