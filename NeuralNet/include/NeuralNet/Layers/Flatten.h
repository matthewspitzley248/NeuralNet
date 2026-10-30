#pragma once

#include <NeuralNet/Layers/Layer.h>

namespace NN::Layers
{
    struct FlattenContext : LayerContext
    {
        Tensor inputShape;
    };

    class Flatten : public Layer
    {
    public:
        Flatten();

        unique_ptr<LayerContext> CreateContext() const override;

        void Initialize(std::mt19937& generator, const std::vector<size_t>& inputShape) override;

        std::vector<size_t> GetOutputShape(const std::vector<size_t>& inputShape) override;

        void Run(const Tensor& input, Tensor& output, LayerContext& context) override;

        void Backpropagate(const Tensor& input, Tensor& output, LayerContext& context) override;

        void ApplyGradients(Optimizer& optimizer, int batchSize) override;

        void ResetGradients(LayerContext& context) override;

        void AccumulateGradients(LayerContext& context) override;

    protected:
        std::vector<size_t> inputShape_;
    };
}