#pragma once

//#include <utility>
#include <NeuralNet/Tensor.h>
#include <NeuralNet/Activation.h>
#include <NeuralNet/Optimizer.h>
#include <random>

namespace NN::Layers
{
    struct LayerContext
    {
        virtual ~LayerContext() = default;
    };

    class Layer
    {
    public:
        virtual ~Layer() = default;

        virtual unique_ptr<LayerContext> CreateContext() const = 0;

        virtual void Initialize(mt19937& generator, const vector<size_t>& inputShape) = 0;
        virtual vector<size_t> GetOutputShape(const vector<size_t>& inputShape) = 0;
        virtual void Run(const Tensor& source, Tensor& dest, LayerContext& context) = 0;
        virtual void Backpropagate(const Tensor& input, Tensor& output, LayerContext& context) = 0;
        virtual void ApplyGradients(Optimizer& optimizer, int batchSize) = 0;
        virtual void ResetGradients(LayerContext& context) = 0;
        virtual void AccumulateGradients(LayerContext& context) = 0;

    protected:
        Tensor weights_;
        Tensor biases_;

        // Accumulated gradients for the current batch
        Tensor weightGradients_;
        Tensor biasGradients_;

        Activation activation_;
    };
}