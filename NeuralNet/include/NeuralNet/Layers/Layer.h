#pragma once

//#include <utility>
#include <NeuralNet/Tensor.h>

namespace NN::Layers
{
    enum class Activation
    {
        None,
        Relu,
        Softmax
    };

    class Layer
    {
    public:
        virtual ~Layer() = default;
        virtual Tensor Run(const Tensor& input) = 0;

    protected:
        Tensor weights_;
        Tensor biases_;
    };
}