#pragma once
#include <NeuralNet/Tensor.h>
#include <cmath>
#include <stdexcept>

using namespace std;
namespace NN
{
    enum class ActivationType
    {
        None,
        Relu,
        Sigmoid,
        Tanh,
        Softmax
    };

    class Activation
    {
    public:
        Tensor Apply(ActivationType type, const Tensor& input) const;

        Tensor ApplyDerivative(ActivationType type, const Tensor& input, const Tensor& gradient);

    protected:
        float ScalarActivation(ActivationType type, float in) const;
        Tensor VectorActivation(ActivationType type, const Tensor& input) const;
    };
}