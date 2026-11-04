#pragma once
#include <NeuralNet/Tensor.h>
#include <cmath>
#include <stdexcept>
#include <string>

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
        void Apply(ActivationType type, Tensor& tensor) const;
        void ApplyDerivative(ActivationType type, const Tensor& postActivation, const Tensor& gradient, Tensor& dest) const;

    protected:
        float ScalarActivation(ActivationType type, float in) const;
        void VectorActivation(ActivationType type, Tensor& input) const;
    };
}