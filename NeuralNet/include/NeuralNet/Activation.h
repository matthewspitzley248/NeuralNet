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
        Tensor Apply(ActivationType type, const Tensor& input) const;
        void ApplyInPlace(ActivationType type, Tensor& tensor) const;

        Tensor ApplyDerivative(ActivationType type, const Tensor& input, const Tensor& gradient);
        void ApplyDerivativeInPlace(
            ActivationType type,
            const Tensor& preActivation,
            const Tensor& gradient,
            Tensor& dest
        ) const;

    protected:
        float ScalarActivation(ActivationType type, float in) const;
        Tensor VectorActivation(ActivationType type, const Tensor& input) const;
    };
}