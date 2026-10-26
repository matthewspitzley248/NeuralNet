#pragma once

#include <NeuralNet/Tensor.h>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace NN
{
    enum class LossType
    {
        CategoricalCrossEntropy
    };

    class Loss
    {
    public:
        float Calculate(const Tensor& prediction, int targetLabel, LossType lossType) const;
        Tensor CalculateGradient(const Tensor& prediction, int targetLabel, LossType lossType) const;

    private:

        float CategoricalCrossEntropy(const Tensor& prediction, int targetLabel) const;
        Tensor CategoricalCrossEntropyGradient(const Tensor& prediction, int targetLabel) const;

    };
}