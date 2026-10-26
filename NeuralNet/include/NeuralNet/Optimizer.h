#pragma once
#include <NeuralNet/Tensor.h>
#include <cmath>
#include <stdexcept>

using namespace std;
namespace NN
{
    enum class OptimizerType
    {
        SGD,
        Adam
    };

    class Optimizer
    {
    public:
        explicit Optimizer(OptimizerType type, float learningRate = 0.001f);

        void Apply(Tensor& parameters, const Tensor& gradients, int batchSize);

        void UpdateLearningRate(float learningRate);

    protected:
        OptimizerType type_;
        float learningRate_;
        
    };
}