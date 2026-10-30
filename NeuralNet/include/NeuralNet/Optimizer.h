#pragma once
#include <NeuralNet/Tensor.h>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

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
        Optimizer() = default;
        Optimizer(OptimizerType type, float learningRate = 0.001f);

        void Apply(Tensor& parameters, const Tensor& gradients, int batchSize);

        void UpdateLearningRate(float learningRate);

        void IncrementTimeStep() { t_++; }

    protected:
        OptimizerType type_;
        float learningRate_;
        
        // Adam Hyperparameters
        float beta1_ = 0.9f;
        float beta2_ = 0.999f;
        float epsilon_ = 1e-10f;

        // Uses the parameter's raw data pointer address as a unique key 
        // to map to its corresponding m and v state buffers.
        std::unordered_map<void*, Tensor> adamM_;
        std::unordered_map<void*, Tensor> adamV_;

        // Tracks the time-step (number of batches processed) for bias corrections
        int t_ = 0;
    };
}