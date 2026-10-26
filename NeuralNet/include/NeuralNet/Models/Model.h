#pragma once

#include <NeuralNet/Layers/Layer.h>
#include <memory>
#include <random>
#include <utility>
#include <thread>
//#include <algorithm>
#include <NeuralNet/ImageDataGenerator.h>
#include <NeuralNet/Loss.h>
#include <NeuralNet/WorkerPool.h>

namespace NN::Models
{
    enum class Metric
    {
        Accuracy
    };

    class Model
    {
    public:
        Model(std::vector<size_t> inputShape);

        Model(std::vector<size_t> inputShape, unsigned int seed);

        Model(std::vector<size_t> inputShape_, std::mt19937& generator);

        //virtual Tensor Run(const Tensor& input) = 0;

        //virtual void Initialize(std::mt19937& generator) = 0;

    protected:
        std::vector<std::unique_ptr<Layers::Layer>> layers_;
        std::vector<size_t> inputShape_;

        std::mt19937 randomGenerator_;

        LossType lossType_;
        Loss loss_;
        Optimizer optimizer_;
        std::vector<Metric> metrics_;
    };
}