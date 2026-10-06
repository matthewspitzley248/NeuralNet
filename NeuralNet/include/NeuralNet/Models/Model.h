#pragma once

#include <NeuralNet/Layers/Layer.h>
#include <memory>
#include <random>

namespace NN::Models
{
    class Model
    {
    public:
        Model() = default;

        Model(unsigned int seed);

        Model(std::mt19937& generator);

        virtual Tensor Run(const Tensor& input) = 0;

        //virtual void Initialize(std::mt19937& generator) = 0;

    protected:
        std::vector<std::unique_ptr<Layers::Layer>> layers_;

        std::mt19937 randomGenerator_;
    };
}