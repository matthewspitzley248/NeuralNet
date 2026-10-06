#pragma once

#include <NeuralNet/Models/Model.h>
#include <memory>

namespace NN::Models
{
    class Sequential : public Model
    {
    public:
        Sequential();
        Sequential(unsigned int seed);
        Sequential(std::mt19937& generator);

        template<typename T, typename... Args>
        void Add(Args&&... args)
        {
            layers_.push_back(std::make_unique<T>(std::forward<Args>(args)...));
        }

        void Compile();

        Tensor Run(const Tensor& input);

    protected:
    };
}