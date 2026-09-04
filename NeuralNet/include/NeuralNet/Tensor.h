#pragma once
#include <vector>


namespace NeuralNet
{
    class Tensor
    {
    public:

        Tensor(std::vector<int> shape);

        float& At(const std::vector<int>& index);

        const std::vector<int>& GetShape() const;

    protected:

        std::vector<float> data_;
        std::vector<int> shape_;
    };
}