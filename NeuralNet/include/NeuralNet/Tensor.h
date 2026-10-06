#pragma once
#include <vector>
#include <stdexcept>


namespace NN
{
    class Tensor
    {
    public:
        Tensor() = default;
        Tensor(std::vector<int> shape);

        float& At(const std::vector<int>& index);
        float& At(const int y, const int x, const int z);

        const std::vector<int>& GetShape() const;
        const size_t& GetSize() const;

    protected:

        std::vector<float> data_;
        std::vector<int> shape_;
    };
}