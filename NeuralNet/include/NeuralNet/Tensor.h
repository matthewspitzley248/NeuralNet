#pragma once
#include <vector>
#include <stdexcept>


namespace NN
{
    class Tensor
    {
    public:
        Tensor() = default;
        Tensor(std::vector<size_t> shape);

        float& At(const std::vector<size_t>& index);
        //float& At(const int y, const int x, const int z);
        float& At(size_t index);
        float& AtFlat(size_t index);

        const float& At(const std::vector<size_t>& index) const;
        //const float& At(const int y, const int x, const int z) const;
        const float& At(size_t index) const;
        const float& AtFlat(size_t index) const;

        const std::vector<size_t>& GetShape() const;
        const size_t& GetSize() const;

        void SetShape(const std::vector<size_t>& newShape);

    protected:

        std::vector<float> data_;
        std::vector<size_t> shape_;
    };
}