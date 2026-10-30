#pragma once
#include <vector>
#include <stdexcept>
#include <numeric>


namespace NN
{
    class Tensor
    {
    public:
        Tensor() = default;
        Tensor(std::vector<size_t> shape);

        Tensor(Tensor&& other) noexcept
            : shape_(std::move(other.shape_)),
            data_(std::move(other.data_))
        {
        }

        //Handles cases like: tensorA = std::move(tensorB);
        Tensor& operator=(Tensor&& other) noexcept;

        // --- COPY SEMANTICS (Explicitly kept or defaulted) ---

        // 4. Copy Constructor
        Tensor(const Tensor& other) = default;

        // 5. Copy Assignment Operator
        Tensor& operator=(const Tensor& other) = default;

        // Destructor
        ~Tensor() = default;

        //float& At(const std::vector<size_t>& index);
        //float& At(const int y, const int x, const int z);
        //float& At(size_t index);
        float& AtFlat(size_t index);

        //const float& At(const std::vector<size_t>& index) const;
        //const float& At(const int y, const int x, const int z) const;
        //const float& At(size_t index) const;
        const float& AtFlat(size_t index) const;

        const std::vector<size_t>& GetShape() const;
        const size_t& GetSize() const;

        std::vector<float>& GetData();
        const std::vector<float>& GetData() const;


        void SetShape(const std::vector<size_t>& newShape);

        void ReshapeAndResize(const std::vector<size_t>& newShape);

    protected:

        std::vector<float> data_;
        std::vector<size_t> shape_;
    };
}