#include <NeuralNet/Layers/Conv2D.h>

namespace NN::Layers
{
    Conv2D::Conv2D(
        int filters,
        std::pair<int, int> kernelSize,
        std::pair<int, int> strides,
        Padding padding
    )
    {
        filters_ = filters;
        kernelSize_ = kernelSize;
        strides_ = strides;
        padding_ = padding;
    }

    Tensor Conv2D::Run(const Tensor& input)
    {

        return input;
    }
}