#include <NeuralNet/Layers/Flatten.h>

namespace NN::Layers
{
    Flatten::Flatten()
    {
    }

    Tensor Flatten::Run(const Tensor& input)
    {
        return input;
    }
}