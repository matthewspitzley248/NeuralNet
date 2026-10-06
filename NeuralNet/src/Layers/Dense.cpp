#include <NeuralNet/Layers/Dense.h>

namespace NN::Layers
{
    Dense::Dense(int units, Activation activation)
        : units_(units),
        activation_(activation)
    {
    }

    Tensor Dense::Run(const Tensor& input)
    {
        size_t inputSize = input.GetSize();
        //set the weights and biases if not set yet
        if (weights_.GetSize() == 0)
        {
            weights_ = Tensor({ units_, static_cast<int>(inputSize) });
            biases_ = Tensor({ units_ });


        }
        return input;
    }
}