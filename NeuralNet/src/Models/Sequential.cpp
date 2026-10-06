#include <NeuralNet/Models/Sequential.h>

namespace NN::Models
{
    Sequential::Sequential()
    {
    }

    Sequential::Sequential(unsigned int seed)
        : Model(seed)
    {
    }

    Sequential::Sequential(std::mt19937& generator)
        : Model(generator)
    {
    }

    Tensor Sequential::Run(const Tensor& input)
    {
        Tensor output = input;

        for (const auto& layer : layers_)
        {
            output = layer->Run(output);
        }

        return output;
    }
}