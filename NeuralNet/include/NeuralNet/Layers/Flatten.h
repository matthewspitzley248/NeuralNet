#pragma once

#include <NeuralNet/Layers/Layer.h>

namespace NN::Layers
{
    class Flatten : public Layer
    {
    public:
        Flatten();

        Tensor Run(const Tensor& input) override;
    protected:

    };
}