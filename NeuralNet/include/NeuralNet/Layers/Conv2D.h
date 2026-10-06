#pragma once

#include <NeuralNet/Layers/Layer.h>
#include <utility>

namespace NN::Layers
{
    enum class Padding
    {
        Valid,
        Same
    };

    class Conv2D : public Layer
    {
    public:
        Conv2D(
            int filters,
            std::pair<int, int> kernalSize,
            std::pair<int, int> strides = {1, 1},
            Padding padding = Padding::Valid
        );

        Tensor Run(const Tensor& input) override;
         
    protected:
        int filters_;
        std::pair<int, int> kernelSize_;
        std::pair<int, int> strides_;
        Padding padding_;
    };
}