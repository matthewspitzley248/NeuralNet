#pragma once

#include "Layers.h"
#include <utility>

namespace NeuralNet::Layers
{
    class Conv2D : public Layers
    {
    public:
        Conv2D(
            int filers,
            std::pair<int, int> kernalSize,
            std::pair<int, int> strides = {1, 1}
        );
    protected:

    };
}