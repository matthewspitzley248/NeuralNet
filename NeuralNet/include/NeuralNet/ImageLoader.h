#pragma once

#include "Tensor.h"
#include <string>
#include <stdexcept>

namespace NN
{
    class ImageLoader
    {
    public:
        Tensor Load(const std::string& path);
    protected:
    };
}