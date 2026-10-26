#pragma once

#include <NeuralNet/Tensor.h>
#include <string>
#include <stdexcept>
#include <filesystem>

namespace NN
{
    class ImageLoader
    {
    public:
        Tensor Load(const std::string& path);
        Tensor Load(const std::filesystem::path& path);
    protected:
    };
}