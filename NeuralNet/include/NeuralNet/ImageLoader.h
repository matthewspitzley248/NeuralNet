#pragma once

#include <NeuralNet/Tensor.h>
#include <string>
#include <stdexcept>
#include <filesystem>
#include <vector>

namespace NN
{
    class ImageLoader
    {
    public:
        Tensor Load(const std::string& path);
        Tensor Load(const std::filesystem::path& path);

        std::vector<size_t> GetImageSize(const std::string& path);
        std::vector<size_t> GetImageSize(const std::filesystem::path& path);
    protected:
    };
}