#include <NeuralNet/ImageLoader.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


namespace NN 
{
    Tensor ImageLoader::Load(const std::string& path)
    {
        int width, height, channels;

        // Load image using stb_image
        unsigned char* img = stbi_load(path.c_str(), &width, &height, &channels, 0);

        if (img == nullptr)
            throw std::runtime_error("Error loading image: " + path);

        const size_t imageWidth = static_cast<size_t>(width);
        const size_t imageHeight = static_cast<size_t>(height);
        const size_t imageChannels = static_cast<size_t>(channels);

        Tensor imgTensor({ imageHeight, imageWidth, imageChannels });

        const size_t pixelCount = imageHeight * imageWidth * imageChannels;

        // Extract the raw data destination pointer
        float* destData = imgTensor.GetData().data();

        // Performance Optimization: Hoist the division out of the loop.
        // Multiplying by a reciprocal is significantly faster than standard division.
        const float invScale = 1.0f / 255.0f;

        // The compiler can now fully auto-vectorize this loop using SIMD instructions.
        for (size_t i = 0; i < pixelCount; ++i)
        {
            destData[i] = static_cast<float>(img[i]) * invScale;
        }

        stbi_image_free(img);

        return imgTensor;
    }


	Tensor ImageLoader::Load(const std::filesystem::path& path)
	{
		return Load(path.string());
	}
}
