#include <NeuralNet/ImageLoader.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


namespace NN 
{
	/// <summary>
	/// Takes in a path to an image and loads it into a Tensor
	/// </summary>
	/// <param name="path">string - location of the image to load into a tensor</param>
	/// <returns>Tensor - the image stored as a Tensor</returns>
	Tensor ImageLoader::Load(const std::string& path)
	{
		int width, height, channels;
		unsigned char* img = stbi_load(path.c_str(), &width, &height, &channels, 0);
		if (img == NULL) {
			throw std::runtime_error("Error loading image: " + path);
		}

		//printf("Loaded image with a width of %dpx, a height of %dpx and %d channels\n", width, height, channels);

		const size_t imageWidth = static_cast<size_t>(width);
		const size_t imageHeight = static_cast<size_t>(height);
		const size_t imageChannels = static_cast<size_t>(channels);

		Tensor imgTensor = Tensor({ imageHeight, imageWidth, imageChannels });

        //loop through the img list and put it into the tensor
        for (size_t y = 0; y < height; y++)
        {
            for (size_t x = 0; x < width; x++)
            {
                for (size_t c = 0; c < channels; c++)
                {
                    int imgIndex = (y * width * channels) + (x * channels) + c;

                    //normalize image to 0-1
					imgTensor.At({ y, x, c }) = static_cast<float>(img[imgIndex]) / 255.0f;
                }
            }
        }
        stbi_image_free(img);

		return imgTensor;
	}


	Tensor ImageLoader::Load(const std::filesystem::path& path)
	{
		return Load(path.string());
	}
}
