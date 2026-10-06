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
			exit(-1);
		}
		printf("Loaded image with a width of %dpx, a height of %dpx and %d channels\n", width, height, channels);

		Tensor imgTensor = Tensor({ height, width, channels });

        //loop through the img list and put it into the tensor
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                for (int c = 0; c < channels; c++)
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
}
