#pragma once
#include <filesystem>
#include <NeuralNet/Tensor.h>
#include <NeuralNet/ImageLoader.h>
#include <random>

using namespace std;

namespace NN
{
    enum class ColorMode
    {
        Grayscale,
        RGB
    };

    enum class ClassMode
    {
        Categorical
    };

    struct ImageSettings
    {
        ColorMode colorMode = ColorMode::RGB;
        float rescale = 1.0f / 255.0f;
        bool horizontalFlip = false;
        bool verticalFlip = false;
    };

    struct GeneratorSettings
    {
        filesystem::path directory;
        int batchSize = 1;
        ClassMode classMode = ClassMode::Categorical;
        bool shuffle = false;
    };

    struct ImageLabel
    {
        Tensor image;
        int labelIndex;
    };

    struct ImageBatch
    {
        vector<ImageLabel> imageLabels;
    };

    struct LabeledImagePath
    {
        int labelIndex;
        filesystem::path imagePath;
    };

    class ImageDataGenerator
    {
    public:
        ImageDataGenerator(ImageSettings& imageSettings, GeneratorSettings& generatorSettings, mt19937& randomGenerator);

        void NewEpoch();

        ImageBatch NextBatch();

        int GetBatchSize();

    protected:
        ImageSettings imageSettings_;
        GeneratorSettings generatorSettings_;

        mt19937& randomGenerator_;

        vector<string> labels;
        vector<LabeledImagePath> labeledImages_;
        size_t currentImageIndex_ = 0;

        ImageLoader imgLoader;
    };
}