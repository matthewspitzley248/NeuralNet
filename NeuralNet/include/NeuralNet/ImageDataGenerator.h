#pragma once
#pragma once
#include <filesystem>
#include <vector>
#include <string>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <exception>
#include <atomic>
#include <utility>
#include <random>

#include <NeuralNet/Tensor.h>
#include <NeuralNet/ImageLoader.h>
#include <NeuralNet/WorkerPool.h>

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
        ImageDataGenerator(ImageSettings& imageSettings, GeneratorSettings& generatorSettings, mt19937& randomGenerator, size_t workerCount =  2);
        ~ImageDataGenerator();

        void NewEpoch();

        ImageBatch NextBatch();

        int GetBatchSize();

        int GetBatchCount();

        int GetLabelSize();

    protected:

        ImageSettings imageSettings_;
        GeneratorSettings generatorSettings_;

        mt19937& randomGenerator_;

        vector<string> labels_;
        vector<LabeledImagePath> labeledImages_;
        atomic<size_t> currentImageIndex_{ 0 };

        ImageLoader imgLoader_;

        //Threaded pool for preloading and sequential loading images
        WorkerPool workerPool_;


        //Background batch preloading
        void BatchLoaderLoop();
        ImageBatch LoadBatch();

        thread batchLoaderThread_;
        exception_ptr batchLoaderException_;

        queue<ImageBatch> preloadedBatches_;
        mutex batchQueueMutex_;
        condition_variable batchQueueCondition_;

        size_t maxPreloadedBatches_ = 1;

        bool stopBatchLoader_ = false;
        bool epochActive_ = false;
        bool epochProductionFinished_ = true;
        bool loadingBatch_ = false;
    };
}