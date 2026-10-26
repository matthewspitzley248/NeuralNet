#pragma once

#include <iostream>
#include <NeuralNet/NeuralNet.h>
#include <filesystem>
using namespace NN::Models;

using namespace NN;
using namespace NN::Layers;
using namespace std;



/// <summary>
/// 
/// </summary>
/// <param name="datasetPath"></param>
/// <param name="trainRatio"></param>
/// <param name="randomGenerator"></param>
static void SeperateDataset(const filesystem::path& datasetPath, const filesystem::path& outputPath, float trainRatio, mt19937& randomGenerator)
{
    //Make sure train ratio is proper
    if (trainRatio <= 0.0f || trainRatio >= 1.0f)
        throw invalid_argument("Train ratio must be between 0 and 1.");

    //Delete and recreate the test and train dataset

    filesystem::remove_all("Dataset/train");
    filesystem::remove_all("Dataset/test");
    filesystem::create_directory("Dataset/train");
    filesystem::create_directory("Dataset/test");


    //for each label in the dataset directory 
    for (const auto& folder : filesystem::directory_iterator(datasetPath))
    {
        if (!folder.is_directory())
            continue;

        const string label = folder.path().filename().string();

        const filesystem::path testDir = outputPath / "test" / label;
        const filesystem::path trainDir = outputPath / "train" / label;

        //create the label directory in test and train
        filesystem::create_directory(testDir);
        filesystem::create_directory(trainDir);

        //Collect the files for the label
        vector<filesystem::path> images;

        for (const auto& file : filesystem::directory_iterator(folder.path()))
        {
            if (file.is_regular_file())
                images.push_back(file.path());
        }

        //Randomize the image order 
        shuffle(images.begin(), images.end(), randomGenerator);

        // Calculate the number of training images 
        const size_t trainCount = static_cast<size_t>(images.size() * trainRatio);

        // Copy each image into its assigned split 
        for (size_t i = 0; i < images.size(); ++i)
        {
            const filesystem::path& destinationDir = (i < trainCount) ? trainDir : testDir;
            filesystem::copy_file(images[i], destinationDir / images[i].filename(), filesystem::copy_options::overwrite_existing);
        }
    }
}



static int testWorkerPool()
{
    NN::WorkerPool pool;

    constexpr size_t workerCount = 8;
    constexpr size_t jobCount = 30;

    std::mutex printMutex;

    printf("Starting worker pool...\n");

    pool.Start(workerCount);

    printf("Created %zu workers.\n\n", pool.GetWorkerCount());

    std::vector<NN::WorkerPool::Job> jobs;

    for (size_t i = 0; i < jobCount; ++i)
    {
        jobs.emplace_back([i, &printMutex](size_t workerIndex)
            {
                {
                    std::lock_guard<std::mutex> lock(printMutex);

                    printf("Worker %zu started job %zu\n",
                        workerIndex, i);
                }

                // Simulate a job that takes time to complete.
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(100));

                {
                    std::lock_guard<std::mutex> lock(printMutex);

                    printf("Worker %zu finished job %zu\n",
                        workerIndex, i);
                }
            });
    }

    printf("\nSubmitting %zu jobs...\n\n", jobs.size());

    pool.SubmitJobs(std::move(jobs));
    pool.WaitForJobs();

    printf("\nAll jobs completed.\n");

    pool.Stop();

    printf("Worker pool stopped.\n");

    return 0;
}





int main()
{
    ImageLoader imgLoader = ImageLoader();

    //cout << filesystem::current_path() << '\n';

    Tensor image = imgLoader.Load(string("Dataset/Zero_full (1).jpg"));

    vector<size_t> inputShape = image.GetShape();

    printf("Image shape: ");
    for (size_t dimension : inputShape)
    {
        printf("%zu ", dimension);
    }
    printf("\n");

    //testWorkerPool();



    //NeuralNet model = NeuralNet();

    mt19937 randomGenerator(10);


    //Seperate the data
    //SeperateDataset("Dataset/HandwrittenDigits0-9", "Dataset", 0.7f, randomGenerator);






    //Creating a datagen for train and test
    ImageSettings imgSettings;

    // Training settings
    GeneratorSettings trainGenSettings;
    trainGenSettings.directory = "Dataset/train";
    trainGenSettings.batchSize = 32;
    trainGenSettings.classMode = ClassMode::Categorical;
    trainGenSettings.shuffle = true;

    // Test settings
    GeneratorSettings testGenSettings;
    testGenSettings.directory = "Dataset/test";
    testGenSettings.batchSize = 64;
    testGenSettings.classMode = ClassMode::Categorical;
    testGenSettings.shuffle = false;

    ImageDataGenerator testImageGen = ImageDataGenerator(imgSettings, testGenSettings, randomGenerator);
    ImageDataGenerator trainImageGen = ImageDataGenerator(imgSettings, trainGenSettings, randomGenerator);









    //model.Sequential();
    Sequential model = Sequential(inputShape, randomGenerator);

    model.Add<Flatten>();
    model.Add<Dense>(200, ActivationType::Relu);
    model.Add<Dense>(100, ActivationType::Relu);
    //output layer
    model.Add<Dense>(10, ActivationType::Softmax);

    model.Compile(LossType::CategoricalCrossEntropy, OptimizerType::SGD, { Metric::Accuracy }, 16);

    //Need to get some sort of early stop and reduce learning rate for the model

    //model fit to train it
    model.Fit(10, trainImageGen, testImageGen);
}