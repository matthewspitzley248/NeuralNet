#include <NeuralNet/Models/Sequential.h>

namespace NN::Models
{
    Sequential::Sequential(vector<size_t> inputShape, float learningRate)
        : Model(inputShape),
        learningRate_(learningRate)
    {
    }

    Sequential::Sequential(vector<size_t> inputShape, unsigned int seed, float learningRate)
        : Model(inputShape, seed),
        learningRate_(learningRate)
    {
    }

    Sequential::Sequential(vector<size_t> inputShape, mt19937& generator, float learningRate)
        : Model(inputShape, generator),
        learningRate_(learningRate)
    {
    }

    void Sequential::Compile(LossType lossType, OptimizerType optimizerType, vector<Metric> metrics, int threadCount)
    {
        optimizer_ = Optimizer(optimizerType, learningRate_);
        lossType_ = lossType;
        metrics_ = move(metrics);

        size_t maxElements = 0;
        //initialze the layers, start with the input shape of the image
        vector<size_t> currentShape = inputShape_;

        for (unique_ptr<Layers::Layer>& layer : layers_)
        {
            //Initialize the layer
            layer->Initialize(randomGenerator_, currentShape);
            //Get the output shape of the layer as the input for the next layer
            currentShape = layer->GetOutputShape(currentShape);

            size_t layerSize = 1;
            for (size_t dim : currentShape) layerSize *= dim;
            if (layerSize > maxElements) maxElements = layerSize;
        }

        CreateWorkers(threadCount);
    }

    void Sequential::Fit(int epochs, ImageDataGenerator& trainData, ImageDataGenerator& testData)
    {
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            printf("\n========== Epoch %d/%d ==========\n", epoch + 1, epochs);


            //Start the newEpoch of batchimages
            trainData.NewEpoch();

            float totalLoss = 0.0f;
            int imageCount = 0;
            int batchCount = 0;

            const size_t totalBatches = trainData.GetBatchCount();
            double totalBatchSeconds = 0.0;
            size_t completedBatches = 0;

            //Get a batch until all data is gone through 
            for (ImageBatch batch = trainData.NextBatch(); !batch.imageLabels.empty(); batch = trainData.NextBatch())
            {
                const auto batchStart = Clock::now();
                batchCount++;
                const size_t batchSize = batch.imageLabels.size();

                //Clear the gradients accumulated during previous batches.
                ResetWorkerGradients();

                //each layer handles its own run pass. It then waits for the layer to finish what it wants before starting the next layer
                // output is overwritten into the ImageBatch batch image Tensor
                //Later will want to change this to have less stops and waits
                //Will want to have it so it gos through all layers until it hits a layer that wants a stop (batch normalization) and loop that 
                for (size_t i = 0; i < layers_.size(); i++)
                {
                    layers_[i]->RunBatch(batch, batchSize, workerContexts_[i], workerPool_);

                    workerPool_.WaitForJobs();
                }




                //Store each image's loss separately so workers never write to the same loss variable.
                alignas(64) vector<float> imageLosses(batchSize, 0.0f);

                //Now get the gradients for the images to send to backprop
                vector<Tensor> batchGradients(batchSize);
                vector<WorkerPool::Job> jobs;
                jobs.reserve(batchSize);

                for (size_t i = 0; i < batchSize; i++)
                {
                    jobs.emplace_back([this, &batch, &imageLosses, &batchGradients, i](size_t workerIndex)
                        {
                            //This now contains the prediction and the label for what the correct would be
                            const ImageLabel& item = batch.imageLabels[i];

                            imageLosses[i] = loss_.Calculate(item.image, item.labelIndex, lossType_);

                            batchGradients[i] = loss_.CalculateGradient(item.image, item.labelIndex, lossType_);
                        });
                }

                workerPool_.SubmitJobs(move(jobs));
                workerPool_.WaitForJobs();


                //Now each layer handles its own Backpropagate same as run
                for (size_t i = layers_.size(); i > 0; i--)
                {
                    layers_[i - 1]->BackpropagateBatch(batchGradients, batchSize, workerContexts_[i - 1], workerPool_);

                    workerPool_.WaitForJobs();
                }

                //Sum the losses after all jobs have finished.
                // Vectorized Reductions: Faster data consolidation using standard pipelines
                totalLoss += std::accumulate(imageLosses.begin(), imageLosses.end(), 0.0f);
                imageCount += static_cast<int>(batchSize);

                //Combine gradients from all workers, then update weights once using the full batch.
                AccumulateWorkerGradients();
                ApplyGradients(static_cast<int>(batchSize));

                const auto batchEnd = Clock::now();

                const double batchSeconds = chrono::duration<double>(batchEnd - batchStart).count();

                totalBatchSeconds += batchSeconds;
                completedBatches++;

                const double averageBatchSeconds = totalBatchSeconds / completedBatches;

                PrintProgress(completedBatches, totalBatches, averageBatchSeconds);
            }
            printf("\n");

            float averageLoss = imageCount > 0 ? totalLoss / imageCount : 0.0f;

            printf("\n[Epoch %d] Training complete.\n", epoch + 1);
            printf("[Epoch %d] Batches: %d, Images: %d\n", epoch + 1, batchCount, imageCount);
            printf("Epoch %d/%d - Loss: %.4f\n", epoch + 1, epochs, averageLoss);

            printf("[Epoch %d] Starting evaluation...\n", epoch + 1);
            Evaluate(testData);
            printf("[Epoch %d] Evaluation complete.\n", epoch + 1);
        }

        printf("\n[Fit] Training finished.\n");
    }

    float Sequential::Evaluate(ImageDataGenerator& testData)
    {
        testData.NewEpoch();

        float totalLoss = 0.0f;
        int correctPredictions = 0;
        int imageCount = 0;

        for (ImageBatch batch = testData.NextBatch(); !batch.imageLabels.empty(); batch = testData.NextBatch())
        {
            const size_t batchSize = batch.imageLabels.size();

            for (size_t i = 0; i < layers_.size(); i++)
            {
                layers_[i]->RunBatch(batch, batchSize, workerContexts_[i], workerPool_);

                workerPool_.WaitForJobs();
            }


            // Cache Alignment Optimization: Pad and structure arrays to clear CPU cache line dependencies
            alignas(64) vector<float> imageLosses(batchSize, 0.0f);
            alignas(64) vector<int> correct(batchSize, 0);

            vector<WorkerPool::Job> jobs;
            jobs.reserve(batchSize);

            for (size_t i = 0; i < batchSize; i++)
            {
                jobs.emplace_back([this, &batch, &imageLosses, &correct, i](size_t workerIndex)
                    {
                        //This now contains the prediction and the label for what the correct would be
                        const ImageLabel& item = batch.imageLabels[i];

                        //Calculate the image's loss.
                        imageLosses[i] = loss_.Calculate(item.image, item.labelIndex, lossType_);

                        // Find the predicted class.
                        int predictedLabel = 0;

                        // SIMD Prediction Parsing Optimization:
                        // Extracting raw data lets standard pointers leverage specialized assembly vector scanning
                        const float* predData = item.image.GetData().data();
                        const size_t predSize = item.image.GetSize();

                        if (predSize > 0)
                        {
                            // std::max_element compiles directly to ultra-fast hardware tracking loops
                            const float* maxPtr = std::max_element(predData, predData + predSize);
                            predictedLabel = static_cast<int>(maxPtr - predData);

                            correct[i] = (predictedLabel == item.labelIndex) ? 1 : 0;
                        }
                    });
            }

            workerPool_.SubmitJobs(move(jobs));
            workerPool_.WaitForJobs();

            // Aggregate results after all jobs have completed.
            // Vectorized Batch Consolidation
            totalLoss += std::accumulate(imageLosses.begin(), imageLosses.end(), 0.0f);
            correctPredictions += std::accumulate(correct.begin(), correct.end(), 0);
            imageCount += static_cast<int>(batchSize);
        }

        float averageLoss = imageCount > 0 ? totalLoss / imageCount : 0.0f;
        float accuracy = imageCount > 0 ? static_cast<float>(correctPredictions) / imageCount : 0.0f;

        printf("[Evaluation] Accuracy: %.4f - Loss: %.4f\n", accuracy, averageLoss);

        return accuracy;
    }

    void Sequential::CreateWorkers(int workerCount)
    {
        //Remove existing workers and recreate with the size
        workerContexts_.clear();
        workerContexts_.resize(layers_.size());

        for (size_t i = 0; i < layers_.size(); i++)
        {
            auto& contexts = workerContexts_[i];
            contexts.reserve(workerCount);

            for (int j = 0; j < workerCount; j++)
            {
                contexts.push_back(layers_[i]->CreateContext());
            }
        }

        // Start the persistent worker threads.
        workerPool_.Start(workerCount);
    }

    void Sequential::ResetWorkerGradients()
    {
        for (size_t layerIndex = 0; layerIndex < layers_.size(); ++layerIndex)
        {
            for (auto& context : workerContexts_[layerIndex])
            {
                layers_[layerIndex]->ResetGradients(*context);
            }
        }
    }

    void Sequential::AccumulateWorkerGradients()
    {
        vector<WorkerPool::Job> jobs;
        jobs.reserve(layers_.size());

        for (size_t i = 0; i < layers_.size(); ++i)
        {
            jobs.emplace_back([this, i](size_t workerIndex)
                {
                    for (auto& context : workerContexts_[i])
                    {
                        layers_[i]->AccumulateGradients(*context);
                    }
                });
        }

        workerPool_.SubmitJobs(std::move(jobs));
        workerPool_.WaitForJobs();
    }

    void Sequential::ApplyGradients(int batchSize)
    {
        /*vector<WorkerPool::Job> jobs;
        jobs.reserve(layers_.size());

        for (size_t i = 0; i < layers_.size(); i++)
        {
            jobs.emplace_back([this, i, batchSize](size_t workerIndex)
                {
                    layers_[i]->ApplyGradients(optimizer_, batchSize);
                });
        }

        workerPool_.SubmitJobs(std::move(jobs));
        workerPool_.WaitForJobs();*/

        // Increment the Adam time-step counter once per full batch step
        optimizer_.IncrementTimeStep();

        for (size_t i = 0; i < layers_.size(); i++)
        {
            layers_[i]->ApplyGradients(optimizer_, batchSize);
        }
    }




    void Sequential::PrintProgress(size_t currentBatch, size_t totalBatches, double averageBatchSeconds)
    {
        constexpr size_t barWidth = 25;
        const double progress = totalBatches > 0 ? static_cast<double>(currentBatch) / totalBatches : 0.0;
        const size_t filled = static_cast<size_t>(progress * barWidth);
        const double remainingSeconds = averageBatchSeconds * (totalBatches - currentBatch);

        // Standard I/O Optimization: Construct a static buffer string in place
        // to minimize system printing interruptions.
        char bar[barWidth + 1];
        for (size_t i = 0; i < barWidth; i++) {
            bar[i] = (i < filled) ? '#' : '-';
        }
        bar[barWidth] = '\0';

        printf("\rBatch %zu/%zu [%s] %3.0f%% | Avg: %.2fs/batch | ETA: ",
            currentBatch, totalBatches, bar, progress * 100.0, averageBatchSeconds);

        const size_t remaining = static_cast<size_t>(remainingSeconds);
        if (remaining >= 3600)
        {
            printf("%02zu:%02zu:%02zu", remaining / 3600, (remaining % 3600) / 60, remaining % 60);
        }
        else if (remaining <= 0)
        {
            printf("00:00");
        }
        else
        {
            printf("%02zu:%02zu", remaining / 60, remaining % 60);
        }

        printf("   ");
        fflush(stdout);
    }
}
