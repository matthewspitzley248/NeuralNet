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

        for (WorkerContext& worker : workers_) {
            worker.scratchBufferA = Tensor({ maxElements });
            worker.scratchBufferB = Tensor({ maxElements });
        }
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

                //printf("\n[Batch %d] Starting batch with %zu images\n", batchCount, batchSize);

                //Clear the gradients accumulated during previous batches.
                ResetWorkerGradients();

                //Store each image's loss separately so workers never write to the same loss variable.
                alignas(64) vector<float> imageLosses(batchSize, 0.0f);

                vector<WorkerPool::Job> jobs;
                jobs.reserve(batchSize);

                //For each image in the batch add it to the jobs list for the threads or workers to handle
                for (size_t i = 0; i < batchSize; i++)
                {
                    jobs.emplace_back([this, &batch, &imageLosses, i](size_t workerIndex)
                        {
                            const ImageLabel& item = batch.imageLabels[i];
                            WorkerContext& worker = workers_[workerIndex];

                            //printf("[Worker %zu][Image %zu] Starting job. Label index: %d\n", workerIndex, i, item.labelIndex);

                            //Forward pass using this worker's contexts.
                            Tensor prediction = Run(item.image, worker);
                            //printf("[Worker %zu][Image %zu] Forward pass complete. Output size: %zu\n", workerIndex, i, prediction.GetSize());

                            //Calculate the image's loss.
                            imageLosses[i] = loss_.Calculate(prediction, item.labelIndex, lossType_);
                            //printf("[Worker %zu][Image %zu] Loss calculated: %.6f\n", workerIndex, i, imageLosses[i]);

                            //Calculate loss gradient.
                            Tensor gradient = loss_.CalculateGradient(prediction, item.labelIndex, lossType_);
                            //printf("[Worker %zu][Image %zu] Loss gradient calculated. Size: %zu\n", workerIndex, i, gradient.GetSize());

                            //Backpropagate into this worker's gradient buffers.
                            Backpropagation(move(gradient), worker);
                            //printf("[Worker %zu][Image %zu] Backpropagation complete. Job finished.\n", workerIndex, i);
                        });
                }

                //Execute all images using the shared worker pool.
                workerPool_.SubmitJobs(move(jobs));
                workerPool_.WaitForJobs();

                //printf("[Batch %d] Wait for Jobs complete.\n", batchCount);

                //Sum the losses after all jobs have finished.
                // Vectorized Reductions: Faster data consolidation using standard pipelines
                totalLoss += std::accumulate(imageLosses.begin(), imageLosses.end(), 0.0f);
                imageCount += static_cast<int>(batchSize);

                //Combine gradients from all workers, then update weights once using the full batch.
                AccumulateWorkerGradients();
                ApplyGradients(static_cast<int>(batchSize));

                //printf("[Batch %d] Parameter update complete.\n", batchCount);

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

            // Cache Alignment Optimization: Pad and structure arrays to clear CPU cache line dependencies
            alignas(64) vector<float> imageLosses(batchSize, 0.0f);
            alignas(64) vector<int> correct(batchSize, 0);

            vector<WorkerPool::Job> jobs;
            jobs.reserve(batchSize);

            for (size_t i = 0; i < batchSize; i++)
            {
                jobs.emplace_back([this, &batch, &imageLosses, &correct, i](size_t workerIndex)
                    {
                        const ImageLabel& item = batch.imageLabels[i];
                        WorkerContext& worker = workers_[workerIndex];
                        //printf("[Worker %zu][Image %zu] Starting job. Label index: %d\n", workerIndex, i, item.labelIndex);

                        //Forward pass using this worker's contexts.
                        Tensor prediction = Run(item.image, worker);
                        //printf("[Worker %zu][Image %zu] Forward pass complete. Output size: %zu\n", workerIndex, i, prediction.GetSize());

                        //Calculate the image's loss.
                        imageLosses[i] = loss_.Calculate(prediction, item.labelIndex, lossType_);
                        //printf("[Worker %zu][Image %zu] Loss calculated: %.6f\n", workerIndex, i, imageLosses[i]);

                        // Find the predicted class.
                        int predictedLabel = 0;

                        // SIMD Prediction Parsing Optimization:
                        // Extracting raw data lets standard pointers leverage specialized assembly vector scanning
                        const float* predData = prediction.GetData().data();
                        const size_t predSize = prediction.GetSize();

                        if (predSize > 0)
                        {
                            // std::max_element compiles directly to ultra-fast hardware tracking loops
                            const float* maxPtr = std::max_element(predData, predData + predSize);
                            int predictedLabel = static_cast<int>(maxPtr - predData);

                            correct[i] = (predictedLabel == item.labelIndex) ? 1 : 0;
                        }

                        //printf("[Worker %zu][Image %zu] Predicted class: %d, Actual class: %d\n", workerIndex, i, predictedLabel, item.labelIndex);
                        //printf("[Worker %zu][Image %zu] Prediction %s. Job finished.\n", workerIndex, i, correct[i] ? "correct" : "incorrect");
                    });
            }

            // Process the batch using the shared worker pool.
            workerPool_.SubmitJobs(std::move(jobs));
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

    Tensor Sequential::Run(const Tensor& input, WorkerContext& worker)
    {
        // Start by copying the raw input image data into our first scratch buffer
        worker.scratchBufferA.ReshapeAndResize(input.GetShape());
        std::copy(input.GetData().begin(), input.GetData().end(), worker.scratchBufferA.GetData().begin());

        // Pointers to alternate back and forth between the two pre-allocated spaces
        Tensor* currentSource = &worker.scratchBufferA;
        Tensor* currentDest = &worker.scratchBufferB;

        for (size_t i = 0; i < layers_.size(); ++i)
        {
            // Pass the references directly. The layer computes math out of 'currentSource'
            // and overwrites 'currentDest' completely in-place.
            layers_[i]->Run(*currentSource, *currentDest, *worker.layerContexts[i]);

            // Swap the pointers for the next layer (Ping-Pong)
            std::swap(currentSource, currentDest);
        }

        // Return a copy of the final prediction vector (usually very small, e.g., 10 or 100 classes)
        return *currentSource;
    }

    Tensor Sequential::Backpropagation(Tensor gradient, WorkerContext& worker)
    {
        // Copy the incoming loss gradient into our first scratch buffer
        worker.scratchBufferA.ReshapeAndResize(gradient.GetShape());
        std::copy(gradient.GetData().begin(), gradient.GetData().end(), worker.scratchBufferA.GetData().begin());

        Tensor* currentSource = &worker.scratchBufferA;
        Tensor* currentDest = &worker.scratchBufferB;

        // Stream backward from the last layer down to the first
        for (size_t i = layers_.size(); i > 0; i--)
        {
            layers_[i - 1]->Backpropagate(*currentSource, *currentDest, *worker.layerContexts[i - 1]);

            // Swap the pointers for the next backward layer step
            std::swap(currentSource, currentDest);
        }

        // Returns the ultimate gradient for the input image layers
        return *currentSource;
    }

    void Sequential::CreateWorkers(int workerCount)
    {
        //Remove existing workers and recreate with the size
        workers_.clear();
        workers_.resize(workerCount);

        //For each worker give it the layer contexts
        for (WorkerContext& worker : workers_)
        {
            worker.layerContexts.reserve(layers_.size());

            for (const auto& layer : layers_)
            {
                worker.layerContexts.push_back(layer->CreateContext());
            }
        }

        // Start the persistent worker threads.
        workerPool_.Start(workerCount);
    }

    void Sequential::ResetWorkerGradients()
    {
        for (WorkerContext& worker : workers_)
        {
            for (size_t i = 0; i < layers_.size(); ++i)
            {
                layers_[i]->ResetGradients(*worker.layerContexts[i]);
            }
        }
    }

    void Sequential::AccumulateWorkerGradients()
    {
        vector<WorkerPool::Job> jobs;
        jobs.reserve(layers_.size());

        for (size_t i = 0; i < layers_.size(); i++)
        {
            jobs.emplace_back([this, i](size_t workerIndex)
                {
                    // Accumulate gradients from every worker for this layer.
                    for (WorkerContext& worker : workers_)
                    {
                        layers_[i]->AccumulateGradients(*worker.layerContexts[i]);
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
