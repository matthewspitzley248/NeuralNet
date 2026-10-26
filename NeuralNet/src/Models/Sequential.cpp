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

        //initialze the layers, start with the input shape of the image
        vector<size_t> currentShape = inputShape_;

        for (unique_ptr<Layers::Layer>& layer : layers_)
        {
            //Initialize the layer
            layer->Initialize(randomGenerator_, currentShape);
            //Get the output shape of the layer as the input for the next layer
            currentShape = layer->GetOutputShape(currentShape);
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

            //Get a batch until all data is gone through 
            for (ImageBatch batch = trainData.NextBatch(); !batch.imageLabels.empty(); batch = trainData.NextBatch())
            {
                batchCount++;
                const size_t batchSize = batch.imageLabels.size();

                printf("\n[Batch %d] Starting batch with %zu images\n", batchCount, batchSize);

                //Clear the gradients accumulated during previous batches.
                ResetWorkerGradients();

                //Store each image's loss separately so workers never write to the same loss variable.
                vector<float> imageLosses(batchSize, 0.0f);

                vector<WorkerPool::Job> jobs;
                jobs.reserve(batchSize);

                //For each image in the batch add it to the jobs list for the threads or workers to handle
                for (size_t i = 0; i < batchSize; i++)
                {
                    jobs.emplace_back([this, &batch, &imageLosses, i] (size_t workerIndex)
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

                //Sum the losses after all jobs have finished.
                for (float loss : imageLosses)
                {
                    totalLoss += loss;
                }

                imageCount += static_cast<int>(batchSize);

                //Combine gradients from all workers, then update weights once using the full batch.
                AccumulateWorkerGradients();
                ApplyGradients(static_cast<int>(batchSize));

                printf("[Batch %d] Parameter update complete.\n", batchCount);
            }

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

            // Each job writes to its own result slot.
            vector<float> imageLosses(batchSize, 0.0f);
            vector<int> correct(batchSize, 0);

            vector<WorkerPool::Job> jobs;
            jobs.reserve(batchSize);

            for (size_t i = 0; i < batchSize; i++)
            {
                jobs.emplace_back([this, &batch, &imageLosses, &correct, i] (size_t workerIndex)
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

                        for (int j = 1;
                            j < static_cast<int>(prediction.GetSize());
                            ++j)
                        {
                            if (prediction.At(j) > prediction.At(predictedLabel))
                            {
                                predictedLabel = j;
                            }
                        }

                        //printf("[Worker %zu][Image %zu] Predicted class: %d, Actual class: %d\n", workerIndex, i, predictedLabel, item.labelIndex);

                        //record if prediction was right
                        correct[i] = (predictedLabel == item.labelIndex);

                        //printf("[Worker %zu][Image %zu] Prediction %s. Job finished.\n", workerIndex, i, correct[i] ? "correct" : "incorrect");
                    });
            }

            // Process the batch using the shared worker pool.
            workerPool_.SubmitJobs(std::move(jobs));
            workerPool_.WaitForJobs();

            // Aggregate results after all jobs have completed.
            for (size_t i = 0; i < batchSize; i++)
            {
                totalLoss += imageLosses[i];

                if (correct[i])
                {
                    correctPredictions++;
                }
            }

            imageCount += static_cast<int>(batchSize);
        }

        if (imageCount == 0)
        {
            printf("Evaluation skipped: no test images found.\n");
            return 0.0f;
        }

        float averageLoss = totalLoss / static_cast<float>(imageCount);
        float accuracy = static_cast<float>(correctPredictions) / static_cast<float>(imageCount);

        printf(" - Test Loss: %.4f - Test Accuracy: %.2f%%\n", averageLoss, accuracy * 100.0f);

        return accuracy;
    }

    Tensor Sequential::Run(const Tensor& input, WorkerContext& worker)
    {
        Tensor output = input;

        for (size_t i = 0; i < layers_.size(); ++i)
        {
            output = layers_[i]->Run(output, *worker.layerContexts[i]);
        }

        return output;
    }

    Tensor Sequential::Backpropagation(Tensor gradient, WorkerContext& worker)
    {
        //go through the model backwards to train it
        Tensor outputGradient = gradient;

        for (size_t i = layers_.size(); i > 0; i--)
        {
            outputGradient = layers_[i-1]->Backpropagate(outputGradient, *worker.layerContexts[i-1]);
        }

        return outputGradient;
    }

    void Sequential::ApplyGradients(int BatchSize)
    {
        for (const auto& layer : layers_)
        {
            layer->ApplyGradients(optimizer_, BatchSize);
        }
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
        for (WorkerContext& worker : workers_)
        {
            for (size_t i = 0; i < layers_.size(); ++i)
            {
                layers_[i]->AccumulateGradients(*worker.layerContexts[i]);
            }
        }
    }
}