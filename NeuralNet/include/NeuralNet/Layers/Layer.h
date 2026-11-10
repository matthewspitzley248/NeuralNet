#pragma once

//#include <utility>
#include <NeuralNet/Tensor.h>
#include <NeuralNet/Activation.h>
#include <NeuralNet/Optimizer.h>
#include <random>
#include <NeuralNet/WorkerPool.h>
#include <NeuralNet/ImageDataGenerator.h>

namespace NN::Layers
{
    struct WorkerContext
    {
        virtual ~WorkerContext() = default;
    };

    struct ImageContext
    {
        virtual ~ImageContext() = default;
    };

    class Layer
    {
    public:
        virtual ~Layer() = default;

        virtual unique_ptr<WorkerContext> CreateWorkerContext() const;

        virtual void Initialize(mt19937& generator, const vector<size_t>& inputShape) = 0;
        virtual vector<size_t> GetOutputShape(const vector<size_t>& inputShape) const = 0;
        virtual void RunBatch(ImageBatch& batch, const size_t batchSize, std::vector<std::unique_ptr<WorkerContext>>& workerContexts, WorkerPool& workerPool);
        /// <summary>
        /// Clean pure Run function. Not for use with learning or training
        /// </summary>
        /// <param name="data">input to run, sets it with the output</param>
        virtual void Run(Tensor& data);
        virtual void BackpropagateBatch(vector<Tensor>& batchGradients, const size_t batchSize, std::vector<std::unique_ptr<WorkerContext>>& workerContexts, WorkerPool& workerPool);
        virtual void ApplyGradients(Optimizer& optimizer, int batchSize);
        virtual void ResetGradients(WorkerContext& workerContext);
        virtual void AccumulateGradients(WorkerContext& workerContext);

        virtual void SetTraining(bool training)
        {
            training_ = training;
        }

    protected:
        //for batch methods to run
        virtual void Run(Tensor& data, WorkerContext& workerContext, ImageContext& imageContext);
        virtual void Backpropagate(Tensor& gradient, WorkerContext& workerContext, ImageContext& imageContext) = 0;
        virtual unique_ptr<ImageContext> CreateImageContext() const;

        std::vector<std::unique_ptr<ImageContext>> imageContexts_;


        Tensor weights_;
        Tensor biases_;

        // Accumulated gradients for the current batch
        Tensor weightGradients_;
        Tensor biasGradients_;

        Activation activation_;

        AdamLayerContext adamContextWeight_;
        AdamLayerContext adamContextBias_;

        bool training_ = false;
    };
}