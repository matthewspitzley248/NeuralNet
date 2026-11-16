#pragma once

#include <NeuralNet/Layers/Layer.h>

namespace NN::Layers
{
    struct BatchNormalizationWorkerContext : public WorkerContext
    {
        // Accumulated gradients for the entire batch
        Tensor gammaGradients;
        Tensor betaGradients;
    };

    struct BatchNormalizationImageContext : ImageContext
    {
        Tensor normalized;
    };

    class BatchNormalization : public Layer
    {
    public:
        BatchNormalization(float momentum = 0.99f, float epsilon = 0.001f);

        unique_ptr<WorkerContext> CreateWorkerContext() const override;
        unique_ptr<ImageContext> CreateImageContext() const override;

        void Initialize(std::mt19937& generator, const std::vector<size_t>& inputShape) override;

        std::vector<size_t> GetOutputShape(const std::vector<size_t>& inputShape) const override;

        void RunBatch(ImageBatch& batch, const size_t batchSize, std::vector<std::unique_ptr<WorkerContext>>& workerContexts, WorkerPool& workerPool) override;

        void RunBatch(ImageBatch& batch, const size_t batchSize, WorkerPool& workerPool) override;

        void BackpropagateBatch(vector<Tensor>& batchGradients, const size_t batchSize, std::vector<std::unique_ptr<WorkerContext>>& workerContexts, WorkerPool& workerPool) override;

        void Run(Tensor& data) override;

        void ApplyGradients(Optimizer& optimizer, int batchSize) override;

        void ResetGradients(WorkerContext& workerContext) override;

        void AccumulateGradients(WorkerContext& workerContext) override;

    protected:
        void Run(Tensor& data, WorkerContext& workerContext, ImageContext& imageContext) override;
        void Backpropagate(Tensor& gradient, WorkerContext& workerContext, ImageContext& imageContext) override;

        float momentum_;
        float epsilon_;

        size_t channels_;

        //Trainable parameters
        Tensor gamma_;
        Tensor beta_;

        //Running statistics for inference
        Tensor movingMean_;
        Tensor movingVariance_;

        //Accumulated gradients
        Tensor gammaGradients_;
        Tensor betaGradients_;

        //Batch statistics
        Tensor batchMean_;
        Tensor batchVariance_;
        Tensor inverseStdDev_;

        Tensor batchBetaGradients_;
        Tensor batchGammaGradients_;

        size_t totalBatchValues_;

        //Inference optimization tracking metrics
        std::vector<float> inferenceScale_;
        std::vector<float> inferenceOffset_;

        //Adam specific optimizer context configurations mapping to internal parameters
        AdamLayerContext adamContextGamma_;
        AdamLayerContext adamContextBeta_;
    };
}