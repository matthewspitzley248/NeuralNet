#pragma once

#include <NeuralNet/Layers/Layer.h>

namespace NN::Layers
{
    struct DenseWorkerContext : WorkerContext
    {
        //Accumulated gradients
        Tensor weightGradients;
        Tensor biasGradients;
        Tensor activationGradient;
    };

    struct DenseImageContext : ImageContext
    {
        //Forward pass info
        Tensor input;
        Tensor output;

        //Tensor activationGradient;
    };

    class Dense : public Layer
    {
    public:
        /// <summary>
        /// Dense layer based on Keras
        /// </summary>
        /// <param name="units">Dimensionality of the output space.</param>
        /// <param name="activation">Activation function to use.</param>
        Dense(int units, ActivationType activationType = ActivationType::None);

        unique_ptr<WorkerContext> CreateWorkerContext() const override;

        void Initialize(std::mt19937& generator, const std::vector<size_t>& inputShape) override;

        std::vector<size_t> GetOutputShape(const std::vector<size_t>& inputShape) const override;

        void Run(Tensor& data) override;

        void ApplyGradients(Optimizer& optimizer, int batchSize) override;

        void ResetGradients(WorkerContext& workerContext) override;

        void AccumulateGradients(WorkerContext& workerContext) override;

    protected:
        void Run(Tensor& data, WorkerContext& workerContext, ImageContext& imageContext) override;
        void Backpropagate(Tensor& gradient, WorkerContext& workerContext, ImageContext& imageContext) override;
        unique_ptr<ImageContext> CreateImageContext() const override;

        size_t units_;
        ActivationType activationType_;

        //Tensor lastInput_;
        //Tensor lastOutput_;
    };
}