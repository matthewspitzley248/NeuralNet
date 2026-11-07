#pragma once

#include <NeuralNet/Layers/Layer.h>

namespace NN::Layers
{
    struct DenseContext : LayerContext
    {
        //Accumulated gradients
        Tensor weightGradients;
        Tensor biasGradients;
        Tensor activationGradient;
    };

    struct DenseImageContext
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

        unique_ptr<LayerContext> CreateContext() const override;

        void Initialize(std::mt19937& generator, const std::vector<size_t>& inputShape) override;

        std::vector<size_t> GetOutputShape(const std::vector<size_t>& inputShape) override;

        void RunBatch(ImageBatch& batch, const size_t batchSize, std::vector<std::unique_ptr<LayerContext>>& contexts, WorkerPool& workerPool) override;

        void Run(Tensor& source, LayerContext& context) override;

        void BackpropagateBatch(vector<Tensor>& batchGradients, const size_t batchSize, std::vector<std::unique_ptr<LayerContext>>& contexts, WorkerPool& workerPool) override;

        void Backpropagate(Tensor& gradient, LayerContext& context) override;

        void ApplyGradients(Optimizer& optimizer, int batchSize) override;

        void ResetGradients(LayerContext& context) override;

        void AccumulateGradients(LayerContext& context) override;

    protected:
        //DenseImageContext CreateBatchContext() const;

        void Run(Tensor& source, LayerContext& gradientContext, DenseImageContext& imageContext);
        void Backpropagate(Tensor& gradient, LayerContext& gradientContext, DenseImageContext& imageContext);

        size_t units_;
        ActivationType activationType_;

        std::vector<DenseImageContext> imageContexts_;

        //Tensor lastInput_;
        //Tensor lastOutput_;
    };
}