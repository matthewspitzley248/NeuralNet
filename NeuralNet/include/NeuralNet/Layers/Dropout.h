#pragma once

#include <NeuralNet/Layers/Layer.h>

namespace NN::Layers
{
    struct DropoutWorkerContext : WorkerContext
    {
        std::mt19937 randomGenerator;
    };

    struct DropoutImageContext : ImageContext
    {
        std::vector<bool> dropoutMask;
    };

    class Dropout : public Layer
    {
    public:
        Dropout(float dropoutRate);

        unique_ptr<WorkerContext> CreateWorkerContext() const override;

        void Initialize(std::mt19937& generator, const std::vector<size_t>& inputShape) override;

        std::vector<size_t> GetOutputShape(const std::vector<size_t>& inputShape) const override;

    protected:
        void Run(Tensor& data, WorkerContext& workerContext, ImageContext& imageContext) override;
        void Backpropagate(Tensor& gradient, WorkerContext& workerContext, ImageContext& imageContext) override;
        unique_ptr<ImageContext> CreateImageContext() const override;

        float dropoutRate_;
        float scale_;

        //used to setup the gens for the worker contexts set in init
        std::mt19937* randomGenerator_ = nullptr;
    };
}