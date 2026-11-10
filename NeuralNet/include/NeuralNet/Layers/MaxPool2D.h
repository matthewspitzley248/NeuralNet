#pragma once

#include <algorithm>
#include <stdexcept>

#include <NeuralNet/Layers/Layer.h>
#include <NeuralNet/Padding.h>

namespace NN::Layers
{
    struct MaxPool2DImageContext : ImageContext
    {
        //Maps each output position to the absolute flat source index of the maximum value
        std::vector<size_t> maxIndices;
    };


    class MaxPool2D : public Layer
    {
    public:
        MaxPool2D(std::pair<size_t, size_t> poolSize = { 2, 2 }, std::pair<size_t, size_t> strides = { 0, 0 }, Padding padding = Padding::Valid);

        void Initialize(std::mt19937& generator, const std::vector<size_t>& inputShape) override;

        std::vector<size_t> GetOutputShape(const std::vector<size_t>& inputShape) const override;

        void Run(Tensor& data) override;

    protected:
        void Run(Tensor& data, WorkerContext& workerContext, ImageContext& imageContext) override;
        void Backpropagate(Tensor& gradient, WorkerContext& workerContext, ImageContext& imageContext) override;
        unique_ptr<ImageContext> CreateImageContext() const override;


        std::pair<size_t, size_t> poolSize_; 
        std::pair<size_t, size_t> strides_; 
        Padding padding_;

        std::vector<size_t> inputShape_;
    };
}