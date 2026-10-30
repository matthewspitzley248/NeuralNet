#include <NeuralNet/Layers/Flatten.h>

namespace NN::Layers
{
    Flatten::Flatten()
    {
    }

    unique_ptr<LayerContext> Flatten::CreateContext() const
    {
        auto context = make_unique<FlattenContext>();
        return context;
    }

    void Flatten::Initialize(std::mt19937& generator, const std::vector<size_t>& inputShape)
    {
        inputShape_ = inputShape;
    }

    std::vector<size_t> Flatten::GetOutputShape(const std::vector<size_t>& inputShape)
    {
        size_t totalSize = 1;

        for (size_t dimension : inputShape)
        {
            totalSize *= dimension;
        }

        return { totalSize };
    }

    void Flatten::Run(const Tensor& input, Tensor& output, LayerContext& context)
    {
        const size_t totalElements = input.GetSize();

        // 1. Force the destination buffer size to match without hitting the OS heap
        output.ReshapeAndResize({ totalElements });

        // 2. Direct fast block memory copy (or pointer swap if optimized further)
        std::copy_n(input.GetData().data(), totalElements, output.GetData().data());
    }

    void Flatten::Backpropagate(const Tensor& input, Tensor& output, LayerContext& context)
    {
        output.ReshapeAndResize(inputShape_);

        // 3. Fast contiguous raw block copy from the source gradient to the destination buffer.
        // Since flattening doesn't change any mathematical numbers, this is a clean L1/L2 cache stream.
        const size_t totalElements = input.GetSize();
        std::copy_n(input.GetData().data(), totalElements, output.GetData().data());
    }

    void Flatten::ApplyGradients(Optimizer& optimizer, int batchSize)
    {
    }

    void Flatten::ResetGradients(LayerContext& context)
    {
    }

    void Flatten::AccumulateGradients(LayerContext& context)
    {
    }
}