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

    Tensor Flatten::Run(const Tensor& input, LayerContext& context)
    {
        auto& flattenContext = dynamic_cast<FlattenContext&>(context);

        Tensor output = input;
        output.SetShape({ static_cast<size_t>(input.GetSize()) });

        return output;
    }

    Tensor Flatten::Backpropagate(const Tensor& gradient, LayerContext& context)
    {
        auto& flattenContext = dynamic_cast<FlattenContext&>(context);

        //Restores the gradient to the original input shape.
        Tensor inputGradient = gradient;
        inputGradient.SetShape(inputShape_);

        return inputGradient;
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