#include <NeuralNet/Activation.h>

namespace NN 
{
    void Activation::Apply(ActivationType type, Tensor& tensor) const
    {
        if (type == ActivationType::Softmax)
        {
            // Compute softmax using the tensor's vector references natively
            VectorActivation(type, tensor);    // tensor = VectorActivation(type, tensor); // Or update VectorActivation to be in-place
            return;
        }

        float* data = tensor.GetData().data();
        const size_t size = tensor.GetSize();

        for (size_t i = 0; i < size; ++i)
        {
            data[i] = ScalarActivation(type, data[i]);
        }
    }

    float Activation::ScalarActivation(ActivationType type, float input) const
    {
        switch (type)
        {
        case ActivationType::None:
            return input;

        case ActivationType::Relu:
            return max(0.0f, input);

        case ActivationType::Sigmoid:
            return 1.0f / (1.0f + exp(-input));

        case ActivationType::Tanh:
            return tanh(input);

        case ActivationType::Softmax:
            throw invalid_argument("Softmax requires a tensor of values.");
        }

        throw invalid_argument("Unknown activation function.");
    }

    void Activation::VectorActivation(ActivationType type, Tensor& input) const
    {
        if (type == ActivationType::Softmax)
        {
            const size_t size = input.GetSize();
            if (size == 0)
            {
                throw std::invalid_argument("Softmax cannot process an empty tensor.");
            }

            Tensor output(input.GetShape());

            float* inData = input.GetData().data();

            // 1. Find Max Value (Vectorized)
            float maxValue = inData[0];
            for (size_t i = 1; i < size; i++)
            {
                maxValue = std::max(maxValue, inData[i]);
            }

            // 2. Compute Exponentials & Sum (Vectorized)
            float sum = 0.0f;
            for (size_t i = 0; i < size; i++)
            {
                const float value = std::exp(inData[i] - maxValue);
                inData[i] = value;
                sum += value;
            }

            // 3. Divide by sum (Vectorized multiplication optimization)
            const float invSum = 1.0f / sum;
            for (size_t i = 0; i < size; i++)
            {
                inData[i] *= invSum;
            }

            return;
        }

        throw std::invalid_argument("Unknown activation function.");
    }

    void Activation::ApplyDerivative(ActivationType type, const Tensor& postActivation, const Tensor& gradient, Tensor& dest) const
    {
        const size_t size = postActivation.GetSize();
        if (size != gradient.GetSize())
        {
            throw std::invalid_argument("Post-activation and gradient must have the same size. "
                "Post-activation size: " + std::to_string(size) +
                ", gradient size: " + std::to_string(gradient.GetSize())
            );
        }

        // ZERO ALLOCATION: Reuse the context's pre-allocated destination tracking block
        dest.ReshapeAndResize(postActivation.GetShape());

        const auto& postActData = postActivation.GetData();
        const auto& gradData = gradient.GetData();
        auto& destData = dest.GetData();

        switch (type)
        {
        case ActivationType::None:
            std::copy_n(gradData.data(), size, destData.data());
            break;

        case ActivationType::Relu:
            for (size_t i = 0; i < size; ++i)
            {
                // Mathematically identical: x > 0 is exactly equivalent to y > 0
                destData[i] = postActData[i] > 0.0f ? gradData[i] : 0.0f;
            }
            break;

        case ActivationType::Sigmoid:
            for (size_t i = 0; i < size; ++i)
            {
                // Faster and memory-clean: derivative using y instead of computing exp(-x)
                const float y = postActData[i];
                destData[i] = gradData[i] * y * (1.0f - y);
            }
            break;

        case ActivationType::Tanh:
            for (size_t i = 0; i < size; ++i)
            {
                // Faster and memory-clean: derivative using 1 - y^2 instead of tanh(x)
                const float y = postActData[i];
                destData[i] = gradData[i] * (1.0f - y * y);
            }
            break;

        case ActivationType::Softmax:
        {
            // SAVED ALLOCATION: In Softmax, postActivation IS the probabilities array.
            // There is no longer a need to run a temporary Tensor probabilities = Apply(...) copy here.
            float dotProduct = 0.0f;
            for (size_t i = 0; i < size; ++i)
            {
                dotProduct += gradData[i] * postActData[i];
            }

            for (size_t i = 0; i < size; ++i)
            {
                destData[i] = postActData[i] * (gradData[i] - dotProduct);
            }
        }
        break;

        default:
            throw std::invalid_argument("Unknown activation function.");
        }
    }
}