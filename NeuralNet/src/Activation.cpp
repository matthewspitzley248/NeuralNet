#include <NeuralNet/Activation.h>

namespace NN 
{
    Tensor Activation::Apply(ActivationType type, const Tensor& input) const
    {
        // Vector Activations
        if (type == ActivationType::Softmax)
        {
            return VectorActivation(type, input);
        }

        // Scalar Activations optimization: Copy structure first
        Tensor output = input;

        const size_t size = input.GetSize();
        const float* inData = input.GetData().data();
        float* outData = output.GetData().data();

        // High-velocity element-wise transform loop
        for (size_t i = 0; i < size; ++i)
        {
            outData[i] = ScalarActivation(type, inData[i]);
        }

        return output;
    }

    void Activation::ApplyInPlace(ActivationType type, Tensor& tensor) const
    {
        if (type == ActivationType::Softmax)
        {
            // Compute softmax using the tensor's vector references natively
            tensor = VectorActivation(type, tensor); // Or update VectorActivation to be in-place
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

    Tensor Activation::VectorActivation(ActivationType type, const Tensor& input) const
    {
        if (type == ActivationType::Softmax)
        {
            const size_t size = input.GetSize();
            if (size == 0)
            {
                throw std::invalid_argument("Softmax cannot process an empty tensor.");
            }

            Tensor output(input.GetShape());

            const float* inData = input.GetData().data();
            float* outData = output.GetData().data();

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
                outData[i] = value;
                sum += value;
            }

            // 3. Divide by sum (Vectorized multiplication optimization)
            const float invSum = 1.0f / sum;
            for (size_t i = 0; i < size; i++)
            {
                outData[i] *= invSum;
            }

            return output;
        }

        throw std::invalid_argument("Unknown activation function.");
    }


    Tensor Activation::ApplyDerivative(ActivationType type, const Tensor& input, const Tensor& gradient)
    {
        const size_t size = input.GetSize();
        if (size != gradient.GetSize())
        {
            std::string inputStrShape = "";
            for (auto s : input.GetShape()) inputStrShape += std::to_string(s) + "x";

            std::string gradStrShape = "";
            for (auto s : gradient.GetShape()) gradStrShape += std::to_string(s) + "x";

            throw std::invalid_argument(
                "Size mismatch! Activation input shape: " + inputStrShape + " (Total: " + std::to_string(size) +
                ") but incoming gradient shape: " + gradStrShape + " (Total: " + std::to_string(gradient.GetSize()) + ")."
            );
        }

        Tensor result(input.GetShape());

        const float* inData = input.GetData().data();
        const float* gradData = gradient.GetData().data();
        float* resData = result.GetData().data();

        switch (type)
        {
        case ActivationType::None:
            // High-speed block copy replacement for None activation pass
            std::copy_n(gradData, size, resData);
            return result;

        case ActivationType::Relu:
            for (size_t i = 0; i < size; ++i)
            {
                resData[i] = inData[i] > 0.0f ? gradData[i] : 0.0f;
            }
            return result;

        case ActivationType::Sigmoid:
            for (size_t i = 0; i < size; ++i)
            {
                const float sigmoid = 1.0f / (1.0f + std::exp(-inData[i]));
                resData[i] = gradData[i] * sigmoid * (1.0f - sigmoid);
            }
            return result;

        case ActivationType::Tanh:
            for (size_t i = 0; i < size; ++i)
            {
                const float tanhValue = std::tanh(inData[i]);
                resData[i] = gradData[i] * (1.0f - tanhValue * tanhValue);
            }
            return result;

        case ActivationType::Softmax:
        {
            Tensor probabilities = Apply(type, input);
            const float* probData = probabilities.GetData().data();

            float dotProduct = 0.0f;
            for (size_t i = 0; i < size; ++i)
            {
                dotProduct += gradData[i] * probData[i];
            }

            for (size_t i = 0; i < size; ++i)
            {
                resData[i] = probData[i] * (gradData[i] - dotProduct);
            }

            return result;
        }
        }

        throw std::invalid_argument("Unknown activation function.");
    }

    void Activation::ApplyDerivativeInPlace(ActivationType type, const Tensor& preActivation, const Tensor& gradient, Tensor& dest) const
    {
        const size_t size = preActivation.GetSize();
        if (size != gradient.GetSize())
        {
            throw std::invalid_argument("Pre-activation and gradient must have the same size.");
        }

        //ZERO ALLOCATION: Resize the pre-allocated context destination tensor in-place
        dest.ReshapeAndResize(preActivation.GetShape());

        // Extract safe vector references to ensure performance and memory safety
        const auto& preActData = preActivation.GetData();
        const auto& gradData = gradient.GetData();
        auto& destData = dest.GetData();

        switch (type)
        {
        case ActivationType::None:
            // High-speed block copy replacement for None activation pass
            std::copy_n(gradData.data(), size, destData.data());
            break;

        case ActivationType::Relu:
            // Pure sequential element-wise derivative: easily auto-vectorized by the compiler
            for (size_t i = 0; i < size; ++i)
            {
                destData[i] = preActData[i] > 0.0f ? gradData[i] : 0.0f;
            }
            break;

        case ActivationType::Sigmoid:
            for (size_t i = 0; i < size; ++i)
            {
                const float sigmoid = 1.0f / (1.0f + std::exp(-preActData[i]));
                destData[i] = gradData[i] * sigmoid * (1.0f - sigmoid);
            }
            break;

        case ActivationType::Tanh:
            for (size_t i = 0; i < size; ++i)
            {
                const float tanhValue = std::tanh(preActData[i]);
                destData[i] = gradData[i] * (1.0f - tanhValue * tanhValue);
            }
            break;

        case ActivationType::Softmax:
        {
            // Softmax derivative depends on the activated probabilities.
            // We reuse the existing ApplyInPlace/Apply code structure safely.
            Tensor probabilities = Apply(type, preActivation);
            const auto& probData = probabilities.GetData();

            float dotProduct = 0.0f;
            for (size_t i = 0; i < size; ++i)
            {
                dotProduct += gradData[i] * probData[i];
            }

            for (size_t i = 0; i < size; ++i)
            {
                destData[i] = probData[i] * (gradData[i] - dotProduct);
            }
        }
        break;

        default:
            throw std::invalid_argument("Unknown activation function.");
        }
    }
}