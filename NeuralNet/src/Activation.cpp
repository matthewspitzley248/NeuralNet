#include <NeuralNet/Activation.h>

namespace NN 
{
    Tensor Activation::Apply(ActivationType type, const Tensor& input) const
    {
        //Vector Activations
        if (type == ActivationType::Softmax)
        {
            return VectorActivation(type, input);
        }

        //Scalar Activations
        Tensor output = input;

        for (size_t i = 0; i < input.GetSize(); ++i)
        {
            output.At({ i }) = ScalarActivation(type, input.At({ i }));
        }

        return output;
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
        //Softmax
        if (type == ActivationType::Softmax)
        {
            const size_t size = input.GetSize();

            if (size == 0)
            {
                throw invalid_argument("Softmax cannot process an empty tensor.");
            }

            Tensor output = Tensor({ size });

            //Subtract the maximum to improve numerical stability.
            float maxValue = input.At({ 0 });

            for (size_t i = 1; i < size; ++i)
            {
                maxValue = max(maxValue, input.At({ i }));
            }

            float sum = 0.0f;

            for (size_t i = 0; i < size; ++i)
            {
                const float value = exp(input.At({ i }) - maxValue);

                output.At({ i }) = value;
                sum += value;
            }

            for (size_t i = 0; i < size; ++i)
            {
                output.At({ i }) /= sum;
            }

            return output;
        }

        throw invalid_argument("Unknown activation function.");
    }


    Tensor Activation::ApplyDerivative(ActivationType type, const Tensor& input, const Tensor& gradient)
    {
        if (input.GetSize() != gradient.GetSize())
        {
            throw invalid_argument("Input and gradient must have the same size.");
        }

        Tensor result(input.GetShape());

        switch (type)
        {
        case ActivationType::None:
            return gradient;

        case ActivationType::Relu:
            for (size_t i = 0; i < input.GetSize(); ++i)
            {
                result.At({ i }) = input.At({ i }) > 0.0f ? gradient.At({ i }) : 0.0f;
            }
            return result;

        case ActivationType::Sigmoid:
            for (size_t i = 0; i < input.GetSize(); ++i)
            {
                float x = input.At({  (i) });
                float sigmoid = 1.0f / (1.0f + exp(-x));

                result.At({ i }) = gradient.At({ i }) * sigmoid * (1.0f - sigmoid);
            }
            return result;

        case ActivationType::Tanh:
            for (size_t i = 0; i < input.GetSize(); ++i)
            {
                float tanhValue = tanh(input.At({  (i) }));

                result.At({ i }) = gradient.At({ i }) * (1.0f - tanhValue * tanhValue);
            }
            return result;

        case ActivationType::Softmax:        
            Tensor probabilities = Apply(type, input);

            float dotProduct = 0.0f;

            for (size_t i = 0; i < input.GetSize(); ++i)
            {
                dotProduct += gradient.At({ i }) * probabilities.At({ i });
            }

            for (size_t i = 0; i < input.GetSize(); ++i)
            {
                size_t index = i;

                result.At({ index }) = probabilities.At({ index }) * (gradient.At({ index }) - dotProduct);
            }

            return result;
        }

        throw invalid_argument("Unknown activation function.");
    }
}