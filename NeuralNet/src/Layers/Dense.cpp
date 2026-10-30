#include <NeuralNet/Layers/Dense.h>

namespace NN::Layers
{
    Dense::Dense(int units, ActivationType activationType)
        : units_(units),
        activationType_(activationType)
    {
    }

    unique_ptr<LayerContext> Dense::CreateContext() const
    {
        auto context = make_unique<DenseContext>();

        context->weightGradients = Tensor(weights_.GetShape());
        context->biasGradients = Tensor(biases_.GetShape());

        return context;
    }

    void Dense::Initialize(mt19937& generator, const vector<size_t>& inputShape)
    {
        //Get the total number of input features.
        size_t inputSize = 1;

        for (int dimension : inputShape)
        {
            inputSize *= static_cast<size_t>(dimension);
        }

        //Create weights and biases.
        weights_ = Tensor({ units_, inputSize });
        biases_ = Tensor({ units_ });

        //Get raw pointers for fast sequential access
        float* wData = weights_.GetData().data();
        float* bData = biases_.GetData().data();

        //Initialize weights and biases with random values.
        uniform_real_distribution<float> distribution(-0.1f, 0.1f);

        for (size_t neuron = 0; neuron < units_; neuron++)
        {
            bData[neuron] = 0.0f;

            const size_t weightOffset = neuron * inputSize;
            for (size_t size = 0; size < inputSize; size++)
            {
                wData[weightOffset + size] = distribution(generator);
            }
        }

        weightGradients_ = Tensor(weights_.GetShape());
        biasGradients_ = Tensor(biases_.GetShape());
    }

    std::vector<size_t> Dense::GetOutputShape(const std::vector<size_t>& inputShape)
    {
        return {units_};
    }

    void Dense::Run(const Tensor& input, Tensor& output, LayerContext& context)
    {
        auto& denseContext = dynamic_cast<DenseContext&>(context);

        const size_t inputSize = input.GetSize();
        denseContext.input.ReshapeAndResize(input.GetShape());
        std::copy_n(input.GetData().data(), inputSize, denseContext.input.GetData().data());

        // Prepare the pre-allocated scratch buffer destination view
        output.ReshapeAndResize({ static_cast<size_t>(units_) });

        const float* xData = input.GetData().data();
        const float* wData = weights_.GetData().data();
        const float* bData = biases_.GetData().data();
        float* outData = output.GetData().data();

        // Pure sequential execution loop: easily auto-vectorized by the compiler
        for (size_t neuron = 0; neuron < units_; neuron++)
        {
            float sum = 0.0f;
            const size_t weightOffset = neuron * inputSize;

            for (size_t size = 0; size < inputSize; size++)
            {
                sum += xData[size] * wData[weightOffset + size];
            }

            outData[neuron] = sum + bData[neuron];
        }

        // Capture the pre-activation state snapshot for the backward pass safely
        denseContext.preActivation.ReshapeAndResize(output.GetShape());
        std::copy_n(outData, units_, denseContext.preActivation.GetData().data());

        // Apply the activation directly onto the destination scratch buffer
        activation_.ApplyInPlace(activationType_, output);
    }

    void Dense::Backpropagate(const Tensor& input, Tensor& output, LayerContext& context)
    {
        auto& denseContext = dynamic_cast<DenseContext&>(context);

        // Calculate activation derivatives natively into a cached tracking variable
        // inside context, or let ApplyDerivative write into a pre-allocated field.
        activation_.ApplyDerivativeInPlace(activationType_, denseContext.preActivation, input, denseContext.activationGradient);

        const size_t inputSize = denseContext.input.GetSize();

        output.ReshapeAndResize(denseContext.input.GetShape());

        const float* actGradData = denseContext.activationGradient.GetData().data();
        const float* inputData = denseContext.input.GetData().data();
        const float* wData = weights_.GetData().data();

        float* biasGradData = denseContext.biasGradients.GetData().data();
        float* weightGradData = denseContext.weightGradients.GetData().data();
        float* inGradData = output.GetData().data();

        std::fill_n(inGradData, inputSize, 0.0f);

        // Completely lock-free execution path. 
        // The compiler can unroll and pipelining this smoothly now.
        for (size_t neuron = 0; neuron < units_; neuron++)
        {
            const float neuronGradient = actGradData[neuron];
            const size_t weightOffset = neuron * inputSize;

            biasGradData[neuron] += neuronGradient;

            for (size_t i = 0; i < inputSize; i++)
            {
                weightGradData[weightOffset + i] += neuronGradient * inputData[i];
                inGradData[i] += wData[weightOffset + i] * neuronGradient;
            }
        }
    }

    void Dense::ApplyGradients(Optimizer& optimizer, int batchSize)
    {
        optimizer.Apply(weights_, weightGradients_, batchSize);
        optimizer.Apply(biases_, biasGradients_, batchSize);

        //ResetGradients();
    }


    void Dense::ResetGradients(LayerContext& context)
    {
        auto& denseContext = dynamic_cast<DenseContext&>(context);

        // Reuse existing allocated memory blocks and clear elements to 0.0f in-place
        std::fill(denseContext.weightGradients.GetData().begin(), denseContext.weightGradients.GetData().end(), 0.0f);
        std::fill(denseContext.biasGradients.GetData().begin(), denseContext.biasGradients.GetData().end(), 0.0f);

        std::fill(weightGradients_.GetData().begin(), weightGradients_.GetData().end(), 0.0f);
        std::fill(biasGradients_.GetData().begin(), biasGradients_.GetData().end(), 0.0f);
    }

    void Dense::AccumulateGradients(LayerContext& context)
    {
        auto& denseContext = dynamic_cast<DenseContext&>(context);

        // Extract raw pointers for contiguous data processing
        float* wGradData = weightGradients_.GetData().data();
        const float* ctxWGradData = denseContext.weightGradients.GetData().data();

        float* bGradData = biasGradients_.GetData().data();
        const float* ctxBGradData = denseContext.biasGradients.GetData().data();

        const size_t wSize = weightGradients_.GetSize();
        const size_t bSize = biasGradients_.GetSize();

        // Fast, localized flat memory accumulation loops
        for (size_t i = 0; i < wSize; i++)
        {
            wGradData[i] += ctxWGradData[i];
        }

        for (size_t i = 0; i < bSize; i++)
        {
            bGradData[i] += ctxBGradData[i];
        }
    }
}