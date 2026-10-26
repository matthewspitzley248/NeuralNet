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

    void Dense::Initialize(std::mt19937& generator, const std::vector<size_t>& inputShape)
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

        //Initialize weights with random values.
        std::uniform_real_distribution<float> distribution(-0.1f, 0.1f);
        for (size_t neuron = 0; neuron < units_; neuron++)
        {
            biases_.At({neuron}) = 0.0f;

            for (size_t size = 0; size < inputSize; size++)
            {
                weights_.At({neuron, size}) = distribution(generator);
            }
        }
    }

    std::vector<size_t> Dense::GetOutputShape(const std::vector<size_t>& inputShape)
    {
        return {units_};
    }

    Tensor Dense::Run(const Tensor& input, LayerContext& context)
    {
        auto& denseContext = dynamic_cast<DenseContext&>(context);

        denseContext.input = input;
        Tensor output = Tensor({units_});

        const size_t inputSize = input.GetSize();

        //For each Neuron
        for (size_t neuron = 0; neuron < units_; neuron++)
        {
            const size_t weightOffset = neuron * inputSize;

            //Sum bias + in0*w0 + in1*w1 +...
            float sum = biases_.AtFlat(neuron);

            //Go through all the inputs and multiply the weights for that neuron with it
            for (size_t i = 0; i < input.GetSize(); i++)
            {
                sum += input.AtFlat(i) * weights_.AtFlat(weightOffset + i);
            }
            //Add the sum to the output for the neuron
            output.AtFlat(neuron) = sum;
        }

        denseContext.preActivation = output;

        //Apply Activation to the output
        return activation_.Apply(activationType_, output);
    }


    Tensor Dense::Backpropagate(const Tensor& gradient, LayerContext& context)
    {
        auto& denseContext = dynamic_cast<DenseContext&>(context);

        //Calculate the gradient before the activation.
        Tensor activationGradient = activation_.ApplyDerivative(activationType_, denseContext.preActivation, gradient);

        const size_t inputSize = denseContext.input.GetSize();

        //Calculate the gradient for the previous layer.
        Tensor inputGradient(denseContext.input.GetShape());

        //Accumulate gradients for weights and biases.
        for (size_t neuron = 0; neuron < units_; neuron++)
        {

            const float neuronGradient = activationGradient.AtFlat(neuron);
            const size_t weightOffset = neuron * inputSize;

            //Accumulate bias gradient.
            denseContext.biasGradients.AtFlat(neuron) += neuronGradient;

            //Accumulate weight gradients.
            for (size_t i = 0; i < inputSize; i++)
            {
                //Accumulate weight gradient
                denseContext.weightGradients.AtFlat(weightOffset + i) += neuronGradient * denseContext.input.AtFlat(i);

                //Calculate gradient for the previous layer
                inputGradient.AtFlat(i) += weights_.AtFlat(weightOffset + i) * neuronGradient;
            }
        }

        return inputGradient;
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

        denseContext.weightGradients = Tensor(weights_.GetShape());
        denseContext.biasGradients = Tensor(biases_.GetShape());

        weightGradients_ = Tensor(weights_.GetShape());
        biasGradients_ = Tensor(biases_.GetShape());
    }

    void Dense::AccumulateGradients(LayerContext& context)
    {
        auto& denseContext = dynamic_cast<DenseContext&>(context);

        for (size_t i = 0; i < weightGradients_.GetSize(); ++i)
        {
            weightGradients_.AtFlat(i) +=  denseContext.weightGradients.AtFlat(i);
        }

        for (size_t i = 0; i < biasGradients_.GetSize(); ++i)
        {
            biasGradients_.AtFlat(i) += denseContext.biasGradients.AtFlat(i);
        }
    }
}