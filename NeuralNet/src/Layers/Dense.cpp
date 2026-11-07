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

    void Dense::RunBatch(ImageBatch& batch, const size_t batchSize, std::vector<std::unique_ptr<LayerContext>>& contexts, WorkerPool& workerPool)
    {
        vector<WorkerPool::Job> jobs;
        jobs.reserve(batchSize);

        imageContexts_.clear();
        imageContexts_.resize(batchSize);
        //imageContexts_.reserve(batchSize);

        //For each image in the batch add it to the jobs list for the threads or workers to handle
        for (size_t i = 0; i < batchSize; i++)
        {
            jobs.emplace_back([this, &batch, &contexts, i](size_t workerIndex)
                {
                    ImageLabel& image = batch.imageLabels[i];

                    //Create a new Context for this image and add it to imageContexts
                    imageContexts_[i] = DenseImageContext();

                    //Forward pass using this worker's contexts and image context
                    Run(image.image, *contexts[workerIndex], imageContexts_[i]);
                });
        }

        workerPool.SubmitJobs(std::move(jobs));
        workerPool.WaitForJobs();
    }

    void Dense::Run(Tensor& source, LayerContext& gradientContext, DenseImageContext& imageContext)
    {
        const size_t inputSize = source.GetSize();
        imageContext.input.ReshapeAndResize(source.GetShape());
        std::copy_n(source.GetData().data(), inputSize, imageContext.input.GetData().data());

        // Resize input to hold the output.
        source.ReshapeAndResize({ units_ });

        const float* xData = imageContext.input.GetData().data();
        const float* wData = weights_.GetData().data();
        const float* bData = biases_.GetData().data();
        float* outData = source.GetData().data();

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
        //denseContext.preActivation.ReshapeAndResize(output.GetShape());
        //std::copy_n(outData, units_, denseContext.preActivation.GetData().data());

        // Apply the activation directly onto the destination
        activation_.Apply(activationType_, source);

        imageContext.output.ReshapeAndResize(source.GetShape());
        std::copy_n(outData, units_, imageContext.output.GetData().data());
    }

    void Dense::Run(Tensor& source, LayerContext& context)
    {
        const size_t inputSize = source.GetSize();

        Tensor input(source.GetShape()); 
        std::copy_n(source.GetData().data(),inputSize, input.GetData().data());

        Tensor output({ units_ });

        const float* xData = input.GetData().data();
        const float* wData = weights_.GetData().data();
        const float* bData = biases_.GetData().data();
        float* outData = output.GetData().data();

        for (size_t neuron = 0; neuron < units_; neuron++)
        {
            float sum = 0.0f;
            const size_t weightOffset = neuron * inputSize;

            for (size_t i = 0; i < inputSize; i++)
            {
                sum += xData[i] * wData[weightOffset + i];
            }

            outData[neuron] = sum + bData[neuron];
        }

        activation_.Apply(activationType_, output);

        source = std::move(output);
    }

    void Dense::BackpropagateBatch(vector<Tensor>& batchGradients, const size_t batchSize, std::vector<std::unique_ptr<LayerContext>>& contexts, WorkerPool& workerPool)
    {
        vector<WorkerPool::Job> jobs;
        jobs.reserve(batchSize);

        //For each image in the batch add it to the jobs list for the threads or workers to handle
        for (size_t i = 0; i < batchSize; i++)
        {
            jobs.emplace_back([this, &batchGradients, &contexts, i](size_t workerIndex)
                {
                    //pass using this worker's contexts.
                    Backpropagate(batchGradients[i], *contexts[workerIndex], imageContexts_[i]);
                });
        }

        workerPool.SubmitJobs(std::move(jobs));
        workerPool.WaitForJobs();
    }

    void Dense::Backpropagate(Tensor& gradient, LayerContext& gradientContext, DenseImageContext& imageContext)
    {
        auto& denseContext = dynamic_cast<DenseContext&>(gradientContext);

        // Calculate activation derivatives natively into a cached tracking variable
        // inside context, or let ApplyDerivative write into a pre-allocated field.

        //Tensor activationGradient;

        activation_.ApplyDerivative(activationType_, imageContext.output, gradient, denseContext.activationGradient);

        const size_t inputSize = imageContext.input.GetSize();

        gradient.ReshapeAndResize(imageContext.input.GetShape());

        const float* actGradData = denseContext.activationGradient.GetData().data();
        const float* inputData = imageContext.input.GetData().data();
        const float* wData = weights_.GetData().data();

        float* biasGradData = denseContext.biasGradients.GetData().data();
        float* weightGradData = denseContext.weightGradients.GetData().data();
        float* gradData = gradient.GetData().data();

        std::fill_n(gradData, inputSize, 0.0f);

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
                gradData[i] += wData[weightOffset + i] * neuronGradient;
            }
        }
    }

    void Dense::Backpropagate(Tensor& gradient, LayerContext& context)
    {
        throw std::invalid_argument("This Backpropagate is not implemented yet.");
    }

    void Dense::ApplyGradients(Optimizer& optimizer, int batchSize)
    {

        //printf("\nWeight before: %.8f\n", weights_.GetData()[0]);
        //printf("Weight gradient: %.8f\n", weightGradients_.GetData()[0]);

        optimizer.Apply(weights_, weightGradients_, batchSize, adamContextWeight_);
        optimizer.Apply(biases_, biasGradients_, batchSize, adamContextBias_);

        //printf("Weight after: %.8f\n", weights_.GetData()[0]);

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