#include <NeuralNet/Layers/Conv2D.h>

namespace NN::Layers
{
    Conv2D::Conv2D(
        size_t filters,
        pair<size_t, size_t> kernelSize,
        pair<size_t, size_t> strides,
        ActivationType activationType,
        Padding padding
    )
        : filters_(filters),
        kernelSize_(kernelSize),
        strides_(strides),
        padding_(padding),
        activationType_(activationType)
    {
    }

    unique_ptr<WorkerContext> Conv2D::CreateWorkerContext() const
    {
        auto context = make_unique<Conv2DContext>();

        context->weightGradients = Tensor(weights_.GetShape());
        context->biasGradients = Tensor(biases_.GetShape());

        return context;
    }

    unique_ptr<ImageContext> Conv2D::CreateImageContext() const
    {
        return std::make_unique<Conv2DImageContext>();
    }

    void Conv2D::Initialize(mt19937& generator, const vector<size_t>& inputShape)
    {
        if (inputShape.size() != 3)
            throw invalid_argument("Conv2D expects a 3D input shape.");

        inputChannels_ = inputShape[2];

        // Validate the input and convolution configuration.
        GetOutputShape(inputShape);

        const size_t kernelHeight = static_cast<size_t>(kernelSize_.first);
        const size_t kernelWidth = static_cast<size_t>(kernelSize_.second);

        // Create weights and biases.
        weights_ = Tensor({static_cast<size_t>(filters_), kernelHeight, kernelWidth, inputChannels_});
        biases_ = Tensor({ static_cast<size_t>(filters_) });

        // Get raw data pointers for initialization updates
        float* wData = weights_.GetData().data();
        float* bData = biases_.GetData().data();

        uniform_real_distribution<float> distribution(-0.1f, 0.1f);
        const size_t filterVolume = kernelHeight * kernelWidth * inputChannels_;

        for (size_t filter = 0; filter < static_cast<size_t>(filters_); filter++)
        {
            bData[filter] = 0.0f;

            const size_t filterOffset = filter * filterVolume;
            for (size_t i = 0; i < filterVolume; ++i)
            {
                wData[filterOffset + i] = distribution(generator);
            }
        }

        weightGradients_ = Tensor(weights_.GetShape());
        biasGradients_ = Tensor(biases_.GetShape());

        // Pre-allocate and populate the cache view once at start
        const size_t numFilters = filters_;
        const size_t patchSize = kernelHeight * kernelWidth * inputChannels_;

        transposedWeights_ = Tensor({ patchSize, numFilters });
        UpdateTransposedWeightsCache();
    }

    vector<size_t> Conv2D::GetOutputShape(const vector<size_t>& inputShape) const
    {
        if (inputShape.size() != 3)
            throw invalid_argument("Conv2D expects input shape (height, width, channels).");

        const size_t inputHeight = inputShape[0];
        const size_t inputWidth = inputShape[1];

        const size_t kernelHeight = static_cast<size_t>(kernelSize_.first);
        const size_t kernelWidth = static_cast<size_t>(kernelSize_.second);

        const size_t strideHeight = static_cast<size_t>(strides_.first);
        const size_t strideWidth = static_cast<size_t>(strides_.second);

        size_t outputHeight;
        size_t outputWidth;

        if (padding_ == Padding::Same)
        {
            outputHeight = (inputHeight + strideHeight - 1) / strideHeight;
            outputWidth = (inputWidth + strideWidth - 1) / strideWidth;
        }
        else
        {
            if (inputHeight < kernelHeight || inputWidth < kernelWidth)
                throw invalid_argument("Conv2D kernel is larger than the input with Valid padding.");

            outputHeight = (inputHeight - kernelHeight) / strideHeight + 1;
            outputWidth = (inputWidth - kernelWidth) / strideWidth + 1;
        }

        return { outputHeight, outputWidth, static_cast<size_t>(filters_) };
    }

    void Conv2D::Run(Tensor& data)
    {
        const auto& inputShape = data.GetShape();

        const size_t inputHeight = inputShape[0];
        const size_t inputWidth = inputShape[1];

        const size_t kernelHeight = static_cast<size_t>(kernelSize_.first);
        const size_t kernelWidth = static_cast<size_t>(kernelSize_.second);

        const size_t strideHeight = static_cast<size_t>(strides_.first);
        const size_t strideWidth = static_cast<size_t>(strides_.second);

        size_t outputHeight = (padding_ == Padding::Same) ?
            ((inputHeight + strideHeight - 1) / strideHeight) : ((inputHeight - kernelHeight) / strideHeight + 1);
        size_t outputWidth = (padding_ == Padding::Same) ?
            ((inputWidth + strideWidth - 1) / strideWidth) : ((inputWidth - kernelWidth) / strideWidth + 1);

        const size_t numFilters = filters_;

        //Continuous spatial image flattening (Pure sequential speed)
        Tensor columnsCache;
        NN::Math::Im2Col(data, columnsCache, kernelSize_, strides_, padding_);

        //Compute via GEMM using pre-transposed weight matrix
        NN::Math::MatrixMultiplication(columnsCache, transposedWeights_, data);

        //Accumulate Biases sequentially (Clean L1/L2 cache streaming)
        float* outData = data.GetData().data();
        const float* bData = biases_.GetData().data();
        const size_t totalElements = data.GetSize();

        for (size_t i = 0; i < totalElements; ++i)
        {
            outData[i] += bData[i % numFilters];
        }

        data.SetShape({ outputHeight, outputWidth, numFilters });

        //Apply the activation function in-place directly on the destination scratch tensor
        activation_.Apply(activationType_, data);
    }

    void Conv2D::Run(Tensor& data, WorkerContext& workerContext, ImageContext& imageContext)
    {
        auto& convWorkerContext = dynamic_cast<Conv2DContext&>(workerContext);
        auto& convImageContext = dynamic_cast<Conv2DImageContext&>(imageContext);

        const auto& inputShape = data.GetShape();

        const size_t inputHeight = inputShape[0];
        const size_t inputWidth = inputShape[1];

        const size_t kernelHeight = static_cast<size_t>(kernelSize_.first);
        const size_t kernelWidth = static_cast<size_t>(kernelSize_.second);

        const size_t strideHeight = static_cast<size_t>(strides_.first);
        const size_t strideWidth = static_cast<size_t>(strides_.second);

        size_t outputHeight = (padding_ == Padding::Same) ?
            ((inputHeight + strideHeight - 1) / strideHeight) : ((inputHeight - kernelHeight) / strideHeight + 1);
        size_t outputWidth = (padding_ == Padding::Same) ?
            ((inputWidth + strideWidth - 1) / strideWidth) : ((inputWidth - kernelWidth) / strideWidth + 1);

        const size_t numFilters = filters_;

        // 1. Move input into context
        convImageContext.input.ReshapeAndResize(inputShape);
        std::copy_n(data.GetData().data(), data.GetSize(), convImageContext.input.GetData().data());

        // 2. Continuous spatial image flattening (Pure sequential speed)
        NN::Math::Im2Col(convImageContext.input, convWorkerContext.columnsCache, kernelSize_, strides_, padding_);

        // 3. Compute via GEMM using our perfectly cached pre-transposed weight matrix
        NN::Math::MatrixMultiplication(convWorkerContext.columnsCache, transposedWeights_, data);

        // 4. Accumulate Biases sequentially (Clean L1/L2 cache streaming)
        float* outData = data.GetData().data();
        const float* bData = biases_.GetData().data();
        const size_t totalElements = data.GetSize();

        for (size_t i = 0; i < totalElements; ++i)
        {
            outData[i] += bData[i % numFilters];
        }

        data.SetShape({ outputHeight, outputWidth, numFilters });

        // 6. Snapshot the pre-activation state safely for the backward pass
        /*convContext.preActivation.ReshapeAndResize(output.GetShape());
        std::copy_n(outData, totalElements, convContext.preActivation.GetData().data());*/

        // 7. Apply the activation function in-place directly on the destination scratch tensor
        activation_.Apply(activationType_, data);

        convImageContext.output.ReshapeAndResize(data.GetShape());
        std::copy_n(outData, totalElements, convImageContext.output.GetData().data());
    }

    void Conv2D::Backpropagate(Tensor& gradient, WorkerContext& workerContext, ImageContext& imageContext)
    {
        auto& convWorkerContext = dynamic_cast<Conv2DContext&>(workerContext);
        auto& convImageContext = dynamic_cast<Conv2DImageContext&>(imageContext);

        // 1. Calculate the activation derivative.
        activation_.ApplyDerivative(activationType_, convImageContext.output, gradient, convWorkerContext.activationGradient);

        const auto& inputShape = convImageContext.input.GetShape();
        const auto outputShape = convImageContext.output.GetShape();

        const size_t inputHeight = inputShape[0];
        const size_t inputWidth = inputShape[1];
        const size_t inputChannels = inputShape[2];

        const size_t outputHeight = outputShape[0];
        const size_t outputWidth = outputShape[1];
        const size_t numFilters = static_cast<size_t>(filters_);

        const size_t kernelHeight = kernelSize_.first;
        const size_t kernelWidth = kernelSize_.second;

        const size_t outputPositions = outputHeight * outputWidth;
        const size_t patchSize = kernelHeight * kernelWidth * inputChannels;

        // 2. Recreate the input-patch matrix using the parallelized math implementation.
        NN::Math::Im2Col(convImageContext.input, convWorkerContext.columnsCache, kernelSize_, strides_, padding_);

        // Extract raw pointers for data-heavy operations
        const float* outputGradData = convWorkerContext.activationGradient.GetData().data();

        float* biasGradData = convWorkerContext.biasGradients.GetData().data();
        float* weightGradData = convWorkerContext.weightGradients.GetData().data();

        // 3. Calculate bias gradients sequentiallytransposedGradCache
        for (size_t position = 0; position < outputPositions; position++)
        {
            const size_t outputOffset = position * numFilters;
            for (size_t filter = 0; filter < numFilters; filter++)
            {
                biasGradData[filter] += outputGradData[outputOffset + filter];
            }
        }

        // 5. Calculate weight gradients via Transposed Gradient Matrix Multiply.
        convWorkerContext.transposedGradCache.ReshapeAndResize({ numFilters, outputPositions });
        float* transposedGradData = convWorkerContext.transposedGradCache.GetData().data();


        //#pragma omp parallel for collapse(2) schedule(static)
        for (size_t position = 0; position < outputPositions; position++)
        {
            for (size_t filter = 0; filter < numFilters; filter++)
            {
                transposedGradData[filter * outputPositions + position] = outputGradData[position * numFilters + filter];
            }
        }

        Tensor weightGradient({ numFilters, patchSize });
        NN::Math::MatrixMultiplication(convWorkerContext.transposedGradCache, convWorkerContext.columnsCache, weightGradient);


        // Accumulate weight gradient map natively
        float* wGradDest = convWorkerContext.weightGradients.GetData().data();
        const float* calculatedWeightGradData = weightGradient.GetData().data();
        const size_t wGradSize = weightGradient.GetSize();


        for (size_t i = 0; i < wGradSize; i++)
        {
            wGradDest[i] += calculatedWeightGradData[i];
        }

        // 6. Calculate gradients for every flattened input patch.
        Tensor flatWeightsView({ numFilters, patchSize });
        flatWeightsView.GetData() = weights_.GetData();

        convWorkerContext.activationGradient.SetShape({ outputPositions, numFilters });

        // Transpose weight tracking multiplication to retrieve the input patch gradients
        // flatActivationGradient shape: {outputPositions, numFilters}
        // flatWeights shape:            {numFilters, patchSize}
        // columnGradient shape:         {outputPositions, patchSize}
        NN::Math::MatrixMultiplication(convWorkerContext.activationGradient, flatWeightsView, convWorkerContext.columnGradCache);

        convWorkerContext.activationGradient.SetShape({ outputHeight, outputWidth, numFilters });

        // 7. Scatter patch gradients back into the original HWC input using the optimized Col2Im helper.
        // We pass the recycled 'gradient' buffer directly into the helper function, enabling it to steal
        // the memory context and perform an allocation-free array reduction.
        NN::Math::Col2Im(convWorkerContext.columnGradCache, gradient, inputShape, kernelSize_, strides_, padding_);
    }

    void Conv2D::ApplyGradients(Optimizer& optimizer, int batchSize)
    {
        optimizer.Apply(weights_, weightGradients_, batchSize, adamContextWeight_);
        optimizer.Apply(biases_, biasGradients_, batchSize, adamContextBias_);

        UpdateTransposedWeightsCache();
    }

    void Conv2D::ResetGradients(WorkerContext& workerContext)
    {
        auto& convContext = dynamic_cast<Conv2DContext&>(workerContext);

        // Reuse existing allocated memory blocks and clear elements to 0.0f in-place
        std::fill(convContext.weightGradients.GetData().begin(), convContext.weightGradients.GetData().end(), 0.0f);
        std::fill(convContext.biasGradients.GetData().begin(), convContext.biasGradients.GetData().end(), 0.0f);

        std::fill(weightGradients_.GetData().begin(), weightGradients_.GetData().end(), 0.0f);
        std::fill(biasGradients_.GetData().begin(), biasGradients_.GetData().end(), 0.0f);
    }

    void Conv2D::AccumulateGradients(WorkerContext& workerContext)
    {
        auto& convContext = dynamic_cast<Conv2DContext&>(workerContext);

        float* wGradData = weightGradients_.GetData().data();
        const float* ctxWGradData = convContext.weightGradients.GetData().data();

        float* bGradData = biasGradients_.GetData().data();
        const float* ctxBGradData = convContext.biasGradients.GetData().data();

        const size_t wSize = weightGradients_.GetSize();
        const size_t bSize = biasGradients_.GetSize();

        // Contiguous spatial data loops ready for auto-vectorization
        for (size_t i = 0; i < wSize; i++)
        {
            wGradData[i] += ctxWGradData[i];
        }

        for (size_t i = 0; i < bSize; i++)
        {
            bGradData[i] += ctxBGradData[i];
        }
    }

    void Conv2D::UpdateTransposedWeightsCache()
    {
        const size_t numFilters = filters_;
        const size_t kernelHeight = kernelSize_.first;
        const size_t kernelWidth = kernelSize_.second;
        const size_t patchSize = kernelHeight * kernelWidth * inputChannels_;

        const float* wData = weights_.GetData().data();
        float* twData = transposedWeights_.GetData().data();

        // Done once per batch - parallel optimization is efficient here
        //#pragma omp parallel for collapse(2) schedule(static)
        for (size_t f = 0; f < numFilters; ++f)
        {
            for (size_t p = 0; p < patchSize; ++p)
            {
                twData[p * numFilters + f] = wData[f * patchSize + p];
            }
        }
    }
}