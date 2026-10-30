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

    unique_ptr<LayerContext> Conv2D::CreateContext() const
    {
        auto context = make_unique<Conv2DContext>();

        context->weightGradients = Tensor(weights_.GetShape());
        context->biasGradients = Tensor(biases_.GetShape());

        return context;
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

    vector<size_t> Conv2D::GetOutputShape(const vector<size_t>& inputShape)
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

    /*Tensor Conv2D::Run(const Tensor& input, LayerContext& context)
    {
        auto& convContext = dynamic_cast<Conv2DContext&>(context);
        const auto& inputShape = input.GetShape();

        if (inputShape.size() != 3 || inputShape[2] != inputChannels_)
            throw invalid_argument("Conv2D input shape does not match initialization.");

        convContext.input = input;

        const size_t inputHeight = inputShape[0];
        const size_t inputWidth = inputShape[1];
        const size_t inputChannels = inputShape[2];

        const size_t kernelHeight = static_cast<size_t>(kernelSize_.first);
        const size_t kernelWidth = static_cast<size_t>(kernelSize_.second);
        const size_t strideHeight = static_cast<size_t>(strides_.first);
        const size_t strideWidth = static_cast<size_t>(strides_.second);

        const vector<size_t> outputShape = GetOutputShape(inputShape);
        const size_t outputHeight = outputShape[0];
        const size_t outputWidth = outputShape[1];
        const size_t numFilters = static_cast<size_t>(filters_);

        //Safely calculate padding using ptrdiff_t to avoid unsigned underflow
        ptrdiff_t paddingTop = 0;
        ptrdiff_t paddingLeft = 0;

        if (padding_ == Padding::Same)
        {
            ptrdiff_t neededHeight = static_cast<ptrdiff_t>((outputHeight - 1) * strideHeight + kernelHeight);
            ptrdiff_t totalPaddingHeight = neededHeight > static_cast<ptrdiff_t>(inputHeight) ? neededHeight - static_cast<ptrdiff_t>(inputHeight) : 0;

            ptrdiff_t neededWidth = static_cast<ptrdiff_t>((outputWidth - 1) * strideWidth + kernelWidth);
            ptrdiff_t totalPaddingWidth = neededWidth > static_cast<ptrdiff_t>(inputWidth) ? neededWidth - static_cast<ptrdiff_t>(inputWidth) : 0;

            //Symmetric/Top-heavy distribution matching standard deep learning frameworks
            paddingTop = totalPaddingHeight / 2;
            paddingLeft = totalPaddingWidth / 2;
        }

        Tensor output(outputShape);

        //Optimized execution loops (HWC Memory Layout Order)
        for (size_t outY = 0; outY < outputHeight; outY++)
        {
            for (size_t outX = 0; outX < outputWidth; outX++)
            {
                const size_t spatialOutputOffset = (outY * outputWidth + outX) * numFilters;

                //Initialize bias values across all filters for this spatial coordinate
                for (size_t filter = 0; filter < numFilters; filter++)
                {
                    output.AtFlat(spatialOutputOffset + filter) = biases_.AtFlat(filter);
                }

                //Iterate through the sliding kernel window
                for (size_t kernelY = 0; kernelY < kernelHeight; kernelY++)
                {
                    const ptrdiff_t paddedY = static_cast<ptrdiff_t>(outY * strideHeight + kernelY);
                    if (paddedY < paddingTop) 
                        continue;

                    const size_t inY = static_cast<size_t>(paddedY - paddingTop);
                    if (inY >= inputHeight) 
                        continue;

                    for (size_t kernelX = 0; kernelX < kernelWidth; kernelX++)
                    {
                        const ptrdiff_t paddedX = static_cast<ptrdiff_t>(outX * strideWidth + kernelX);
                        if (paddedX < paddingLeft) 
                            continue;

                        const size_t inX = static_cast<size_t>(paddedX - paddingLeft);
                        if (inX >= inputWidth) 
                            continue;

                        //Compute spatial source offset once for all channels/filters
                        const size_t inputOffset = (inY * inputWidth + inX) * inputChannels;

                        //Keep filter and channel variations tight to leverage CPU L1/L2 caches
                        for (size_t filter = 0; filter < numFilters; filter++)
                        {
                            const size_t weightOffset = ((filter * kernelHeight + kernelY) * kernelWidth + kernelX) * inputChannels;
                            float sum = 0.0f;

                            for (size_t channel = 0; channel < inputChannels; channel++)
                            {
                                sum += input.AtFlat(inputOffset + channel) * weights_.AtFlat(weightOffset + channel);
                            }

                            output.AtFlat(spatialOutputOffset + filter) += sum;
                        }
                    }
                }
            }
        }

        convContext.preActivation = output;
        return activation_.Apply(activationType_, output);
    }*/

    void Conv2D::Run(const Tensor& input, Tensor& output, LayerContext& context)
    {
        auto& convContext = dynamic_cast<Conv2DContext&>(context);
        const auto& inputShape = input.GetShape();

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
        convContext.input.ReshapeAndResize(inputShape);
        std::copy_n(input.GetData().data(), input.GetSize(), convContext.input.GetData().data());

        // 2. Continuous spatial image flattening (Pure sequential speed)
        NN::Math::Im2Col(convContext.input, convContext.columnsCache, kernelSize_, strides_, padding_);

        // 3. Compute via GEMM using our perfectly cached pre-transposed weight matrix
        NN::Math::MatrixMultiplication(convContext.columnsCache, transposedWeights_, output);

        // 4. Accumulate Biases sequentially (Clean L1/L2 cache streaming)
        float* outData = output.GetData().data();
        const float* bData = biases_.GetData().data();
        const size_t totalElements = output.GetSize();

        for (size_t i = 0; i < totalElements; ++i)
        {
            outData[i] += bData[i % numFilters];
        }

        output.SetShape({ outputHeight, outputWidth, numFilters });

        // 6. Snapshot the pre-activation state safely for the backward pass
        convContext.preActivation.ReshapeAndResize(output.GetShape());
        std::copy_n(outData, totalElements, convContext.preActivation.GetData().data());

        // 7. Apply the activation function in-place directly on the destination scratch tensor
        activation_.ApplyInPlace(activationType_, output);
    }


    /*Tensor Conv2D::Backpropagate(const Tensor& gradient, LayerContext& context)
    {
        auto& convContext = dynamic_cast<Conv2DContext&>(context);

        //Calculate activation derivative
        Tensor activationGradient = activation_.ApplyDerivative(activationType_, convContext.preActivation, gradient);

        const auto& inputShape = convContext.input.GetShape();
        const auto outputShape = GetOutputShape(inputShape);

        const size_t inputHeight = inputShape[0];
        const size_t inputWidth = inputShape[1];
        const size_t inputChannels = inputShape[2];

        const size_t outputHeight = outputShape[0];
        const size_t outputWidth = outputShape[1];
        const size_t numFilters = static_cast<size_t>(filters_);

        const size_t kernelHeight = static_cast<size_t>(kernelSize_.first);
        const size_t kernelWidth = static_cast<size_t>(kernelSize_.second);
        const size_t strideHeight = static_cast<size_t>(strides_.first);
        const size_t strideWidth = static_cast<size_t>(strides_.second);


        //Safely calculate padding bounds using signed integers to avoid underflow
        ptrdiff_t paddingTop = 0;
        ptrdiff_t paddingLeft = 0;

        if (padding_ == Padding::Same)
        {
            ptrdiff_t neededHeight = static_cast<ptrdiff_t>((outputHeight - 1) * strideHeight + kernelHeight);
            ptrdiff_t totalPaddingHeight = neededHeight > static_cast<ptrdiff_t>(inputHeight) ? neededHeight - static_cast<ptrdiff_t>(inputHeight) : 0;

            ptrdiff_t neededWidth = static_cast<ptrdiff_t>((outputWidth - 1) * strideWidth + kernelWidth);
            ptrdiff_t totalPaddingWidth = neededWidth > static_cast<ptrdiff_t>(inputWidth) ? neededWidth - static_cast<ptrdiff_t>(inputWidth) : 0;

            paddingTop = totalPaddingHeight / 2;
            paddingLeft = totalPaddingWidth / 2;
        }

        Tensor inputGradient(inputShape);

        //Cache-Optimized Spatial Loop Execution
        for (size_t outY = 0; outY < outputHeight; outY++)
        {
            for (size_t outX = 0; outX < outputWidth; outX++)
            {
                const size_t spatialOutputOffset = (outY * outputWidth + outX) * numFilters;

                //First pass: Accumulate biases efficiently and cache output gradients locally
                for (size_t filter = 0; filter < numFilters; filter++)
                {
                    const float outputGrad = activationGradient.AtFlat(spatialOutputOffset + filter);
                    convContext.biasGradients.AtFlat(filter) += outputGrad;
                }

                //Slide kernel window over spatial input coordinate
                for (size_t kernelY = 0; kernelY < kernelHeight; kernelY++)
                {
                    const ptrdiff_t paddedY = static_cast<ptrdiff_t>(outY * strideHeight + kernelY);
                    if (paddedY < paddingTop) continue;

                    const size_t inY = static_cast<size_t>(paddedY - paddingTop);
                    if (inY >= inputHeight) continue;

                    for (size_t kernelX = 0; kernelX < kernelWidth; kernelX++)
                    {
                        const ptrdiff_t paddedX = static_cast<ptrdiff_t>(outX * strideWidth + kernelX);
                        if (paddedX < paddingLeft) continue;

                        const size_t inX = static_cast<size_t>(paddedX - paddingLeft);
                        if (inX >= inputWidth) continue;

                        const size_t inputOffset = (inY * inputWidth + inX) * inputChannels;

                        //Tight innermost loops: Maximize memory locality for filters and channels
                        for (size_t filter = 0; filter < numFilters; filter++)
                        {
                            const float outputGrad = activationGradient.AtFlat(spatialOutputOffset + filter);
                            const size_t weightOffset = ((filter * kernelHeight + kernelY) * kernelWidth + kernelX) * inputChannels;

                            for (size_t channel = 0; channel < inputChannels; channel++)
                            {
                                const size_t inputIndex = inputOffset + channel;
                                const size_t weightIndex = weightOffset + channel;
                                const float inputVal = convContext.input.AtFlat(inputIndex);

                                //Accumulate weight gradient
                                convContext.weightGradients.AtFlat(weightIndex) += outputGrad * inputVal;

                                //Propagate error backward to the previous layer's input data
                                inputGradient.AtFlat(inputIndex) += weights_.AtFlat(weightIndex) * outputGrad;
                            }
                        }
                    }
                }
            }
        }

        return inputGradient;
    }*/

    void Conv2D::Backpropagate(const Tensor& input, Tensor& output, LayerContext& context)
    {
        auto& convContext = dynamic_cast<Conv2DContext&>(context);

        // 1. Calculate the activation derivative.
        activation_.ApplyDerivativeInPlace(activationType_, convContext.preActivation, input, convContext.activationGradient);

        const auto& inputShape = convContext.input.GetShape();
        const auto outputShape = convContext.preActivation.GetShape();

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
        NN::Math::Im2Col(convContext.input, convContext.columnsCache, kernelSize_, strides_, padding_);

        // Extract raw pointers for data-heavy operations
        const float* outputGradData = convContext.activationGradient.GetData().data();

        float* biasGradData = convContext.biasGradients.GetData().data();
        float* weightGradData = convContext.weightGradients.GetData().data();

        // 3. Calculate bias gradients sequentially
        for (size_t position = 0; position < outputPositions; position++)
        {
            const size_t outputOffset = position * numFilters;
            for (size_t filter = 0; filter < numFilters; filter++)
            {
                biasGradData[filter] += outputGradData[outputOffset + filter];
            }
        }

        // 5. Calculate weight gradients via Transposed Gradient Matrix Multiply.
        convContext.transposedGradCache.ReshapeAndResize({ numFilters, outputPositions });
        float* transposedGradData = convContext.transposedGradCache.GetData().data();


        //#pragma omp parallel for collapse(2) schedule(static)
        for (size_t position = 0; position < outputPositions; position++)
        {
            for (size_t filter = 0; filter < numFilters; filter++)
            {
                transposedGradData[filter * outputPositions + position] = outputGradData[position * numFilters + filter];
            }
        }

        Tensor weightGradient({ numFilters, patchSize });
        NN::Math::MatrixMultiplication(convContext.transposedGradCache, convContext.columnsCache, weightGradient);


        // Accumulate weight gradient map natively
        float* wGradDest = convContext.weightGradients.GetData().data();
        const float* calculatedWeightGradData = weightGradient.GetData().data();
        const size_t wGradSize = weightGradient.GetSize();

        
        for (size_t i = 0; i < wGradSize; i++)
        {
            wGradDest[i] += calculatedWeightGradData[i];
        }

        // 6. Calculate gradients for every flattened input patch.
        Tensor flatWeightsView({ numFilters, patchSize });
        flatWeightsView.GetData() = weights_.GetData();

        convContext.activationGradient.SetShape({ outputPositions, numFilters });

        // Transpose weight tracking multiplication to retrieve the input patch gradients
        // flatActivationGradient shape: {outputPositions, numFilters}
        // flatWeights shape:            {numFilters, patchSize}
        // columnGradient shape:         {outputPositions, patchSize}
        NN::Math::MatrixMultiplication(convContext.activationGradient, flatWeightsView, convContext.columnGradCache);

        convContext.activationGradient.SetShape({ outputHeight, outputWidth, numFilters });

        // 7. Scatter patch gradients back into the original HWC input using the optimized Col2Im helper.
        // We pass the recycled 'gradient' buffer directly into the helper function, enabling it to steal
        // the memory context and perform an allocation-free array reduction.
        NN::Math::Col2Im(convContext.columnGradCache, output, inputShape, kernelSize_, strides_, padding_);
    }

    void Conv2D::ApplyGradients(Optimizer& optimizer, int batchSize)
    {
        optimizer.Apply(weights_, weightGradients_, batchSize);
        optimizer.Apply(biases_, biasGradients_, batchSize);

        UpdateTransposedWeightsCache();
    }

    void Conv2D::ResetGradients(LayerContext& context)
    {
        auto& convContext = dynamic_cast<Conv2DContext&>(context);

        // Reuse existing allocated memory blocks and clear elements to 0.0f in-place
        std::fill(convContext.weightGradients.GetData().begin(), convContext.weightGradients.GetData().end(), 0.0f);
        std::fill(convContext.biasGradients.GetData().begin(), convContext.biasGradients.GetData().end(), 0.0f);

        std::fill(weightGradients_.GetData().begin(), weightGradients_.GetData().end(), 0.0f);
        std::fill(biasGradients_.GetData().begin(), biasGradients_.GetData().end(), 0.0f);
    }

    void Conv2D::AccumulateGradients(LayerContext& context)
    {
        auto& convContext = dynamic_cast<Conv2DContext&>(context);

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