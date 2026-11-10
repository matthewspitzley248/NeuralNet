#include <NeuralNet/Layers/MaxPool2D.h>

namespace NN::Layers
{
    MaxPool2D::MaxPool2D(std::pair<size_t, size_t> poolSize, std::pair<size_t, size_t> strides, Padding padding)
        : poolSize_(poolSize),
        strides_(strides),
        padding_(padding)
    {
        // If stride is unset (0,0), standard framework practice defaults it to match the pool size
        if (strides_.first == 0 || strides_.second == 0)
        {
            strides_ = poolSize_;
        }
    }

    unique_ptr<ImageContext> MaxPool2D::CreateImageContext() const
    {
        return std::make_unique<MaxPool2DImageContext>();
    }

    void MaxPool2D::Initialize(std::mt19937& generator, const std::vector<size_t>& inputShape)
    {
        if (inputShape.size() != 3)
            throw std::invalid_argument("MaxPool2D expects a 3D input shape (height, width, channels).");

        // MaxPool layers do not contain trainable weights or biases.
        // We only call GetOutputShape to validate dimensions ahead of time.
        GetOutputShape(inputShape);

        inputShape_ = inputShape;
    }

    std::vector<size_t> MaxPool2D::GetOutputShape(const std::vector<size_t>& inputShape) const
    {
        if (inputShape.size() != 3)
            throw std::invalid_argument("MaxPool2D expects input shape (height, width, channels).");

        const size_t inputHeight = inputShape[0];
        const size_t inputWidth = inputShape[1];
        const size_t channels = inputShape[2];

        size_t outputHeight;
        size_t outputWidth;

        if (padding_ == Padding::Same)
        {
            outputHeight = (inputHeight + strides_.first - 1) / strides_.first;
            outputWidth = (inputWidth + strides_.second - 1) / strides_.second;
        }
        else // Padding::Valid
        {
            if (inputHeight < poolSize_.first || inputWidth < poolSize_.second)
                throw std::invalid_argument("MaxPool2D pool size is larger than the input window.");

            outputHeight = (inputHeight - poolSize_.first) / strides_.first + 1;
            outputWidth = (inputWidth - poolSize_.second) / strides_.second + 1;
        }

        return { outputHeight, outputWidth, channels };
    }

    void MaxPool2D::Run(Tensor& data)
    {
        const size_t inputHeight = inputShape_[0];
        const size_t inputWidth = inputShape_[1];
        const size_t channels = inputShape_[2];

        const std::vector<size_t> outputShape = GetOutputShape(inputShape_);
        const size_t outputHeight = outputShape[0];
        const size_t outputWidth = outputShape[1];

        //Maybe can just calculate the size and set it to read and write to the same vector if it doesnt change it?
        Tensor output(outputShape);

        output.ReshapeAndResize(outputShape);
        const size_t outputSize = output.GetSize();

        const float* inData = data.GetData().data();
        float* outData = output.GetData().data();

        ptrdiff_t padTop = 0;
        ptrdiff_t padLeft = 0;

        if (padding_ == Padding::Same)
        {
            ptrdiff_t neededHeight = static_cast<ptrdiff_t>((outputHeight - 1) * strides_.first + poolSize_.first);
            ptrdiff_t totalPadY = neededHeight > static_cast<ptrdiff_t>(inputHeight) ? neededHeight - static_cast<ptrdiff_t>(inputHeight) : 0;

            ptrdiff_t neededWidth = static_cast<ptrdiff_t>((outputWidth - 1) * strides_.second + poolSize_.second);
            ptrdiff_t totalPadX = neededWidth > static_cast<ptrdiff_t>(inputWidth) ? neededWidth - static_cast<ptrdiff_t>(inputWidth) : 0;

            padTop = totalPadY / 2;
            padLeft = totalPadX / 2;
        }

        // Each spatial HxW output slot updates its unique flat offset space safely.
        for (size_t outY = 0; outY < outputHeight; ++outY)
        {
            for (size_t outX = 0; outX < outputWidth; ++outX)
            {
                const size_t outSpatialOffset = (outY * outputWidth + outX) * channels;

                // Pre-calculate spatial window visibility ranges once per coordinate block.
                // This hoists structural branching out of the performance-critical inner loops!
                const ptrdiff_t startPaddedY = static_cast<ptrdiff_t>(outY * strides_.first) - padTop;
                const size_t poolStartY = static_cast<size_t>(std::max(ptrdiff_t(0), -startPaddedY));
                const size_t poolEndY = static_cast<size_t>(std::min(static_cast<ptrdiff_t>(poolSize_.first), static_cast<ptrdiff_t>(inputHeight) - startPaddedY));

                const ptrdiff_t startPaddedX = static_cast<ptrdiff_t>(outX * strides_.second) - padLeft;
                const size_t poolStartX = static_cast<size_t>(std::max(ptrdiff_t(0), -startPaddedX));
                const size_t poolEndX = static_cast<size_t>(std::min(static_cast<ptrdiff_t>(poolSize_.second), static_cast<ptrdiff_t>(inputWidth) - startPaddedX));

                // Run channel operations natively. Inner calculations map continuously.
                for (size_t c = 0; c < channels; ++c)
                {
                    const size_t outLinearIndex = outSpatialOffset + c;

                    float maxVal = -std::numeric_limits<float>::infinity();
                    size_t maxFlatIdx = 0;

                    for (size_t poolY = poolStartY; poolY < poolEndY; ++poolY)
                    {
                        const size_t inYOffset = static_cast<size_t>(startPaddedY + poolY) * inputWidth;

                        for (size_t poolX = poolStartX; poolX < poolEndX; ++poolX)
                        {
                            const size_t inLinearIdx = (inYOffset + static_cast<size_t>(startPaddedX + poolX)) * channels + c;
                            const float val = inData[inLinearIdx];

                            if (val > maxVal)
                            {
                                maxVal = val;
                                maxFlatIdx = inLinearIdx;
                            }
                        }
                    }

                    outData[outLinearIndex] = (maxVal == -std::numeric_limits<float>::infinity()) ? 0.0f : maxVal;
                }
            }
        }

        //set input
        data = std::move(output);
    }

    void MaxPool2D::Run(Tensor& data, WorkerContext& workerContext, ImageContext& imageContext)
    {
        auto& maxPoolImageContext = dynamic_cast<MaxPool2DImageContext&>(imageContext);

        const size_t inputHeight = inputShape_[0];
        const size_t inputWidth = inputShape_[1];
        const size_t channels = inputShape_[2];

        const std::vector<size_t> outputShape = GetOutputShape(inputShape_);
        const size_t outputHeight = outputShape[0];
        const size_t outputWidth = outputShape[1];

        //Maybe can just calculate the size and set it to read and write to the same vector if it doesnt change it?
        Tensor output(outputShape);

        output.ReshapeAndResize(outputShape);
        const size_t outputSize = output.GetSize();

        maxPoolImageContext.maxIndices.resize(outputSize);

        const float* inData = data.GetData().data();
        float* outData = output.GetData().data();
        size_t* maxIdxData = maxPoolImageContext.maxIndices.data();

        ptrdiff_t padTop = 0;
        ptrdiff_t padLeft = 0;

        if (padding_ == Padding::Same)
        {
            ptrdiff_t neededHeight = static_cast<ptrdiff_t>((outputHeight - 1) * strides_.first + poolSize_.first);
            ptrdiff_t totalPadY = neededHeight > static_cast<ptrdiff_t>(inputHeight) ? neededHeight - static_cast<ptrdiff_t>(inputHeight) : 0;

            ptrdiff_t neededWidth = static_cast<ptrdiff_t>((outputWidth - 1) * strides_.second + poolSize_.second);
            ptrdiff_t totalPadX = neededWidth > static_cast<ptrdiff_t>(inputWidth) ? neededWidth - static_cast<ptrdiff_t>(inputWidth) : 0;

            padTop = totalPadY / 2;
            padLeft = totalPadX / 2;
        }

        // Each spatial HxW output slot updates its unique flat offset space safely.
        for (size_t outY = 0; outY < outputHeight; ++outY)
        {
            for (size_t outX = 0; outX < outputWidth; ++outX)
            {
                const size_t outSpatialOffset = (outY * outputWidth + outX) * channels;

                // Pre-calculate spatial window visibility ranges once per coordinate block.
                // This hoists structural branching out of the performance-critical inner loops!
                const ptrdiff_t startPaddedY = static_cast<ptrdiff_t>(outY * strides_.first) - padTop;
                const size_t poolStartY = static_cast<size_t>(std::max(ptrdiff_t(0), -startPaddedY));
                const size_t poolEndY = static_cast<size_t>(std::min(static_cast<ptrdiff_t>(poolSize_.first), static_cast<ptrdiff_t>(inputHeight) - startPaddedY));

                const ptrdiff_t startPaddedX = static_cast<ptrdiff_t>(outX * strides_.second) - padLeft;
                const size_t poolStartX = static_cast<size_t>(std::max(ptrdiff_t(0), -startPaddedX));
                const size_t poolEndX = static_cast<size_t>(std::min(static_cast<ptrdiff_t>(poolSize_.second), static_cast<ptrdiff_t>(inputWidth) - startPaddedX));

                // Run channel operations natively. Inner calculations map continuously.
                for (size_t c = 0; c < channels; ++c)
                {
                    const size_t outLinearIndex = outSpatialOffset + c;

                    float maxVal = -std::numeric_limits<float>::infinity();
                    size_t maxFlatIdx = 0;

                    for (size_t poolY = poolStartY; poolY < poolEndY; ++poolY)
                    {
                        const size_t inYOffset = static_cast<size_t>(startPaddedY + poolY) * inputWidth;

                        for (size_t poolX = poolStartX; poolX < poolEndX; ++poolX)
                        {
                            const size_t inLinearIdx = (inYOffset + static_cast<size_t>(startPaddedX + poolX)) * channels + c;
                            const float val = inData[inLinearIdx];

                            if (val > maxVal)
                            {
                                maxVal = val;
                                maxFlatIdx = inLinearIdx;
                            }
                        }
                    }

                    outData[outLinearIndex] = (maxVal == -std::numeric_limits<float>::infinity()) ? 0.0f : maxVal;
                    maxIdxData[outLinearIndex] = maxFlatIdx;
                }
            }
        }

        //set input
        data = std::move(output);
    }

    void MaxPool2D::Backpropagate(Tensor& gradient, WorkerContext& workerContext, ImageContext& imageContext)
    {
        auto& maxPoolImageContext = dynamic_cast<MaxPool2DImageContext&>(imageContext);

        const size_t gradSize = gradient.GetSize();
        const size_t totalInputElements = inputShape_[0] * inputShape_[1] * inputShape_[2];

        Tensor outGradient(inputShape_);

        // 2. Extract vector references securely
        const auto& inGradData = gradient.GetData(); // 'source' holds incoming gradients from the next layer
        auto& outGradData = outGradient.GetData();   // 'dest' is our destination scratch canvas
        const auto& maxIdxData = maxPoolImageContext.maxIndices;

        // 3. Fast contiguous buffer zeroing
        std::fill(outGradData.begin(), outGradData.end(), 0.0f);

        // 4. O(N) Sequential Scatter:
        // Simply route the incoming gradient error elements directly to their tracked flat destination positions.
        // Highly vectorizable and cache-friendly.
        for (size_t i = 0; i < gradSize; ++i)
        {
            const size_t targetFlatInputIndex = maxIdxData[i];
            outGradData[targetFlatInputIndex] += inGradData[i];
        }

        gradient = std::move(outGradient);
    }
}