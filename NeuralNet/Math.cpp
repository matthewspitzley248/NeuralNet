#include <NeuralNet/Math.h>

namespace NN
{
    void Math::MatrixMultiplication(const Tensor& A, const Tensor& B, Tensor& dest)
    {
        const auto& aShape = A.GetShape();
        const auto& bShape = B.GetShape();

        if (aShape.size() != 2 || bShape.size() != 2)
        {
            throw std::invalid_argument("MatrixMultiplication requires 2D tensors.");
        }

        const size_t aRows = aShape[0];
        const size_t aCols = aShape[1];
        const size_t bRows = bShape[0];
        const size_t bCols = bShape[1];

        if (aCols != bRows)
        {
            throw std::invalid_argument("Matrix multiplication invalid: Columns of A (" + std::to_string(aCols) +
                ") must equal Rows of B (" + std::to_string(bRows) + ").");
        }

        //ZERO ALLOCATION: Adjust scratch target metadata boundaries in-place
        dest.ReshapeAndResize({ aRows, bCols });

        const auto& aData = A.GetData();
        const auto& bData = B.GetData();
        auto& destData = dest.GetData();

        // Safe in-place memory clear prior to GEMM accumulation
        std::fill(destData.begin(), destData.end(), 0.0f);

        constexpr size_t BLOCK_SIZE = 32;

        // Cache-blocked sequential matrix multiply execution loop
        for (size_t iBlock = 0; iBlock < aRows; iBlock += BLOCK_SIZE)
        {
            const size_t iEnd = std::min(iBlock + BLOCK_SIZE, aRows);
            for (size_t jBlock = 0; jBlock < bCols; jBlock += BLOCK_SIZE)
            {
                const size_t jEnd = std::min(jBlock + BLOCK_SIZE, bCols);
                for (size_t kBlock = 0; kBlock < aCols; kBlock += BLOCK_SIZE)
                {
                    const size_t kEnd = std::min(kBlock + BLOCK_SIZE, aCols);

                    for (size_t i = iBlock; i < iEnd; i++)
                    {
                        const size_t aRowOffset = i * aCols;
                        const size_t cRowOffset = i * bCols;

                        for (size_t k = kBlock; k < kEnd; k++)
                        {
                            const float aValue = aData[aRowOffset + k];
                            const size_t bRowOffset = k * bCols;

                            for (size_t j = jBlock; j < jEnd; j++)
                            {
                                destData[cRowOffset + j] += aValue * bData[bRowOffset + j];
                            }
                        }
                    }
                }
            }
        }
    }

    void Math::Im2Col(const Tensor& input, Tensor& dest,
        std::pair<size_t, size_t> kernelSize,
        std::pair<size_t, size_t> strides,
        Padding padding)
    {
        const std::vector<size_t>& inputShape = input.GetShape();

        const size_t kernelHeight = kernelSize.first;
        const size_t kernelWidth = kernelSize.second;
        const size_t strideHeight = strides.first;
        const size_t strideWidth = strides.second;

        if (inputShape.size() != 3)
            throw std::invalid_argument("Im2Col requires a 3D HWC input tensor.");

        const size_t inputHeight = inputShape[0];
        const size_t inputWidth = inputShape[1];
        const size_t inputChannels = inputShape[2];

        size_t outputHeight;
        size_t outputWidth;

        if (padding == Padding::Same)
        {
            outputHeight = (inputHeight + strideHeight - 1) / strideHeight;
            outputWidth = (inputWidth + strideWidth - 1) / strideWidth;
        }
        else // Padding::Valid
        {
            if (inputHeight < kernelHeight || inputWidth < kernelWidth)
                throw std::invalid_argument("Im2Col kernel is larger than the input with Valid padding.");

            outputHeight = (inputHeight - kernelHeight) / strideHeight + 1;
            outputWidth = (inputWidth - kernelWidth) / strideWidth + 1;
        }

        size_t padTop = 0;
        size_t padLeft = 0;

        if (padding == Padding::Same)
        {
            size_t requiredHeight = (outputHeight - 1) * strideHeight + kernelHeight;
            size_t requiredWidth = (outputWidth - 1) * strideWidth + kernelWidth;

            size_t totalPadHeight = (requiredHeight > inputHeight) ? (requiredHeight - inputHeight) : 0;
            size_t totalPadWidth = (requiredWidth > inputWidth) ? (requiredWidth - inputWidth) : 0;

            padTop = totalPadHeight / 2;
            padLeft = totalPadWidth / 2;
        }

        const size_t outputPositions = outputHeight * outputWidth;
        const size_t patchSize = kernelHeight * kernelWidth * inputChannels;

        //ZERO ALLOCATION: Adjust pre-allocated columns cache size in-place
        dest.ReshapeAndResize({ outputPositions, patchSize });

        const auto& inputData = input.GetData();
        auto& columnData = dest.GetData();

        // Clear the destination matrix to ensure padding fields are properly zeroed out
        std::fill(columnData.begin(), columnData.end(), 0.0f);

        const ptrdiff_t limitY = static_cast<ptrdiff_t>(inputHeight);
        const ptrdiff_t limitX = static_cast<ptrdiff_t>(inputWidth);

        // Sequential spatial pixel parsing stream loop
        for (size_t outY = 0; outY < outputHeight; ++outY)
        {
            const size_t baseInputY = outY * strideHeight;
            for (size_t outX = 0; outX < outputWidth; ++outX)
            {
                const size_t baseInputX = outX * strideWidth;
                const size_t row = outY * outputWidth + outX;
                const size_t patchStart = row * patchSize;

                for (size_t kernelY = 0; kernelY < kernelHeight; ++kernelY)
                {
                    const ptrdiff_t inputY = static_cast<ptrdiff_t>(baseInputY + kernelY) - static_cast<ptrdiff_t>(padTop);
                    const size_t kernelY_offset = kernelY * kernelWidth;

                    if (inputY >= 0 && inputY < limitY)
                    {
                        const size_t inputY_stride = static_cast<size_t>(inputY) * inputWidth;

                        for (size_t kernelWidthIdx = 0; kernelWidthIdx < kernelWidth; ++kernelWidthIdx)
                        {
                            const ptrdiff_t inputX = static_cast<ptrdiff_t>(baseInputX + kernelWidthIdx) - static_cast<ptrdiff_t>(padLeft);

                            if (inputX >= 0 && inputX < limitX)
                            {
                                const size_t patchIndex = patchStart + (kernelY_offset + kernelWidthIdx) * inputChannels;
                                const size_t inputIndex = (inputY_stride + static_cast<size_t>(inputX)) * inputChannels;

                                // Fast continuous data block transfer operation
                                std::copy_n(inputData.data() + inputIndex, inputChannels, columnData.data() + patchIndex);
                            }
                        }
                    }
                }
            }
        }
    }

    void Math::Col2Im(const Tensor& columnGradient, Tensor& dest,
        const std::vector<size_t>& inputShape,
        std::pair<size_t, size_t> kernelSize,
        std::pair<size_t, size_t> strides,
        Padding padding)
    {
        const size_t inputHeight = inputShape[0];
        const size_t inputWidth = inputShape[1];
        const size_t inputChannels = inputShape[2];

        const size_t kernelHeight = kernelSize.first;
        const size_t kernelWidth = kernelSize.second;
        const size_t strideHeight = strides.first;
        const size_t strideWidth = strides.second;

        size_t outputHeight;
        size_t outputWidth;

        if (padding == Padding::Same)
        {
            outputHeight = (inputHeight + strideHeight - 1) / strideHeight;
            outputWidth = (inputWidth + strideWidth - 1) / strideWidth;
        }
        else // Padding::Valid
        {
            outputHeight = (inputHeight - kernelHeight) / strideHeight + 1;
            outputWidth = (inputWidth - kernelWidth) / strideWidth + 1;
        }

        size_t padTop = 0;
        size_t padLeft = 0;

        if (padding == Padding::Same)
        {
            size_t requiredHeight = (outputHeight - 1) * strideHeight + kernelHeight;
            size_t requiredWidth = (outputWidth - 1) * strideWidth + kernelWidth;

            size_t totalPadHeight = (requiredHeight > inputHeight) ? (requiredHeight - inputHeight) : 0;
            size_t totalPadWidth = (requiredWidth > inputWidth) ? (requiredWidth - inputWidth) : 0;

            padTop = totalPadHeight / 2;
            padLeft = totalPadWidth / 2;
        }

        const size_t patchSize = kernelHeight * kernelWidth * inputChannels;

        //ZERO ALLOCATION: Format the pre-allocated scratch destination tensor view directly
        dest.ReshapeAndResize(inputShape);

        const auto& columnGradData = columnGradient.GetData();
        auto& inputGradData = dest.GetData();

        // Wipe target gradient canvas clean
        std::fill(inputGradData.begin(), inputGradData.end(), 0.0f);

        // Lock-free row-isolated error accumulation loop sequence
        for (size_t inY = 0; inY < inputHeight; ++inY)
        {
            const size_t inputYOffset = inY * inputWidth;
            for (size_t outY = 0; outY < outputHeight; ++outY)
            {
                const ptrdiff_t kernelY = static_cast<ptrdiff_t>(inY + padTop) - static_cast<ptrdiff_t>(outY * strideHeight);
                if (kernelY < 0 || kernelY >= static_cast<ptrdiff_t>(kernelHeight)) continue;

                const size_t outYOffset = outY * outputWidth;
                const size_t kernelYOffset = static_cast<size_t>(kernelY) * kernelWidth;


                for (size_t inX = 0; inX < inputWidth; ++inX)
                {
                    const size_t inputOffset = (inputYOffset + inX) * inputChannels;
                    for (size_t outX = 0; outX < outputWidth; ++outX)
                    {
                        const ptrdiff_t kernelX = static_cast<ptrdiff_t>(inX + padLeft) - static_cast<ptrdiff_t>(outX * strideWidth);
                        if (kernelX < 0 || kernelX >= static_cast<ptrdiff_t>(kernelWidth)) continue;
                        const size_t position = outYOffset + outX;
                        const size_t patchOffset = position * patchSize;
                        const size_t kernelOffset = (kernelYOffset + static_cast<size_t>(kernelX)) * inputChannels;
                        const size_t srcIndex = patchOffset + kernelOffset;
                        for (size_t channel = 0; channel < inputChannels; ++channel)
                        {
                            inputGradData[inputOffset + channel] += columnGradData[srcIndex + channel];
                        }
                    }
                }
            }
        }
    }
}